"""Chẩn đoán FAISS CLIP index đã build sẵn.

Hai lớp kiểm tra:
1) mapping: row của FAISS -> keyframe_id từ id_map.json (chỉ đọc, không suy đoán).
2) semantic (--semantic): text caption -> ảnh CLIP.

   Caption trong DB là tiếng Việt, nên dùng clip-ViT-B-32-multilingual-v1
   (sentence-transformers) để encode -- ĐÚNG model mà backend production
   dùng (xem backend/rag/pipeline.py::ClipQueryEmbedder), không dùng thẳng
   text encoder gốc của OpenAI CLIP (chỉ hiểu tiếng Anh, sẽ cho hit rate
   thấp giả tạo dù index/mapping hoàn toàn đúng). Model multilingual này
   được distill để giữ đúng không gian embedding ảnh của OpenAI CLIP
   ViT-B/32 (khớp checkpoint "BTC provided" dùng lúc build faiss_clip.index).

   Không phải bằng chứng tuyệt đối cho mapping vì caption không phải vector
   ảnh gốc, nhưng đủ tin cậy để xác nhận -- nếu hit thấp bất thường dù
   verify_clip_self_consistency.py đã PASS, khả năng cao là do caption
   diễn đạt khác với nội dung ảnh hơn là do mapping sai.

Dùng:
    python -m pipelines.verify_clip_mapping --semantic
    python -m pipelines.verify_clip_mapping --semantic --sample 20 --top-k 10
"""

import argparse
import os
import sqlite3

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from pipelines.clip_mapping import load_clip_id_map
from pipelines.config import DB_PATH, INDEX_ROOT

MODEL_NAME = "clip-ViT-B-32-multilingual-v1"


def _paths(cli_index=None, cli_map=None):
    """Trả về (index_path, map_path), cho phép override qua CLI hoặc env var,
    mặc định dùng data/index/faiss_clip.index + id_map.json."""
    index_path = cli_index or os.environ.get("AIC_CLIP_INDEX_PATH", f"{INDEX_ROOT}/faiss_clip.index")
    map_path = cli_map or os.environ.get("AIC_CLIP_ID_MAP_PATH", f"{INDEX_ROOT}/id_map.json")
    missing = [p for p in (index_path, map_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Không tìm thấy artifact: " + ", ".join(missing) +
            ". Hãy truyền --index và --id-map bằng đường dẫn tuyệt đối."
        )
    return index_path, map_path


def check_mapping(index_path: str, map_path: str) -> np.ndarray:
    """Kiểm tra cơ bản: số dòng khớp, dim đúng 512. Trả về id_map để dùng tiếp
    cho semantic test (nếu có)."""
    index = faiss.read_index(index_path)
    id_map = load_clip_id_map(map_path)

    if index.ntotal != len(id_map):
        raise ValueError(f"FAISS ntotal={index.ntotal} nhưng id_map có {len(id_map)} dòng")
    if index.d != 512:
        raise ValueError(f"Index dim={index.d}, cần 512 cho {MODEL_NAME}")

    print(f"index={index_path}")
    print(f"id_map={map_path}")
    print(f"ntotal={index.ntotal}, dim={index.d}, "
          f"map_first={id_map[:5].tolist()}, map_last={id_map[-5:].tolist()}")
    print("Mapping row->keyframe_id đã đọc thành công; chưa kết luận semantic từ caption.\n")

    return index, id_map


def run_semantic_test(index, id_map: np.ndarray, sample: int, top_k: int):
    """Encode caption tiếng Việt bằng clip-ViT-B-32-multilingual-v1 (đúng
    model backend production dùng), query vào index, so rank với
    expected_row theo id_map.

    In thêm video_id của expected_row và của top-1 để phân biệt 2 tình
    huống: (a) top-1 khác video hoàn toàn -> nghi ngờ semantic thật sự yếu
    hoặc mapping sai; (b) top-1 CÙNG video_id với expected -> gần như chắc
    chắn là false-negative do 2 keyframe liền kề trong cùng video trông
    gần giống hệt nhau (rất bình thường với keyframe sampling dày), không
    phải bug.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT keyframe_id, caption_text FROM captions "
        "WHERE caption_text IS NOT NULL AND caption_text != '' "
        "ORDER BY RANDOM() LIMIT ?", (sample,)
    ).fetchall()

    def video_of(keyframe_id: int):
        row = conn.execute(
            "SELECT video_id FROM keyframes WHERE id = ?", (keyframe_id,)
        ).fetchone()
        return row["video_id"] if row else None

    device = "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    semantic_hits = 0
    same_video_near_miss = 0
    for r in rows:
        keyframe_id = int(r["keyframe_id"])
        matches = np.flatnonzero(id_map == keyframe_id)
        if len(matches) == 0:
            print(f"keyframe_id={keyframe_id}: NOT_IN_MAP")
            continue
        expected_row = int(matches[0])

        q = model.encode(r["caption_text"], convert_to_numpy=True, normalize_embeddings=True)
        q = q.reshape(1, -1).astype("float32")

        scores, idxs = index.search(q, max(top_k, 1))
        top_rows = idxs[0].tolist()
        top_scores = scores[0].tolist()
        expected_score = float(np.dot(q[0], index.reconstruct(expected_row)))
        found = expected_row in top_rows
        semantic_hits += found
        rank = top_rows.index(expected_row) + 1 if found else None
        preview = [(int(row), int(id_map[row]), round(float(score), 4))
                   for row, score in zip(top_rows[:5], top_scores[:5])]

        expected_video = video_of(keyframe_id)
        top1_keyframe_id = int(id_map[top_rows[0]])
        top1_video = video_of(top1_keyframe_id)
        near_miss = (not found) and (top1_video == expected_video)
        same_video_near_miss += near_miss

        print(
            f"keyframe_id={keyframe_id} (video={expected_video}), expected_row={expected_row}, "
            f"expected_score={expected_score:.4f}, rank={rank}, top5={preview}\n"
            f"  top1_video={top1_video}"
            f"{'  <- CÙNG VIDEO với expected (near-miss do frame liền kề, không phải bug)' if near_miss else ''}\n"
            f"  caption={r['caption_text'][:100]!r}"
        )

    conn.close()
    print(f"\nsemantic caption hit@{top_k}: {semantic_hits}/{len(rows)}")
    print(f"trong số miss, số case top1 CÙNG video (near-miss do frame liền kề): "
          f"{same_video_near_miss}/{len(rows) - semantic_hits}")
    print("Lưu ý: hit thấp chỉ chứng minh caption-text không kéo đúng ảnh vào top-k; "
          "không tự chứng minh id_map sai. Để verify mapping tuyệt đối cần encode lại "
          "ảnh keyframe bằng đúng image encoder/preprocessing đã tạo index.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--semantic", action="store_true",
                     help="Chạy thêm test caption->image; không dùng để kết luận mapping")
    ap.add_argument("--index", default=None, help="Đường dẫn tới faiss_clip.index")
    ap.add_argument("--id-map", default=None, help="Đường dẫn tới id_map.json hoặc clip_id_map.npy")
    args = ap.parse_args()

    index_path, map_path = _paths(args.index, args.id_map)
    index, id_map = check_mapping(index_path, map_path)

    if not args.semantic:
        print("PASS: mapping-only verification hoàn tất; không chạy semantic caption test.")
        return

    run_semantic_test(index, id_map, args.sample, args.top_k)


if __name__ == "__main__":
    main()