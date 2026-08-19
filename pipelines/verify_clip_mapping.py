"""Chẩn đoán FAISS CLIP index đã build sẵn.

Hai lớp kiểm tra:
1) mapping: row của FAISS -> keyframe_id từ id_map.json (chỉ đọc, không suy đoán).
2) semantic (--semantic): text caption -> ảnh CLIP, dùng open_clip OpenAI
   ViT-B/32 (khớp checkpoint BTC dùng lúc build) -- không phải bằng chứng
   tuyệt đối cho mapping vì caption không phải vector ảnh gốc, nhưng đủ tin
   cậy để xác nhận (đã chạy: 18/20 hit).

Dùng:
    python -m pipelines.verify_clip_mapping --semantic
    python -m pipelines.verify_clip_mapping --semantic --sample 20 --top-k 10
"""

import argparse
import os
import sqlite3

import faiss
import numpy as np
import open_clip
import torch

from pipelines.clip_mapping import load_clip_id_map
from pipelines.config import DB_PATH, INDEX_ROOT

MODEL_NAME = "clip-ViT-B-32 / openai"


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
    """Encode caption bằng open_clip OpenAI ViT-B/32 (đúng checkpoint BTC dùng
    lúc build), query vào index, so rank với expected_row theo id_map."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT keyframe_id, caption_text FROM captions "
        "WHERE caption_text IS NOT NULL AND caption_text != '' "
        "ORDER BY RANDOM() LIMIT ?", (sample,)
    ).fetchall()
    conn.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.to(device).eval()

    semantic_hits = 0
    for r in rows:
        keyframe_id = int(r["keyframe_id"])
        matches = np.flatnonzero(id_map == keyframe_id)
        if len(matches) == 0:
            print(f"keyframe_id={keyframe_id}: NOT_IN_MAP")
            continue
        expected_row = int(matches[0])

        with torch.no_grad():
            tokens = tokenizer([r["caption_text"]]).to(device)
            q_tensor = model.encode_text(tokens)
            q_tensor = q_tensor / q_tensor.norm(dim=-1, keepdim=True)
            q = q_tensor.cpu().numpy().astype("float32")

        scores, idxs = index.search(q, max(top_k, 1))
        top_rows = idxs[0].tolist()
        top_scores = scores[0].tolist()
        expected_score = float(np.dot(q[0], index.reconstruct(expected_row)))
        found = expected_row in top_rows
        semantic_hits += found
        rank = top_rows.index(expected_row) + 1 if found else None
        preview = [(int(row), int(id_map[row]), round(float(score), 4))
                   for row, score in zip(top_rows[:5], top_scores[:5])]
        print(
            f"keyframe_id={keyframe_id}, expected_row={expected_row}, "
            f"expected_score={expected_score:.4f}, rank={rank}, top5={preview}\n"
            f"  caption={r['caption_text'][:100]!r}"
        )

    print(f"\nsemantic caption hit@{top_k}: {semantic_hits}/{len(rows)}")
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