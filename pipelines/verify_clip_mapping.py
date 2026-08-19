"""Kiểm chứng clip_id_map.npy đúng: lấy vài keyframe có caption rõ ràng,
encode caption bằng CLIP text encoder, query vào faiss_clip.index, xem row
top-1/top-5 trả về có khớp đúng keyframe đó không."""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import sqlite3
from pipelines.config import DB_PATH, INDEX_ROOT


def main():
    """In ra rank tìm thấy cho 10 keyframe ngẫu nhiên -- hits cao (>=7/10)
    nghĩa là mapping đáng tin, hits gần 0 nghĩa là thứ tự build sai giả định."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT keyframe_id, caption_text FROM captions "
        "WHERE caption_text IS NOT NULL ORDER BY RANDOM() LIMIT 10"
    ).fetchall()

    id_map = np.load(f"{INDEX_ROOT}/clip_id_map.npy")
    index = faiss.read_index(f"{INDEX_ROOT}/faiss_clip.index")
    model = SentenceTransformer("clip-ViT-B-32", device="cpu")

    hits = 0
    for r in rows:
        expected_keyframe_id = r["keyframe_id"]
        expected_row = np.where(id_map == expected_keyframe_id)[0][0]

        query_vec = model.encode(r["caption_text"]).astype("float32").reshape(1, -1)
        _, idxs = index.search(query_vec, 5)
        top_rows = idxs[0].tolist()

        found = expected_row in top_rows
        hits += found
        rank = top_rows.index(expected_row) + 1 if found else "không thấy trong top-5"
        print(f"keyframe_id={expected_keyframe_id} caption=\"{r['caption_text'][:50]}...\" "
              f"-- expected_row={expected_row}, rank={rank}")

    print(f"\n{hits}/{len(rows)} đúng trong top-5.")


if __name__ == "__main__":
    main()