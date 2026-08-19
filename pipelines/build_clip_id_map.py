"""Tạo clip_id_map.npy dựa trên giả định: faiss_clip.index được build theo
đúng thứ tự keyframes.id tăng dần (row i -> keyframe_id i+1)."""

import sqlite3
import faiss
import numpy as np
from pipelines.config import DB_PATH, INDEX_ROOT   


def main():
    conn = sqlite3.connect(DB_PATH)
    ids = [r[0] for r in conn.execute("SELECT id FROM keyframes ORDER BY id ASC").fetchall()]
    conn.close()

    index = faiss.read_index(f"{INDEX_ROOT}/faiss_clip.index")
    if index.ntotal != len(ids):
        print(f"[CẢNH BÁO] faiss_clip.index có {index.ntotal} vector nhưng sqlite có {len(ids)} keyframe — LỆCH, không nên tin id_map này.")
        return
    
    id_map = np.array(ids, dtype="int64")
    np.save(f"{INDEX_ROOT}/clip_id_map.npy", id_map)
    print(f"Đã ghi clip_id_map.npy ({len(id_map)} phần tử, keyframe_id {ids[0]}..{ids[-1]})")


if __name__ == "__main__":
    main()