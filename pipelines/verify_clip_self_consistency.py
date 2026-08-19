"""Test tự-nhất-quán faiss_clip.index: lấy thẳng vector gốc từ clip_b32.f32.npy
(vector đã dùng để build index), query ngược lại vào chính index đó.

Nếu KHÔNG tự khớp (top-1 != đúng row) -- lỗi nằm ở cách build FAISS index
(thứ tự add lệch, hoặc normalize sai), không liên quan gì đến model encode
câu query hay mapping keyframe_id.

Nếu tự khớp đúng nhưng verify_clip_mapping.py (dùng model encode query) vẫn
fail -- lỗi nằm ở việc model encode câu query không cùng không gian embedding
với model BTC dùng lúc build (vd OpenAI CLIP gốc khác sentence-transformers)."""

import numpy as np
import faiss
from pipelines.config import INDEX_ROOT

N_SAMPLES = 20
SEED = 0


def main():
    raw_vecs = np.load(f"{INDEX_ROOT}/clip_b32.f32.npy").astype("float32")
    index = faiss.read_index(f"{INDEX_ROOT}/faiss_clip.index")

    print(f"clip_b32.f32.npy: {raw_vecs.shape}")
    print(f"faiss_clip.index: {index.ntotal} vector, dim={index.d}")

    if raw_vecs.shape[0] != index.ntotal:
        print("CẢNH BÁO: số vector trong .npy và .index không khớp -- ngay cả trước khi test")

    rng = np.random.default_rng(SEED)
    test_rows = rng.choice(raw_vecs.shape[0], min(N_SAMPLES, raw_vecs.shape[0]), replace=False)

    hits = 0
    for row in test_rows:
        q = raw_vecs[row:row + 1]
        _, idxs = index.search(q, 1)
        match = (idxs[0][0] == row)
        hits += match
        print(f"row={row} top1={idxs[0][0]} -- {'KHỚP' if match else 'SAI'}")

    print(f"\n{hits}/{len(test_rows)} tự-khớp")
    if hits >= len(test_rows) * 0.8:
        print("=> Index build đúng. Nếu verify_clip_mapping.py vẫn fail, lỗi nằm ở "
              "model encode query (khác không gian embedding với model BTC dùng).")
    else:
        print("=> Index có vấn đề khi build (thứ tự lệch hoặc normalize sai). "
              "Nên bỏ faiss_clip.index gốc, tự build lại từ clip_b32.f32.npy.")


if __name__ == "__main__":
    main()