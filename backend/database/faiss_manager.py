"""Lớp query 2 FAISS index đã build sẵn:
- faiss_text.index: caption + asr gộp chung (BGE-small, 384-dim), phân biệt
  bằng field "kind" trong text_embedding_metadata.jsonl (đọc song song, 1 dòng
  = 1 vector, theo đúng thứ tự .add() lúc build).
- faiss_clip.index: ảnh keyframe (CLIP ViT-B/32, 512-dim), map row -> keyframe_id
  qua clip_id_map.npy (build từ build_clip_id_map.py, đã verify)."""

import faiss
import json
import numpy as np


class FaissManager:
    def __init__(self, text_index_path: str, text_metadata_path: str,
                 clip_index_path: str, clip_id_map_path: str):
        """Load cả 2 index + metadata vào RAM lúc khởi động app. assert ở đây
        để fail sớm và rõ ràng nếu index và metadata/id_map bị lệch số dòng
        (dấu hiệu build lỗi, không nên chạy tiếp)."""
        self.text_index = faiss.read_index(text_index_path)
        with open(text_metadata_path, "r", encoding="utf-8") as f:
            self.text_metadata = [json.loads(line) for line in f if line.strip()]
        assert self.text_index.ntotal == len(self.text_metadata), \
            "faiss_text.index và metadata lệch số dòng"

        self.clip_index = faiss.read_index(clip_index_path)
        self.clip_id_map = np.load(clip_id_map_path)
        assert self.clip_index.ntotal == len(self.clip_id_map), \
            "faiss_clip.index và clip_id_map lệch số dòng"

    def search_text(self, query_vector: np.ndarray, top_k: int = 50) -> list[tuple]:
        """Tìm top_k vector gần nhất trong faiss_text.index (inner product,
        vector đã normalize sẵn nên tương đương cosine similarity).
        Trả về list[(metadata_entry, kind, score)] -- kind là "caption" hoặc
        "asr", lấy trực tiếp từ metadata, không suy luận theo vị trí row."""
        q = query_vector.reshape(1, -1).astype("float32")
        scores, idxs = self.text_index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:  # FAISS trả -1 khi không đủ top_k kết quả
                continue
            entry = self.text_metadata[idx]
            results.append((entry, entry["kind"], float(score)))
        return results

    def search_clip(self, query_vector: np.ndarray, top_k: int = 50) -> list[tuple]:
        """Tìm top_k vector gần nhất trong faiss_clip.index. Trả về
        list[(keyframe_id, score)], đã map sẵn qua clip_id_map -- caller không
        cần biết về row index nội bộ của FAISS."""
        q = query_vector.reshape(1, -1).astype("float32")
        scores, idxs = self.clip_index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((int(self.clip_id_map[idx]), float(score)))
        return results