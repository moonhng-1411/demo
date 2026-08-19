import faiss
import json
import numpy as np


class FaissManager:
    def __init__(self, text_index_path: str, text_metadata_path: str,
                 clip_index_path: str, clip_id_map_path: str):
        """clip_id_map_path giờ trỏ tới id_map.json (đã verify đúng, 18/20 hit),
        KHÔNG dùng clip_id_map.npy tự build (đã xác nhận sai)."""
        self.text_index = faiss.read_index(text_index_path)
        with open(text_metadata_path, "r", encoding="utf-8") as f:
            self.text_metadata = [json.loads(line) for line in f if line.strip()]
        assert self.text_index.ntotal == len(self.text_metadata), \
            "faiss_text.index và metadata lệch số dòng"

        self.clip_index = faiss.read_index(clip_index_path)
        self.clip_id_map = self._load_id_map(clip_id_map_path)
        assert self.clip_index.ntotal == len(self.clip_id_map), \
            "faiss_clip.index và clip id map lệch số dòng"

    @staticmethod
    def _load_id_map(path: str) -> np.ndarray:
        """Đọc id_map.json (list [{"row":..,"keyframe_id":..}]) hoặc .npy,
        tự nhận diện theo phần mở rộng."""
        if path.endswith(".npy"):
            return np.load(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw_sorted = sorted(raw, key=lambda x: x["row"])
        return np.array([r["keyframe_id"] for r in raw_sorted], dtype="int64")

    def search_text(self, query_vector: np.ndarray, top_k: int = 50):
        q = query_vector.reshape(1, -1).astype("float32")
        scores, idxs = self.text_index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            entry = self.text_metadata[idx]
            results.append((entry, entry["kind"], float(score)))
        return results

    def search_clip(self, query_vector: np.ndarray, top_k: int = 50):
        q = query_vector.reshape(1, -1).astype("float32")
        scores, idxs = self.clip_index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((int(self.clip_id_map[idx]), float(score)))
        return results