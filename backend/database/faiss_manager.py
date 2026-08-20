import faiss
import json
import os
import numpy as np


class FaissManager:
    def __init__(self, text_index_path: str, text_metadata_path: str,
                 clip_index_path: str, clip_id_map_path: str):
        """clip_id_map_path giờ trỏ tới id_map.json (đã verify đúng, 18/20 hit),
        KHÔNG dùng clip_id_map.npy tự build (đã xác nhận sai)."""
        self.text_index = faiss.read_index(text_index_path)
        self.text_metadata = self._load_text_metadata(text_metadata_path)
        assert self.text_index.ntotal == len(self.text_metadata), \
            "faiss_text.index và metadata lệch số dòng"

        self.clip_index = faiss.read_index(clip_index_path)
        self.clip_id_map = self._load_id_map(clip_id_map_path)
        assert self.clip_index.ntotal == len(self.clip_id_map), \
            "faiss_clip.index và clip id map lệch số dòng"

    @staticmethod
    def _resolve_text_metadata_path(path: str) -> str:
        """Tự dò file thật nếu path truyền vào không tồn tại nhưng có phiên bản
        đuôi khác (.json <-> .jsonl) tồn tại -- tránh lệch tên như đã thấy giữa
        code (mặc định .jsonl) và text_embedding_manifest.json (ghi tên .json)."""
        if os.path.isfile(path):
            return path
        base, ext = os.path.splitext(path)
        for alt_ext in (".json", ".jsonl"):
            alt_path = base + alt_ext
            if os.path.isfile(alt_path):
                return alt_path
        return path  # để nguyên -- sẽ raise FileNotFoundError rõ ràng lúc open()

    @staticmethod
    def _load_text_metadata(path: str) -> list[dict]:
        """Đọc metadata text embedding, tự nhận diện JSONL (mỗi dòng 1 object)
        hoặc JSON thường (mảng object, hoặc object chứa mảng ở 1 key con) --
        không giả định cứng định dạng theo phần mở rộng, vì tên file thật
        (theo text_embedding_manifest.json) có thể không khớp quy ước .jsonl
        mà code trước đây giả định."""
        resolved = FaissManager._resolve_text_metadata_path(path)
        with open(resolved, "r", encoding="utf-8") as f:
            raw = f.read()

        stripped = raw.strip()
        if not stripped:
            raise ValueError(f"File metadata rỗng: {resolved}")

        # Thử JSON nguyên khối trước (mảng, hoặc object bọc ngoài 1 mảng).
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for value in parsed.values():
                    if isinstance(value, list):
                        return value
            raise ValueError(
                f"JSON hợp lệ nhưng không tìm thấy list entry nào trong {resolved}"
            )
        except json.JSONDecodeError:
            pass

        # Không phải JSON nguyên khối hợp lệ -- thử JSONL (mỗi dòng 1 object).
        entries = [json.loads(line) for line in stripped.splitlines() if line.strip()]
        if not entries:
            raise ValueError(f"Không parse được metadata (thử cả JSON và JSONL): {resolved}")
        return entries

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