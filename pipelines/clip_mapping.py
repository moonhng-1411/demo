"""Hàm dùng chung để load mapping row -> keyframe_id cho faiss_clip.index.
Dùng lại được cả ở pipelines/ (script chẩn đoán, build) và backend/
(FaissManager) để không lặp logic đọc id_map.json ở 2 nơi."""

import json
import numpy as np


def load_clip_id_map(path: str) -> np.ndarray:
    """Đọc id_map.json (list các dict có "row" và "keyframe_id") hoặc file
    .npy (mảng int64 đã sort theo row), tự nhận diện theo phần mở rộng.
    Trả về mảng numpy int64, index i = keyframe_id ứng với FAISS row i."""
    if path.endswith(".npy"):
        return np.load(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw_sorted = sorted(raw, key=lambda x: x["row"])
    return np.array([r["keyframe_id"] for r in raw_sorted], dtype="int64")