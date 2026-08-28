"""
Chẩn đoán "hubness": kiểm tra xem 1 frame có phải là hub trong không gian
CLIP visual embedding không -- tức có similarity TRUNG BÌNH cao bất thường
với RẤT NHIỀU vector khác trong index (không riêng gì các query liên quan),
khiến nó tự nhiên trồi lên top-k của hầu hết mọi truy vấn bất kể nội dung.

Cách đo: dùng chính faiss_clip.index (đã build sẵn, không cần encode gì
thêm) -- lấy vector của frame nghi ngờ, tính cosine similarity trung bình
với N vector khác lấy ngẫu nhiên trong CÙNG index, rồi làm y hệt với vài
chục frame ngẫu nhiên khác để có baseline so sánh. Nếu avg_similarity của
frame nghi ngờ nằm ở percentile rất cao so với baseline -> xác nhận hub
thật, không phải trùng hợp.

Chạy: python check_hubness.py --data-dir ./data --video-id L21_V014 --frame-idx 21063
"""

import argparse
import json
import os
import sqlite3

import faiss
import numpy as np


def load_id_map(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw_sorted = sorted(raw, key=lambda x: x["row"])
    return np.array([r["keyframe_id"] for r in raw_sorted], dtype="int64")


def reconstruct_all_normalized(index) -> np.ndarray:
    """Lấy toàn bộ vector trong index (giả định đã normalize lúc build, vì
    manifest dùng inner product = cosine -- xem docstring FaissManager)."""
    n = index.ntotal
    vecs = np.zeros((n, index.d), dtype="float32")
    for i in range(n):
        vecs[i] = index.reconstruct(i)
    return vecs


def avg_sim_to_sample(vec: np.ndarray, all_vecs: np.ndarray, sample_idx: np.ndarray,
                       exclude_row: int) -> float:
    sample = sample_idx[sample_idx != exclude_row]
    sims = all_vecs[sample] @ vec
    return float(np.mean(sims))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="./data/index", help="Thư mục chứa faiss_clip.index, id_map.json (mặc định data/index theo .env.example)")
    ap.add_argument("--db-path", default="./data/aic.sqlite", help="Đường dẫn aic.sqlite (mặc định data/aic.sqlite, KHÁC thư mục với index)")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--frame-idx", type=int, required=True)
    ap.add_argument("--sample-size", type=int, default=3000, help="Số vector ngẫu nhiên dùng để tính avg similarity")
    ap.add_argument("--baseline-frames", type=int, default=30, help="Số frame ngẫu nhiên khác dùng làm baseline so sánh")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clip_index_path = os.path.join(args.index_dir, "faiss_clip.index")
    id_map_json = os.path.join(args.index_dir, "id_map.json")
    id_map_npy = os.path.join(args.index_dir, "clip_id_map.npy")
    db_path = args.db_path

    id_map_path = id_map_json if os.path.isfile(id_map_json) else id_map_npy
    for p, label in [(clip_index_path, "faiss_clip.index"), (id_map_path, "clip id map"), (db_path, "aic.sqlite")]:
        if not os.path.isfile(p):
            raise SystemExit(f"Không tìm thấy {label} tại {p} -- kiểm tra --index-dir/--db-path")

    print(f"Đang load {clip_index_path} ...")
    index = faiss.read_index(clip_index_path)
    id_map = load_id_map(id_map_path)
    print(f"Index có {index.ntotal} vector, dim={index.d}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    kf = conn.execute(
        "SELECT id FROM keyframes WHERE video_id = ? AND frame_idx = ?",
        (args.video_id, args.frame_idx),
    ).fetchone()
    if kf is None:
        raise SystemExit(f"Không tìm thấy video_id={args.video_id} frame_idx={args.frame_idx} trong DB")
    target_frame_id = kf["id"]

    rows_for_target = np.where(id_map == target_frame_id)[0]
    if len(rows_for_target) == 0:
        raise SystemExit(f"frame_id={target_frame_id} không có trong CLIP index (có thể chưa được index)")
    target_row = int(rows_for_target[0])

    print(f"Đang load toàn bộ {index.ntotal} vector vào RAM (có thể mất vài giây/phút tuỳ kích thước index)...")
    all_vecs = reconstruct_all_normalized(index)

    rng = np.random.default_rng(args.seed)
    sample_size = min(args.sample_size, index.ntotal - 1)
    sample_idx = rng.choice(index.ntotal, size=sample_size, replace=False)

    target_vec = all_vecs[target_row]
    target_avg_sim = avg_sim_to_sample(target_vec, all_vecs, sample_idx, target_row)

    baseline_rows = rng.choice(index.ntotal, size=min(args.baseline_frames, index.ntotal - 1), replace=False)
    baseline_avgs = []
    for row in baseline_rows:
        if row == target_row:
            continue
        v = all_vecs[row]
        baseline_avgs.append(avg_sim_to_sample(v, all_vecs, sample_idx, row))

    baseline_avgs = np.array(baseline_avgs)
    percentile = float(np.mean(baseline_avgs < target_avg_sim) * 100)

    print("\n=== KẾT QUẢ ===")
    print(f"frame_id={target_frame_id} (video_id={args.video_id}, frame_idx={args.frame_idx})")
    print(f"avg cosine similarity với {sample_size} vector ngẫu nhiên: {target_avg_sim:.4f}")
    print(f"baseline ({len(baseline_avgs)} frame ngẫu nhiên khác): "
          f"mean={baseline_avgs.mean():.4f} std={baseline_avgs.std():.4f} "
          f"min={baseline_avgs.min():.4f} max={baseline_avgs.max():.4f}")
    print(f"-> frame nghi ngờ nằm ở percentile {percentile:.1f}% so với baseline "
          f"(percentile càng gần 100% càng chắc chắn là hub)")

    if percentile >= 90:
        print("\n>>> XÁC NHẬN: đây là hub thật -- similarity trung bình cao hơn hẳn "
              "phần lớn frame khác, giải thích vì sao nó lọt top-k của nhiều query "
              "bất kể nội dung.")
    elif percentile >= 70:
        print("\n>>> Có dấu hiệu hub nhưng chưa cực đoan -- có thể kết hợp với caption "
              "modality mới đủ giải thích tần suất xuất hiện.")
    else:
        print("\n>>> KHÔNG phải hub rõ rệt trong visual embedding -- nguyên nhân có "
              "khả năng nằm ở modality caption/asr thay vì visual. Chạy lại kiểm tra "
              "tương tự trên faiss_text.index nếu cần.")


if __name__ == "__main__":
    main()