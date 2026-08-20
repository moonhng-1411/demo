"""Lớp truy vấn aic.sqlite -- cung cấp thông tin frame (video_id, timestamp,
đường dẫn ảnh trên MinIO), object labels, và text tổng hợp (caption/ocr/asr)
cho 1 keyframe cụ thể. Được Retriever và Reranker dùng chung."""

import sqlite3
from functools import lru_cache


class SqliteManager:
    def __init__(self, db_path: str):
        """Mở kết nối SQLite dùng chung suốt vòng đời app (check_same_thread=False
        vì FastAPI xử lý request trên nhiều thread)."""
        self._conn_obj = sqlite3.connect(db_path, check_same_thread=False)
        self._conn_obj.row_factory = sqlite3.Row

    def _conn(self):
        return self._conn_obj

    @lru_cache(maxsize=8192)
    def get_frame_info(self, frame_id: int) -> dict:
        """Trả metadata của một keyframe.

        ``id`` là keyframe_id toàn cục dùng để truy vấn database; ``n`` là số
        thứ tự keyframe trong video dùng để dựng tên ảnh MinIO; ``frame_idx`` là
        chỉ số frame theo video và là trường được trả ra cho frontend.
        """
        row = self._conn().execute(
            "SELECT id, video_id, n, frame_idx, pts_time, s3_bucket, s3_key, file_name "
            "FROM keyframes WHERE id = ?", (frame_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"frame_id không tồn tại: {frame_id}")
        return dict(row)

    def resolve_asr_to_frame(self, video_id: str, start_s: float, end_s: float) -> int:
        """Map một ASR segment sang keyframe gần nhất trong cùng video.

        Metadata JSONL hiện tại chỉ có ``video_id``, ``start_s`` và ``end_s``;
        không có ``transcript_id``. Ưu tiên keyframe nằm trong segment, sau đó
        fallback về keyframe gần điểm giữa segment để không làm hỏng toàn bộ
        truy vấn khi sampling frame thưa.
        """
        video_id = str(video_id)
        start_s = float(start_s)
        end_s = float(end_s)
        mid_s = (start_s + end_s) / 2.0

        frame = self._conn().execute(
            "SELECT id FROM keyframes "
            "WHERE video_id = ? AND pts_time BETWEEN ? AND ? "
            "ORDER BY ABS(pts_time - ?) LIMIT 1",
            (video_id, start_s, end_s, mid_s),
        ).fetchone()
        if frame is None:
            frame = self._conn().execute(
                "SELECT id FROM keyframes WHERE video_id = ? "
                "ORDER BY ABS(pts_time - ?) LIMIT 1",
                (video_id, mid_s),
            ).fetchone()
        if frame is None:
            raise KeyError(f"không tìm thấy keyframe cho video {video_id}")
        return int(frame["id"])

    def resolve_transcript_to_frame(self, transcript_id: int) -> int:
        """Resolver tương thích ngược cho metadata cũ có transcript_id."""
        row = self._conn().execute(
            "SELECT video_id, start_s, end_s FROM transcripts WHERE id = ?",
            (transcript_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"transcript_id không tồn tại: {transcript_id}")
        return self.resolve_asr_to_frame(row["video_id"], row["start_s"], row["end_s"])

    def resolve_frame_position(self, video_id: str, frame_idx: int) -> int:
        """Resolve video_id + frame_idx thành keyframe_id nội bộ cho endpoint ảnh."""
        row = self._conn().execute(
            "SELECT id FROM keyframes WHERE video_id = ? AND frame_idx = ? LIMIT 1",
            (str(video_id), int(frame_idx)),
        ).fetchone()
        if row is None:
            raise KeyError(f"không tìm thấy frame_idx={frame_idx} trong video {video_id}")
        return int(row["id"])

    @lru_cache(maxsize=8192)
    def get_frame_objects(self, frame_id: int, min_score: float = 0.3) -> list[str]:
        """Trả về list nhãn object (entity) detect được trên frame, lọc theo
        score tối thiểu, sort giảm dần theo score. Dùng cho object_score trong
        merge() (fusion.py) -- không chấm điểm object có score quá thấp."""
        rows = self._conn().execute(
            "SELECT entity FROM objects WHERE keyframe_id = ? AND score >= ? ORDER BY score DESC",
            (frame_id, min_score),
        ).fetchall()
        return [r["entity"] for r in rows]

    @lru_cache(maxsize=8192)
    def get_frame_texts(self, frame_id: int) -> dict:
        """Trả về {"caption_text", "ocr_text", "asr_text"} cho 1 frame, dùng
        để Reranker build document cho cross-encoder. asr_text lấy từ segment
        transcript có khoảng thời gian chứa đúng pts_time của frame; ocr_text
        luôn None vì schema hiện chưa có bảng OCR."""
        conn = self._conn()

        cap = conn.execute(
            "SELECT caption_text FROM captions WHERE keyframe_id = ?", (frame_id,)
        ).fetchone()

        info = self.get_frame_info(frame_id)
        seg = conn.execute(
            "SELECT text FROM transcripts WHERE video_id = ? AND start_s <= ? AND end_s >= ? "
            "ORDER BY start_s LIMIT 1",
            (info["video_id"], info["pts_time"], info["pts_time"]),
        ).fetchone()

        return {
            "caption_text": cap["caption_text"] if cap else None,
            "ocr_text": None,
            "asr_text": seg["text"] if seg else None,
        }

    def clear_caches(self) -> None:
        """Xóa toàn bộ lru_cache (frame_info/frame_objects/frame_texts).

        Cần gọi sau khi aic.sqlite được cập nhật lúc runtime (ví dụ upload
        thêm keyframe/caption mới) để tránh trả về dữ liệu cũ đã cache từ
        trước đó -- các cache này không tự hết hạn theo thời gian.
        """
        self.get_frame_info.cache_clear()
        self.get_frame_objects.cache_clear()
        self.get_frame_texts.cache_clear()