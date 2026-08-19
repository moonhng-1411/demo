"""Lớp truy vấn aic.sqlite -- cung cấp thông tin frame (video_id, timestamp,
đường dẫn ảnh trên MinIO), object labels, và text tổng hợp (caption/ocr/asr)
cho 1 keyframe cụ thể. Được Retriever và Reranker dùng chung."""

import sqlite3


class SqliteManager:
    def __init__(self, db_path: str):
        """Mở kết nối SQLite dùng chung suốt vòng đời app (check_same_thread=False
        vì FastAPI xử lý request trên nhiều thread)."""
        self._conn_obj = sqlite3.connect(db_path, check_same_thread=False)
        self._conn_obj.row_factory = sqlite3.Row

    def _conn(self):
        return self._conn_obj

    def get_frame_info(self, frame_id: int) -> dict:
        """Trả về video_id, pts_time, và thông tin ảnh (s3_bucket/s3_key/file_name)
        cho 1 keyframe_id. Raise KeyError nếu frame_id không tồn tại."""
        row = self._conn().execute(
            "SELECT video_id, pts_time, s3_bucket, s3_key, file_name "
            "FROM keyframes WHERE id = ?", (frame_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"frame_id không tồn tại: {frame_id}")
        return dict(row)

    def resolve_transcript_to_frame(self, transcript_id: int) -> int:
        """ASR không map 1-1 vào keyframe (1 segment có thể phủ nhiều frame) --
        hàm này lấy mốc giữa (start_s+end_s)/2 của transcript rồi tìm keyframe
        có pts_time gần nhất trong CÙNG video. Dùng khi Retriever xử lý hit
        có modality == "asr" từ faiss_text.index."""
        row = self._conn().execute(
            "SELECT video_id, (start_s + end_s) / 2.0 AS mid_s FROM transcripts WHERE id = ?",
            (transcript_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"transcript_id không tồn tại: {transcript_id}")

        frame = self._conn().execute(
            "SELECT id FROM keyframes WHERE video_id = ? ORDER BY ABS(pts_time - ?) LIMIT 1",
            (row["video_id"], row["mid_s"]),
        ).fetchone()
        if frame is None:
            raise KeyError(f"không tìm thấy keyframe cho video {row['video_id']}")
        return frame["id"]

    def get_frame_objects(self, frame_id: int, min_score: float = 0.3) -> list[str]:
        """Trả về list nhãn object (entity) detect được trên frame, lọc theo
        score tối thiểu, sort giảm dần theo score. Dùng cho object_score trong
        merge() (fusion.py) -- không chấm điểm object có score quá thấp."""
        rows = self._conn().execute(
            "SELECT entity FROM objects WHERE keyframe_id = ? AND score >= ? ORDER BY score DESC",
            (frame_id, min_score),
        ).fetchall()
        return [r["entity"] for r in rows]

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