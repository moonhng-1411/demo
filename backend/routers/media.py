"""Các endpoint phục vụ media keyframe từ MinIO."""

import os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from minio import Minio
from minio.error import S3Error

from dependencies import get_sqlite_manager, get_minio_client, presigned_keyframe_url
from database.sqlite_manager import SqliteManager

router = APIRouter(prefix="/api", tags=["media"])

FRAMES_BUCKET = os.environ.get("MINIO_BUCKET_FRAMES", "original-images")
MEDIA_DEBUG = os.environ.get("RAG_MEDIA_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


def _image_key(video_id: str, number: int) -> str:
    """Tạo key chuẩn theo thứ tự keyframe trong video, ví dụ ``001.jpg``."""
    return f"{video_id}/{int(number):03d}.jpg"


def _clean_db_key(bucket: str, key: str | None) -> str | None:
    """Chuẩn hóa s3_key lưu trong SQLite thành object key tương đối bucket."""
    if not key:
        return None
    value = str(key).strip().replace("\\", "/").lstrip("/")
    if value.startswith("s3://"):
        value = value[5:]
        parts = value.split("/", 1)
        value = parts[1] if len(parts) == 2 else ""
    bucket_prefix = f"{bucket.strip('/')}/"
    if value.startswith(bucket_prefix):
        value = value[len(bucket_prefix):]
    return value or None


def _candidate_locations(info: dict, video_id: str) -> list[tuple[str, str]]:
    """Sinh các vị trí có thể có của ảnh, ưu tiên metadata SQLite.

    SQLite có thể đã được cập nhật với s3_key thật, còn các object upload theo
    convention mới dùng ``n:03d``. Hai dạng global keyframe id được giữ làm
    fallback để tương thích dữ liệu đã upload trước đó.
    """
    locations: list[tuple[str, str]] = []
    db_bucket = str(info.get("s3_bucket") or FRAMES_BUCKET)
    db_key = _clean_db_key(db_bucket, info.get("s3_key"))
    if db_key:
        locations.append((db_bucket, db_key))

    n = info.get("n")
    keyframe_id = info.get("id", info.get("keyframe_id"))
    if n is not None:
        locations.append((FRAMES_BUCKET, _image_key(video_id, int(n))))
    if keyframe_id is not None:
        locations.append((FRAMES_BUCKET, _image_key(video_id, int(keyframe_id))))
        locations.append((FRAMES_BUCKET, f"{video_id}/{int(keyframe_id)}.jpg"))

    # Giữ thứ tự ưu tiên nhưng loại bỏ bản ghi trùng.
    return list(dict.fromkeys(locations))


def _find_uploaded_image(
    info: dict,
    video_id: str,
    minio_client: Minio,
) -> tuple[str, str] | None:
    """Tìm candidate object thực sự tồn tại trên MinIO bằng HEAD request."""
    checked: list[str] = []
    for bucket, key in _candidate_locations(info, video_id):
        checked.append(f"{bucket}/{key}")
        try:
            minio_client.stat_object(bucket, key)
            if MEDIA_DEBUG:
                print(f"[MEDIA] found {video_id} frame_idx={info.get('frame_idx')} -> {bucket}/{key}")
            return bucket, key
        except S3Error:
            continue
        except Exception as exc:
            if MEDIA_DEBUG:
                print(f"[MEDIA] check failed {bucket}/{key}: {exc}")
            continue
    if MEDIA_DEBUG:
        print(f"[MEDIA] not found video_id={video_id} frame_idx={info.get('frame_idx')} checked={checked}")
    return None


def _redirect_if_available(
    info: dict,
    video_id: str,
    minio_client: Minio,
):
    """Proxy bytes ảnh qua backend thay vì redirect trình duyệt ra thẳng
    MinIO/ngrok -- request browser-to-ngrok trực tiếp bị chặn bởi trang
    cảnh báo interstitial của ngrok free tier (chỉ chặn traffic có vẻ đến
    từ browser, request server-to-server qua minio SDK thì không bị)."""
    location = _find_uploaded_image(info, video_id, minio_client)
    if location is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Ảnh chưa có trên MinIO cho video_id={video_id}, "
                f"frame_idx={info.get('frame_idx')}"
            ),
        )
    bucket, key = location
    response = minio_client.get_object(bucket, key)
    return StreamingResponse(
        response.stream(32 * 1024),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/keyframe/{video_id}/{frame_idx}/image")
def keyframe_image_by_position(
    video_id: str,
    frame_idx: int,
    sqlite_manager: SqliteManager = Depends(get_sqlite_manager),
    minio_client: Minio = Depends(get_minio_client),
):
    """Lấy ảnh bằng cặp công khai ``video_id + frame_idx``."""
    try:
        keyframe_id = sqlite_manager.resolve_frame_position(video_id, frame_idx)
        info = sqlite_manager.get_frame_info(keyframe_id)
        return _redirect_if_available(info, video_id, minio_client)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/keyframe/{frame_id}/image")
def keyframe_image_legacy(
    frame_id: int,
    sqlite_manager: SqliteManager = Depends(get_sqlite_manager),
    minio_client: Minio = Depends(get_minio_client),
):
    """Tương thích ngược cho client cũ dùng keyframe_id nội bộ."""
    try:
        info = sqlite_manager.get_frame_info(frame_id)
        return _redirect_if_available(info, info["video_id"], minio_client)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc