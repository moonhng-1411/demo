"""Router phục vụ ảnh keyframe -- redirect sang presigned URL MinIO."""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import RedirectResponse

from dependencies import get_sqlite_manager, presigned_keyframe_url
from database.sqlite_manager import SqliteManager

router = APIRouter(prefix="/api", tags=["media"])


@router.get("/keyframe/{frame_id}/image")
def keyframe_image(frame_id: int, sqlite_manager: SqliteManager = Depends(get_sqlite_manager)):
    """Redirect sang presigned URL của ảnh keyframe trên MinIO (hết hạn sau 1h)."""
    info = sqlite_manager.get_frame_info(frame_id)
    if not info["s3_key"]:
        raise HTTPException(status_code=404, detail="Ảnh chưa được upload lên MinIO")
    url = presigned_keyframe_url(info["s3_bucket"], info["s3_key"])
    return RedirectResponse(url)