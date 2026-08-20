"""Endpoint quản trị nội bộ -- hiện chỉ có xóa cache SQLite sau khi cập nhật
dữ liệu lúc runtime. Bảo vệ bằng shared-secret ADMIN_TOKEN (biến môi trường);
nếu ADMIN_TOKEN không được set, endpoint bị vô hiệu hóa hoàn toàn (404) để
tránh vô tình public một endpoint quản trị không bảo vệ."""

import os
from fastapi import APIRouter, HTTPException, Header
from dependencies import get_sqlite_manager

router = APIRouter(prefix="/api/admin", tags=["admin"])

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")


@router.post("/clear-cache")
def clear_cache(x_admin_token: str | None = Header(default=None)):
    """Xóa lru_cache của SqliteManager. Gọi sau khi cập nhật aic.sqlite
    (thêm keyframe/caption mới) mà không muốn restart backend."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=404, detail="Not found")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Token không hợp lệ")
    get_sqlite_manager().clear_caches()
    return {"status": "ok"}