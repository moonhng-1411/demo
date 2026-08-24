"""Điểm vào FastAPI -- khởi tạo app, CORS, gắn routers. Logic thật nằm trong
routers/, dependencies dùng chung nằm trong dependencies.py."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import retrieval, media, admin

# INFO trở lên cho toàn bộ log của app (translate/rewrite/split_clauses trong
# rag/pipeline.py dùng logger.info -- mặc định root logger ở WARNING nên nếu
# không set ở đây các log debug này sẽ không bao giờ hiện ra, kể cả khi chạy
# dưới uvicorn).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="AIC26 RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(retrieval.router)
app.include_router(media.router)
app.include_router(admin.router)


@app.get("/api/health")
def health():
    """Health check đơn giản cho docker-compose/monitoring."""
    return {"status": "ok"}