"""Điểm vào FastAPI -- khởi tạo app, CORS, gắn routers. Logic thật nằm trong
routers/, dependencies dùng chung nằm trong dependencies.py."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import retrieval, media

app = FastAPI(title="AIC26 RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(retrieval.router)
app.include_router(media.router)


@app.get("/api/health")
def health():
    """Health check đơn giản cho docker-compose/monitoring."""
    return {"status": "ok"}