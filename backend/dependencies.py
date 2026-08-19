"""Khởi tạo các object dùng chung (SqliteManager, FaissManager, RagPipeline)
1 lần duy nhất lúc app start -- routers import từ đây, không tự tạo lại."""

import os
from datetime import timedelta
from minio import Minio

from database.sqlite_manager import SqliteManager
from database.faiss_manager import FaissManager
from rag.pipeline import Retriever, Reranker, RagPipeline, GroqClient

DB_PATH = os.environ.get("AIC_DB_PATH", "data/aic.sqlite")
INDEX_ROOT = os.environ.get("AIC_INDEX_ROOT", "data/index")

sqlite_manager = SqliteManager(DB_PATH)
faiss_manager = FaissManager(
    text_index_path=f"{INDEX_ROOT}/faiss_text.index",
    text_metadata_path=f"{INDEX_ROOT}/text_embedding_metadata.jsonl",
    clip_index_path=f"{INDEX_ROOT}/faiss_clip.index",
    clip_id_map_path=f"{INDEX_ROOT}/clip_id_map.npy",
)
retriever = Retriever(faiss_manager, sqlite_manager)
reranker = Reranker(sqlite_manager)
llm_client = GroqClient() if os.environ.get("GROQ_API_KEY") else None
pipeline = RagPipeline(retriever, reranker, llm_client=llm_client)

minio_public_client = Minio(
    os.environ.get("MINIO_PUBLIC_ENDPOINT", "localhost:9000"),
    access_key=os.environ["MINIO_ACCESS_KEY"],
    secret_key=os.environ["MINIO_SECRET_KEY"],
    secure=False,
)


def get_pipeline() -> RagPipeline:
    """FastAPI dependency -- inject pipeline vào route thay vì import biến global trực tiếp."""
    return pipeline


def get_sqlite_manager() -> SqliteManager:
    return sqlite_manager


def get_minio_client() -> Minio:
    return minio_public_client


def presigned_keyframe_url(bucket: str, key: str) -> str:
    """Sinh presigned URL cho 1 ảnh keyframe trên MinIO, hết hạn sau 1h."""
    return minio_public_client.presigned_get_object(bucket, key, expires=timedelta(hours=1))