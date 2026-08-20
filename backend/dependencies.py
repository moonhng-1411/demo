"""Khởi tạo các object dùng chung (SqliteManager, FaissManager, RagPipeline)
1 lần duy nhất lúc app start -- routers import từ đây, không tự tạo lại."""

import inspect
import os
from datetime import timedelta
from minio import Minio

from database.sqlite_manager import SqliteManager
from database.faiss_manager import FaissManager
from rag.pipeline import Retriever, Reranker, RagPipeline, GroqClient, QueryTranslator

DB_PATH = os.environ.get("AIC_DB_PATH", "data/aic.sqlite")
INDEX_ROOT = os.environ.get("AIC_INDEX_ROOT", "data/index")
_DEFAULT_JSON_MAP = f"{INDEX_ROOT}/id_map.json"
_DEFAULT_NPY_MAP = f"{INDEX_ROOT}/clip_id_map.npy"
CLIP_ID_MAP_PATH = os.environ.get(
    "AIC_CLIP_ID_MAP_PATH",
    _DEFAULT_JSON_MAP if os.path.exists(_DEFAULT_JSON_MAP) else _DEFAULT_NPY_MAP,
)

sqlite_manager = SqliteManager(DB_PATH)
faiss_manager = FaissManager(
    text_index_path=f"{INDEX_ROOT}/faiss_text.index",
    text_metadata_path=f"{INDEX_ROOT}/text_embedding_metadata.jsonl",
    clip_index_path=f"{INDEX_ROOT}/faiss_clip.index",
    clip_id_map_path=CLIP_ID_MAP_PATH,
)

# Dịch query VI->EN để khớp trực tiếp với caption/object label (chủ yếu
# tiếng Anh) thay vì chỉ trông chờ khả năng cross-lingual của multilingual
# embedder/cross-encoder. Tắt bằng cách không set GROQ_API_KEY hoặc set
# RAG_TRANSLATE_QUERY=0 -- Retriever/Reranker đều xử lý translator=None
# một cách an toàn (bỏ qua bước dịch, không lỗi).
TRANSLATE_QUERY = os.environ.get("RAG_TRANSLATE_QUERY", "1") == "1"
query_translator = (
    QueryTranslator() if TRANSLATE_QUERY and os.environ.get("GROQ_API_KEY") else None
)
if TRANSLATE_QUERY and query_translator is None:
    print("[WARN] RAG_TRANSLATE_QUERY=1 nhưng thiếu GROQ_API_KEY -- bỏ qua dịch query, "
          "chạy như cũ (chỉ multilingual embedder/cross-encoder).")

retriever = Retriever(faiss_manager, sqlite_manager, translator=query_translator)
reranker = Reranker(sqlite_manager)
llm_client = GroqClient() if os.environ.get("GROQ_API_KEY") else None
# 20 mỗi modality thường đủ cho KIS và giảm đáng kể số cặp đưa vào CrossEncoder.
# Có thể tăng lên 30/50 nếu ưu tiên recall hơn tốc độ.
TOP_K_RETRIEVE = int(os.environ.get("RAG_TOP_K_RETRIEVE", "20"))
FAST_KIS = os.environ.get("RAG_FAST_KIS", "0") == "1"
pipeline_kwargs = {
    "llm_client": llm_client,
    "top_k_retrieve": TOP_K_RETRIEVE,
}
if "fast_kis" in inspect.signature(RagPipeline).parameters:
    pipeline_kwargs["fast_kis"] = FAST_KIS
else:
    print("[WARN] pipeline.py chưa hỗ trợ fast_kis; đang chạy compatibility mode. Hãy cập nhật pipeline.py để bật RAG_FAST_KIS.")
pipeline = RagPipeline(retriever, reranker, **pipeline_kwargs)

MINIO_SECURE = os.environ.get("MINIO_SECURE", "false").lower() in {"1", "true", "yes", "on"}
_MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY")
_MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY")
if not _MINIO_ACCESS_KEY or not _MINIO_SECRET_KEY:
    raise RuntimeError(
        "Thiếu MINIO_ACCESS_KEY / MINIO_SECRET_KEY trong biến môi trường (.env). "
        "Backend cần MinIO để phục vụ ảnh keyframe, hãy cấu hình trước khi khởi động."
    )
minio_public_client = Minio(
    os.environ.get("MINIO_PUBLIC_ENDPOINT", os.environ.get("MINIO_ENDPOINT", "localhost:9000")),
    access_key=_MINIO_ACCESS_KEY,
    secret_key=_MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)


def get_pipeline() -> RagPipeline:
    """FastAPI dependency -- inject pipeline vào route thay vì import biến global trực tiếp."""
    return pipeline


def get_sqlite_manager() -> SqliteManager:
    return sqlite_manager


def get_minio_client() -> Minio:
    return minio_public_client


def presigned_keyframe_url(bucket: str, key: str, client: Minio | None = None) -> str:
    """Sinh presigned URL bằng đúng public client dùng cho endpoint media."""
    signer = client or minio_public_client
    return signer.presigned_get_object(bucket, key, expires=timedelta(hours=1))