"""FastAPI app -- nối SqliteManager + FaissManager vào RagPipeline, expose qua
HTTP cho frontend gọi. Ảnh keyframe serve qua presigned URL từ MinIO."""

import os
from datetime import timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from minio import Minio

from database.sqlite_manager import SqliteManager
from database.faiss_manager import FaissManager
from rag.pipeline import Retriever, Reranker, RagPipeline, GroqClient

app = FastAPI(title="AIC26 RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

minio_client = Minio(
    os.environ["MINIO_ENDPOINT"],
    access_key=os.environ["MINIO_ACCESS_KEY"],
    secret_key=os.environ["MINIO_SECRET_KEY"],
    secure=False,
)


class KISRequest(BaseModel):
    """Body cho /api/kis."""
    query: str
    top_n: int = 10


class QARequest(BaseModel):
    """Body cho /api/qa."""
    query: str
    top_n: int = 10


class TRAKERequest(BaseModel):
    """Body cho /api/trake -- events phải đúng thứ tự thời gian mong muốn."""
    events: list[str]
    top_n: int = 5


@app.post("/api/kis")
def kis(req: KISRequest):
    """Known-Item Search -- trả về list frame/timestamp, không gọi LLM."""
    pipeline.top_n_rerank = req.top_n
    try:
        return {"results": pipeline.run_kis(req.query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/qa")
def qa(req: QARequest):
    """Q&A/VQA -- trả về answer (text) + sources (list frame đã rerank)."""
    if pipeline.llm_client is None:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY chưa được set trên server")
    pipeline.top_n_rerank = req.top_n
    try:
        return pipeline.run_qa(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trake")
def trake(req: TRAKERequest):
    """TRAKE -- trả về list kết quả cho từng event, theo đúng thứ tự events truyền vào."""
    pipeline.top_n_rerank = req.top_n
    try:
        return {"results": pipeline.run_trake(req.events)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/keyframe/{frame_id}/image")
def keyframe_image(frame_id: int):
    """Redirect sang presigned URL của ảnh keyframe trên MinIO (hết hạn sau 1h)."""
    info = sqlite_manager.get_frame_info(frame_id)
    if not info["s3_key"]:
        raise HTTPException(status_code=404, detail="Ảnh chưa được upload lên MinIO")
    url = minio_client.presigned_get_object(
        info["s3_bucket"], info["s3_key"], expires=timedelta(hours=1)
    )
    return RedirectResponse(url)


@app.get("/api/health")
def health():
    """Health check đơn giản cho docker-compose/monitoring."""
    return {"status": "ok"}