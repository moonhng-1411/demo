"""Router cho 3 loại truy vấn AIC: KIS, Q&A, TRAKE."""

import logging

from fastapi import APIRouter, HTTPException, Depends

from schemas import KISRequest, QARequest, TRAKERequest
from dependencies import get_pipeline
from rag.pipeline import RagPipeline

router = APIRouter(prefix="/api", tags=["retrieval"])
logger = logging.getLogger(__name__)


@router.post("/kis")
def kis(req: KISRequest, pipeline: RagPipeline = Depends(get_pipeline)):
    """Known-Item Search -- trả về list frame/timestamp, không gọi LLM."""
    try:
        return {"results": pipeline.run_kis(req.query, top_n=req.top_n, translate=req.translate)}
    except Exception:
        logger.exception("Lỗi khi xử lý /api/kis")
        raise HTTPException(status_code=500, detail="Lỗi nội bộ khi xử lý KIS")


@router.post("/qa")
def qa(req: QARequest, pipeline: RagPipeline = Depends(get_pipeline)):
    """Q&A/VQA -- trả về answer (text) + sources (list frame đã rerank)."""
    if pipeline.llm_client is None:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY chưa được set trên server")
    try:
        return pipeline.run_qa(req.query, top_n=req.top_n, translate=req.translate)
    except Exception:
        logger.exception("Lỗi khi xử lý /api/qa")
        raise HTTPException(status_code=500, detail="Lỗi nội bộ khi xử lý Q&A")


@router.post("/trake")
def trake(req: TRAKERequest, pipeline: RagPipeline = Depends(get_pipeline)):
    """TRAKE -- trả về list kết quả cho từng event, theo đúng thứ tự events truyền vào."""
    try:
        return {"results": pipeline.run_trake(req.events, top_n=req.top_n, translate=req.translate)}
    except Exception:
        logger.exception("Lỗi khi xử lý /api/trake")
        raise HTTPException(status_code=500, detail="Lỗi nội bộ khi xử lý TRAKE")