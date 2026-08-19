"""Router cho 3 loại truy vấn AIC: KIS, Q&A, TRAKE."""

from fastapi import APIRouter, HTTPException, Depends

from schemas import KISRequest, QARequest, TRAKERequest
from dependencies import get_pipeline
from rag.pipeline import RagPipeline

router = APIRouter(prefix="/api", tags=["retrieval"])


@router.post("/kis")
def kis(req: KISRequest, pipeline: RagPipeline = Depends(get_pipeline)):
    """Known-Item Search -- trả về list frame/timestamp, không gọi LLM."""
    pipeline.top_n_rerank = req.top_n
    try:
        return {"results": pipeline.run_kis(req.query)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qa")
def qa(req: QARequest, pipeline: RagPipeline = Depends(get_pipeline)):
    """Q&A/VQA -- trả về answer (text) + sources (list frame đã rerank)."""
    if pipeline.llm_client is None:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY chưa được set trên server")
    pipeline.top_n_rerank = req.top_n
    try:
        return pipeline.run_qa(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trake")
def trake(req: TRAKERequest, pipeline: RagPipeline = Depends(get_pipeline)):
    """TRAKE -- trả về list kết quả cho từng event, theo đúng thứ tự events truyền vào."""
    pipeline.top_n_rerank = req.top_n
    try:
        return {"results": pipeline.run_trake(req.events)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))