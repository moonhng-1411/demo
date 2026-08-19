import os
import re
from dataclasses import dataclass, replace, asdict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class Candidate:
    """Một kết quả retrieval thô, trước khi fusion.

    modality: "caption" | "asr" | "visual" (từ FAISS) | "fused" (sau merge()).
    """
    frame_id: int
    video_id: str
    timestamp: float
    score: float
    modality: str


class TextEmbedder:
    """Encode câu truy vấn text sang vector BGE-small (384-dim), dùng cho
    faiss_text.index (caption + asr gộp chung)."""

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, text: str) -> np.ndarray:
        """Trả về vector float32 đã normalize (để dùng inner product = cosine)."""
        return np.asarray(self.model.encode(text, normalize_embeddings=True), dtype="float32")


class ClipQueryEmbedder:
    """Encode câu truy vấn text sang vector CLIP (512-dim), dùng cho
    faiss_clip.index (ảnh keyframe) -- cùng không gian embedding với ảnh
    nên có thể query text -> ảnh trực tiếp."""

    def __init__(self, model_name="clip-ViT-B-32", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode(text), dtype="float32")


class Retriever:
    """Gộp kết quả từ 2 FAISS index (text: caption+asr, clip: ảnh) thành
    1 dict {modality: [Candidate]} để đưa vào fusion."""

    def __init__(self, faiss_manager, sqlite_manager, text_embedder=None, clip_embedder=None):
        self.faiss_manager = faiss_manager
        self.sqlite_manager = sqlite_manager
        self.text_embedder = text_embedder or TextEmbedder()
        self.clip_embedder = clip_embedder or ClipQueryEmbedder()

    def _text_candidate(self, entry: dict, modality: str, score: float) -> Candidate:
        """Chuyển 1 hit từ faiss_text.index thành Candidate.

        entry["kind"] == "asr": entry không có sẵn keyframe_id (chỉ có
        transcript_id) -- phải resolve sang keyframe gần nhất theo thời gian.
        entry["kind"] == "caption": đã có sẵn keyframe_id 1-1.
        """
        frame_id = (
            self.sqlite_manager.resolve_transcript_to_frame(entry["transcript_id"])
            if modality == "asr" else entry["keyframe_id"]
        )
        info = self.sqlite_manager.get_frame_info(frame_id)
        return Candidate(frame_id, info["video_id"], info["pts_time"], float(score), modality)

    def _visual_candidate(self, frame_id: int, score: float) -> Candidate:
        """Chuyển 1 hit từ faiss_clip.index (đã map sẵn ra keyframe_id qua
        clip_id_map.npy) thành Candidate."""
        info = self.sqlite_manager.get_frame_info(frame_id)
        return Candidate(frame_id, info["video_id"], info["pts_time"], float(score), "visual")

    def search(self, query: str, top_k: int = 50) -> dict:
        """Truy vấn cả 2 index (text + clip) song song, trả về:
        {"caption": [Candidate], "asr": [Candidate], "visual": [Candidate]}
        Dùng cho 1 câu query đơn (KIS/Q&A). Với TRAKE, gọi search() nhiều lần
        qua search_events()."""
        results = {"caption": [], "asr": [], "visual": []}

        text_vec = self.text_embedder.encode(query)
        for entry, modality, score in self.faiss_manager.search_text(text_vec, top_k=top_k):
            results[modality].append(self._text_candidate(entry, modality, score))

        clip_vec = self.clip_embedder.encode(query)
        for frame_id, score in self.faiss_manager.search_clip(clip_vec, top_k=top_k):
            results["visual"].append(self._visual_candidate(frame_id, score))

        return results

    def search_events(self, events: list[str], top_k: int = 50) -> list[dict]:
        """Dùng cho TRAKE -- search riêng từng event, giữ đúng thứ tự events
        truyền vào (không được sắp xếp lại)."""
        return [self.search(e, top_k=top_k) for e in events]


DEFAULT_MODALITY_WEIGHTS = {"caption": 1.0, "asr": 0.8, "visual": 1.0}
DEFAULT_OBJECT_WEIGHT = 0.5
RRF_K = 60  # hằng số chuẩn của Reciprocal Rank Fusion (Cormack et al.), ít nhạy với giá trị cụ thể


def _tokenize(text: str) -> list[str]:
    """Tách text thành list token chữ/số, lowercase -- dùng cho object score."""
    return re.findall(r"\w+", text.lower())


def _object_score(query: str, object_labels: list[str]) -> float:
    """Tỉ lệ token trong query khớp với nhãn object detect được trên frame.
    Trả về 0.0 nếu không có object hoặc query rỗng sau tokenize."""
    if not object_labels:
        return 0.0
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    labels_joined = " ".join(l.lower() for l in object_labels)
    matched = sum(1 for tok in query_tokens if tok in labels_joined)
    return matched / len(query_tokens)


def _rrf_scores(ranked_frame_ids: list[int], k: int = RRF_K) -> dict:
    """Reciprocal Rank Fusion: frame xếp hạng càng cao (rank nhỏ) thì điểm
    càng lớn, không phụ thuộc vào thang điểm gốc khác nhau giữa các modality."""
    return {fid: 1.0 / (k + rank) for rank, fid in enumerate(ranked_frame_ids, start=1)}


def merge(candidates_by_modality: dict, query: str = None, sqlite_manager=None,
          modality_weights: dict = None, object_weight: float = DEFAULT_OBJECT_WEIGHT) -> list:
    """Gộp candidate từ nhiều modality (caption/asr/visual) thành 1 list đã
    xếp hạng theo fused score = RRF theo từng modality (có trọng số) +
    object_weight * object_score (nếu có query + sqlite_manager).

    Trả về list[Candidate] với modality="fused", sort giảm dần theo score.
    """
    weights = modality_weights or DEFAULT_MODALITY_WEIGHTS
    frame_lookup = {}   # frame_id -> Candidate gốc (lấy video_id/timestamp)
    fused_scores = {}   # frame_id -> fused score

    for modality, candidates in candidates_by_modality.items():
        if not candidates:
            continue
        rrf = _rrf_scores([c.frame_id for c in candidates])
        w = weights.get(modality, 1.0)
        for c in candidates:
            frame_lookup.setdefault(c.frame_id, c)
            fused_scores[c.frame_id] = fused_scores.get(c.frame_id, 0.0) + w * rrf[c.frame_id]

    if query and sqlite_manager is not None:
        for fid in list(fused_scores.keys()):
            labels = sqlite_manager.get_frame_objects(fid)
            fused_scores[fid] += object_weight * _object_score(query, labels)

    merged = [replace(frame_lookup[fid], score=fused_scores[fid], modality="fused")
              for fid in fused_scores]
    merged.sort(key=lambda c: c.score, reverse=True)
    return merged


@dataclass
class RerankedResult:
    """Kết quả sau cross-encoder rerank -- document_text giữ lại để
    prompt_builder dùng làm ngữ cảnh cho LLM ở bước Q&A."""
    frame_id: int
    video_id: str
    timestamp: float
    rerank_score: float
    document_text: str


class Reranker:
    """Rerank list candidate đã fusion bằng cross-encoder, dựa trên văn bản
    tổng hợp (caption + ocr + asr) của từng frame."""

    def __init__(self, sqlite_manager, cross_encoder=None,
                 model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"):
        self.sqlite_manager = sqlite_manager
        self.cross_encoder = cross_encoder or CrossEncoder(model_name, device=device)

    def _build_document(self, c: Candidate) -> str:
        """Ghép caption/ocr/asr của 1 frame thành 1 đoạn text để cross-encoder
        chấm điểm liên quan với query."""
        texts = self.sqlite_manager.get_frame_texts(c.frame_id)
        parts = [texts.get(k) for k in ("caption_text", "ocr_text", "asr_text")]
        return ". ".join(p.strip() for p in parts if p and p.strip())

    @staticmethod
    def _dedupe(candidates: list) -> list:
        """Loại candidate trùng frame_id, giữ candidate xuất hiện đầu tiên
        (đề phòng fusion chưa dedupe hết)."""
        seen, unique = set(), []
        for c in candidates:
            if c.frame_id not in seen:
                seen.add(c.frame_id)
                unique.append(c)
        return unique

    def rerank(self, query: str, candidates: list, top_n: int = 10) -> list:
        """Rerank candidates đã fusion theo mức độ liên quan thật sự với query.
        Bỏ qua candidate không có text nào (document rỗng, cross-encoder
        không chấm được). Dùng cho cả KIS và Q&A/VQA."""
        unique = self._dedupe(candidates)
        docs = [self._build_document(c) for c in unique]
        valid = [(c, d) for c, d in zip(unique, docs) if d]
        if not valid:
            return []

        pairs = [(query, d) for _, d in valid]
        scores = [float(s) for s in self.cross_encoder.predict(pairs)]

        results = [RerankedResult(c.frame_id, c.video_id, c.timestamp, score, d)
                   for (c, d), score in zip(valid, scores)]
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_n]

    def rerank_events(self, events: list[str], per_event_candidates: list, top_n: int = 5) -> list:
        """Dùng cho TRAKE -- rerank riêng từng event, giữ nguyên thứ tự event
        truyền vào (không gộp candidate giữa các event với nhau)."""
        return [self.rerank(e, cands, top_n=top_n) for e, cands in zip(events, per_event_candidates)]


class GroqClient:
    """Wrapper gọi Groq API (LLM free-tier) để sinh câu trả lời cho Q&A."""

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, model=None, api_key=None, temperature=0.2, max_tokens=1024):
        from groq import Groq
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model = model or self.DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        """Gọi Groq chat completion, trả về text câu trả lời."""
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return r.choices[0].message.content


SYSTEM_PROMPT_QA = (
    "Bạn là trợ lý AI trả lời câu hỏi dựa trên nội dung video. "
    "Bạn CHỈ được dùng thông tin trong phần NGỮ CẢNH được cung cấp. "
    "Nếu ngữ cảnh không đủ, hãy nói rõ là không đủ thông tin, KHÔNG bịa thêm. "
    "Sau khi trả lời, LUÔN liệt kê nguồn (video_id và timestamp)."
)


def _format_context(results: list) -> str:
    """Ghép document_text của các RerankedResult thành 1 khối ngữ cảnh,
    đánh số nguồn rõ ràng để LLM trích dẫn lại."""
    return "\n\n".join(
        f"[Nguồn {i}] video_id={r.video_id}, timestamp={r.timestamp:.1f}s\nNội dung: {r.document_text}"
        for i, r in enumerate(results, start=1)
    )


def build_qa_answer(query: str, results: list, llm_client: GroqClient) -> str:
    """Sinh câu trả lời Q&A từ context đã rerank. Trả về thông báo cố định
    nếu không có kết quả nào (tránh gọi LLM vô ích khi chắc chắn không đủ ngữ cảnh)."""
    if not results:
        return "Không tìm thấy thông tin liên quan trong video để trả lời câu hỏi này."
    prompt = f"NGỮ CẢNH:\n{_format_context(results)}\n\nCÂU HỎI: {query}\n\n"
    return llm_client.generate_answer(SYSTEM_PROMPT_QA, prompt)


class RagPipeline:
    """Điều phối toàn bộ flow retrieve -> fuse -> rerank -> (LLM nếu cần),
    theo 3 loại truy vấn của AIC: KIS, QA/VQA, TRAKE."""

    def __init__(self, retriever, reranker, llm_client=None, top_k_retrieve=50, top_n_rerank=10):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client  # None hợp lệ nếu chỉ dùng KIS
        self.top_k_retrieve = top_k_retrieve
        self.top_n_rerank = top_n_rerank

    def run_kis(self, query: str) -> list[dict]:
        """Known-Item Search -- dừng lại ở rerank, không gọi LLM, để giảm
        độ trễ (yêu cầu quan trọng cho KIS trong thi AIC)."""
        cands = self.retriever.search(query, top_k=self.top_k_retrieve)
        fused = merge(cands, query=query, sqlite_manager=self.retriever.sqlite_manager)
        return [asdict(r) for r in self.reranker.rerank(query, fused, top_n=self.top_n_rerank)]

    def run_qa(self, query: str) -> dict:
        """Q&A/VQA -- đi tiếp bước gọi LLM sinh câu trả lời tự nhiên,
        kèm sources đã rerank để hiển thị bằng chứng."""
        if self.llm_client is None:
            raise ValueError("run_qa() cần llm_client (GroqClient) -- KIS thì không cần.")
        cands = self.retriever.search(query, top_k=self.top_k_retrieve)
        fused = merge(cands, query=query, sqlite_manager=self.retriever.sqlite_manager)
        reranked = self.reranker.rerank(query, fused, top_n=self.top_n_rerank)
        return {"answer": build_qa_answer(query, reranked, self.llm_client),
                "sources": [asdict(r) for r in reranked]}

    def run_trake(self, events: list[str]) -> list[list[dict]]:
        """TRAKE (multi-event/temporal) -- search+rerank riêng từng event,
        CHƯA align timestamp giữa các event trong cùng 1 video (TODO)."""
        cands_per_event = self.retriever.search_events(events, top_k=self.top_k_retrieve)
        fused_per_event = [merge(c, query=e, sqlite_manager=self.retriever.sqlite_manager)
                            for c, e in zip(cands_per_event, events)]
        reranked = self.reranker.rerank_events(events, fused_per_event, top_n=self.top_n_rerank)
        return [[asdict(r) for r in ev] for ev in reranked]

    def run(self, query_type: str, query: str = None, events: list[str] = None):
        """Entry point chung, chọn nhánh xử lý theo query_type ("KIS"|"QA"|"VQA"|"TRAKE")."""
        query_type = query_type.upper()
        if query_type == "KIS":
            return self.run_kis(query)
        if query_type in ("QA", "VQA"):
            return self.run_qa(query)
        if query_type == "TRAKE":
            return self.run_trake(events)
        raise ValueError(f"query_type không hợp lệ: {query_type}")