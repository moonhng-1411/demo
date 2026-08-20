import os
import re
import time
from dataclasses import dataclass, replace, asdict

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class Candidate:
    """Một kết quả retrieval thô, trước khi fusion.

    ``frame_id`` chỉ dùng nội bộ để join database; ``frame_idx`` là chỉ số
    frame theo video và là field công khai cho frontend.
    """
    frame_id: int
    video_id: str
    frame_idx: int
    timestamp: float
    score: float
    modality: str


class TextEmbedder:
    """Encode câu truy vấn text sang vector (384-dim), dùng cho faiss_text.index
    (caption + asr gộp chung).
    """

    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, text: str) -> np.ndarray:
        """Trả về vector float32 đã normalize (để dùng inner product = cosine)."""
        return np.asarray(self.model.encode(text, normalize_embeddings=True), dtype="float32")


class ClipQueryEmbedder:
    """Encode câu truy vấn text sang vector CLIP (512-dim), dùng cho
    faiss_clip.index (ảnh keyframe) -- cùng không gian embedding với ảnh
    nên có thể query text -> ảnh trực tiếp.
    """

    def __init__(self, model_name="clip-ViT-B-32-multilingual-v1", device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, text: str) -> np.ndarray:
        """Trả về vector float32 đã normalize (để dùng inner product = cosine),
        khớp đúng normalize=True lúc build index (metric cosine qua inner product)."""
        vector = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vector, dtype="float32")


class Retriever:
    """Gộp kết quả từ 2 FAISS index (text: caption+asr, clip: ảnh) thành
    1 dict {modality: [Candidate]} để đưa vào fusion."""

    def __init__(self, faiss_manager, sqlite_manager, text_embedder=None, clip_embedder=None):
        self.faiss_manager = faiss_manager
        self.sqlite_manager = sqlite_manager
        self.text_embedder = text_embedder or TextEmbedder()
        self.clip_embedder = clip_embedder or ClipQueryEmbedder()

    def _text_candidate(self, entry: dict, modality: str, score: float) -> Candidate:
        """Chuyển một hit text thành Candidate.

        ASR metadata hiện tại có ``video_id``, ``start_s`` và ``end_s`` nhưng
        không có ``transcript_id``. Vẫn hỗ trợ schema cũ nếu entry có
        transcript_id để không phá các index cũ.
        """
        if modality == "asr":
            transcript_id = entry.get("transcript_id")
            if transcript_id is not None:
                frame_id = self.sqlite_manager.resolve_transcript_to_frame(transcript_id)
            else:
                required = ("video_id", "start_s", "end_s")
                missing = [key for key in required if entry.get(key) is None]
                if missing:
                    raise KeyError("ASR metadata thiếu trường " + ", ".join(missing))
                frame_id = self.sqlite_manager.resolve_asr_to_frame(
                    entry["video_id"], entry["start_s"], entry["end_s"]
                )
        else:
            frame_id = entry.get("keyframe_id")
            if frame_id is None:
                raise KeyError("caption metadata thiếu keyframe_id")

        info = self.sqlite_manager.get_frame_info(int(frame_id))
        return Candidate(
            int(frame_id), info["video_id"], info["frame_idx"], info["pts_time"],
            float(score), modality,
        )

    def _visual_candidate(self, frame_id: int, score: float) -> Candidate:
        """Chuyển 1 hit từ faiss_clip.index (đã map sẵn ra keyframe_id qua
        clip_id_map.npy) thành Candidate."""
        info = self.sqlite_manager.get_frame_info(frame_id)
        return Candidate(
            int(frame_id), info["video_id"], info["frame_idx"], info["pts_time"],
            float(score), "visual",
        )

    def search(self, query: str, top_k: int = 50) -> dict:
        """Truy vấn cả hai index text và CLIP."""
        results = {"caption": [], "asr": [], "visual": []}
        profile = os.environ.get("RAG_PROFILE", "0") == "1"
        started = time.perf_counter()

        text_vec = self.text_embedder.encode(query)
        text_encoded = time.perf_counter()
        for entry, modality, score in self.faiss_manager.search_text(text_vec, top_k=top_k):
            results[modality].append(self._text_candidate(entry, modality, score))
        text_searched = time.perf_counter()

        clip_vec = self.clip_embedder.encode(query)
        clip_encoded = time.perf_counter()
        for frame_id, score in self.faiss_manager.search_clip(clip_vec, top_k=top_k):
            results["visual"].append(self._visual_candidate(frame_id, score))
        finished = time.perf_counter()

        if profile:
            print(
                "[RAG_PROFILE] "
                f"text_encode={text_encoded-started:.2f}s "
                f"text_search_map={text_searched-text_encoded:.2f}s "
                f"clip_encode={clip_encoded-text_searched:.2f}s "
                f"clip_search_map={finished-clip_encoded:.2f}s "
                f"total_retrieve={finished-started:.2f}s"
            )
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
    """Kết quả nội bộ sau cross-encoder rerank.

    ``frame_id`` giữ lại để lấy caption/ASR và ảnh; response API sẽ loại field
    này và chỉ công khai video_id, frame_idx, score.
    """
    frame_id: int
    video_id: str
    frame_idx: int
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

        results = [RerankedResult(
            c.frame_id, c.video_id, c.frame_idx, c.timestamp, score, d
        ) for (c, d), score in zip(valid, scores)]
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_n]

    def rerank_events(self, events: list[str], per_event_candidates: list, top_n: int = 5) -> list:
        """Dùng cho TRAKE -- rerank riêng từng event, giữ nguyên thứ tự event
        truyền vào (không gộp candidate giữa các event với nhau)."""
        return [self.rerank(e, cands, top_n=top_n) for e, cands in zip(events, per_event_candidates)]


class GroqClient:
    """Wrapper gọi Groq API (LLM free-tier) để sinh câu trả lời cho Q&A."""

    # Groq đã deprecate llama-3.3-70b-versatile (thông báo 17/6/2026);
    # openai/gpt-oss-120b là model thay thế được Groq khuyến nghị.
    # Có thể override bằng biến môi trường GROQ_MODEL nếu Groq đổi model lần nữa.
    DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

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

    @staticmethod
    def _public_result(result: RerankedResult) -> dict:
        """Response sau CrossEncoder; không lộ keyframe_id nội bộ."""
        return {
            "video_id": result.video_id,
            "frame_idx": result.frame_idx,
            "score": float(result.rerank_score),
        }

    @staticmethod
    def _public_candidate(candidate: Candidate) -> dict:
        """Response nhanh dùng điểm fusion, không chạy CrossEncoder."""
        return {
            "video_id": candidate.video_id,
            "frame_idx": candidate.frame_idx,
            "score": float(candidate.score),
        }

    def __init__(self, retriever, reranker, llm_client=None, top_k_retrieve=50,
                 top_n_rerank=10, fast_kis=False):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client  # None hợp lệ nếu chỉ dùng KIS
        self.top_k_retrieve = top_k_retrieve
        self.top_n_rerank = top_n_rerank
        self.fast_kis = fast_kis

    def run_kis(self, query: str, top_n: int | None = None) -> list[dict]:
        """KIS; có thể bỏ CrossEncoder bằng ``RAG_FAST_KIS=1``."""
        top_n = top_n if top_n is not None else self.top_n_rerank
        started = time.perf_counter()
        cands = self.retriever.search(query, top_k=self.top_k_retrieve)
        fused = merge(cands, query=query, sqlite_manager=self.retriever.sqlite_manager)
        if self.fast_kis:
            results = [self._public_candidate(c) for c in fused[:top_n]]
        else:
            results = [self._public_result(r)
                       for r in self.reranker.rerank(
                           query, fused, top_n=top_n
                       )]
        if os.environ.get("RAG_PROFILE", "0") == "1":
            print(
                f"[RAG_PROFILE] total_kis={time.perf_counter()-started:.2f}s "
                f"fast_kis={self.fast_kis} results={len(results)}"
            )
        return results

    def run_qa(self, query: str, top_n: int | None = None) -> dict:
        """Q&A/VQA -- đi tiếp bước gọi LLM sinh câu trả lời tự nhiên,
        kèm sources đã rerank để hiển thị bằng chứng."""
        top_n = top_n if top_n is not None else self.top_n_rerank
        if self.llm_client is None:
            raise ValueError("run_qa() cần llm_client (GroqClient) -- KIS thì không cần.")
        cands = self.retriever.search(query, top_k=self.top_k_retrieve)
        fused = merge(cands, query=query, sqlite_manager=self.retriever.sqlite_manager)
        reranked = self.reranker.rerank(query, fused, top_n=top_n)
        return {"answer": build_qa_answer(query, reranked, self.llm_client),
                "sources": [self._public_result(r) for r in reranked]}

    _ALIGN_POOL_MULTIPLIER = 3
    _MAX_ALIGN_POOL = 50

    @staticmethod
    def _align_trake_events(events_results: list) -> list:
        """Ưu tiên chuỗi frame có timestamp tăng dần khớp đúng thứ tự event,
        trong cùng 1 video (đây là ý nghĩa thực sự của TRAKE).

        Với mỗi video xuất hiện trong TẤT CẢ các event, chọn tham lam khung
        hình sớm nhất có timestamp lớn hơn khung hình đã chọn ở event trước.
        Nếu tìm được chuỗi hợp lệ cho video đó, các candidate thuộc chuỗi
        được đẩy lên đầu danh sách kết quả của từng event; các candidate còn
        lại giữ nguyên thứ tự rerank cũ ở phía sau.
        """
        if len(events_results) < 2:
            return events_results

        video_sets = [set(r.video_id for r in ev) for ev in events_results]
        common_videos = set.intersection(*video_sets) if all(video_sets) else set()

        aligned_ids_per_event = [set() for _ in events_results]
        for video_id in common_videos:
            prev_ts = float("-inf")
            chosen = []
            for ev in events_results:
                candidates = [r for r in ev if r.video_id == video_id and r.timestamp > prev_ts]
                if not candidates:
                    chosen = []
                    break
                best = min(candidates, key=lambda r: r.timestamp)
                chosen.append(best)
                prev_ts = best.timestamp
            if chosen:
                for idx, r in enumerate(chosen):
                    aligned_ids_per_event[idx].add(r.frame_id)

        aligned_results = []
        for ev, aligned_ids in zip(events_results, aligned_ids_per_event):
            aligned = [r for r in ev if r.frame_id in aligned_ids]
            rest = [r for r in ev if r.frame_id not in aligned_ids]
            aligned_results.append(aligned + rest)
        return aligned_results

    def run_trake(self, events: list[str], top_n: int | None = None) -> list[list[dict]]:
        """TRAKE (multi-event/temporal) -- search+rerank riêng từng event, sau đó
        align timestamp giữa các event trong cùng video để ưu tiên đúng chuỗi
        sự kiện theo thứ tự thời gian."""
        top_n = top_n if top_n is not None else self.top_n_rerank
        pool_n = min(max(top_n * self._ALIGN_POOL_MULTIPLIER, top_n), self._MAX_ALIGN_POOL)
        cands_per_event = self.retriever.search_events(events, top_k=self.top_k_retrieve)
        fused_per_event = [merge(c, query=e, sqlite_manager=self.retriever.sqlite_manager)
                            for c, e in zip(cands_per_event, events)]
        reranked = self.reranker.rerank_events(events, fused_per_event, top_n=pool_n)
        aligned = self._align_trake_events(reranked)
        truncated = [ev[:top_n] for ev in aligned]
        return [[self._public_result(r) for r in ev] for ev in truncated]

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