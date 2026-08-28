"""Điểm vào giao diện -- chỉ còn control flow, logic UI/API đã tách sang
components.py / api.py / styles.py."""

import os
import time

import streamlit as st
from dotenv import load_dotenv
from api import AicApiClient
from components import (
    render_sidebar,
    render_results,
    render_kis_input,
    render_qa_input,
    render_trake_input,
)
from styles import inject_css

load_dotenv()

st.set_page_config(page_title="AIC26 Video Retrieval", layout="wide", page_icon="🎥")
inject_css(st)
st.title("🎥 AIC26 — Video Retrieval")

API_URL = os.environ.get("API_URL") or st.secrets.get("API_URL", "http://localhost:8000")
api = AicApiClient(API_URL)
mode, top_n, n_cols, translate, show_boxes, neighbor_window = render_sidebar()

# Cache kết quả truy vấn gần nhất mỗi mode vào session_state. Cần thiết vì
# n_cols/top_n là widget nằm ngoài form (sidebar) -- đổi giá trị của chúng
# tự rerun toàn bộ script nhưng KHÔNG submit lại form, nên nếu không cache,
# kết quả cũ sẽ biến mất mỗi khi chỉnh "Số cột hiển thị" và người dùng phải
# bấm tìm lại. Render luôn đọc từ cache để áp n_cols mới ngay lập tức mà
# không gọi lại API.
if "cache" not in st.session_state:
    st.session_state.cache = {}


def _run_query(mode_key: str, spinner_text: str, fetch_fn, top_n: int, query_key):
    """Gọi API, lưu kết quả + thời gian + top_n/query đã dùng vào session_state.cache[mode_key].

    query_key được lưu lại để có thể refetch tự động khi top_n đổi mà
    form KHÔNG được submit lại (xem _maybe_refetch)."""
    with st.spinner(spinner_text):
        try:
            started = time.perf_counter()
            payload = fetch_fn()
            api_elapsed = time.perf_counter() - started
            st.session_state.cache[mode_key] = {
                "payload": payload,
                "api_elapsed": api_elapsed,
                "error": None,
                "top_n": top_n,
                "query_key": query_key,
            }
        except Exception as e:
            st.session_state.cache[mode_key] = {
                "payload": None, "api_elapsed": None, "error": str(e),
                "top_n": top_n, "query_key": query_key,
            }


def _maybe_refetch(mode_key: str, new_query, top_n: int, spinner_text: str, fetch_fn, query_key=None):
    """Quyết định có cần gọi lại API hay không, và gọi nếu cần.

    - new_query có giá trị (vừa bấm nút submit) -> luôn fetch với query mới.
    - new_query là None (rerun do đổi slider/n_cols) nhưng đã có cache trước
      đó với top_n khác -> refetch lại bằng query đã lưu trong cache, để
      slider "Số kết quả" có tác dụng ngay cả khi không bấm lại nút Tìm.
    """
    cached = st.session_state.cache.get(mode_key)
    if new_query:
        _run_query(mode_key, spinner_text, fetch_fn, top_n, query_key if query_key is not None else new_query)
    elif cached is not None and cached.get("error") is None and cached.get("top_n") != top_n:
        st.session_state.cache[mode_key]["top_n"] = top_n  # tránh loop nếu fetch_fn ném lỗi
        _run_query(mode_key, spinner_text, fetch_fn, top_n, cached.get("query_key"))


def _render_cached(mode_key: str, render_fn):
    """Render lại payload đã cache (nếu có) cho mode hiện tại, dùng n_cols
    mới nhất -- được gọi lại mỗi lần rerun kể cả khi chỉ đổi slider."""
    cached = st.session_state.cache.get(mode_key)
    if cached is None:
        return
    if cached["error"] is not None:
        st.error(cached["error"])
        return
    st.caption(f"Thời gian truy vấn API (lần gần nhất): {cached['api_elapsed']:.2f} giây")
    render_fn(cached["payload"])


if mode == "KIS":
    query = render_kis_input()
    cached_kis = st.session_state.cache.get("KIS")
    effective_query = query or (cached_kis.get("query_key") if cached_kis else None)
    if effective_query:
        _maybe_refetch(
            "KIS", query, top_n, "Đang tìm...",
            lambda: api.search_kis(effective_query, top_n=top_n, translate=translate),
            query_key=effective_query,
        )
    _render_cached(
        "KIS",
        lambda results: render_results(results, api, n_cols, show_boxes, neighbor_window, key_prefix="kis"),
    )

elif mode == "Q&A":
    query = render_qa_input()
    cached_qa = st.session_state.cache.get("QA")
    effective_query = query or (cached_qa.get("query_key") if cached_qa else None)
    if effective_query:
        _maybe_refetch(
            "QA", query, top_n, "Đang suy nghĩ...",
            lambda: api.ask_qa(effective_query, top_n=top_n, translate=translate),
            query_key=effective_query,
        )

    def _render_qa(data):
        reasoning = (data.get("reasoning") or "").strip()
        if reasoning:
            with st.expander("Đang suy nghĩ...", expanded=False):
                st.markdown(reasoning)
        st.markdown(f"### Trả lời\n{data['answer']}")
        st.divider()
        render_results(data["sources"], api, n_cols, show_boxes, neighbor_window, key_prefix="qa")

    _render_cached("QA", _render_qa)

elif mode == "TRAKE":
    events = render_trake_input()
    cached_trake = st.session_state.cache.get("TRAKE")
    effective_events = events or (cached_trake.get("query_key") if cached_trake else None)
    if effective_events:
        _maybe_refetch(
            "TRAKE", events, top_n, "Đang tìm...",
            lambda: {
                "events": effective_events,
                "results_per_event": api.search_trake(effective_events, top_n=top_n, translate=translate),
            },
            query_key=effective_events,
        )

    def _render_trake(payload):
        for ev_i, (event_text, event_results) in enumerate(zip(payload["events"], payload["results_per_event"])):
            st.subheader(f"📌 {event_text}")
            render_results(
                event_results, api, n_cols, show_boxes, neighbor_window,
                key_prefix=f"trake-{ev_i}",
            )
            st.divider()

    _render_cached("TRAKE", _render_trake)