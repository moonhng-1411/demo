"""Điểm vào giao diện -- chỉ còn control flow, logic UI/API đã tách sang
components.py / api.py / styles.py."""

import time

import streamlit as st
from api import AicApiClient
from components import (
    render_sidebar,
    render_results,
    render_kis_input,
    render_qa_input,
    render_trake_input,
)
from styles import inject_css

st.set_page_config(page_title="AIC26 Video Retrieval", layout="wide", page_icon="🎥")
inject_css(st)
st.title("🎥 AIC26 — Video Retrieval")

api = AicApiClient(st.secrets.get("API_URL", "http://localhost:8000"))
mode, top_n, n_cols, translate = render_sidebar()

# Cache kết quả truy vấn gần nhất mỗi mode vào session_state. Cần thiết vì
# n_cols/top_n là widget nằm ngoài form (sidebar) -- đổi giá trị của chúng
# tự rerun toàn bộ script nhưng KHÔNG submit lại form, nên nếu không cache,
# kết quả cũ sẽ biến mất mỗi khi chỉnh "Số cột hiển thị" và người dùng phải
# bấm tìm lại. Render luôn đọc từ cache để áp n_cols mới ngay lập tức mà
# không gọi lại API.
if "cache" not in st.session_state:
    st.session_state.cache = {}


def _run_query(mode_key: str, spinner_text: str, fetch_fn):
    """Gọi API, lưu kết quả + thời gian vào session_state.cache[mode_key]."""
    with st.spinner(spinner_text):
        try:
            started = time.perf_counter()
            payload = fetch_fn()
            api_elapsed = time.perf_counter() - started
            st.session_state.cache[mode_key] = {
                "payload": payload,
                "api_elapsed": api_elapsed,
                "error": None,
            }
        except Exception as e:
            st.session_state.cache[mode_key] = {"payload": None, "api_elapsed": None, "error": str(e)}


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
    if query:
        _run_query("KIS", "Đang tìm...", lambda: api.search_kis(query, top_n=top_n, translate=translate))
    _render_cached("KIS", lambda results: render_results(results, api, n_cols))

elif mode == "Q&A":
    query = render_qa_input()
    if query:
        _run_query("QA", "Đang suy nghĩ...", lambda: api.ask_qa(query, top_n=top_n, translate=translate))

    def _render_qa(data):
        st.markdown(f"### Trả lời\n{data['answer']}")
        st.divider()
        render_results(data["sources"], api, n_cols)

    _render_cached("QA", _render_qa)

elif mode == "TRAKE":
    events = render_trake_input()
    if events:
        _run_query(
            "TRAKE", "Đang tìm...",
            lambda: {"events": events, "results_per_event": api.search_trake(events, top_n=top_n, translate=translate)},
        )

    def _render_trake(payload):
        for event_text, event_results in zip(payload["events"], payload["results_per_event"]):
            st.subheader(f"📌 {event_text}")
            render_results(event_results, api, n_cols)
            st.divider()

    _render_cached("TRAKE", _render_trake)