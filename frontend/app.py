"""Điểm vào giao diện -- chỉ còn control flow, logic UI/API đã tách sang
components.py / api.py / styles.py."""

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
mode, top_n, n_cols = render_sidebar()

if mode == "KIS":
    query = render_kis_input()
    if query:
        with st.spinner("Đang tìm..."):
            try:
                results = api.search_kis(query, top_n=top_n)
                render_results(results, api, n_cols)
            except Exception as e:
                st.error(str(e))

elif mode == "Q&A":
    query = render_qa_input()
    if query:
        with st.spinner("Đang suy nghĩ..."):
            try:
                data = api.ask_qa(query, top_n=top_n)
                st.markdown(f"### Trả lời\n{data['answer']}")
                st.divider()
                render_results(data["sources"], api, n_cols)
            except Exception as e:
                st.error(str(e))

elif mode == "TRAKE":
    events = render_trake_input()
    if events:
        with st.spinner("Đang tìm..."):
            try:
                results_per_event = api.search_trake(events, top_n=top_n)
                for event_text, event_results in zip(events, results_per_event):
                    st.subheader(f"📌 {event_text}")
                    render_results(event_results, api, n_cols)
                    st.divider()
            except Exception as e:
                st.error(str(e))