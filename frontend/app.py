"""Streamlit UI -- gọi backend API, không chứa logic retrieval nào ở đây."""

import streamlit as st
import requests

API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AIC26 Video Retrieval", layout="wide")
st.title("🎥 AIC26 — Video Retrieval")

mode = st.sidebar.radio("Loại truy vấn", ["KIS", "Q&A", "TRAKE"])
top_n = st.sidebar.slider("Số kết quả", 1, 20, 10)


def render_results(results: list[dict]):
    """Hiển thị lưới ảnh keyframe (4 cột) cho 1 list kết quả đã rerank."""
    cols = st.columns(4)
    for i, r in enumerate(results):
        with cols[i % 4]:
            img_url = f"{API_URL}/api/keyframe/{r['frame_id']}/image"
            st.image(img_url, use_container_width=True)
            st.caption(f"{r['video_id']} @ {r['timestamp']:.1f}s\nscore={r.get('rerank_score', 0):.3f}")


if mode == "KIS":
    query = st.text_input("Nhập mô tả cảnh cần tìm")
    if st.button("Tìm") and query:
        with st.spinner("Đang tìm..."):
            resp = requests.post(f"{API_URL}/api/kis", json={"query": query, "top_n": top_n})
        if resp.ok:
            render_results(resp.json()["results"])
        else:
            st.error(resp.text)

elif mode == "Q&A":
    query = st.text_input("Nhập câu hỏi")
    if st.button("Hỏi") and query:
        with st.spinner("Đang suy nghĩ..."):
            resp = requests.post(f"{API_URL}/api/qa", json={"query": query, "top_n": top_n})
        if resp.ok:
            data = resp.json()
            st.markdown(data["answer"])
            st.divider()
            render_results(data["sources"])
        else:
            st.error(resp.text)

elif mode == "TRAKE":
    events_raw = st.text_area("Nhập các sự kiện, mỗi dòng 1 sự kiện, theo đúng thứ tự")
    if st.button("Tìm chuỗi sự kiện") and events_raw.strip():
        events = [e.strip() for e in events_raw.splitlines() if e.strip()]
        with st.spinner("Đang tìm..."):
            resp = requests.post(f"{API_URL}/api/trake", json={"events": events, "top_n": top_n})
        if resp.ok:
            for event_text, event_results in zip(events, resp.json()["results"]):
                st.subheader(event_text)
                render_results(event_results)
        else:
            st.error(resp.text)