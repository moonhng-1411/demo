"""Các component Streamlit cho giao diện tìm kiếm."""

import streamlit as st
from api import AicApiClient


def render_sidebar() -> tuple[str, int, int]:
    """Sidebar chọn loại truy vấn + tham số hiển thị."""
    with st.sidebar:
        st.header("Tuỳ chọn")
        mode = st.radio("Loại truy vấn", ["KIS", "Q&A", "TRAKE"])
        top_n = st.slider("Số kết quả", 1, 20, 10)
        n_cols = st.slider("Số cột hiển thị", 2, 6, 4)
    return mode, top_n, n_cols


def render_results(results: list[dict], api: AicApiClient, n_cols: int = 4):
    """Hiển thị kết quả với schema công khai video_id/frame_idx/score.

    Ảnh được lấy riêng qua backend. Nếu key chưa tồn tại trên MinIO, metadata
    của kết quả vẫn được hiển thị và card chỉ hiện placeholder bằng text.
    """
    if not results:
        st.info("Không có kết quả.")
        return

    st.markdown(f'<div class="result-count">{len(results)} kết quả</div>', unsafe_allow_html=True)

    cols = st.columns(n_cols)
    for i, result in enumerate(results):
        video_id = result.get("video_id", "-")
        frame_idx = result.get("frame_idx", "-")
        score = float(result.get("score", result.get("rerank_score", 0.0)))
        with cols[i % n_cols]:
            st.markdown('<div class="frame-card">', unsafe_allow_html=True)
            image = api.get_keyframe_image(video_id, frame_idx) if frame_idx != "-" else None
            if image is not None:
                st.image(image, use_container_width=True)
            else:
                st.info("Ảnh chưa có trên MinIO")
            st.markdown(
                f'<div class="frame-meta">'
                f'<span class="score-badge">score {score:.3f}</span><br>'
                f'<b>Video ID:</b> {video_id}<br>'
                f'<b>Frame idx:</b> {frame_idx}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)


def render_kis_input() -> str | None:
    """Ô input cho KIS -- trả về query khi bấm Tìm."""
    query = st.text_input("Nhập mô tả cảnh cần tìm", placeholder="vd: người đàn ông mặc áo đỏ đứng cạnh xe hơi")
    return query if st.button("🔍 Tìm", type="primary") and query else None


def render_qa_input() -> str | None:
    """Ô input cho Q&A -- trả về query khi bấm Hỏi."""
    query = st.text_input("Nhập câu hỏi", placeholder="vd: người đó đang làm gì trong video?")
    return query if st.button("💬 Hỏi", type="primary") and query else None


def render_trake_input() -> list[str] | None:
    """Ô input cho TRAKE -- trả về list events khi bấm Tìm."""
    events_raw = st.text_area(
        "Nhập các sự kiện, mỗi dòng 1 sự kiện, theo đúng thứ tự",
        placeholder="người bước vào phòng\nngười ngồi xuống ghế\nngười mở laptop",
        height=120,
    )
    if st.button("🔗 Tìm chuỗi sự kiện", type="primary") and events_raw.strip():
        return [e.strip() for e in events_raw.splitlines() if e.strip()]
    return None
