"""Các component Streamlit cho giao diện tìm kiếm."""

import io

import streamlit as st
from PIL import Image, ImageDraw
from api import AicApiClient

# Màu box theo entity, cycle qua danh sách này để phân biệt nhiều object khác nhau
_BOX_COLORS = ["#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE", "#00C7BE"]


def render_sidebar() -> tuple[str, int, int, bool, bool]:
    """Sidebar chọn loại truy vấn + tham số hiển thị."""
    with st.sidebar:
        st.header("Tuỳ chọn")
        mode = st.radio("Loại truy vấn", ["KIS", "Q&A", "TRAKE"])
        top_n = st.slider("Số kết quả", 1, 20, 10)
        n_cols = st.slider("Số cột hiển thị", 2, 6, 5)
        translate = st.toggle(
            "🌐 Dịch query sang tiếng Anh",
            value=True,
            help=(
                "Bật: query được dịch VI->EN trước khi tìm kiếm, thường khớp tốt hơn "
                "với caption/nhãn vật thể (chủ yếu tiếng Anh). "
                "Tắt: tìm kiếm bằng đúng câu gốc, không gọi bước dịch."
            ),
        )
        show_boxes = st.toggle(
            "🔲 Hiện bounding box object",
            value=False,
            help="Vẽ box các object đã detect được lên ảnh kết quả (dữ liệu có sẵn từ SQLite, vẽ ở frontend nên không tốn tài nguyên backend).",
        )
    return mode, top_n, n_cols, translate, show_boxes


def _draw_bboxes(image_bytes: bytes, objects: list[dict]) -> bytes:
    """Vẽ bounding box (toạ độ normalized 0-1) + nhãn entity lên ảnh.
    Trả về ảnh gốc nguyên trạng nếu không parse được hoặc không có object."""
    if not objects:
        return image_bytes
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return image_bytes
    w, h = img.size
    draw = ImageDraw.Draw(img)
    for i, obj in enumerate(objects):
        bbox = obj.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        xmin, ymin, xmax, ymax = bbox
        box_px = (xmin * w, ymin * h, xmax * w, ymax * h)
        color = _BOX_COLORS[i % len(_BOX_COLORS)]
        draw.rectangle(box_px, outline=color, width=3)
        label = f'{obj.get("entity", "?")} {obj.get("score", 0):.2f}'
        text_y = max(0, box_px[1] - 14)
        draw.rectangle((box_px[0], text_y, box_px[0] + 8 * len(label), text_y + 14), fill=color)
        draw.text((box_px[0] + 2, text_y), label, fill="white")
    out = io.BytesIO()
    img.save(out, format="JPEG")
    return out.getvalue()


def render_results(results: list[dict], api: AicApiClient, n_cols: int = 4, show_boxes: bool = False):
    """Hiển thị kết quả với schema công khai video_id/frame_idx/score.

    Ảnh được lấy riêng qua backend. Nếu key chưa tồn tại trên MinIO, metadata
    của kết quả vẫn được hiển thị và card chỉ hiện placeholder bằng text.
    ``show_boxes``: vẽ bounding box các object đã detect (field "objects" từ
    backend, đã kèm sẵn bbox normalized) trực tiếp ở frontend bằng PIL --
    không tốn thêm resource backend.
    """
    if not results:
        st.info("Không có kết quả.")
        return

    st.markdown(f'<div class="result-count">{len(results)} kết quả</div>', unsafe_allow_html=True)

    cols = st.columns(n_cols)
    for i, result in enumerate(results):
        video_id = result.get("video_id", "-")
        frame_idx = result.get("frame_idx", "-")
        keyframe_id = result.get("keyframe_id", "-")
        score = float(result.get("score", result.get("rerank_score", 0.0)))
        with cols[i % n_cols]:
            st.markdown('<div class="frame-card">', unsafe_allow_html=True)
            image = api.get_keyframe_image(video_id, frame_idx) if frame_idx != "-" else None
            if image is not None:
                if show_boxes:
                    image = _draw_bboxes(image, result.get("objects", []))
                st.image(image, use_container_width=True)
            else:
                st.info("Ảnh chưa có trên MinIO")
            st.markdown(
                f'<div class="frame-meta">'
                f'<span class="score-badge">score {score:.3f}</span><br>'
                f'<b>Video ID:</b> {video_id}<br>'
                f'<b>Frame idx:</b> {frame_idx}<br>'
                f'<b>Keyframe ID:</b> {keyframe_id}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)


def render_kis_input() -> str | None:
    """Ô input cho KIS -- trả về query khi bấm Tìm hoặc nhấn Enter.

    Bọc trong st.form vì text_input đứng riêng không tự "bấm" nút khi nhấn
    Enter (chỉ rerun app) -- st.form_submit_button mới bắt được Enter đúng
    hành vi submit form chuẩn của Streamlit.
    """
    with st.form("kis_form", clear_on_submit=False):
        query = st.text_input(
            "Nhập mô tả cảnh cần tìm",
            placeholder="vd: người đàn ông mặc áo đỏ đứng cạnh xe hơi",
        )
        submitted = st.form_submit_button("🔍 Tìm", type="primary")
    return query if submitted and query else None


def render_qa_input() -> str | None:
    """Ô input cho Q&A -- trả về query khi bấm Hỏi hoặc nhấn Enter."""
    with st.form("qa_form", clear_on_submit=False):
        query = st.text_input("Nhập câu hỏi", placeholder="vd: người đó đang làm gì trong video?")
        submitted = st.form_submit_button("💬 Hỏi", type="primary")
    return query if submitted and query else None


def render_trake_input() -> list[str] | None:
    """Ô input cho TRAKE -- trả về list events khi bấm Tìm hoặc nhấn Ctrl+Enter
    (text_area dùng Ctrl+Enter để submit form, do Enter đơn thuần dùng để
    xuống dòng nhập nhiều event)."""
    with st.form("trake_form", clear_on_submit=False):
        events_raw = st.text_area(
            "Nhập các sự kiện, mỗi dòng 1 sự kiện, theo đúng thứ tự (Ctrl+Enter để tìm)",
            placeholder="người bước vào phòng\nngười ngồi xuống ghế\nngười mở laptop",
            height=120,
        )
        submitted = st.form_submit_button("🔗 Tìm chuỗi sự kiện", type="primary")
    if submitted and events_raw.strip():
        return [e.strip() for e in events_raw.splitlines() if e.strip()]
    return None