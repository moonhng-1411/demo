"""Các component Streamlit cho giao diện tìm kiếm."""

import io

import streamlit as st
from PIL import Image, ImageDraw
from api import AicApiClient

# Màu box theo entity, cycle qua danh sách này để phân biệt nhiều object khác nhau
_BOX_COLORS = ["#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE", "#00C7BE"]


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_keyframe_image(_api: AicApiClient, video_id, frame_idx) -> bytes | None:
    """Cache ảnh keyframe theo (video_id, frame_idx) trong session.

    Tham số ``_api`` đặt dấu gạch dưới để Streamlit KHÔNG hash object này khi
    tính cache key (AicApiClient không hashable theo nghĩa hữu ích) -- chỉ
    video_id/frame_idx mới quyết định cache hit. Nhờ vậy khi rerun do mở
    dialog xem lân cận, các card kết quả khác không tải lại ảnh đã có."""
    return _api.get_keyframe_image(video_id, frame_idx)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_keyframe_neighbors(_api: AicApiClient, video_id, frame_idx, window: int) -> list[dict]:
    """Cache metadata neighbor theo (video_id, frame_idx, window), cùng lý do
    dùng ``_api`` như trên."""
    return _api.get_keyframe_neighbors(video_id, frame_idx, window=window)


def render_sidebar() -> tuple[str, int, int, bool, bool, int]:
    """Sidebar chọn loại truy vấn + tham số hiển thị.

    ``neighbor_window`` trả về 0 nếu tắt tính năng xem keyframe lân cận
    (không hiện expander trên frame-card), >0 là số keyframe lấy về mỗi
    phía khi mở expander."""
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
        show_neighbors = st.toggle(
            "🎞️ Cho phép xem keyframe lân cận",
            value=True,
            help="Hiện nút mở rộng trên mỗi kết quả để xem các keyframe ngay trước/sau nó trong cùng video.",
        )
        neighbor_window = (
            st.slider("Số keyframe lấy mỗi phía khi bấm xem", 1, 10, 10) if show_neighbors else 0
        )
    return mode, top_n, n_cols, translate, show_boxes, neighbor_window


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


@st.dialog("🎞️ Keyframe lân cận", width="large")
def _neighbor_dialog(video_id, frame_idx, api: AicApiClient, window: int, key_prefix: str = "res"):
    """Modal kiểu thư viện ảnh (Google Photos): ảnh chính phóng to ở trên,
    2 nút mũi tên ◀▶ để trượt qua lại, VÀ filmstrip toàn bộ neighbor bên
    dưới (ảnh đang xem được viền sáng) -- bấm thumbnail cũng nhảy tới đó.

    ``st.dialog`` chạy nội dung như 1 fragment riêng (Streamlit >=1.37):
    tương tác widget bên trong dialog chỉ rerun dialog, KHÔNG rerun toàn bộ
    app -- nên ``st.rerun(scope="fragment")`` ở đây không đụng tới lưới kết
    quả phía sau (không tải lại ảnh đáp án)."""
    neighbors = _cached_keyframe_neighbors(api, video_id, frame_idx, window)
    if not neighbors:
        st.caption("Không tìm được keyframe lân cận (video có thể chỉ có 1 keyframe, hoặc lỗi kết nối).")
        return

    idx_key = f"nb_pos-{key_prefix}-{video_id}-{frame_idx}"
    target_pos = next((k for k, nb in enumerate(neighbors) if nb.get("is_target")), len(neighbors) // 2)
    pos = st.session_state.get(idx_key, target_pos)
    pos = max(0, min(pos, len(neighbors) - 1))

    current = neighbors[pos]
    cur_frame_idx = current.get("frame_idx")
    main_image = _cached_keyframe_image(api, video_id, cur_frame_idx)

    st.markdown(
        """
        <style>
        @keyframes nbSlideIn { from { opacity: 0; transform: translateX(var(--nb-slide-x, 24px)); }
                                to   { opacity: 1; transform: translateX(0); } }
        .nb-slide { animation: nbSlideIn 0.2s ease-out; }
        .nb-thumb-active img { border: 3px solid #FF9500 !important; border-radius: 4px; }
        /* Nút chọn frame trong filmstrip: chữ không xuống dòng, cỡ nhỏ lại
           để không vỡ layout khi có nhiều cột hẹp (window lớn). */
        .nb-strip button p { white-space: nowrap !important; font-size: 0.78rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Ảnh chính + mũi tên trượt ---
    nav_prev, nav_main, nav_next = st.columns([1, 8, 1])
    with nav_prev:
        st.write("")
        st.write("")
        if st.button("◀", key=f"{idx_key}-prev", disabled=pos == 0, width="stretch"):
            st.session_state[idx_key] = pos - 1
            st.rerun(scope="fragment")
    with nav_next:
        st.write("")
        st.write("")
        if st.button("▶", key=f"{idx_key}-next", disabled=pos == len(neighbors) - 1, width="stretch"):
            st.session_state[idx_key] = pos + 1
            st.rerun(scope="fragment")
    with nav_main:
        slide_dir = "-24px" if pos < target_pos else "24px"
        st.markdown(f'<div class="nb-slide" style="--nb-slide-x:{slide_dir}">', unsafe_allow_html=True)
        if main_image is not None:
            st.image(main_image, width="stretch")
        else:
            st.info("Chưa có ảnh")
        st.markdown("</div>", unsafe_allow_html=True)

    caption = f"Video {video_id} — frame idx {cur_frame_idx} ({pos + 1}/{len(neighbors)})"
    if current.get("is_target"):
        caption += "  ➡️ gốc"
    st.caption(caption)

    # --- Filmstrip toàn bộ neighbor, ảnh đang xem viền cam ---
    # Giới hạn tối đa _STRIP_MAX_COLS cột/hàng: window lớn (vd 10 -> tối đa
    # 21 neighbor) nếu nhét hết vào 1 hàng thì mỗi cột quá hẹp, chữ "idx xxx"
    # vỡ dòng xấu (đã gặp thực tế). Nhiều hơn thì tự xuống hàng tiếp theo.
    _STRIP_MAX_COLS = 8
    st.divider()
    st.markdown('<div class="nb-strip">', unsafe_allow_html=True)
    for row_start in range(0, len(neighbors), _STRIP_MAX_COLS):
        row_neighbors = list(enumerate(neighbors))[row_start:row_start + _STRIP_MAX_COLS]
        strip_cols = st.columns(len(row_neighbors))
        for col, (j, nb) in zip(strip_cols, row_neighbors):
            with col:
                nb_frame_idx = nb.get("frame_idx")
                thumb = _cached_keyframe_image(api, video_id, nb_frame_idx)
                css_class = "nb-thumb-active" if j == pos else ""
                st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
                if thumb is not None:
                    st.image(thumb, width="stretch")
                else:
                    st.info("—")
                st.markdown("</div>", unsafe_allow_html=True)
                thumb_label = f"{nb_frame_idx}"
                if nb.get("is_target"):
                    thumb_label = f"➡️ {thumb_label}"
                if st.button(thumb_label, key=f"{idx_key}-pick-{nb_frame_idx}", width="stretch"):
                    st.session_state[idx_key] = j
                    st.rerun(scope="fragment")
    st.markdown("</div>", unsafe_allow_html=True)


@st.fragment
def _neighbor_trigger(video_id, frame_idx, api: AicApiClient, window: int, key_prefix: str):
    """Nút mở dialog xem lân cận, bọc trong fragment riêng để bấm nút này
    (và tương tác bên trong dialog) không kéo theo rerun toàn bộ lưới kết
    quả -- các ảnh đáp án khác trên trang giữ nguyên, không tải/vẽ lại."""
    if st.button(
        "🔍 Xem lân cận",
        key=f"nb_open-{key_prefix}-{video_id}-{frame_idx}",
        width='stretch',
    ):
        _neighbor_dialog(video_id, frame_idx, api, window, key_prefix=key_prefix)


def render_results(
    results: list[dict],
    api: AicApiClient,
    n_cols: int = 4,
    show_boxes: bool = False,
    neighbor_window: int = 0,
    key_prefix: str = "res",
):
    """Hiển thị kết quả với schema công khai video_id/frame_idx/score.

    Ảnh được lấy riêng qua backend. Nếu key chưa tồn tại trên MinIO, metadata
    của kết quả vẫn được hiển thị và card chỉ hiện placeholder bằng text.
    ``show_boxes``: vẽ bounding box các object đã detect (field "objects" từ
    backend, đã kèm sẵn bbox normalized) trực tiếp ở frontend bằng PIL --
    không tốn thêm resource backend.
    ``neighbor_window``: 0 để tắt hẳn expander "keyframe lân cận"; >0 là số
    keyframe lấy mỗi phía khi người dùng mở expander (xem
    ``_render_neighbor_strip``).
    ``key_prefix``: tiền tố duy nhất cho key Streamlit của expander mỗi
    card -- cần thiết khi cùng 1 kết quả có thể lặp lại ở nhiều nơi trong
    trang (vd nhiều event của TRAKE) để tránh đụng key.
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
            image = _cached_keyframe_image(api, video_id, frame_idx) if frame_idx != "-" else None
            if image is not None:
                if show_boxes:
                    image = _draw_bboxes(image, result.get("objects", []))
                st.image(image, width='stretch')
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
            if neighbor_window > 0 and frame_idx != "-":
                _neighbor_trigger(video_id, frame_idx, api, neighbor_window, key_prefix)
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