"""CSS tuỳ biến cho giao diện -- tách riêng để đổi theme không phải sửa app.py."""

CUSTOM_CSS = """
<style>
.frame-card {
    border: 1px solid #333;
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 12px;
    background-color: #1e1e1e;
}
.frame-meta {
    font-size: 0.82rem;
    color: #ccc;
    margin-top: 4px;
    line-height: 1.4;
}
.frame-meta b { color: #fff; }
.score-badge {
    display: inline-block;
    background-color: #2e7d32;
    color: white;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: 600;
}
.result-count {
    color: #999;
    font-size: 0.85rem;
    margin-bottom: 8px;
}
</style>
"""


def inject_css(st):
    """Nạp CSS vào trang -- gọi 1 lần đầu app.py."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)