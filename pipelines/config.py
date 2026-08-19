"""Đường dẫn dùng chung cho toàn bộ script trong pipelines/, đọc từ env var
để mỗi máy trong nhóm có thể trỏ tới data/ khác nhau nếu cần.
Mặc định các path đều tính từ thư mục gốc project (nơi chạy python -m)."""

import os

DB_PATH = os.environ.get("AIC_DB_PATH", "data/aic.sqlite")
INDEX_ROOT = os.environ.get("AIC_INDEX_ROOT", "data/index")

os.makedirs(INDEX_ROOT, exist_ok=True)