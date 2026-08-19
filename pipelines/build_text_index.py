"""Build lại faiss_text.index + text_embedding_metadata.jsonl từ đầu, gộp
caption + transcript trong CÙNG 1 lần chạy để đảm bảo index và metadata luôn
khớp số dòng (tránh lặp lại lỗi id_map.json cũ bị thiếu phần ASR).

CHỈ CẦN CHẠY LẠI khi có data mới (video mới, caption/transcript mới) --
bản hiện tại trong data/index/ đã đủ 278,503 dòng (177,321 caption +
101,182 asr), không cần build lại cho lần setup ban đầu."""

import sqlite3
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import DB_PATH, INDEX_ROOT

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64


def main():
    """Đọc toàn bộ caption + transcript có text, encode bằng BGE-small,
    ghi index + metadata theo đúng thứ tự (row i trong index khớp dòng i
    trong file .jsonl)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    metadata = []
    texts = []

    captions = conn.execute(
        "SELECT keyframe_id, video_id, caption_text FROM captions WHERE caption_text IS NOT NULL"
    ).fetchall()
    for c in captions:
        metadata.append({"kind": "caption", "keyframe_id": c["keyframe_id"],
                          "video_id": c["video_id"], "transcript_id": None})
        texts.append(c["caption_text"])

    transcripts = conn.execute(
        "SELECT id, video_id, text FROM transcripts WHERE text IS NOT NULL AND text != ''"
    ).fetchall()
    for t in transcripts:
        metadata.append({"kind": "asr", "keyframe_id": None,
                          "video_id": t["video_id"], "transcript_id": t["id"]})
        texts.append(t["text"])

    print(f"{len(captions)} caption + {len(transcripts)} asr = {len(texts)} dòng để encode")

    vecs = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                         show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    faiss.write_index(index, f"{INDEX_ROOT}/faiss_text.index")
    with open(f"{INDEX_ROOT}/text_embedding_metadata.jsonl", "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    assert index.ntotal == len(metadata), "index và metadata lệch số dòng -- không được xảy ra"
    print(f"Đã ghi faiss_text.index ({index.ntotal} vector) + metadata.jsonl khớp nhau")
    conn.close()


if __name__ == "__main__":
    main()