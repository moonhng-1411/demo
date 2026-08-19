"""Client gọi backend AIC26 API -- tách riêng để app.py không lẫn logic HTTP
với logic hiển thị."""

import requests

API_URL_DEFAULT = "http://localhost:8000"


class ApiError(Exception):
    """Bọc lỗi HTTP thành exception rõ ràng, kèm status code + message từ backend."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{status_code}] {detail}")


class AicApiClient:
    def __init__(self, base_url: str = API_URL_DEFAULT):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        resp = requests.post(f"{self.base_url}{path}", json=payload)
        if not resp.ok:
            raise ApiError(resp.status_code, resp.text)
        return resp.json()

    def health(self) -> bool:
        """True nếu backend đang sống, False nếu không kết nối được."""
        try:
            resp = requests.get(f"{self.base_url}/api/health", timeout=3)
            return resp.ok
        except requests.RequestException:
            return False

    def search_kis(self, query: str, top_n: int = 10) -> list[dict]:
        """Known-Item Search -- trả về list kết quả đã rerank."""
        data = self._post("/api/kis", {"query": query, "top_n": top_n})
        return data["results"]

    def ask_qa(self, query: str, top_n: int = 10) -> dict:
        """Q&A -- trả về {"answer": str, "sources": list[dict]}."""
        return self._post("/api/qa", {"query": query, "top_n": top_n})

    def search_trake(self, events: list[str], top_n: int = 5) -> list[list[dict]]:
        """TRAKE -- trả về list kết quả cho từng event, đúng thứ tự events truyền vào."""
        data = self._post("/api/trake", {"events": events, "top_n": top_n})
        return data["results"]

    def keyframe_image_url(self, frame_id: int) -> str:
        """URL ảnh keyframe (backend redirect sang presigned MinIO URL)."""
        return f"{self.base_url}/api/keyframe/{frame_id}/image"