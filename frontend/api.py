"""HTTP client cho backend AIC, tách khỏi logic hiển thị."""

from urllib.parse import quote

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
        resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=120)
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
        """Known-Item Search -- trả về video_id, frame_idx và score."""
        data = self._post("/api/kis", {"query": query, "top_n": top_n})
        return data["results"]

    def ask_qa(self, query: str, top_n: int = 10) -> dict:
        """Q&A -- trả về {answer: str, sources: list[dict]}."""
        return self._post("/api/qa", {"query": query, "top_n": top_n})

    def search_trake(self, events: list[str], top_n: int = 5) -> list[list[dict]]:
        """TRAKE -- trả về kết quả cho từng event."""
        data = self._post("/api/trake", {"events": events, "top_n": top_n})
        return data["results"]

    def keyframe_image_url(self, video_id: str, frame_idx: int) -> str:
        """URL ảnh theo cặp field công khai video_id + frame_idx."""
        safe_video_id = quote(str(video_id), safe="")
        return f"{self.base_url}/api/keyframe/{safe_video_id}/{int(frame_idx)}/image"

    def get_keyframe_image(self, video_id: str, frame_idx: int) -> bytes | None:
        """Tải ảnh nếu đã có; trả None cho ảnh chưa upload hoặc lỗi kết nối."""
        try:
            resp = requests.get(
                self.keyframe_image_url(video_id, frame_idx),
                timeout=15,
                allow_redirects=True,
            )
            if not resp.ok or not resp.content:
                return None
            return resp.content
        except requests.RequestException:
            return None
