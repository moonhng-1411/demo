"""Pydantic models cho request body -- tách khỏi router để dùng lại được
giữa nhiều router nếu cần, và test riêng schema không phải import cả app."""

from pydantic import BaseModel


class KISRequest(BaseModel):
    query: str
    top_n: int = 10
    translate: bool = True  # dịch query VI->EN trước khi search (xem QueryTranslator)


class QARequest(BaseModel):
    query: str
    top_n: int = 10
    translate: bool = True


class TRAKERequest(BaseModel):
    events: list[str]
    top_n: int = 5
    translate: bool = True