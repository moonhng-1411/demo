"""Pydantic models cho request body -- tách khỏi router để dùng lại được
giữa nhiều router nếu cần, và test riêng schema không phải import cả app."""

from pydantic import BaseModel


class KISRequest(BaseModel):
    query: str
    top_n: int = 10


class QARequest(BaseModel):
    query: str
    top_n: int = 10


class TRAKERequest(BaseModel):
    events: list[str]
    top_n: int = 5