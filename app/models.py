from pydantic import BaseModel
from typing import Optional

class IngestRequest(BaseModel):
    file_path: str

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3  # quantos chunks retornar

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]        # trechos usados pra responder
    tokens_used: int