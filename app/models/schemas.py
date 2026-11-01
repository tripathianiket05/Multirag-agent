from typing import List, Optional
from pydantic import BaseModel, Field

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    status: str
    pages: Optional[int] = None
    evidence_items: int = 0
    error: Optional[str] = None

class QueryRequest(BaseModel):
    question: str = Field(min_length=2)
    document_ids: Optional[List[str]] = None

class SourceCitation(BaseModel):
    document_id: str
    page: int
    content_type: str
    vector_score: float
    rerank_score: float
    text: str
    source_path: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    best_match_type: str
    sources: List[SourceCitation]
    request_id: Optional[str] = None