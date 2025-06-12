from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Request/Response 모델 정의
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class RAGSearchResultItem(BaseModel):
    score: float
    content: str
    metadata: Dict[str, Any]
    media_type: str

class RAGSearchResponse(BaseModel):
    success: bool
    query: str
    results: List[RAGSearchResultItem]
    response: Optional[str] = None
    images: List[Dict[str, str]] = []
    error: Optional[str] = None

# 호환성을 위한 별칭
SearchResponse = RAGSearchResponse
RAGRequest = SearchRequest
RAGResponse = RAGSearchResponse


# 7. API 정의 (FastAPI)
app = FastAPI(title="이미지 캡션 RAG API")

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 오리진 허용 (프로덕션에서는 특정 도메인으로 제한하는 것이 좋습니다)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# Export models for use in other modules
__all__ = [
    'SearchRequest',
    'RAGSearchResultItem',
    'RAGSearchResponse',
    'RAGRequest',
    'RAGResponse',
    'SearchResponse'
]