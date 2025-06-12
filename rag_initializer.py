from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional

# Import RAG components
from rag_config import RAGConfig
from rag_search import RAGSearchEngine

# Import models
from rag_fastapi import (
    SearchRequest,
    SearchResponse,
    RAGRequest,
    RAGResponse,
    RAGSearchResultItem,
    RAGSearchResponse
)

# Import other components
from rag_integration import SearchIntegration
from ollamagen import OllamaGenerator

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize configuration and search engine
config = RAGConfig()
search_engine = RAGSearchEngine(config)

# Initialize generator and search integration
generator = OllamaGenerator()
search_integration = SearchIntegration(search_engine, generator)

@app.post("/search", response_model=RAGSearchResponse)
async def search(request: SearchRequest):
    """검색 API 엔드포인트"""
    query = request.query
    top_k = request.top_k or 5
    generate_answer = getattr(request, "generate_answer", True)
    
    # 검색 실행
    search_results = search_engine.search(query, top_k=top_k)
    
    # 결과 포맷팅
    formatted_results = [
        RAGSearchResultItem(
            score=result.get("score", 0.0),
            content=result.get("text", ""),
            metadata=result.get("metadata", {}),
            media_type=result.get("media_type", "text")
        )
        for result in search_results
    ]
    
    # 응답 준비
    response = RAGSearchResponse(
        success=True,
        query=query,
        results=formatted_results,
        images=[]
    )
    
    # 답변 생성이 요청된 경우
    if generate_answer and search_results:
        try:
            # 검색 결과를 기반으로 답변 생성
            context = "\n".join([r.get("text", "") for r in search_results])
            prompt = f"질문: {query}\n\n맥락:\n{context}\n\n위 맥락을 바탕으로 질문에 대한 답변을 생성해주세요."
            
            # Ollama를 통한 답변 생성
            answer = generator.generate(prompt)
            response.response = answer
        except Exception as e:
            print(f"답변 생성 중 오류 발생: {str(e)}")
            response.error = "죄송합니다. 답변을 생성하는 데 실패했습니다."
    
    return response

@app.post("/chat", response_model=RAGSearchResponse)
async def chat(request: RAGRequest):
    """채팅 메시지 처리 엔드포인트
    
    Args:
        request: RAGRequest 모델을 따르는 요청 데이터
        
    Returns:
        RAGSearchResponse: 검색 결과와 생성된 답변을 포함한 응답
            "history": list   # 업데이트된 대화 기록
        }
    """
    try:
        print(f"[DEBUG] 채팅 요청 수신: {request}")
        user_message = request.get("message", "")
        history = request.get("history", [])
        
        if not user_message.strip():
            raise ValueError("메시지가 비어있습니다.")
        
        print(f"[DEBUG] 검색 엔진으로 검색 시작: {user_message}")
        # 검색 엔진을 사용하여 관련 문서 검색
        try:
            search_results = search_engine.search(user_message, top_k=5)
            print(f"[DEBUG] 검색 결과 수: {len(search_results.get('texts', []))}개")
        except Exception as e:
            print(f"[ERROR] 검색 중 오류 발생: {str(e)}")
            search_results = {'texts': []}
        
        # 검색 결과를 컨텍스트로 변환
        context = []
        if search_results.get('texts'):
            for i, result in enumerate(search_results['texts'][:3], 1):
                context.append({
                    "title": result.get("title", f"문서 {i}"),
                    "content": result.get("content", "")[:500]  # 내용이 너무 길 수 있으니 일부만 사용
                })
        
        print(f"[DEBUG] 컨텍스트 준비 완료: {len(context)}개 항목")
        
        # Ollama 생성기 초기화
        try:
            generator = OllamaGenerator(config)
            print(f"[DEBUG] Ollama 생성기 초기화 완료. 연결 상태: {generator.connected}")
            
            # 컨텍스트가 없는 경우 기본 응답 생성
            if not context and generator.connected:
                print("[DEBUG] 컨텍스트가 없어 일반적인 답변 생성 시도...")
                response = generator.generate(
                    prompt=f"사용자 질문에 답변해주세요: {user_message}",
                    system_prompt="당신은 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 최대한 정확하고 유용한 정보를 제공해주세요.",
                    temperature=0.7
                )
                print("[DEBUG] 일반적인 답변 생성 완료")
            # 컨텍스트가 있는 경우 컨텍스트 기반 응답 생성
            elif generator.connected:
                print("[DEBUG] Ollama로 컨텍스트 기반 응답 생성 시도...")
                response = generator.generate_from_context(
                    query=user_message,
                    context=context
                )
                print("[DEBUG] Ollama 응답 생성 완료")
            else:
                response = "죄송합니다. 현재 답변 생성 서비스를 사용할 수 없습니다."
                print("[WARNING] Ollama 서버에 연결할 수 없음")
        except Exception as e:
            print(f"[ERROR] Ollama 응답 생성 중 오류: {str(e)}")
            response = "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."
        
        # 대화 기록 업데이트 (최대 20개 메시지 유지)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response})
        history = history[-20:]  # 최근 10번의 대화(메시지 20개)만 유지
        
        result = {
            "response": response,
            "history": history,
            "context": context  # 디버깅을 위해 컨텍스트도 반환
        }
        print("[DEBUG] 채팅 응답 반환")
        return result
        
    except Exception as e:
        error_msg = f"채팅 처리 중 오류 발생: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return {
            "response": "죄송합니다. 요청을 처리하는 중에 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "history": history if 'history' in locals() else [],
            "error": str(e)  # 디버깅을 위해 오류 메시지 포함
        }