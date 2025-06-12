from typing import Optional

# Import RAG components
from rag_config import RAGConfig
from ollamagen import OllamaGenerator


# 쿼리 재작성
# --- 헬퍼 함수: rewrite_query ---
def rewrite_query(query: str, config: RAGConfig, generator: 'OllamaGenerator') -> str:
    """쿼리 리라이팅을 수행합니다."""
    if not config.USE_QUERY_REWRITING or not hasattr(generator, 'current_url'):
        print(f"[rewrite_query] 리라이팅 비활성화 또는 Ollama 생성기 사용 불가. 원본 쿼리 사용: {query}")
        return query

    system_prompt = """You are a helpful assistant that rewrites Korean search queries to improve document retrieval performance.  
        The input will always be in Korean. Rewrite the query using clear, concise language and include only the most relevant keywords for maximum search relevance.  
        Respond ONLY with the rewritten Korean query and nothing else.  
        If the input query is already optimal, return it as is. 영어나 다른 언어로 번역하지 마세요. 한글로 응답하세요."""
    
    rewrite_prompt = f"""
        Given the following user query, rewrite it to be more effective for document retrieval.
        The rewritten query should be concise, clear, and contain keywords that maximize search relevance.
        If the query is already optimal, return it as is. 영어나 다른 언어로 번역하지 마세요. 한글로 응답하세요.
        Example:
        Original Query: "회사 소개 자료 찾아줘"
        Rewritten Query: "회사 소개, 기업 개요, 회사 연혁"

        Original Query: "{query}"
        Rewritten Query:
        """.strip()
    
    try:
        print(f"[rewrite_query] 원본 쿼리: {query}")
        # Set the model before generating
        generator.model = config.REWRITER_MODEL
        rewritten_query = generator.generate(
            prompt=rewrite_prompt,
            temperature=0.1,
            max_tokens=64
        )
        rewritten_query = rewritten_query.replace("Rewritten Query:", "").replace("재작성 된 질문 :", "").strip()
        print(f"[rewrite_query] 리라이팅된 쿼리: {rewritten_query}")
        return rewritten_query
    except Exception as e:
        print(f"쿼리 리라이팅 중 오류: {e}. 원본 쿼리 사용.")
        return query