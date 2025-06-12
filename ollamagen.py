from typing import Optional, Dict, Any, List
import requests
import json
import time
from rag_config import RAGConfig

class OllamaGenerator:
    """Ollama API를 사용한 텍스트 생성기"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Ollama 생성기 초기화
        
        Args:
            config: RAG 시스템 설정 (선택사항, 기본값 사용 가능)
        """
        self.config = config or RAGConfig()
        self.base_url = getattr(self.config, 'OLLAMA_API_URL', 'http://localhost:11434').rstrip('/')
        self.fallback_url = getattr(self.config, 'OLLAMA_FALLBACK_URL', 'http://localhost:11435').rstrip('/')
        self.model = getattr(self.config, 'OLLAMA_MODEL', 'llama3')
        self.current_url = None  # 현재 사용 중인 URL
        self.max_retries = 3  # 최대 재시도 횟수
        self.connected = False
        
        # API 연결 확인
        self.connected = self._check_connection()
        if not self.connected:
            print("경고: Ollama 서버에 연결할 수 없습니다. 텍스트 생성이 제한됩니다.")
        else:
            print(f"Ollama 생성기가 성공적으로 초기화되었습니다. 사용 중인 URL: {self.current_url}")
    
    def _check_connection(self) -> bool:
        """Ollama API 연결 확인 및 URL 설정"""
        urls_to_try = [self.base_url, self.fallback_url]
        
        for url in urls_to_try:
            if not url:
                continue
                
            try:
                print(f"Ollama 서버에 연결을 시도합니다: {url}")
                response = requests.get(f"{url}/api/tags", timeout=10)
                
                if response.status_code == 200:
                    self.current_url = url
                    print(f"Ollama API에 성공적으로 연결되었습니다: {self.current_url}")
                    
                    # 사용 가능한 모델 확인
                    try:
                        models = response.json().get('models', [])
                        if models:
                            model_names = [model.get("name", "unknown") for model in models]
                            print(f"사용 가능한 모델: {', '.join(model_names)}")
                            
                            # 요청된 모델이 사용 가능한지 확인
                            model_available = any(self.model in model.get("name", "") for model in models)
                            if not model_available:
                                print(f"경고: 요청된 모델 '{self.model}'을(를) 찾을 수 없습니다.")
                                print(f"사용 가능한 모델 중 하나를 선택하거나 다음 명령으로 모델을 다운로드하세요:")
                                print(f"ollama pull {self.model}")
                                return False
                            
                            return True
                        else:
                            print("경고: 사용 가능한 모델이 없습니다. Ollama에 모델을 다운로드하세요.")
                            
                    except Exception as e:
                        print(f"모델 정보 파싱 중 오류: {str(e)}")
                    
                    return True
                else:
                    print(f"Ollama API 연결 실패 ({url}): HTTP 상태 코드 {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Ollama API 연결 오류 ({url}): {str(e)}")
            except Exception as e:
                print(f"Ollama 연결 확인 중 예상치 못한 오류 ({url}): {str(e)}")
            
            # 잠시 대기 후 다음 URL 시도
            time.sleep(1)
        
        print("경고: 모든 Ollama API URL에 연결할 수 없습니다.")
        print("Ollama 서버가 실행 중인지 확인하세요. 다음 명령으로 서버를 시작할 수 있습니다:")
        print("ollama serve")
        return False
    
    def generate(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Ollama API를 사용하여 텍스트 생성 - analyze_images_new.py 스타일로 구현
        
        Args:
            prompt: 생성할 텍스트의 프롬프트
            system_prompt: 시스템 프롬프트 (선택 사항)
            temperature: 생성 다양성 (0.0 ~ 1.0)
            max_tokens: 생성할 최대 토큰 수
            
        Returns:
            생성된 텍스트 또는 오류 메시지
        """
        if not self.connected or not self.current_url:
            return "죄송합니다. 현재 답변 생성 서비스를 사용할 수 없습니다. 나중에 다시 시도해주세요."
            
        # 요청 데이터 구성
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": max(0.1, min(1.0, temperature)),  # 0.1 ~ 1.0 사이로 제한
            "max_tokens": max(10, min(2048, max_tokens)),  # 10 ~ 2048 사이로 제한
            "stream": False
        }
        
        # 시스템 프롬프트가 제공된 경우 추가
        if system_prompt:
            request_data["system"] = system_prompt
        
        # 시스템 프롬프트가 제공된 경우 추가
        if system_prompt:
            request_data["system"] = system_prompt
        
        # 재시도 로직
        for attempt in range(self.max_retries):
            try:
                # 대기 시간 계산 (지수 백오프)
                wait_time = 1 * (2 ** attempt)
                
                # analyze_images_new.py와 동일한 방식으로 API 호출
                response = requests.post(
                    f"{self.current_url}/api/generate",
                    json=request_data,
                    timeout=60,  # 타임아웃 값 (초)
                    headers={'Connection': 'keep-alive'}  # keep-alive 사용 (중요!)
                )
                
                # 결과 확인
                if response.status_code == 200:
                    result = response.json()
                    if "response" in result:
                        return result["response"]
                    else:
                        print(f"Ollama API 응답에 'response' 필드가 없습니다: {result}")
                else:
                    error_msg = response.text
                    print(f"Ollama API 오류 (시도 {attempt+1}/{self.max_retries}): {error_msg}")
                    
                    # 리다이렉션 오류가 발생한 경우 URL 전환
                    if "127.0.0.1" in error_msg and self.current_url == self.base_url:
                        print(f"리다이렉션 오류 감지, 대체 URL로 전환: {self.fallback_url}")
                        self.current_url = self.fallback_url
                        continue  # 즉시 다음 시도로 넘어감
                
                # 마지막 시도가 아니면 대기 후 재시도
                if attempt < self.max_retries - 1:
                    print(f"{wait_time}초 후 재시도합니다...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                print(f"Ollama API 호출 중 예외 발생 (시도 {attempt+1}/{self.max_retries}): {str(e)}")
                
                # 마지막 시도가 아니면 대기 후 재시도
                if attempt < self.max_retries - 1:
                    print(f"{wait_time}초 후 재시도합니다...")
                    time.sleep(wait_time)
            
        return "죄송합니다. 텍스트 생성 중 오류가 발생했습니다."
    

    def generate_from_context(self, query: str, context: List[Dict[str, Any]], max_tokens: int = 1024) -> str:
        """검색 결과 컨텍스트를 기반으로 응답 생성

        Args:
            query: 사용자 질의
            context: 검색 결과 컨텍스트 목록
            max_tokens: 생성할 최대 토큰 수

        Returns:
            생성된 응답
        """
        system_prompt = None  # 시스템 프롬프트는 영어 프롬프트에 포함되므로 별도 사용 안 함

        # context 포맷팅
        def format_context(docs):
            return "\n\n".join([
                f"con_type: {doc['con_type']}\nfile_path: {doc.get('file_path', '')}\ncontent: {doc.get('text', '')}"
                for doc in docs
            ])
        
        context_text = format_context(context)

        # 최종 프롬프트
        prompt = f"""
    You are an AI assistant. Your goal is to answer the user's question using the documents below.

    - Each document has a field called con_type which is either "text" or "image".
    - Each document, including images, is provided with a similarity field, indicating its relevance to the user's question.
    - For text documents, **prioritize using data blocks where the similarity is close to 1** to build the answer, ensuring the most relevant information is utilized. Aim to provide a comprehensive answer from these text documents.
    - For image documents, if their similarity is close to 1 (e.g., higher than 0.1) AND the text content within the con_type image document is highly relevant to the generated answer text, then **include their file_path** in a section called Relevant Images at the very end of your response, formatted as Relevant Images: path : [file_path]. If no images meet these criteria, do not include any image-related text.
    - Do NOT describe the contents of the image.
    - Exclude images that are only tangentially related, redundant, or do not add significant value to the answer beyond the specified similarity and relevance criteria.
    - If the answer is not found in the documents, respond: "제공된 자료에서 해당 정보를 찾을 수 없습니다."
    - You MUST respond in Korean ONLY. Do not use English or any other language. All explanations, terms, and sentences must be in Korean. This is an absolute rule that must be followed.

    Question:
    {query}

    Retrieved documents:
    {context_text}
    """.strip()

        return self.generate(prompt, system_prompt, temperature=0.3, max_tokens=max_tokens)
