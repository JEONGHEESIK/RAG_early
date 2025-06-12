import os
import json
import sys
import re
import time
import datetime
import requests
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel

# Import RAG components
from rag_search import RAGSearchEngine
from rag_config import RAGConfig
from rag_rewriter import rewrite_query
from rag_utils import load_json, save_json, format_search_results
from ollamagen import OllamaGenerator

# 모델 임포트
from rag_fastapi import SearchRequest, RAGSearchResultItem, RAGSearchResponse

# 9. 챗봇 통합
class SearchIntegration:
    def __init__(self, search_engine: RAGSearchEngine = None, generator=None):
        if search_engine is None:
            from rag_search import RAGSearchEngine
            from rag_config import RAGConfig
            config = RAGConfig()
            self.search_engine = RAGSearchEngine(config)
        else:
            self.search_engine = search_engine
            
        self.config = self.search_engine.config
        
        # 로그 디렉토리 설정
        self.log_dir = "/home/nextits2/.conda/envs/workbox/zone/logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.log_dir, "rag_search_log.txt")
        
        # Ollama 생성기 설정
        self.generator = generator
        self.generator_available = generator is not None
        
        # Ollama 서버가 실행 중인지 확인
        try:
            response = requests.get(f"{self.config.OLLAMA_API_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                print("Ollama 서버에 연결되었습니다.")
                self.generator = OllamaGenerator(self.config)
                if hasattr(self.generator, 'current_url') and self.generator.current_url:
                    self.generator_available = True
                    print("Ollama 생성기가 성공적으로 초기화되었습니다.")
                else:
                    print("Ollama 생성기 초기화는 성공했지만, 현재 URL이 설정되지 않았습니다.")
            else:
                print(f"Ollama 서버에 연결할 수 없습니다. 상태 코드: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Ollama 서버 연결 실패: {str(e)}")
            print("Ollama 서버가 실행 중인지 확인해주세요. 기본 URL:", self.config.OLLAMA_API_URL)
        except Exception as e:
            print(f"Ollama 생성기 초기화 중 오류 발생: {str(e)}")
        
        if not self.generator_available:
            print("경고: Ollama 생성기를 사용할 수 없습니다. 쿼리 리라이팅 및 답변 생성 기능이 제한됩니다.")
    
    def _log_search_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """검색 결과를 로그 파일에 기록
        
        Args:
            query: 검색 쿼리 문자열
            results: 검색 결과 목록
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 이미지와 텍스트 결과 분리
        image_results = [r for r in results if r.get('con_type') == 'image']
        text_results = [r for r in results if r.get('con_type') == 'text']
        
        # 이미지 결과 로깅
        if image_results:
            image_log_path = os.path.join(self.log_dir, "image_search.log")
            with open(image_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"\n[이미지 검색] {timestamp} 검색어: {query}\n")
                f.write(f"검색 결과: {len(image_results)}개\n")
                
                for i, result in enumerate(image_results, 1):
                    f.write(f"\n--- 이미지 결과 {i} ---\n")
                    f.write(f"유사도: {result.get('similarity', 'N/A')}\n")
                    f.write(f"제목: {result.get('title', '제목 없음')}\n")
                    if result.get('page_num'):
                        f.write(f"페이지: {result.get('page_num')}\n")
                    f.write(f"파일 경로: {result.get('file_path', '파일 경로 없음')}\n")
                    
                    # 이미지 설명이 있으면 추가
                    if result.get('text'):
                        desc = result['text']
                        if len(desc) > 300:  # 설명이 너무 길면 자르기
                            desc = desc[:300] + "..."
                        f.write(f"설명: {desc}\n")
        
        # 텍스트 결과 로깅
        if text_results:
            text_log_path = os.path.join(self.log_dir, "text_search.log")
            with open(text_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"\n[텍스트 검색] {timestamp} 검색어: {query}\n")
                f.write(f"검색 결과: {len(text_results)}개\n")
                
                for i, result in enumerate(text_results, 1):
                    f.write(f"\n--- 텍스트 결과 {i} ---\n")
                    f.write(f"유사도: {result.get('similarity', 'N/A')}\n")
                    f.write(f"제목: {result.get('title', '제목 없음')}\n")
                    if result.get('page_num'):
                        f.write(f"페이지: {result.get('page_num')}\n")
                    
                    # 텍스트 내용 기록
                    text = result.get('text', '')
                    if text:
                        if len(text) > 500:  # 내용이 너무 길면 자르기
                            text = text[:500] + "..."
                        f.write(f"내용: {text}\n")
                    
                    # 태그가 있으면 추가
                    if result.get('tags'):
                        tags = ", ".join(result['tags']) if isinstance(result['tags'], list) else result['tags']
                        f.write(f"태그: {tags}\n")
        
        # 기존 로그 파일에도 모든 결과 기록 (하위 호환성 유지)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"\n[{timestamp}] 검색어: {query}\n")
            f.write(f"검색 결과: {len(results)}개 (이미지: {len(image_results)}, 텍스트: {len(text_results)})\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"\n--- 결과 {i} ---\n")
                f.write(f"타입: {result.get('con_type', '알 수 없음')}\n")
                f.write(f"유사도: {result.get('similarity', 'N/A')}\n")
                f.write(f"제목: {result.get('title', '제목 없음')}\n")
                if result.get('page_num'):
                    f.write(f"페이지: {result.get('page_num')}\n")
                
                # 내용 기록
                text = result.get('text', '')
                if text:
                    if len(text) > 300:
                        text = text[:300] + "..."
                    f.write(f"내용: {text}\n")
                
                # 이미지 경로 기록 (이미지인 경우)
                if result.get('con_type') == 'image':
                    f.write(f"파일 경로: {result.get('file_path', '파일 경로 없음')}\n")
    
    def _filter_by_media_type(self, result: Dict[str, Any], media_type: str = None) -> bool:
        """미디어 타입에 따라 결과를 필터링
        
        Args:
            result: 검색 결과 항목
            media_type: 필터링할 미디어 타입 ('image', 'text', None)
            
        Returns:
            필터링 후 포함 여부 (True: 포함, False: 제외)
        """
        # 미디어 타입이 지정되지 않은 경우 모든 결과 포함
        if media_type is None:
            return True
            
        # con_type 필드로 미디어 타입 확인
        if "con_type" in result:
            if media_type == "image":
                return result["con_type"] == "image"
            elif media_type == "text":
                return result["con_type"] == "text"
        
        # con_type 필드가 없는 경우 (예외 처리)
        return False
    
    def process_query(self, query: str, media_type: str = None, top_k: int = None, generate_answer: bool = True) -> Dict[str, Any]:
        """쿼리 처리 및 관련 이미지/콘텐츠 검색
        
        Args:
            query: 검색 쿼리 문자열
            media_type: 필터링할 미디어 타입 ('image', 'text', None)
                        'image'는 이미지만 반환, 'text'는 텍스트만 반환, None은 모두 반환
            top_k: 반환할 최대 결과 수 (None이면 기본값 사용)
            generate_answer: 응답 생성 여부 (True면 응답 생성, False면 검색 결과만 반환)
        
        Returns:
            검색 결과 데이터
        """
        # top_k가 None이면 설정에서 기본값 사용
        if top_k is None:
            top_k = self.config.TOP_K
        # 0. 쿼리 리라이팅 적용
        original_query = query
        if self.config.USE_QUERY_REWRITING and self.generator_available and hasattr(self, 'generator'):
            try:
                print("쿼리 리라이팅을 시도합니다...")
                query = rewrite_query(query, self.config, self.generator)
                print(f"리라이팅된 쿼리: {query}")
            except Exception as e:
                print(f"쿼리 리라이팅 중 오류 발생: {str(e)}")
                print("원본 쿼리로 계속 진행합니다.")
                query = original_query
        
        # 1. 정확한 텍스트 매칭 검색 수행 (자동으로 적용)
        print("\n정확한 텍스트 매칭 검색 수행 중...")
        exact_results = []
        if hasattr(self.search_engine, 'exact_match_search'):
            exact_results = self.search_engine.exact_match_search(query, min(top_k, self.config.TOP_K))
        else:
            print("경고: search_engine에 exact_match_search 메서드가 없습니다.")
        
        # 2. 임베딩 기반 검색 수행
        print(f"\n임베딩 기반 검색 수행 중... (top_k: {top_k})")
        embedding_results = self.search_engine.search(query, top_k=top_k, media_type=media_type)
        
        # 3. 결과 합치기 (중복 제거)
        combined_results = []
        seen_ids = set()
        
        # 결과를 미디어 타입별로 분류
        exact_text_results = []
        exact_image_results = []
        embedding_text_results = []
        embedding_image_results = []
        
        # 정확한 텍스트 매칭 결과 분류
        for result in exact_results:
            # id가 있는 경우만 처리
            if "id" in result:
                current_id = result["id"]
                # 이미 처리된 ID가 아닌 경우만 처리
                if current_id not in seen_ids:
                    # 미디어 타입에 따라 분류
                    if "con_type" in result:
                        if result["con_type"] == "text" and (media_type is None or media_type == "text"):
                            exact_text_results.append(result)
                            seen_ids.add(current_id)
                        elif result["con_type"] == "image" and (media_type is None or media_type == "image"):
                            exact_image_results.append(result)
                            seen_ids.add(current_id)
        
        # 임베딩 기반 결과 분류
        for result in embedding_results:
            # id가 있는 경우만 처리
            if "id" in result:
                current_id = result["id"]
                # 이미 처리된 ID가 아닌 경우만 처리
                if current_id not in seen_ids:
                    # 미디어 타입에 따라 분류
                    if "con_type" in result:
                        if result["con_type"] == "text" and (media_type is None or media_type == "text"):
                            embedding_text_results.append(result)
                            seen_ids.add(current_id)
                        elif result["con_type"] == "image" and (media_type is None or media_type == "image"):
                            embedding_image_results.append(result)
                            seen_ids.add(current_id)
        
        # 결과 결합: 텍스트 우선, 이미지 나중에
        # 1. 정확한 텍스트 매칭 결과 (텍스트)
        combined_results.extend(exact_text_results)
        
        # 2. 정확한 텍스트 매칭 결과 (이미지)
        combined_results.extend(exact_image_results)
        
        # 3. 임베딩 기반 결과 (텍스트)
        combined_results.extend(embedding_text_results)
        
        # 4. 임베딩 기반 결과 (이미지)
        combined_results.extend(embedding_image_results)
        
        # TOP_K개의 결과만 포함
        combined_results = combined_results[:self.config.TOP_K]
        
        # 4. 이미지 정보 추출
        images = []
        for result in combined_results[:self.config.TOP_K]:
            # 이미지 URL이 없을 경우 file_path 사용
            image_url = result.get("image_url", "")
            if not image_url and "file_path" in result:
                image_url = result["file_path"]
            
            # 이미지 파일인지 확인
            is_image = image_url and any(image_url.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"])
            
            # 이미지만 필터링하거나 모든 미디어 타입을 포함하는 경우
            if is_image and (media_type == "image" or media_type is None):
                # 캡션이 없을 경우 기본값 사용
                caption = result.get("caption", result.get("text", "이미지 캡션 없음"))
                
                images.append({
                    "url": image_url,
                    "caption": caption,
                    "title": result.get("title", ""),
                    "page_num": result.get("page_num", ""),
                    "similarity": result.get("similarity", 0.0)
                })
        
        # 5. 검색 결과 포맷팅
        formatted_results = format_search_results(combined_results)
        
        # 결과 반환 데이터 준비
        response_data = {
            "results": combined_results,
            "formatted_results": formatted_results,
            "has_image": len(images) > 0,
            "images": images,
            "media_type": media_type or "all",  # 사용된 미디어 타입 정보 추가
            "original_query": original_query,  # 원본 쿼리 추가
            "query": query,  # 재작성된 쿼리 (또는 원본 쿼리)
            "is_rewritten": original_query != query,  # 쿼리 재작성 여부
            "total_results": len(combined_results),
            "exact_match_count": len(exact_text_results),
            "embedding_match_count": len(embedding_results)
        }
        
        # 검색 결과를 로그 파일에 기록
        self._log_search_results(query, combined_results)
        
        # Ollama 생성기가 사용 가능하면 응답 생성
        if hasattr(self, 'generator_available') and self.generator_available and combined_results:
            try:
                print("Ollama로 응답 생성 중...")
                response = self.generator.generate_from_context(query, combined_results)
                response_data["generated_answer"] = response
                print("응답 생성 완료")
                print(response)
            except Exception as e:
                print(f"응답 생성 오류: {str(e)}")
                response_data["generated_answer"] = "응답 생성 중 오류가 발생했습니다."
        else:
            if hasattr(self, 'generator_available') and not self.generator_available:
                print("Ollama 생성기를 사용할 수 없습니다.")
            elif not combined_results:
                print("검색 결과가 없어 응답을 생성할 수 없습니다.")
        
        # 결과 반환
        return response_data

    
    # 6. 검색 결과 포맷팅 유틸리티 함수
    def format_search_results(search_results: List[Dict[str, Any]]) -> str:
        """검색 결과를 읽기 쉬운 형식으로 포맷팅"""
        formatted_text = "검색 결과:\n\n"
        
        for i, result in enumerate(search_results):
            formatted_text += f"[결과 {i+1}] "
            
            if "title" in result and result["title"]:
                formatted_text += f"제목: {result['title']}"
                if "page_num" in result and result["page_num"]:
                    formatted_text += f" (페이지: {result['page_num']})"
                formatted_text += "\n"
            
            if "text" in result and result["text"]:
                # 텍스트가 너무 길면 잘라서 표시
                text = result["text"]
                if len(text) > 300:
                    text = text[:300] + "..."
                formatted_text += f"내용: {text}\n"
            
            if "similarity" in result:
                formatted_text += f"유사도: {result['similarity']:.4f}\n"
            
            formatted_text += "\n"
        
        return formatted_text