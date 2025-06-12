import os
import json
import sys
import re
import time
import datetime
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
import argparse
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor, Qwen2VLForConditionalGeneration
import faiss
from fastapi import FastAPI, Request

# Import RAGConfig from rag_config
from rag_config import RAGConfig

# 5. RAG 검색 엔진
class RAGSearchEngine:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None # 모델 변수 초기화
        self.tokenizer = None # 토크나이저 변수 초기화
        # self.processor는 로드 로직 안에서 설정되므로 여기서 초기화하지 않아도 됩니다.

        # 쿼리 임베딩 생성을 위한 모델 로드 - CUDA 강제 사용
        try:
            # 메모리 클리어
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            print(f"멀티모달 모델 로드 중: {config.MULTIMODAL_MODEL}...")
            
            # Qwen2-VL 모델의 경우 Qwen2VLForConditionalGeneration 사용
            # device_map="auto"를 사용하여 자동으로 디바이스 맵 추론.
            # 이 설정은 단일 GPU 및 멀티 GPU 환경 모두에서 자동으로 디바이스를 할당합니다.
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.MULTIMODAL_MODEL,
                torch_dtype=torch.float16,  # CUDA에서는 float16 사용 (기존과 동일하게 유지)
                device_map="auto",          # 여기가 핵심 변경! "auto"로 설정하여 accelerate가 최적의 디바이스 맵을 추론하도록 합니다.
                low_cpu_mem_usage=True
            )
            
            # 프로세서 로드 (모델 로드 후에도 동일하게 유지)
            self.processor = AutoProcessor.from_pretrained(config.MULTIMODAL_MODEL)
            # 토크나이저 로드 (모델 로드 후에도 동일하게 유지)
            self.tokenizer = AutoTokenizer.from_pretrained(config.MULTIMODAL_MODEL)
            
            print(f"멀티모달 모델이 성공적으로 로드되었습니다. 모델 디바이스: {self.model.device}")

            # 모델을 반드시 CUDA로 이동 (device_map="auto"를 사용하면 대부분 불필요하지만, 명시적 확인)
            if torch.cuda.is_available():
                print("CUDA 사용 확인")
                self.model.eval() # 추론 모드로 설정
                config.DEVICE = "cuda" # 설정의 DEVICE 값도 "cuda"로 업데이트
            else:
                raise RuntimeError("CUDA가 사용 불가능합니다. GPU가 있는지 확인하세요.")

        except Exception as e:
            print(f"모델 로드 오류: {str(e)}")
            raise # 오류를 다시 발생시켜 프로그램이 중단되도록 함
        
        # FAISS 인덱스 로드
        self.index_path = "/home/nextits2/.conda/envs/workbox/faiss_test/text_index"
        self.load_index()
    
    def load_index(self):
        """FAISS 인덱스 및 메타데이터 로드"""
        # 기본 인덱스 초기화 (실패 시에도 사용할 수 있도록)
        self.index = faiss.IndexFlatIP(self.config.VECTOR_DIMENSION)
        self.metadata_list = []
        
        try:
            # 인덱스 로드
            index_file = f"{self.index_path}.index"
            if not os.path.exists(index_file):
                print(f"[ERROR] 인덱스 파일이 존재하지 않습니다: {index_file}")
                return False
            
            # print(f"[DEBUG] FAISS 인덱스 로드 시도: {index_file}")
            self.index = faiss.read_index(index_file)
            print(f"[DEBUG] FAISS 인덱스 로드 성공. 차원 수: {self.index.d}, 벡터 수: {self.index.ntotal}")
            
            # 메타데이터 로드
            meta_file = f"{self.index_path}.meta.json"
            if not os.path.exists(meta_file):
                print(f"[ERROR] 메타데이터 파일이 존재하지 않습니다: {meta_file}")
                return False
            
            # print(f"[DEBUG] 메타데이터 파일 로드 시도: {meta_file}")
            with open(meta_file, "r", encoding="utf-8") as f:
                self.metadata_list = json.load(f)
            
            # 메타데이터와 인덱스의 벡터 수가 일치하는지 확인
            if len(self.metadata_list) != self.index.ntotal:
                print(f"[WARNING] 메타데이터 수({len(self.metadata_list)})와 인덱스의 벡터 수({self.index.ntotal})가 일치하지 않습니다.")
            
            # print(f"[SUCCESS] 인덱스와 메타데이터가 성공적으로 로드되었습니다: {self.index_path}")
            print(f"[INFO] 로드된 벡터 수: {self.index.ntotal}, 메타데이터 수: {len(self.metadata_list)}")
            
            # 인덱스가 비어있는지 확인
            if self.index.ntotal == 0:
                print("[WARNING] 로드된 인덱스에 벡터가 없습니다.")
                return False
                
            return True
            
        except Exception as e:
            import traceback
            print(f"[ERROR] 인덱스 로드 중 예외 발생: {str(e)}")
            print("[TRACEBACK]")
            traceback.print_exc()
            print("[WARNING] 기본 인덱스를 사용합니다.")
            return False
    

    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """쿼리 텍스트에서 임베딩 생성 (검색용) - CUDA 강제 사용"""
        # 기본 반환값 설정 (오류 발생 시 사용)
        default_embedding = np.zeros(self.config.VECTOR_DIMENSION)
        
        try:
            # 입력 텍스트 유효성 검사
            if not text or not isinstance(text, str):
                print(f"[ERROR] 유효하지 않은 입력 텍스트: {text}")
                return default_embedding
            
            # 텍스트 길이 제한
            if len(text) > 1000:
                print(f"[WARNING] 텍스트 길이 제한: {len(text)} -> 1000")
                text = text[:1000]
            
            # CUDA 메모리 관리 최적화
            if torch.cuda.is_available():
                # 모든 GPU 메모리 정리
                for gpu_idx in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu_idx):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
            
            # print("[DEBUG] CUDA에서 임베딩 생성 중...")
            with torch.no_grad():
                # 텍스트 처리
                inputs = self.processor(
                    text=text, 
                    return_tensors="pt", 
                    max_length=512,
                    truncation=True,
                    padding="max_length"
                )
                
                # 모델의 디바이스 확인
                model_device = next(self.model.parameters()).device
                # print(f"[DEBUG] 모델 디바이스: {model_device}")
                
                # 입력을 모델의 디바이스로 이동
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
                
                # 모델 실행 (output_hidden_states=True로 설정)
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Qwen2-VL의 경우 hidden states에서 마지막 레이어의 [CLS] 토큰 사용
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    # 마지막 레이어의 hidden states (batch_size, seq_len, hidden_size)
                    last_hidden = outputs.hidden_states[-1]
                    # [CLS] 토큰의 임베딩 추출 (batch_size, hidden_size)
                    cls_embedding = last_hidden[:, 0]
                    embedding = cls_embedding.detach().cpu().numpy()
                    print(f"[DEBUG] [CLS] 토큰 임베딩 차원: {embedding.shape}")
                else:
                    # hidden_states가 없는 경우, 마지막 hidden state 사용 시도
                    print("[WARNING] hidden_states를 찾을 수 없습니다. outputs의 속성:", 
                          [attr for attr in dir(outputs) if not attr.startswith('_')])
                    return default_embedding
                
                # 차원 확인 및 조정
                if len(embedding.shape) > 2:
                    print(f"[WARNING] 예상치 못한 임베딩 차원: {embedding.shape}, 평균 풀링 적용")
                    embedding = embedding.mean(axis=1)  # 시퀀스 차원 평균
                
                # 1차원으로 변환 (batch 차원 제거)
                if len(embedding) == 1:
                    embedding = embedding[0]
                
                # 차원 확인
                if embedding.shape[0] != self.config.VECTOR_DIMENSION:
                    print(f"[WARNING] 임베딩 차원 불일치: {embedding.shape[0]} (기대값: {self.config.VECTOR_DIMENSION})")
                    # 모델의 실제 출력 차원을 사용하기 위해 패딩/자르지 않음
                    # 대신 config를 업데이트하거나, 인덱스를 재생성해야 함을 알림
                    if embedding.shape[0] < self.config.VECTOR_DIMENSION:
                        print("[WARNING] 임베딩 차원이 설정값보다 작습니다. 인덱스를 재생성해야 할 수 있습니다.")
                    else:
                        print("[WARNING] 임베딩 차원이 설정값보다 큽니다. 설정값을 업데이트하세요.")
                
                # 정규화 (NaN 방지)
                norm = np.linalg.norm(embedding)
                if norm > 1e-10:
                    embedding = embedding / norm
                else:
                    print("[WARNING] 임베딩 norm이 너무 작아 정규화를 건너뜁니다.")

                print(f"[DEBUG] 최종 임베딩 차원: {embedding.shape}")
                return embedding
                
        except Exception as e:
            import traceback
            print(f"[ERROR] 텍스트 임베딩 생성 실패: {str(e)}")
            print("[TRACEBACK]")
            traceback.print_exc()
            # 오류 발생 시 기본 임베딩 반환
            return default_embedding
    
    def search(self, query: str, top_k: int = None, media_type: str = None) -> Dict[str, Any]:
        """사용자 쿼리로 관련 이미지와 텍스트 검색
        
        Args:
            query: 검색 쿼리 문자열
            top_k: 반환할 최대 결과 수 (None이면 기본값 사용)
            media_type: 필터링할 미디어 타입 ('image', 'text', None)
                    'image'는 이미지만 반환, 'text'는 텍스트만 반환, None은 이미지와 텍스트 모두 반환
        
        Returns:
            딕셔너리 형태의 검색 결과:
            {
                'images': [이미지 결과 리스트],
                'texts': [텍스트 결과 리스트],
                'query': 원본 쿼리,
                'top_k': 요청된 top_k 값
            }
        """
        if not query.strip():
            return {'images': [], 'texts': [], 'query': query, 'top_k': top_k or self.config.TOP_K}
        
        # 기본값 설정
        if top_k is None:
            top_k = self.config.TOP_K
            
        # 쿼리 임베딩 생성
        try:
            # print("[DEBUG] 쿼리 임베딩 생성 중...")
            query_embedding = self.generate_text_embedding(query)
            
            if query_embedding is None:
                print("[ERROR] 쿼리 임베딩 생성 실패: generate_text_embedding이 None을 반환했습니다.")
                return {'images': [], 'texts': [], 'query': query, 'top_k': top_k}
                
            # print(f"[DEBUG] 쿼리 임베딩 생성 완료. 차원: {query_embedding.shape}")
            
            # 정규화
            try:
                norm = np.linalg.norm(query_embedding)
                # print(f"[DEBUG] 쿼리 임베딩 norm: {norm}")
                
                if norm > 1e-10:
                    query_embedding = query_embedding / norm
                else:
                    print("[WARNING] 쿼리 임베딩의 norm이 너무 작아 정규화를 건너뜁니다.")
            except Exception as norm_error:
                print(f"[WARNING] 쿼리 임베딩 정규화 실패: {str(norm_error)}")
                # 정규화 실패해도 계속 진행
                
        except Exception as e:
            import traceback
            error_msg = f"[ERROR] 쿼리 임베딩 생성 실패: {str(e)}"
            print(error_msg)
            print("[TRACEBACK]")
            traceback.print_exc()
            return {'images': [], 'texts': [], 'query': query, 'top_k': top_k, 'error': str(e)}
        
        # 이미지와 텍스트를 분리하여 각각 검색
        image_results = []
        text_results = []

        # 이미지 검색 (con_type이 'image'인 경우)
        try:
            print("[DEBUG] 이미지 검색 시작")
            image_indices = [i for i, meta in enumerate(self.metadata_list) 
                          if meta.get('con_type') == 'image' and 'embedding' in meta]
            
            if image_indices:
                print(f"[DEBUG] {len(image_indices)}개의 이미지 인덱스 찾음")
                try:
                    # 이미지 임베딩 추출 (embedding이 있는 항목만 필터링)
                    valid_image_indices = []
                    image_embeddings = []
                    
                    for idx in image_indices:
                        if 'embedding' in self.metadata_list[idx]:
                            valid_image_indices.append(idx)
                            image_embeddings.append(self.metadata_list[idx]['embedding'])
                    
                    if not image_embeddings:
                        print("[WARNING] 유효한 이미지 임베딩을 찾을 수 없습니다.")
                    else:
                        image_embeddings = np.array(image_embeddings)
                        print(f"[DEBUG] 이미지 임베딩 배열 차원: {image_embeddings.shape}")
                        
                        # FAISS 검색 수행
                        k = min(top_k, len(valid_image_indices))
                        print(f"[DEBUG] 이미지 검색: 상위 {k}개 결과 요청 (유효한 이미지 수: {len(valid_image_indices)})")
                        
                        distances, indices = self.index.search(
                            query_embedding.reshape(1, -1).astype(np.float32), 
                            k
                        )
                        
                        # 결과 처리
                        for dist, idx in zip(distances[0], indices[0]):
                            if 0 <= idx < len(valid_image_indices):
                                meta_idx = valid_image_indices[idx]
                                result = self.metadata_list[meta_idx].copy()
                                result["similarity"] = float(1.0 - dist)  # 거리를 유사도로 변환 (1 - 거리)
                                result["search_fields"] = ["page_num", "text", "tags", "index"]
                                image_results.append(result)
                        
                        print(f"[DEBUG] 이미지 검색 완료: {len(image_results)}개 결과")
                        
                except Exception as img_search_error:
                    print(f"[ERROR] 이미지 검색 중 오류: {str(img_search_error)}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"[ERROR] 이미지 검색 처리 중 예외 발생: {str(e)}")
            import traceback
            traceback.print_exc()

        # 텍스트 검색 (con_type이 'text'인 경우)
        try:
            print("[DEBUG] 텍스트 검색 시작")
            text_indices = [i for i, meta in enumerate(self.metadata_list) 
                          if meta.get('con_type') == 'text' and 'embedding' in meta]
            
            if text_indices:
                print(f"[DEBUG] {len(text_indices)}개의 텍스트 인덱스 찾음")
                try:
                    # 텍스트 임베딩 추출 (embedding이 있는 항목만 필터링)
                    valid_text_indices = []
                    text_embeddings = []
                    
                    for idx in text_indices:
                        if 'embedding' in self.metadata_list[idx]:
                            valid_text_indices.append(idx)
                            text_embeddings.append(self.metadata_list[idx]['embedding'])
                    
                    if not text_embeddings:
                        print("[WARNING] 유효한 텍스트 임베딩을 찾을 수 없습니다.")
                    else:
                        text_embeddings = np.array(text_embeddings)
                        print(f"[DEBUG] 텍스트 임베딩 배열 차원: {text_embeddings.shape}")
                        
                        # FAISS 검색 수행
                        k = min(top_k, len(valid_text_indices))
                        print(f"[DEBUG] 텍스트 검색: 상위 {k}개 결과 요청 (유효한 텍스트 수: {len(valid_text_indices)})")
                        
                        distances, indices = self.index.search(
                            query_embedding.reshape(1, -1).astype(np.float32), 
                            k
                        )
                        
                        # 결과 처리
                        for dist, idx in zip(distances[0], indices[0]):
                            if 0 <= idx < len(valid_text_indices):
                                meta_idx = valid_text_indices[idx]
                                result = self.metadata_list[meta_idx].copy()
                                result["similarity"] = float(1.0 - dist)  # 거리를 유사도로 변환 (1 - 거리)
                                result["search_fields"] = ["page_num", "text", "tags"]
                                text_results.append(result)
                        
                        print(f"[DEBUG] 텍스트 검색 완료: {len(text_results)}개 결과")
                        
                except Exception as txt_search_error:
                    print(f"[ERROR] 텍스트 검색 중 오류: {str(txt_search_error)}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"[ERROR] 텍스트 검색 처리 중 예외 발생: {str(e)}")
            import traceback
            traceback.print_exc()

        # 유사도 순으로 정렬
        image_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        text_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        # 상위 top_k개만 선택
        image_results = image_results[:top_k]
        text_results = text_results[:top_k]

        # 결과 포맷팅
        images = self._format_results(image_results)
        texts = self._format_results(text_results)

        return {
            'images': images,
            'texts': texts,
            'query': query,
            'top_k': top_k
        }
    
    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """검색 결과를 클라이언트에 적합한 형태로 포맷팅"""
        formatted_results = []
        
        for result in results:
            # 결과 형식 확인 - 새 검색 결과는 바로 메타데이터와 유사도를 포함
            if "metadata" in result and "similarity" in result:
                # 이전 형식의 결과
                metadata = result["metadata"]
                similarity = result["similarity"]
                rank = result.get("rank", 0)
                
                # 기본 필드 설정
                formatted_item = {
                    "similarity": similarity,
                    "rank": rank
                }
                
                # 메타데이터 필드 복사
                for key, value in metadata.items():
                    formatted_item[key] = value
            else:
                # 새 형식의 결과 (결과가 이미 메타데이터와 유사도를 포함)
                formatted_item = result.copy()
            
            # 파일 경로가 있는지 확인하고 정규화
            if "file_path" in formatted_item and formatted_item["file_path"]:
                # 파일 경로 정규화 (중복 슬래시 제거 등)
                file_path = os.path.normpath(formatted_item["file_path"])
                formatted_item["file_path"] = file_path
                
                # 이미지 URL 설정
                if file_path.startswith(self.config.IMAGE_BASE_PATH):
                    rel_path = file_path[len(self.config.IMAGE_BASE_PATH):].lstrip(os.sep)
                else:
                    rel_path = os.path.basename(file_path)
                
                # URL 인코딩 적용
                rel_path = rel_path.replace(os.sep, '/')
                formatted_item["image_url"] = f"/static/{rel_path}"
                
                # 이미지 여부 확인 및 타입 설정
                is_image = any(file_path.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"])
                formatted_item["type"] = "image" if is_image else "text"
                formatted_item["con_type"] = "image" if is_image else "text"  # 하위 호환성을 위해 con_type도 설정
            
            # 캡션 설정
            if "caption" not in formatted_item:
                if "text" in formatted_item and formatted_item["text"]:
                    formatted_item["caption"] = formatted_item["text"]
                else:
                    formatted_item["caption"] = formatted_item.get("title", "이미지 캡션 없음")
            
            # 타입이 아직 설정되지 않은 경우 기본값 설정
            if "type" not in formatted_item:
                formatted_item["type"] = "text"
                formatted_item["con_type"] = "text"  # 하위 호환성을 위해
            
            formatted_results.append(formatted_item)
        
        return formatted_results

    # 정확한 텍스트 매칭 검색 함수
    def exact_match_search(self, query: str, top_k: int = 5):
        """키워드 기반 검색을 수행하는 함수
        
        Args:
            query: 검색 쿼리 문자열 (쉼표나 공백으로 구분된 키워드들)
            top_k: 반환할 최대 결과 수
            
        Returns:
            키워드 기반 검색 결과 목록
        """
        results_dict = {}  # 중복 결과를 방지하기 위한 딕셔너리
        
        # 클래스의 metadata_list 사용
        if not hasattr(self, 'metadata_list') or not self.metadata_list:
            print("경고: 메타데이터가 없습니다.")
            return []
            
        metadata_list = self.metadata_list
        
        # 쿼리를 키워드로 분리 (쉼표, 공백 등으로 구분)
        keywords = [kw.strip() for kw in re.split(r'[,\s]+', query) if kw.strip()]
        print(f"검색 키워드: {keywords}")
        
        if not keywords:
            keywords = [query]  # 키워드가 추출되지 않으면 원래 쿼리 사용
        
        # 각 메타데이터에 대해 모든 키워드 검색
        for metadata in metadata_list:
            if "text" in metadata and metadata["text"]:
                text_lower = metadata["text"].lower()
                metadata_id = metadata.get("id", str(id(metadata)))
                
                # 각 키워드에 대해 검색
                keyword_matches = 0
                total_similarity = 0.0
                
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in text_lower:
                        keyword_matches += 1
                        
                        # 유사도 점수 계산 (텍스트 내 위치에 따라 계산)
                        position = text_lower.find(keyword_lower)
                        # 시작 부분에 가까울수록 높은 점수 부여
                        similarity = 1.0 - (position / (len(text_lower) + 1))
                        total_similarity += similarity
                
                # 하나 이상의 키워드가 일치하면 결과에 추가
                if keyword_matches > 0:
                    # 평균 유사도 계산
                    avg_similarity = total_similarity / len(keywords)
                    # 키워드 일치 비율 계산 (가중치 부여)
                    match_ratio = keyword_matches / len(keywords)
                    # 최종 유사도 = 평균 유사도 * 키워드 일치 비율 (더 많은 키워드가 일치할수록 높은 점수)
                    final_similarity = avg_similarity * match_ratio
                    
                    result = metadata.copy()
                    result["similarity"] = final_similarity
                    result["keyword_matches"] = keyword_matches
                    result["total_keywords"] = len(keywords)
                    
                    # 중복 결과 방지 (더 높은 유사도 점수로 업데이트)
                    if metadata_id not in results_dict or results_dict[metadata_id]["similarity"] < final_similarity:
                        results_dict[metadata_id] = result
        
        # 결과 리스트로 변환
        results = list(results_dict.values())
        
        # 유사도 점수에 따라 정렬
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        print(f"검색된 결과 수: {len(results)}")
        
        # top_k 결과만 반환
        return results[:top_k]


    # 메타데이터 목록 출력 함수
    def list_metadata():
        """인덱스에 포함된 메타데이터 목록을 출력하는 함수"""
        print("\n인덱스에 포함된 메타데이터 목록:")
        print(f"총 항목 수: {len(search_engine.metadata_list)}")
        
        for i, metadata in enumerate(search_engine.metadata_list):
            print(f"\n[항목 {i+1}]")
            # 주요 메타데이터 필드 출력
            if "title" in metadata:
                print(f"제목: {metadata['title']}")
            if "page_num" in metadata:
                print(f"페이지: {metadata['page_num']}")
            if "id" in metadata:
                print(f"ID: {metadata['id']}")
            if "file_path" in metadata:
                print(f"파일 경로: {metadata['file_path']}")
            if "text" in metadata:
                # 텍스트가 너무 길면 잘라서 표시
                text = metadata["text"]
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"텍스트 미리보기: {preview}")