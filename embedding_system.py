### 명령어 cd /home/nextits2/.conda/envs/workbox
### python /home/nextits2/.conda/envs/workbox/zone/embedding_system.py --json /home/nextits2/.conda/envs/workbox/combined_metadata_2.json --type image
### python /home/nextits2/.conda/envs/workbox/zone/embedding_system.py --json /home/nextits2/.conda/envs/workbox/test_text_2.json --type text

import os
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import faiss
from tqdm import tqdm

class VectorDBService:
    """
    RAG 시스템을 위한 벡터 DB 서비스
    이미지와 텍스트 메타데이터를 위한 별도의 FAISS 인덱스를 관리하며, 
    임베딩 생성 및 벡터 DB 저장/검색 기능을 제공합니다.
    """
    def __init__(self, 
                 model_name="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct", 
                 device=None,
                 vector_dimension=4096,
                 index_path="/home/nextits2/.conda/envs/workbox/faiss_index",
                 index_type="IP"):
        """
        벡터 DB 서비스 초기화
        
        Args:
            model_name: 사용할 멀티모달 모델 이름
            device: 사용할 디바이스 (None이면 자동 감지)
            vector_dimension: 벡터 차원 수
            index_path: 인덱스 저장 경로 (디렉토리)
            index_type: 인덱스 유형 ("L2" 또는 "IP" - 내적)
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.vector_dimension = vector_dimension
        self.index_path = index_path
        self.index_type = index_type
        
        # 이미지와 텍스트 메타데이터를 위한 별도 리스트
        self.image_metadata_list = []
        self.text_metadata_list = []
        
        # 모델 및 프로세서 로드
        self._load_model()
        
        # FAISS 인덱스 초기화
        self._init_indices()
    
    def _load_model(self):
        """모델 및 프로세서 로드"""
        print(f"모델 로드 중: {self.model_name} (디바이스: {self.device})")
        
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # 모델을 평가 모드로 설정
            self.model.eval()
            
            # 모델 구성에서 임베딩 차원 확인 및 설정
            if hasattr(self.model.config, 'hidden_size'):
                self.vector_dimension = self.model.config.hidden_size
            elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'hidden_size'):
                self.vector_dimension = self.model.config.text_config.hidden_size
            elif hasattr(self.model.config, 'projection_dim'):
                self.vector_dimension = self.model.config.projection_dim
            
            print(f"모델 로드 완료: {self.model_name} (임베딩 차원: {self.vector_dimension})")
            
        except Exception as e:
            print(f"모델 로드 실패: {str(e)}")
            raise
    
    def _init_indices(self):
        """이미지와 텍스트를 위한 별도의 FAISS 인덱스 초기화"""
        # 디렉토리 생성
        os.makedirs(self.index_path, exist_ok=True)
        
        # 이미지 인덱스 초기화
        if self.index_type == "IP":
            self.image_index = faiss.IndexFlatIP(self.vector_dimension)
            self.text_index = faiss.IndexFlatIP(self.vector_dimension)
        else:
            self.image_index = faiss.IndexFlatL2(self.vector_dimension)
            self.text_index = faiss.IndexFlatL2(self.vector_dimension)
            
        # 인덱스 저장 경로
        self.image_index_path = os.path.join(self.index_path, "image_index.faiss")
        self.text_index_path = os.path.join(self.index_path, "text_index.faiss")
        
        # 저장된 인덱스가 있으면 로드
        if os.path.exists(self.image_index_path):
            self.image_index = faiss.read_index(self.image_index_path)
            print(f"이미지 인덱스 로드 완료: {self.image_index_path}")
            
        if os.path.exists(self.text_index_path):
            self.text_index = faiss.read_index(self.text_index_path)
            print(f"텍스트 인덱스 로드 완료: {self.text_index_path}")
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        텍스트에서 임베딩 생성
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            텍스트 임베딩 벡터 (numpy 배열)
        """
        try:
            # 메모리 관리 개선
            if self.device == "cuda":
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
            with torch.no_grad():
                # 텍스트 처리 - 길이 제한 추가
                inputs = self.processor(
                    text=text, 
                    return_tensors="pt",
                    max_length=512,  # 최대 길이 제한
                    truncation=True,
                    padding="max_length"
                )
                
                # 안전하게 디바이스로 이동
                if self.device == "cuda":
                    try:
                        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e) or "illegal memory access" in str(e):
                            print("CUDA 메모리 오류 발생, CPU로 전환합니다...")
                            self.model = self.model.to("cpu")
                            self.device = "cpu"
                            inputs = {k: v.to("cpu") for k, v in inputs.items()}
                        else:
                            raise
                
                # 모델 실행
                try:
                    outputs = self.model(**inputs)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) or "illegal memory access" in str(e):
                        print("모델 실행 중 CUDA 오류 발생, CPU로 전환합니다...")
                        self.model = self.model.to("cpu")
                        self.device = "cpu"
                        inputs = {k: v.to("cpu") for k, v in inputs.items()}
                        outputs = self.model(**inputs)
                    else:
                        raise
                
                # 마지막 히든 상태 가져오기
                embedding = None
                if hasattr(outputs, 'last_hidden_state'):
                    # 마지막 히든 상태의 [CLS] 토큰 임베딩 사용
                    embedding = outputs.last_hidden_state[:, 0].detach().cpu().numpy()
                elif hasattr(outputs, 'text_embeds'):
                    # 텍스트 임베딩 사용
                    embedding = outputs.text_embeds.detach().cpu().numpy()
                else:
                    # 다른 속성 사용 시도
                    for attr_name in dir(outputs):
                        if attr_name.startswith('_'):
                            continue
                        attr = getattr(outputs, attr_name)
                        if isinstance(attr, torch.Tensor) and attr.ndim >= 2:
                            embedding = attr.mean(dim=1).detach().cpu().numpy()
                            break
                    else:
                        # 임베딩을 찾을 수 없는 경우
                        print(f"사용 가능한 속성: {[a for a in dir(outputs) if not a.startswith('_')]}")
                        raise ValueError("적절한 임베딩을 찾을 수 없습니다.")
                
                # NaN/Inf 체크 추가
                if not np.isfinite(embedding).all():
                    print("경고: 임베딩에 NaN 또는 Inf 값이 포함되어 있습니다. 기본값으로 대체합니다.")
                    return np.zeros(self.vector_dimension)
                
                # 정규화 - 0으로 나누기 방지 로직 추가
                norm = np.linalg.norm(embedding, axis=1, keepdims=True)
                norm = np.where(norm > 1e-10, norm, 1e-10)  # 0으로 나누기 방지
                embedding = embedding / norm
                
                # 메모리 클리어
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                return embedding[0]  # 배치 차원 제거
        except Exception as e:
            print(f"텍스트 임베딩 생성 실패: {str(e)}")
            import traceback
            print(traceback.format_exc())  # 상세 오류 정보 출력
            # 오류 발생 시 임시 임베딩 반환 (0으로 채워진 벡터)
            return np.zeros(self.vector_dimension)
    
    def generate_batch_embeddings(self, texts: List[str], batch_size=4) -> np.ndarray:
        """
        여러 텍스트의 임베딩을 배치로 생성
        
        Args:
            texts: 임베딩할 텍스트 목록
            batch_size: 배치 처리 크기
            
        Returns:
            텍스트 임베딩 벡터 배열 (numpy 배열)
        """
        all_embeddings = []
        
        try:
            # 배치 단위로 처리
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                with torch.no_grad():
                    # 텍스트 처리
                    inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                    
                    if self.device == "cuda":
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Qwen2VLModel은 get_text_features 대신 다른 방법 사용
                    outputs = self.model(**inputs)
                    
                    # 임베딩 추출
                    if hasattr(outputs, 'last_hidden_state'):
                        # 마지막 히든 상태의 [CLS] 토큰 임베딩 사용
                        embeddings = outputs.last_hidden_state[:, 0].detach().cpu().numpy()
                    elif hasattr(outputs, 'text_embeds'):
                        # 텍스트 임베딩 사용
                        embeddings = outputs.text_embeds.detach().cpu().numpy()
                    else:
                        # 다른 속성 사용 시도
                        for attr_name in dir(outputs):
                            if attr_name.startswith('_'):
                                continue
                            attr = getattr(outputs, attr_name)
                            if isinstance(attr, torch.Tensor) and attr.ndim >= 2:
                                embeddings = attr.mean(dim=1).detach().cpu().numpy()
                                break
                        else:
                            # 임베딩을 찾을 수 없는 경우
                            if i == 0:  # 처음 배치에서만 상세 정보 출력
                                print(f"사용 가능한 속성: {[a for a in dir(outputs) if not a.startswith('_')]}")
                            # 빈 배열 반환
                            return np.zeros((len(texts), self.vector_dimension))
                    
                    # 정규화
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    
                    # 메모리 클리어
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    
                    all_embeddings.append(embeddings)
            
            # 모든 배치 결합
            return np.vstack(all_embeddings) if all_embeddings else np.array([])
        except Exception as e:
            print(f"배치 임베딩 생성 실패: {str(e)}")
    
    def add_to_index(self, embedding: np.ndarray, metadata: Dict[str, Any], content_type: str = "text"):
        """
        임베딩과 메타데이터를 적절한 인덱스에 추가
        
        Args:
            embedding: 추가할 임베딩 벡터
            metadata: 해당 임베딩의 메타데이터 (con_type 필드 필요)
            content_type: 콘텐츠 유형 ("image" 또는 "text")
            
        Returns:
            추가 성공 여부 (bool)
        """
        if content_type not in ["image", "text"]:
            raise ValueError("content_type은 'image' 또는 'text'여야 합니다.")
            
        try:
            # 2D 배열로 변환 (FAISS는 2D 배열을 기대함)
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
            
            # NaN/Inf 값 검사
            if not np.isfinite(embedding).all():
                print("경고: 임베딩에 NaN 또는 Inf 값이 포함되어 있습니다. 이 항목은 건너뜁니다.")
                return False
            
            # 정규화 (내적 유사도 인덱스인 경우)
            if self.index_type == "IP":
                norm = np.linalg.norm(embedding, axis=1, keepdims=True)
                norm = np.where(norm > 1e-10, norm, 1e-10)  # 0으로 나누기 방지
                embedding = embedding / norm
            
            # 콘텐츠 유형에 따라 적절한 인덱스에 추가
            if content_type == "image":
                self.image_index.add(embedding.astype('float32'))
                self.image_metadata_list.append(metadata)
            else:  # text
                self.text_index.add(embedding.astype('float32'))
                self.text_metadata_list.append(metadata)
            
            return True
            
        except Exception as e:
            print(f"인덱스에 항목 추가 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_json_data(self, json_file_path, image_base_path="static", append_mode=True, content_type="text"):
        """
        JSON 형식의 이미지 캡션 데이터를 처리하여 벡터 DB에 저장
        키워드 검색에 최적화된 텍스트 임베딩 생성
        
        Args:
            json_file_path: JSON 파일 경로
            image_base_path: 이미지 파일 기본 경로
            append_mode: True이면 기존 인덱스에 추가, False이면 덮어쓰기
            content_type: 처리할 데이터 유형 ("image" 또는 "text")
            
        Returns:
            인덱싱된 항목 수
        """
        if content_type not in ["image", "text"]:
            raise ValueError("content_type은 'image' 또는 'text'여야 합니다.")
        # 데이터 파일 존재 확인
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {json_file_path}")
        
        # 데이터 로드
        print(f"데이터 파일 로드 중: {json_file_path}")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"JSON 파일 로드 완료: {json_file_path}")
        except Exception as e:
            raise ValueError(f"JSON 파일 로드 실패: {str(e)}")
        
        # 텍스트 데이터와 메타데이터 추출
        texts_to_embed = []
        metadata_list = []
        
        # 데이터 구조 확인
        items = []
        
        # "metadata" 키가 있는지 확인
        if isinstance(data, dict) and "metadata" in data:
            print("JSON 구조: metadata 키 포함")
            # metadata 내의 각 아이템 처리
            for key, item in data["metadata"].items():
                item["id"] = key
                items.append(item)
        else:
            print("JSON 형식 오류: 'metadata' 필드가 없습니다.")
            return 0
        
        print(f"처리할 항목 수: {len(items)}")
        
        # 각 항목 처리
        for item in items:
            # 텍스트 추출
            text = item.get("text", "")
            caption = item.get("caption", "")
            
            # 텍스트가 없으면 건너뛰기
            if not text and not caption:
                continue
            
            # 주요 텍스트 결정 (text 우선, 없으면 caption 사용)
            main_text = text if text else caption
            
            # 이미지 경로 구성
            file_path = item.get("file_path", "")
            file_name = item.get("file_name", "")
            
            # 이미지 경로가 없으면 파일명으로 구성
            if not file_path and file_name:
                file_path = os.path.join(image_base_path, file_name)
            
            # 태그 추출
            tags = item.get("tags", [])
            
            # 제목 및 페이지 추출
            title = item.get("title", item.get("title(book)", ""))
            page_num = item.get("page_num", "")
            
            # 키워드 기반 검색에 최적화된 텍스트 구성
            # 1. 키워드(태그)를 여러 번 반복하여 가중치 크게 부여
            # 2. 문서 식별 정보를 텍스트 시작 부분에 배치
            # 3. 원본 텍스트 보존
            
            # 메타데이터 구성 (먼저 생성)
            metadata = {
                "text": main_text,
                "file_path": file_path,
                "tags": tags,
                "title": title,
                "page_num": page_num,
                "id": item.get("id", "")
            }
            
            # 추가 메타데이터 복사
            for key, value in item.items():
                if key not in ["text", "caption", "file_path", "file_name", "tags", "title", "title(book)", "page_num", "id"]:
                    metadata[key] = value
            
            # 임베딩을 위한 텍스트 조합
            combined_text = ""
        
            # 1. 모든 메타데이터 필드를 텍스트로 변환 (created_at, modified_at 제외)
            exclude_fields = ['created_at', 'modified_at']
            
            for key, value in metadata.items():
                if key in exclude_fields:
                    continue
                    
                if isinstance(value, (str, int, float, bool)) and value:
                    combined_text += f"{key}: {value}\n"
                elif isinstance(value, list) and value:
                    combined_text += f"{key}: {', '.join(map(str, value))}\n"
            
            # 2. 메인 텍스트 추가
            combined_text += "\n내용:\n" + main_text
            
            # 디버깅 정보
            print(f"항목 '{item.get('id', '')}' 처리:")
            print(f"  - 제목: {title}")
            print(f"  - 페이지: {page_num}")
            print(f"  - 태그: {tags}")
            print(f"  - 텍스트 시작: {main_text[:50]}..." if len(main_text) > 50 else f"  - 텍스트: {main_text}")
        
            texts_to_embed.append(combined_text)
            metadata_list.append(metadata)
        
        if not texts_to_embed:
            print("인덱싱할 텍스트가 없습니다.")
            return 0
        
        # 기존 인덱스가 있고 append_mode가 True이면 기존 인덱스에 추가
        if append_mode:
            # 기존 인덱스 로드 시도
            index_file = f"{self.index_path}.index"
            meta_file = f"{self.index_path}.meta.json"
            
            if os.path.exists(index_file) and os.path.exists(meta_file):
                print("기존 인덱스와 메타데이터 로드 중...")
                self.load_index()
        
        # 임베딩 생성 및 인덱싱
        print(f"{content_type} 임베딩 생성 중 ({len(texts_to_embed)} 항목)...")
        embeddings = self.generate_batch_embeddings(texts_to_embed)
        
        # 메모리 클리어
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # 벡터 DB에 추가
        print(f"{content_type} 벡터 DB에 임베딩 추가 중...")
        indexed_count = 0
        for emb, meta in zip(embeddings, metadata_list):
            if self.add_to_index(emb, meta, content_type=content_type):
                indexed_count += 1
        
        # 인덱스 저장
        if indexed_count > 0:
            self.save_index(content_type_to_save=content_type)
        
        print(f"데이터 인덱싱 완료: {indexed_count} 항목 인덱싱됨")
        return indexed_count
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        텍스트 임베딩 생성 (RAG 시스템에서 사용)
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            텍스트 임베딩 벡터 (numpy 배열)
        """
        return self.generate_text_embedding(text)
        
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """
        이미지 임베딩 생성 (RAG 시스템에서 사용)
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            이미지 임베딩 벡터 (numpy 배열)
        """
        return self.generate_image_embedding(image_path)
        
    def search_in_index(self, query_embedding: np.ndarray, content_type: str = "text", top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        인덱스에서 검색 (RAG 시스템에서 사용)
        
        Args:
            query_embedding: 검색 쿼리 임베딩
            content_type: 검색할 콘텐츠 유형 ("image" 또는 "text")
            top_k: 반환할 최대 결과 수
            
        Returns:
            (distances, indices): 거리와 인덱스의 튜플
        """
        if content_type not in ["image", "text"]:
            raise ValueError("content_type은 'image' 또는 'text'여야 합니다.")
            
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # 정규화 (내적 유사도 인덱스인 경우)
        if self.index_type == "IP":
            norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            norm = np.where(norm > 1e-10, norm, 1e-10)
            query_embedding = query_embedding / norm
            
        if content_type == "image" and hasattr(self, 'image_index'):
            return self.image_index.search(query_embedding.astype('float32'), top_k)
        elif content_type == "text" and hasattr(self, 'text_index'):
            return self.text_index.search(query_embedding.astype('float32'), top_k)
        else:
            raise ValueError(f"{content_type} 인덱스를 찾을 수 없습니다.")
            
    def get_metadata(self, index: int, content_type: str) -> Dict[str, Any]:
        """
        인덱스에 해당하는 메타데이터 조회 (RAG 시스템에서 사용)
        
        Args:
            index: 메타데이터 인덱스
            content_type: 콘텐츠 유형 ("image" 또는 "text")
            
        Returns:
            메타데이터 딕셔너리
        """
        if content_type == "image" and 0 <= index < len(self.image_metadata_list):
            return self.image_metadata_list[index]
        elif content_type == "text" and 0 <= index < len(self.text_metadata_list):
            return self.text_metadata_list[index]
        else:
            raise IndexError(f"{content_type} 메타데이터에서 인덱스 {index}를 찾을 수 없습니다.")
            
    def save_index(self, content_type_to_save: str):
        """지정된 content_type에 해당하는 인덱스와 메타데이터만 저장"""
        try:
            os.makedirs(self.index_path, exist_ok=True)
            action_taken = False

            if content_type_to_save == "image":
                if hasattr(self, 'image_index') and hasattr(self, 'image_index_path') and hasattr(self, 'image_metadata_list'):
                    image_meta_path = os.path.join(self.index_path, "image_metadata.json")
                    
                    faiss.write_index(self.image_index, self.image_index_path)
                    with open(image_meta_path, 'w', encoding='utf-8') as f:
                        json.dump(self.image_metadata_list, f, ensure_ascii=False, indent=4)
                    print(f"이미지 인덱스 ({self.image_index.ntotal}개 항목) 및 메타데이터 저장 완료: {self.image_index_path}")
                    action_taken = True
                else:
                    print("경고: 이미지 인덱스, 경로 또는 메타데이터 리스트가 제대로 초기화되지 않아 저장할 수 없습니다.")

            elif content_type_to_save == "text":
                if hasattr(self, 'text_index') and hasattr(self, 'text_index_path') and hasattr(self, 'text_metadata_list'):
                    text_meta_path = os.path.join(self.index_path, "text_metadata.json")
                    
                    faiss.write_index(self.text_index, self.text_index_path)
                    with open(text_meta_path, 'w', encoding='utf-8') as f:
                        json.dump(self.text_metadata_list, f, ensure_ascii=False, indent=4)
                    print(f"텍스트 인덱스 ({self.text_index.ntotal}개 항목) 및 메타데이터 저장 완료: {self.text_index_path}")
                    action_taken = True
                else:
                    print("경고: 텍스트 인덱스, 경로 또는 메타데이터 리스트가 제대로 초기화되지 않아 저장할 수 없습니다.")
            else:
                print(f"경고: 알 수 없는 content_type_to_save 값: {content_type_to_save}. 인덱스를 저장하지 않습니다.")
                return False

            if not action_taken:
                # 이 경우는 위에서 이미 hasattr 등으로 처리되었을 가능성이 높음
                print(f"{content_type_to_save} 유형에 대해 저장할 인덱스 데이터가 없거나, 초기화 문제가 있습니다.")
            return action_taken
            
        except Exception as e:
            print(f"인덱스 저장 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def load_index(self):
        """저장된 인덱스와 메타데이터 로드"""
        try:
            # 이미지 인덱스 로드
            if os.path.exists(self.image_index_path):
                self.image_index = faiss.read_index(self.image_index_path)
                # 이미지 메타데이터 로드
                meta_path = os.path.join(self.index_path, "image_metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        self.image_metadata_list = json.load(f)
                print(f"이미지 인덱스 로드 완료: {self.image_index.ntotal} 개의 벡터")
            
            # 텍스트 인덱스 로드
            if os.path.exists(self.text_index_path):
                self.text_index = faiss.read_index(self.text_index_path)
                # 텍스트 메타데이터 로드
                meta_path = os.path.join(self.index_path, "text_metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        self.text_metadata_list = json.load(f)
                print(f"텍스트 인덱스 로드 완료: {self.text_index.ntotal} 개의 벡터")
            
            return True
            
        except Exception as e:
            print(f"인덱스 로드 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# 사용 예시
if __name__ == "__main__":
    import argparse

    # 명령행 인수 처리
    parser = argparse.ArgumentParser(description="임베딩 생성 시스템")
    parser.add_argument("--json", "-j", help="처리할 JSON 파일 경로")
    parser.add_argument("--index", "-i", default="faiss_index", help="인덱스 저장 경로")
    parser.add_argument("--model", "-m", default="Alibaba-NLP/gme-Qwen2-VL-7B-Instruct", help="사용할 모델 이름")
    parser.add_argument("--overwrite", "-o", action="store_true", help="기존 인덱스를 덮어씀")
    parser.add_argument("--type", "-t", choices=["image", "text"], default="text", 
                       help="처리할 데이터 유형 (image 또는 text)")
    
    args = parser.parse_args()
    
    # 벡터 DB 서비스 초기화
    index_path = "/home/nextits2/.conda/envs/workbox/faiss_index" if args.index == "text_index" else args.index
    vector_db_service = VectorDBService(
        model_name=args.model,
        index_path=index_path
    )
    
    # JSON 파일 처리
    if args.json:
        try:
            # append_mode 설정 (--overwrite 옵션이 있으면 False, 없으면 True)
            append_mode = not args.overwrite
            indexed_count = vector_db_service.process_json_data(
                args.json, 
                append_mode=append_mode,
                content_type=args.type
            )
            
            if indexed_count > 0:
                # 인덱스 저장 (처리된 content_type 명시)
                if vector_db_service.save_index(content_type_to_save=args.type):
                    print(f"인덱스 저장 완료: {vector_db_service.index_path} (유형: {args.type})")
                    print(f"생성된 인덱스는 rag_system.py에서 검색할 수 있습니다.")
                else:
                    print(f"{args.type} 유형의 인덱스 저장 실패.")
            elif args.type == "image" and not vector_db_service.image_metadata_list and args.overwrite:
                 # Overwrite 모드이고 이미지 데이터가 없는 경우, 명시적으로 이미지 인덱스/메타데이터를 빈 상태로 저장 시도
                if vector_db_service.save_index(content_type_to_save=args.type):
                    print(f"Overwrite 모드: {args.type} 유형의 인덱스가 비어있는 상태로 저장되었습니다.")
                else:
                    print(f"Overwrite 모드: {args.type} 유형의 빈 인덱스 저장 실패.")
            elif args.type == "text" and not vector_db_service.text_metadata_list and args.overwrite:
                # Overwrite 모드이고 텍스트 데이터가 없는 경우, 명시적으로 텍스트 인덱스/메타데이터를 빈 상태로 저장 시도
                if vector_db_service.save_index(content_type_to_save=args.type):
                    print(f"Overwrite 모드: {args.type} 유형의 인덱스가 비어있는 상태로 저장되었습니다.")
                else:
                    print(f"Overwrite 모드: {args.type} 유형의 빈 인덱스 저장 실패.")
            else:
                print("처리할 데이터가 없거나, 인덱싱된 항목이 없습니다.")
        except Exception as e:
            print(f"JSON 데이터 처리 실패: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    # 사용법 출력
    if not args.json:
        print("사용법:")
        print("  임베딩 생성: python embedding_system.py --json <json_file_path> [--index <index_path>] [--model <model_name>]")
        print("  예시: python embedding_system.py --json /home/nextits2/.conda/envs/workbox/test.json --index test_index")
        print("\n참고: 검색 기능은 rag_system.py에서 제공합니다.")

