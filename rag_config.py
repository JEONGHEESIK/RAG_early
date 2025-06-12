import os
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
import torch

# 1. 설정 및 초기화 클래스
class RAGConfig:
    """RAG 시스템 설정"""
    def __init__(self):
        # 기본 설정
        self.DATA_PATH = "/home/nextits2/.conda/envs/workbox/test.json"
        self.IMAGE_BASE_PATH = "/home/nextits2/completed_images_2"  # 실제 이미지가 저장된 기본 경로
        
        # 모델 설정
        self.MULTIMODAL_MODEL = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"  # 멀티모달 임베딩 모델
        
        # 쿼리 리라이팅 모델 설정
        self.REWRITER_MODEL = "mistral:7b-instruct-v0.3-fp16"  # 쿼리 리라이팅용 모델
        self.USE_QUERY_REWRITING = True  # 쿼리 리라이팅 사용 여부
        
        # 벡터 차원 - Qwen2-VL 모델의 실제 임베딩 차원 (3584)으로 설정
        self.VECTOR_DIMENSION = 3584
        
        # 텍스트 처리 설정
        self.MAX_LENGTH = 512  # 최대 텍스트 길이
        
        # 검색 설정
        self.TOP_K = 5
        self.IMAGE_RELEVANCE_THRESHOLD = 0.1  # 이미지 검색 임계값
        self.TEXT_RELEVANCE_THRESHOLD = 0.6  # 텍스트 검색 임계값
        
        # Ollama 설정 - analyze_images_new.py와 동일하게 맞춤
        self.OLLAMA_API_URL = "http://localhost:11434"  # 기본 URL
        self.OLLAMA_FALLBACK_URL = "http://localhost:11434"  # 도커 컨테이너 이름 사용
        self.OLLAMA_MODEL = "llama4:17b-scout-16e-instruct-q4_K_M"  # 정확한 모델 이름 사용
        
        # API 엔드포인트 경로 (analyze_images_new.py와 동일)
        self.GENERATE_ENDPOINT = "/api/generate"  # 텍스트 생성 엔드포인트
        self.MODELS_ENDPOINT = "/api/tags"      # 모델 목록 엔드포인트  # 사용할 모델 (더 가벼운 버전으로 변경)
        
        # 모델 설정
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.BATCH_SIZE = 4
        
        # GPU 최적화 설정
        self.GPU_COUNT = torch.cuda.device_count()
        self.USE_MULTI_GPU = self.GPU_COUNT > 1
        self.MAIN_GPU = 0  # 주 GPU 인덱스
        self.SECOND_GPU = 1 if self.USE_MULTI_GPU else 0  # 보조 GPU 인덱스
        
        # 모델 분할 설정 - 레이어 수에 따라 조정
        self.MODEL_SPLIT_LAYERS = 14  # 첫 14개 레이어는 GPU 0, 나머지는 GPU 1에 할당
