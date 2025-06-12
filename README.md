<div align="center">
<br>

![python](https://img.shields.io/badge/python-3.11~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux-pink.svg)

</div>

# 멀티모달 RAG 시스템 (Multi-Modal RAG System)

이미지와 텍스트를 모두 처리할 수 있는 고성능 멀티모달 RAG(Retrieval-Augmented Generation) 시스템입니다. Ollama와 Qwen2-VL 모델을 활용하여 대용량 데이터에서도 빠르고 정확한 검색 및 생성 기능을 제공합니다.

## 목차

1.  [주요 특징 및 용도](#1-주요-특징-및-용도)
2.  [시스템 아키텍처](#2-시스템-아키텍처)
3.  [기술 스택](#3-기술-스택)
4.  [설치 및 실행 가이드](#4-설치-및-실행-가이드)
    - [시스템 요구사항](#시스템-요구사항)
    - [설치 절차](#설치-절차)
5.  [환경 설정](#5-환경-설정)
6.  [사용 가이드](#6-사용-가이드)
    - [CLI 도구 사용법](#cli-도구-사용법)
    - [API 서버 사용법](#api-서버-사용법)
7.  [문제 해결](#7-문제-해결)
8.  [프로젝트 심화 정보](#8-프로젝트-심화-정보)
    - [프로젝트 구조](#프로젝트-구조)
    - [핵심 컴포넌트 상세](#핵심-컴포넌트-상세)
    - [성능 최적화 및 확장성](#성능-최적화-및-확장성)
9.  [향후 계획](#9-향후-계획)
10. [기여 안내](#10-기여-안내)
11. [추가 정보](#11-추가-정보)

## 1. 주요 특징 및 용도

### 주요 특징

-   **멀티모달 검색**: 텍스트와 이미지를 통합적으로 검색하는 하이브리드 검색 엔진
-   **고성능 임베딩**: Qwen2-VL 모델 기반의 정교한 시맨틱 임베딩
-   **지능형 쿼리 최적화**: Mistral 7B 모델을 활용한 자동 쿼리 리라이팅
-   **GPU 가속**: CUDA 기반의 병렬 처리로 초고속 검색 가능
-   **RESTful API**: FastAPI 기반의 확장 가능한 API 아키텍처
-   **컨테이너 지원**: Docker를 통한 손쉬운 배포 및 확장

### 주요 용도

-   문서 및 이미지 기반 지식 검색 시스템
-   AI 어시스턴트를 위한 지식 베이스
-   대화형 Q&A 시스템
-   멀티모달 콘텐츠 분석 플랫폼

## 2. 시스템 아키텍처

### 데이터 흐름

1.  **쿼리 수신**: 사용자가 API 또는 CLI를 통해 텍스트 또는 이미지 쿼리를 입력합니다.
2.  **쿼리 최적화 (선택 사항)**: `Mistral 7B` 모델이 검색 정확도를 높이기 위해 쿼리를 재작성합니다.
3.  **멀티모달 임베딩**: `Qwen2-VL` 모델이 쿼리를 벡터 임베딩으로 변환합니다.
4.  **유사도 검색**: `FAISS` 벡터 데이터베이스에서 쿼리 벡터와 유사한 텍스트 및 이미지 벡터를 검색합니다.
5.  **결과 후처리**: 검색된 결과를 순위화하고 컨텍스트를 구성합니다.
6.  **답변 생성**: 구성된 컨텍스트를 기반으로 `Ollama`의 LLM이 최종 답변을 생성합니다.
7.  **응답 반환**: 최종 검색 결과와 생성된 답변을 사용자에게 반환합니다.

## 3. 기술 스택

| 구분 | 기술 | 설명 |
| :--- | :--- | :--- |
| **언어** | **Python 3.11+** | 프로젝트의 주 개발 언어입니다. |
| **ML/DL** | **Transformers** | `Alibaba-NLP/gme-Qwen2-VL-7B-Instruct` 모델 로딩 및 사용 |
| | **FAISS** | Facebook AI의 고성능 벡터 유사도 검색 라이브러리 |
| | **PyTorch** | 딥러닝 모델의 백엔드 프레임워크 |
| **LLM 인프라**| **Ollama** | 로컬 환경에서 LLM을 실행하기 위한 플랫폼 |
| | `llama4:17b-scout-16e`| 메인 답변 생성을 위한 모델 |
| | `mistral:7b-instruct`| 쿼리 최적화를 위한 모델 |
| **웹 프레임워크**| **FastAPI** | 고성능 비동기 API 서버 구축 |
| | **Uvicorn** | ASGI 서버 |
| **유틸리티**| **Pydantic** | 데이터 유효성 검사 및 설정 관리 |
| | **NumPy, Pillow** | 수치 연산 및 이미지 처리 |
| | **Docker** | Ollama 서버 등 서비스의 컨테이너화 및 배포 |

## 4. 설치 및 실행 가이드

### 시스템 요구사항

-   **OS**: Linux (Rocky Linux 8.9에서 테스트 완료)
-   **Python**: 3.11.8 이상
-   **하드웨어**:
    -   **GPU**: NVIDIA H100 또는 유사 사양. `Alibaba-NLP/gme-Qwen2-VL-7B-Instruct` 모델 및 FAISS의 원활한 구동을 위해 **최소 100GB 이상의 VRAM**을 강력히 권장합니다.
-   **소프트웨어**:
    -   Docker (v26.1.3 이상)
    -   NVIDIA Container Toolkit (GPU 가속 사용 시 필수)
    -   CUDA 12.6

### 설치 절차

1.  **저장소 복제 및 환경 변수 설정**
    ```bash
    # 저장소 클론
    git clone <repository-url> && cd rag_system

    # 환경 변수 파일 생성 (필요 시 .env 파일 내용 수정)
    cp .env.example .env
    ```

2.  **가상환경 생성 및 활성화**
    ```bash
    # Linux/macOS
    python -m venv venv
    source venv/bin/activate

    # Windows
    # .\venv\Scripts\activate
    ```

3.  **필수 패키지 설치**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Ollama 설치 및 모델 다운로드**
    ```bash
    # Ollama 설치 (Linux)
    curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh

    # Docker를 사용한 Ollama 서버 실행 (GPU 사용)
    docker run -d --gpus=all \
      -p 11434:11434 \
      --name ollama \
      -v ollama:/root/.ollama \
      ollama/ollama

    # 필수 LLM 모델 다운로드 (시간이 소요될 수 있음)
    docker exec ollama ollama pull mistral:7b-instruct-v0.3-fp16
    docker exec ollama ollama pull llama4:17b-scout-16e-instruct-q4_K_M
    ```

5.  **데이터 임베딩 및 인덱싱**
    `embedding_system.py`를 사용하여 검색 대상 데이터의 벡터 인덱스를 생성합니다.

    ```bash
    # 텍스트 데이터 인덱싱 예시
    python embedding_system.py --json /path/to/text_data.json --type text

    # 이미지 데이터 인덱싱 예시
    python embedding_system.py --json /path/to/image_data.json --type image
    ```

## 5. 환경 설정

시스템의 주요 설정은 `rag_config.py` 파일의 `RAGConfig` 클래스에서 관리합니다.

```python
# ===== 데이터 설정 =====
DATA_PATH = "/path/to/your/data.json"
IMAGE_BASE_PATH = "/path/to/images"
EMBEDDING_CACHE_DIR = "./embeddings"

# ===== 모델 설정 =====
MULTIMODAL_MODEL = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
OLLAMA_MODEL = "llama4:17b-scout-16e-instruct-q4_K_M"
QUERY_REWRITER_MODEL = "mistral:7b-instruct-v0.3-fp16"

# ===== 검색 설정 =====
TOP_K = 5
SIMILARITY_THRESHOLD = 0.6
USE_QUERY_REWRITING = True

# ===== 성능 설정 =====
BATCH_SIZE = 4
USE_GPU = True
```

## 6. 사용 가이드

### CLI 도구 사용법

`rag_clihandler.py`를 통해 커맨드 라인에서 시스템을 사용할 수 있습니다.

-   **기본 검색**
    ```bash
    python rag_clihandler.py -q "검색어"
    ```
-   **상위 10개 결과 표시**
    ```bash
    python rag_clihandler.py -q "검색어" -k 10
    ```
-   **이미지만 검색**
    ```bash
    python rag_clihandler.py -q "검색어" --media-type image
    ```
-   **AI 답변 생성 비활성화**
    ```bash
    python rag_clihandler.py -q "검색어" --no-ai
    ```
-   **대화형 모드 실행**
    ```bash
    python rag_clihandler.py -i
    ```

### API 서버 사용법

1.  **FastAPI 서버 실행**
    ```bash
    uvicorn rag_fastapi:app --host 0.0.0.0 --port 8000 --reload
    ```
    서버 실행 후, 브라우저에서 `http://localhost:8000/docs`를 통해 API 문서를 확인할 수 있습니다.

2.  **API 엔드포인트**

    -   `POST /api/v1/search`: 문서 검색 및 답변 생성
    -   `POST /chat`: 대화형 채팅 인터페이스 (README에 명시되어 있으나 상세 내용은 `rag_fastapi.py` 확인 필요)

3.  **API 요청 예시 (`/api/v1/search`)**

    **Request:**
    ```json
    {
      "query": "검색할 질문 또는 키워드",
      "top_k": 5,
      "threshold": 0.6,
      "use_images": true,
      "rewrite_query": true
    }
    ```

    **Response:**
    ```json
    {
      "success": true,
      "query": "원본 쿼리 문자열",
      "results": [
        {
          "id": "문서 ID",
          "content": "문서 내용",
          "similarity": 0.95,
          "metadata": { "source": "출처" },
          "type": "text"
        }
      ],
      "response": "생성된 답변 (AI가 생성한 경우)",
      "images": [
        { "id": "이미지 ID", "url": "이미지 URL" }
      ],
      "error": null
    }
    ```

## 7. 문제 해결

-   **CUDA 메모리 부족**: `rag_config.py` 파일에서 `BATCH_SIZE`를 더 작은 값(예: 2 또는 1)으로 조정하세요.
-   **Ollama 연결 오류**:
    ```bash
    # Docker 컨테이너 상태 확인 및 재시작
    docker ps | grep ollama
    docker restart ollama
    docker logs -f ollama
    ```

## 8. 프로젝트 심화 정보

### 프로젝트 구조

#### 주요 파일 구성

```/rag_system/
├── README.md              # 프로젝트 문서
├── embedding_system.py    # 벡터 임베딩 생성 및 관리 시스템
├── image_incoder.py       # 이미지 인코딩 처리 모듈
├── ollamagen.py          # Ollama API를 통한 텍스트 생성 모듈
├── rag_clihandler.py     # 커맨드 라인 인터페이스 처리기
├── rag_config.py         # 시스템 설정 관리
├── rag_fastapi.py        # FastAPI 기반 REST API 서버
├── rag_formatter.py      # 검색 결과 포맷팅 모듈
├── rag_initializer.py    # 시스템 초기화 및 설정 모듈
├── rag_integration.py    # 외부 시스템 통합 모듈
├── rag_rewriter.py       # 쿼리 재작성 모듈
├── rag_search.py         # 핵심 검색 엔진 모듈
├── rag_utils.py          # 유틸리티 함수 모듈
├── requirements.txt      # 의존성 패키지 목록
└── telepathy_rag.py      # 메인 RAG 시스템 통합 모듈
```

### 핵심 컴포넌트 상세

-   **RAGSearchEngine (`rag_search.py`)**: FAISS를 활용하여 멀티모달 임베딩의 유사도 검색을 수행하는 핵심 엔진입니다.
-   **VectorDBService (`embedding_system.py`)**: 텍스트와 이미지 데이터를 임베딩하고, 별도의 FAISS 인덱스로 관리하며, 저장/로드 기능을 제공합니다.
-   **OllamaGenerator (`ollamagen.py`)**: Ollama API를 호출하여 검색된 컨텍스트 기반의 답변을 생성합니다.
-   **QueryRewriter (`rag_rewriter.py`)**: Mistral 모델을 사용하여 사용자 쿼리를 더 명확하고 풍부하게 재작성하여 검색 성능을 향상시킵니다.
-   **SearchIntegration (`rag_integration.py`)**: 쿼리 재작성, 검색, 답변 생성 등 RAG의 전체 워크플로우를 통합하고 조율합니다.

### 성능 최적화 및 확장성

-   **성능 최적화**: HNSW 인덱스 사용, 모델 양자화, 쿼리 캐싱, 배치 처리 등을 통해 성능을 향상시켰습니다.
-   **확장성**: 분산 벡터 검색(샤딩), 로드 밸런싱 등을 통해 수평 확장이 가능하며, 새로운 모델이나 데이터베이스를 쉽게 추가할 수 있는 플러그인 아키텍처를 고려하고 있습니다.

## 9. 향후 계획

-   [ ] 멀티모달 임베딩 성능 최적화
-   [ ] Kubernetes 기반 자동 확장(Auto-scaling) 기능 구현
-   [ ] 시스템 상태 모니터링을 위한 대시보드 개발
-   [ ] 지식 증류(Knowledge Distillation)를 통한 모델 경량화 연구

## 10. 기여 안내

이 프로젝트에 기여하고 싶으시다면 Pull Request를 환영합니다.

1.  저장소를 Fork한 후 새로운 기능 브랜치(`feature/your-feature`)를 생성합니다.
2.  코드 스타일(PEP 8, Google Docstring)을 준수하여 코드를 작성합니다.
3.  모든 함수에 타입 힌트를 사용하고, 테스트 커버리지를 80% 이상 유지합니다.
4.  작업 완료 후 PR(Pull Request)을 제출합니다. 최소 1명 이상의 리뷰어 승인이 필요합니다.

버그 리포트나 기능 제안은 GitHub 이슈 트래커를 이용해 주세요.

## 11. 추가 정보

### 감사의 말씀

이 프로젝트는 다음의 훌륭한 오픈소스 프로젝트들 덕분에 가능했습니다.

-   [Hugging Face Transformers](https://github.com/huggingface/transformers)
-   [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
-   [Ollama](https://ollama.ai/)
-   [FastAPI](https://fastapi.tiangolo.com/)

### 연락처

-   **Email**: `jeongnext@hnextits.com`, `junseung_lim@hnextits.com`, `freak91uk@hnextits.com`
-   **Issues**: [GitHub Issues](https://github.com/hnextits/telepathy_rag/issues)
