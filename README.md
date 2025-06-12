<div align="center">
<img src="https://github.com/user-attachments/assets/4547f2d9-b5d7-4dbe-8871-bef3576b0862" width="500" height="350"/>

![python](https://img.shields.io/badge/python-3.11~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux-pink.svg)

</div>

# 멀티모달 RAG 시스템(Multi-Modal RAG System)

## 프로젝트 개요

이미지와 텍스트를 모두 처리할 수 있는 고성능 멀티모달 RAG 시스템입니다. Ollama와 Qwen2-VL 모델을 활용한 고급 검색 및 생성 기능을 제공하며, 대용량 데이터에서도 빠르고 정확한 검색이 가능합니다.

### 주요 특징

- **멀티모달 검색**: 텍스트와 이미지를 통합적으로 검색하는 하이브리드 검색 엔진
- **고성능 임베딩**: Qwen2-VL 모델 기반의 정교한 시맨틱 임베딩
- **지능형 쿼리 최적화**: Mistral 7B 모델을 활용한 자동 쿼리 리라이팅
- **GPU 가속**: CUDA 기반의 병렬 처리로 초고속 검색 가능
- **RESTful API**: FastAPI 기반의 확장 가능한 API 아키텍처
- **컨테이너 지원**: Docker를 통한 손쉬운 배포 및 확장

### 주요 용도

- 문서 및 이미지 기반 지식 검색 시스템
- AI 어시스턴트를 위한 지식 베이스
- 대화형 Q&A 시스템
- 멀티모델 콘텐츠 분석 플랫폼

## 개발환경

- **Python**: 3.11.8 이상
- **Linux** : Rocky Linux 8.9
- **하드웨어**:
  - GPU: NVIDIA H100 기반
- **소프트웨어**:
  - Docker Docker version 26.1.3
  - NVIDIA Container Toolkit (GPU 가속 사용 시)
  - CUDA 12.6

## 설치 가이드

### 1. 저장소 복제 및 설정
```bash
# 저장소 클론
git clone <repository-url> && cd rag_system

# 환경 변수 파일 생성 (필요 시 수정)
cp .env.example .env
```

### 2. 가상환경 설정
```bash
# 가상환경 생성 및 활성화 (Linux/macOS)
python -m venv venv
source venv/bin/activate

# Windows의 경우
# .\venv\Scripts\activate
```

### 3. 의존성 설치
```bash
# 필수 패키지 설치
pip install --upgrade pip
pip install -r requirements.txt

# 개발용 추가 패키지 (선택사항)
pip install -r requirements-dev.txt
```

### 4. Ollama 서버 설정
```bash
# Docker를 사용한 Ollama 서버 실행
docker run -d --gpus=all \
  -p 11434:11434 \
  --name ollama \
  -v ollama:/root/.ollama \
  ollama/ollama

# 필요한 모델 다운로드 (대기시간 발생 가능)
docker exec ollama ollama pull mistral:7b-instruct-v0.3-fp16
docker exec ollama ollama pull llama4:17b-scout-16e-instruct-q4_K_M

# 모델 다운로드 상태 확인
docker logs -f ollama
```

## 환경 설정

시스템 설정은 `rag_config.py` 파일에서 관리됩니다. 주요 설정 항목은 다음과 같습니다:

```python
# ===== 데이터 설정 =====
DATA_PATH = "/path/to/your/data.json"      # 검색 대상 JSON 데이터 경로
IMAGE_BASE_PATH = "/path/to/images"        # 이미지 파일이 저장된 기본 경로
EMBEDDING_CACHE_DIR = "./embeddings"      # 임베딩 캐시 디렉토리

# ===== 모델 설정 =====
MULTIMODAL_MODEL = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"  # 멀티모달 모델
OLLAMA_MODEL = "llama4:17b-scout-16e-instruct-q4_K_M"      # Ollama 생성 모델
QUERY_REWRITER_MODEL = "mistral:7b-instruct-v0.3-fp16"     # 쿼리 최적화 모델

# ===== 검색 설정 =====
TOP_K = 5                                   # 기본 검색 결과 수
SIMILARITY_THRESHOLD = 0.6                 # 유사도 임계값 (0.0 ~ 1.0)
MAX_CONTEXT_LENGTH = 4096                   # 컨텍스트 최대 길이 (토큰 수)
USE_QUERY_REWRITING = True                  # 쿼리 리라이팅 활성화 여부

# ===== 성능 설정 =====
BATCH_SIZE = 4                             # 배치 처리 크기
USE_GPU = True                             # GPU 사용 여부
MAX_WORKERS = 4                            # 병렬 처리 작업자 수
```

## 사용 가이드

### 1. CLI 도구 사용

기본 검색을 수행하려면 다음 명령어를 사용하세요:
```bash
python rag_clihandler.py "검색어"
```

#### 고급 옵션

```bash
# 상위 N개 결과 표시 (기본값: 5)
python rag_clihandler.py "검색어" --top_k 5

# 특정 유형의 결과만 필터링
python rag_clihandler.py "검색어" --content_type text  # 텍스트만 검색
python rag_clihandler.py "검색어" --content_type image  # 이미지만 검색

# 고급 검색 옵션
python rag_clihandler.py "검색어" \
  --no_rewrite \          # 쿼리 리라이팅 비활성화
  --threshold 0.7 \       # 유사도 임계값 설정 (0.0 ~ 1.0)
  --max_length 2000 \     # 결과 텍스트 최대 길이 설정
  --debug                 # 디버그 모드 활성화

# 배치 모드로 여러 쿼리 처리
echo -e "첫 번째 검색어\n두 번째 검색어" | python rag_clihandler.py --batch

# 결과를 JSON 형식으로 출력
python rag_clihandler.py "검색어" --output_format json > results.json
```

### 2. FastAPI 서버 실행

#### 개발 서버 시작
```bash
# 기본 실행 (개발 모드)
uvicorn rag_fastapi:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 환경 (Gunicorn 사용 권장)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker rag_fastapi:app --bind 0.0.0.0:8000
```

#### API 문서
서버가 시작되면 다음 주소에서 대화형 API 문서를 확인할 수 있습니다:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI 스키마**: `http://localhost:8000/openapi.json`

### 3. API 레퍼런스

#### 검색 API (`POST /api/v1/search`)

**요청 본문 (JSON):**

| 매개변수        | 타입    | 필수 | 기본값 | 설명                                      |
|----------------|---------|------|--------|------------------------------------------|
| `query`       | string  | 예   | -      | 검색할 쿼리 문자열                        |
| `top_k`       | integer | 아니오 | 5      | 반환할 결과의 최대 개수                  |
| `threshold`   | float   | 아니오 | 0.6    | 유사도 임계값 (0.0 ~ 1.0)                |
| `use_images`  | boolean | 아니오 | true   | 이미지 검색 포함 여부                     |
| `rewrite_query` | boolean | 아니오 | true   | 쿼리 재작성 사용 여부                   |
| `filters`     | object  | 아니오 | {}     | 추가 필터링 조건 (메타데이터 기반)       |


## 시스템 아키텍처

### 개요

멀티모달 RAG 시스템은 다음과 같은 주요 컴포넌트로 구성됩니다:

1. **Frontend Layer**
   - 사용자 인터페이스 (웹/CLI)
   - API Gateway (FastAPI)
   - 인증/인가 서비스

2. **Application Layer**
   - 검색 서비스
   - 쿼리 처리기
   - 결과 정렬 및 필터링
   - 캐시 관리

3. **Model Layer**
   - 멀티모델 임베딩 (Qwen2-VL)
   - LLM 기반 쿼리 최적화 (Mistral 7B)
   - 벡터 검색 (FAISS)

4. **Data Layer**
   - 벡터 데이터베이스
   - 문서 저장소
   - 메타데이터 인덱스

### 데이터 흐름

1. 사용자 쿼리 수신
2. (선택사항) 쿼리 재작성 및 최적화
3. 멀티모달 임베딩 생성
4. 벡터 유사도 검색
5. 결과 후처리 및 정렬
6. 최종 응답 반환

## 기여 가이드

### 개발 환경 설정

1. 저장소 포크 및 클론
2. 개발 브랜치 생성: `git checkout -b feature/your-feature-name`
3. 코드 수정 및 테스트
4. 코드 포맷팅: `black . && isort .`
5. 테스트 실행: `pytest tests/`
6. PR 제출

### 코드 스타일

- **Python**: PEP 8 준수
- **Docstring**: Google 스타일 사용
- **타입 힌트**: 모든 함수에 타입 힌트 사용
- **테스트 커버리지**: 80% 이상 유지

### 이슈 템플릿

버그 리포트나 기능 요청 시 다음 정보를 포함해 주세요:

- **버전**: 시스템 및 패키지 버전
- **재현 방법**: 단계별 재현 방법
- **예상 동작**: 기대한 정상 동작
- **실제 동작**: 발생한 문제
- **스크린샷/로그**: 관련 로그 또는 스크린샷

## 연락처

문의사항이 있으시면 다음 연락처로 문의해 주세요:
- 이메일: jeongnext@hnextits.com, junseung_lim@hnextits.com, freak91uk@hnextits.com
- 이슈 트래커: [GitHub Issues](https://github.com/hnextits/telepathy_rag/issues) 사용

## 감사의 말씀

이 프로젝트는 다음과 같은 오픈소스 프로젝트에 기반하고 있습니다:
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.ai/)
- [FastAPI](https://fastapi.tiangolo.com/)

## 임베딩 시스템 (embedding_system.py)

### 개요

`embedding_system.py`는 멀티모달 RAG 시스템의 핵심 컴포넌트로, 텍스트와 이미지 데이터를 위한 임베딩을 생성하고 관리합니다. 이 모듈은 `VectorDBService` 클래스를 통해 다음과 같은 기능을 제공합니다:

- 멀티모달 모델(Qwen2-VL)을 활용한 텍스트 및 이미지 임베딩 생성
- 이미지와 텍스트를 위한 별도의 FAISS 인덱스 관리
- 배치 처리를 통한 대용량 데이터 임베딩 생성
- 인덱스 및 메타데이터 저장/로드 기능

### 주요 기능

#### 1. 임베딩 생성
- **텍스트 임베딩**: `generate_text_embedding()` 메서드를 통해 텍스트 데이터의 임베딩 생성
- **이미지 임베딩**: `get_image_embedding()` 메서드를 통해 이미지 파일의 임베딩 생성
- **배치 처리**: `generate_batch_embeddings()` 메서드로 대용량 텍스트 데이터의 효율적인 처리

#### 2. 인덱스 관리
- **이원화된 인덱스**: 텍스트와 이미지를 위한 별도의 FAISS 인덱스 관리
- **메타데이터 저장**: 각 임베딩에 대한 메타데이터 관리
- **인덱스 저장/로드**: 생성된 인덱스와 메타데이터를 파일로 저장하고 필요시 로드

#### 3. 검색 기능
- **유사도 검색**: `search_in_index()` 메서드를 통한 벡터 유사도 기반 검색
- **메타데이터 조회**: `get_metadata()` 메서드로 검색 결과의 메타데이터 조회

### 사용 방법

#### 1. 임베딩 생성 및 인덱싱

```bash
# 텍스트 데이터 임베딩 생성 및 인덱싱
python embedding_system.py --json /path/to/text_data.json --type text

# 이미지 데이터 임베딩 생성 및 인덱싱
python embedding_system.py --json /path/to/image_data.json --type image

# 기존 인덱스 덮어쓰기
python embedding_system.py --json /path/to/data.json --type text --overwrite

# 커스텀 인덱스 경로 지정
python embedding_system.py --json /path/to/data.json --index /path/to/custom_index
```

#### 2. 명령행 인수

| 인수 | 설명 |
|------|------|
| `--json`, `-j` | 처리할 JSON 파일 경로 |
| `--index`, `-i` | 인덱스 저장 경로 (기본값: faiss_index) |
| `--model`, `-m` | 사용할 모델 이름 (기본값: Alibaba-NLP/gme-Qwen2-VL-7B-Instruct) |
| `--overwrite`, `-o` | 기존 인덱스를 덮어쓰기 (기본값: False) |
| `--type`, `-t` | 처리할 데이터 유형 (image 또는 text) |

#### 3. JSON 데이터 형식

텍스트 데이터 예시:
```json
[
  {
    "id": "doc1",
    "content": "텍스트 내용",
    "metadata": {
      "source": "문서 출처",
      "category": "카테고리"
    }
  }
]
```

이미지 데이터 예시:
```json
[
  {
    "id": "img1",
    "image_path": "images/example.jpg",
    "caption": "이미지 설명",
    "metadata": {
      "source": "이미지 출처",
      "category": "카테고리"
    }
  }
]
```

### 성능 최적화

- **배치 크기 조정**: 메모리 사용량과 처리 속도 간의 균형을 위해 배치 크기 조정
- **GPU 가속**: CUDA 지원 환경에서 자동으로 GPU 사용
- **인덱스 유형 선택**: 유사도 측정 방식에 따라 "IP"(내적) 또는 "L2"(유클리드 거리) 선택 가능

## 문제 해결

### 1. CUDA 메모리 부족
```bash
# rag_config.py에서 배치 사이즈 조정
BATCH_SIZE = 2  # 더 작은 값으로 조정
USE_GPU = True  # GPU 사용 여부 확인
```

### 2. Ollama 연결 오류
```bash
# Docker 컨테이너 상태 확인
docker ps | grep ollama

# 컨테이너 로그 확인
docker logs ollama

# 컨테이너 재시작
docker restart ollama
```

### 3. 모델 다운로드 문제
```bash
# 수동으로 모델 다운로드 시도
"""python -c "
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained('Alibaba-NLP/gme-Qwen2-VL-7B-Instruct')
processor = AutoProcessor.from_pretrained('Alibaba-NLP/gme-Qwen2-VL-7B-Instruct')
""""

## 프로젝트 구조


### 주요 컴포넌트

1. **RAGSearchEngine** (`rag_search.py`)
   - 멀티모달 임베딩 생성 및 벡터 검색
   - FAISS를 활용한 고속 유사도 검색
   - 이미지 및 텍스트 데이터 처리

2. **VectorDBService** (`embedding_system.py`)
   - 멀티모달 임베딩 생성 및 관리
   - 이미지와 텍스트를 위한 별도의 FAISS 인덱스 관리
   - 배치 처리를 통한 대용량 데이터 임베딩 생성
   - 인덱스 저장 및 로드 기능

3. **OllamaGenerator** (`models/generator.py`)
   - Ollama API를 통한 텍스트 생성
   - 다중 엔드포인트 지원 및 자동 장애 조치
   - 스트리밍 응답 처리

4. **QueryRewriter** (`rag_rewriter.py`)
   - 사용자 쿼리 최적화
   - Mistral 모델 기반 쿼리 재작성
   - 검색 정확도 향상을 위한 쿼리 확장

5. **FastAPI 애플리케이션** (`rag_fastapi.py`)
   - RESTful API 엔드포인트 제공
   - 비동기 요청 처리
   - Swagger/OpenAPI 문서 자동 생성

6. **CLI 인터페이스** (`rag_clihandler.py`)
   - 명령줄에서의 상호작용 지원
   - 대화형 모드
   - 배치 처리 기능

7. **SearchIntegration** (`rag_integration.py`)
   - RAG 워크플로우 조율
   - 쿼리 재작성, 검색, 답변 생성 통합
   - 로깅 및 결과 처리


### 데이터 흐름

1. 사용자 쿼리 수신
2. (선택사항) 쿼리 재작성 및 최적화
3. 멀티모달 임베딩 생성
4. 벡터 유사도 검색
5. 결과 후처리 및 정렬 (Rerank 진행 중 - 선택사항)
6. 최종 응답 반환

## 기술 스택

### 핵심 기술

#### 프로그래밍 언어
- **Python 3.8+**: 고성능 시스템 개발에 적합한 동적 타이핑 언어

#### 머신러닝/딥러닝
- **Transformers (Hugging Face)**:
  - `Alibaba-NLP/gme-Qwen2-VL-7B-Instruct`: 멀티모달 임베딩 생성
  - `Qwen2VLForConditionalGeneration`: Qwen2-VL 모델 로딩
- **FAISS (Facebook AI Similarity Search)**:
  - 고성능 벡터 유사도 검색
  - 대규모 데이터셋에서의 효율적인 검색 지원
  - 이미지 및 텍스트를 위한 별도의 인덱스 관리

#### LLM 인프라
- **Ollama**:
  - 로컬 LLM 실행 환경
  - 지원 모델:
    - `llama4:17b-scout-16e-instruct-q4_K_M`: 메인 답변 생성
    - `mistral:7b-instruct-v0.3-fp16`: 쿼리 최적화

#### 웹 프레임워크
- **FastAPI**:
  - 고성능 비동기 API 서버
  - 자동 문서화(Swagger/OpenAPI)
  - Pydantic 기반 데이터 검증

#### 유틸리티
- **Pillow (PIL)**: 이미지 처리
- **NumPy**: 수치 연산
- **Pydantic**: 데이터 유효성 검사
- **tqdm**: 진행률 표시

## 3. 설치 및 실행

### 3.1. 환경 설정

1.  **Python 설치**: Python 3.8 이상 버전이 설치되어 있어야 합니다.

2.  **Ollama 설치 및 모델 다운로드**:
    ```bash
    # Ollama 설치 (Linux)
    curl -fsSL https://ollama.com/install.sh | sh
    
    # 필요한 모델 다운로드
    ollama pull mistral:7b-instruct-v0.3-fp16
    ollama pull llama4:17b-scout-16e-instruct-q4_K_M
    ```

3. **Python 의존성 설치**
```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 필수 패키지 설치
pip install -r requirements.txt

# 주요 의존성 목록
# - fastapi: API 서버 프레임워크
# - uvicorn: ASGI 서버
# - pydantic: 데이터 검증
# - pillow: 이미지 처리
# - numpy: 수치 연산
# - httpx: 비동기 HTTP 클라이언트
# - transformers: 멀티모달 모델 지원
# - faiss-cpu/faiss-gpu: 벡터 검색 엔진
# - torch: 딥러닝 프레임워크
# - python-multipart: 멀티파트 폼 데이터 처리
```

4. API 사용법 (FastAPI)
```python
# 1. 기본 검색 API 호출 예시
import requests
import json

# 텍스트 검색 요청
response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": "검색할 질문 또는 키워드",
        "top_k": 5  # 반환할 결과 수 (선택사항)
    }
)

# 응답 처리
results = response.json()
print(f"검색 결과: {len(results['results'])}개 항목 찾음")
print(f"생성된 답변: {results['response']}")

# 2. 채팅 API 호출 예시
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "사용자 메시지",
        "history": []  # 이전 대화 기록 (선택사항)
    }
)

# 응답 처리
chat_result = response.json()
print(f"AI 응답: {chat_result['response']}")
print(f"대화 기록: {len(chat_result['history'])}개 메시지")
```


### API 엔드포인트

- **POST /search**: 문서 검색 및 답변 생성
  - 요청 본문: `{"query": "검색어", "top_k": 5}`
  - 응답: 검색 결과 및 생성된 답변

- **POST /chat**: 대화형 채팅 인터페이스
  - 요청 본문: `{"message": "사용자 메시지", "history": [이전 대화 기록]}`
  - 응답: AI 응답 및 업데이트된 대화 기록

5. 파일 구조 및 역할
```
├── image_incoder.py      # 이미지 인코딩 및 메타데이터 추출
├── ollamagen.py          # Ollama API를 이용한 LLM 제너레이터
├── rag_clihandler.py     # 명령줄 인터페이스(CLI) 처리 및 인자 파싱
├── rag_config.py         # RAG 시스템의 전반적인 설정 정의
├── rag_fastapi.py        # FastAPI 요청/응답 모델 정의 및 CORS 설정
├── rag_formatter.py      # 검색 결과 포맷팅 유틸리티
├── rag_initializer.py    # FastAPI 앱 초기화 및 메인 API 엔드포인트 정의
├── rag_integration.py    # 검색 엔진과 LLM 제너레이터 통합, 전체 워크플로우 관리
├── rag_rewriter.py       # 쿼리 재작성 로직 (Ollama 사용)
├── rag_search.py         # 벡터 데이터베이스 (FAISS) 기반 검색 엔진
├── rag_utils.py          # JSON 파일 로드/저장 등 공통 유틸리티 함수
└── telepathy_rag.py      # CLI 모드 메인 실행 스크립트, 모듈 로딩 담당
```
rag_config.py:
역할: 프로젝트 전반에 걸친 설정(데이터 경로, 모델 이름, 임베딩 차원, 검색 임계값, Ollama URL 등)을 관리하는 클래스입니다. 시스템의 유연성을 위해 모든 중요한 파라미터가 여기에 정의됩니다.


## 프로젝트 구조

```rag_system/
├── api/ # API 엔드포인트 정의
│   ├── init.py
│   ├── endpoints/ # API 엔드포인트
│   └── schemas.py # Pydantic 모델 (요청/응답 데이터 정의)
├── core/ # 핵심 기능
│   ├── init.py
│   ├── engine.py # RAGSearchEngine 클래스 (검색 및 생성 로직 통합)
│   ├── models.py # 데이터 모델 (내부 데이터 구조 정의)
│   └── utils.py # 유틸리티 함수 (공통적으로 사용되는 함수)
├── models/ # 모델 관련 코드
│   ├── init.py
│   ├── embedding.py # 임베딩 모델 래퍼 (텍스트를 벡터로 변환)
│   └── generator.py # 텍스트 생성 모델 (질문에 답변 생성)
├── services/ # 비즈니스 로직
│   ├── init.py
│   ├── search.py # 검색 서비스 (외부/내부 검색 엔진 연동)
│   └── query_rewriter.py # 쿼리 최적화 서비스 (질문 개선)
├── storage/ # 데이터 저장소
│   ├── init.py
│   ├── vector_store.py # 벡터 저장소 인터페이스 (추상화된 벡터 저장소)
│   └── faiss_store.py # FAISS 구현체 (실제 FAISS 기반 벡터 저장소)
├── tests/ # 단위/통합 테스트
│   ├── init.py
│   ├── test_engine.py
│   └── test_api.py
├── config.py # 설정 관리 (환경 변수, 상수 등)
├── main.py # 애플리케이션 진입점 (FastAPI 앱 실행)
└── requirements.txt # 의존성 목록 (프로젝트에 필요한 라이브러리)```

### 주요 컴포넌트 설명

1. **RAGSearchEngine** (`core/engine.py`)
   - 멀티모달 임베딩 생성 및 벡터 검색을 담당
   - Qwen2-VL 모델을 사용한 텍스트/이미지 임베딩
   - FAISS를 활용한 고성능 유사도 검색
   - GPU 가속을 통한 배치 처리 최적화

2. **OllamaGenerator** (`models/generator.py`)
   - Ollama API를 통한 텍스트 생성
   - 다중 엔드포인트 지원 및 자동 장애 조치
   - 스트리밍 응답 처리

3. **QueryRewriter** (`services/query_rewriter.py`)
   - 사용자 쿼리를 검색에 최적화된 형태로 변환
   - Mistral 7B 모델 기반의 자연어 이해
   - 도메인 특화 프롬프트 엔지니어링

4. **VectorStore** (`storage/vector_store.py`)
   - 벡터 검색을 위한 추상 인터페이스
   - FAISS, Milvus 등 다양한 백엔드 지원
   - 자동 인덱싱 및 압축

## 성능 최적화

### 벡터 검색 최적화
- **계층적 탐색**: HNSW(Hierarchical Navigable Small World) 인덱스 사용
- **양자화**: 8/4비트 양자화로 메모리 사용량 감소
- **배치 처리**: GPU 활용 효율성을 위한 병렬 처리

### 쿼리 처리 최적화
- **캐싱**: 자주 사용되는 쿼리 결과 캐싱
- **지연 로딩**: 필요 시점에만 모델 로드
- **프리페치**: 예상되는 다음 쿼리 사전 처리

### 리소스 관리
- **메모리 풀링**: 자주 할당/해제되는 객체 재사용
- **연결 풀링**: 데이터베이스/API 연결 관리
- **그레이스풀 셧다운**: 리소스 정리 후 안전한 종료

## 📈 확장성

### 수평 확장
- **분산 벡터 검색**: 여러 노드에 인덱스 샤딩
- **로드 밸런싱**: 트래픽 분산을 위한 로드 밸런서 구성
- **서비스 메시**: Istio를 활용한 서비스 간 통신 관리

### 플러그인 아키텍처
- **모델 플러그인**: 새로운 임베딩/생성 모델 손쉽게 추가
- **저장소 플러그인**: 다양한 벡터 데이터베이스 지원
- **프로세서 플러그인**: 사용자 정의 전처리/후처리 파이프라인

## 🔮 향후 계획

### 예정된 기능
- [ ] 멀티모달 임베딩 최적화
- [ ] 자동 확장 기능
- [ ] 대화형 튜토리얼
- [ ] 모니터링 대시보드

### 연구 주제
- 효율적인 멀티모델 검색 알고리즘
- 지식 증류를 통한 모델 경량화
- 자가 학습을 통한 성능 개선

## 기여자 가이드라인

### 코드 리뷰 프로세스
1. PR 생성 시 CI/CD 파이프라인 자동 실행
2. 최소 1명의 리뷰어 승인 필요
3. 코드 커버리지 80% 이상 유지
4. 주요 변경사항에 대한 문서 업데이트

### 버전 관리 전략
- **메이저**: 하위 호환성 없는 변경
- **마이너**: 하위 호환성 있는 기능 추가
- **패치**: 버그 수정 및 보안 패치

### 문서화 표준
- 모듈/클래스/함수별 상세 문서화
- 예제 코드 포함
- API 참조 자동 생성

## 추가 자료

### 관련 논문
- 

### 유용한 링크
- [Hugging Face Models](https://huggingface.co/models)
- [FAISS Documentation](https://faiss.ai/)
- [Ollama GitHub](https://github.com/ollama/ollama)

## 변경 이력

### [1.0.0] - 2025-05-30
#### 추가됨
- 초기 버전 출시
- 멀티모델 검색 기능
- REST API 엔드포인트
- CLI 도구

## 감사의 말씀

이 프로젝트는 다음과 같은 오픈소스 프로젝트에 기반하고 있습니다:
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.ai/)
- [FastAPI](https://fastapi.tiangolo.com/)

## 문제 해결
image_incoder.py:
역할: 이미지 파일을 Base64로 인코딩하고, 이미지 파일명(예: 119_1_Picture_1.00.png)에서 구조화된 메타데이터(제목, 인덱스, 페이지 번호)를 추출하는 유틸리티 함수들을 제공합니다. 이미지 검색 및 활용에 필수적인 모듈입니다. 이미지 크기 조정 및 최적화 기능도 포함합니다.
rag_formatter.py:
### 유틸리티 모듈

- **rag_formatter.py**: 검색 결과를 사용자 친화적인 형식으로 포맷팅
- **rag_utils.py**: JSON 파일 로드/저장과 같은 범용 유틸리티 함수 제공
- **rag_logger.py**: 로깅 설정 및 관리

## 프로젝트 구조 상세

### 핵심 모듈 설명

#### rag_fastapi.py
- **역할**: FastAPI 애플리케이션의 Request 및 Response 모델 정의
- **주요 기능**:
  - Pydantic BaseModel을 사용한 데이터 유효성 검사
  - CORS(Cross-Origin Resource Sharing) 미들웨어 설정
  - API 엔드포인트의 데이터 구조 정의

#### rag_initializer.py
- **역할**: FastAPI 애플리케이션의 메인 진입점
- **주요 기능**:
  - FastAPI 앱 초기화
  - 핵심 컴포넌트 인스턴스화
  - `/search` API 엔드포인트 정의
  - 대화 기록 관리

#### rag_integration.py
- **역할**: RAG 워크플로우 통합 관리
- **주요 기능**:
  - RAGSearchEngine과 OllamaGenerator 통합
  - 쿼리 재작성, 검색, 답변 생성 프로세스 조율
  - 상세한 로깅 기능 제공
  - CLI 및 FastAPI 인터페이스 지원

#### rag_clihandler.py:
- **역할**: argparse를 사용하여 명령줄 인자를 파싱하고, SearchIntegration을 호출하여 CLI 모드의 검색, 대화형 모드, 메타데이터 목록 출력 등의 기능을 구현합니다. 사용자가 명령줄에서 시스템과 상호작용할 수 있는 인터페이스를 제공합니다.

#### telepathy_rag.py:
- **역할**: 이 프로젝트의 메인 실행 스크립트 중 하나로, 특히 CLI 모드를 위한 진입점입니다. 파일명에 숫자가 붙은 다른 모듈들을 importlib.machinery.SourceFileLoader를 사용하여 동적으로 로드하고, CLIHandler를 초기화하여 명령줄 인자에 따라 적절한 RAG 기능을 실행합니다. 프로젝트의 여러 부분을 결합하여 단일 실행 가능한 엔트리포인트를 제공합니다.

6. 성능 및 메모리 고려 사항
GPU 권장: Alibaba-NLP/gme-Qwen2-VL-7B-Instruct 모델 및 FAISS는 대규모 데이터를 처리하므로, GPU 환경에서 최적의 성능을 발휘합니다. 최소 16GB 이상의 GPU 메모리(VRAM)를 권장합니다.
Ollama 리소스: Ollama 모델(llama4, mistral) 또한 상당한 시스템 리소스를 요구합니다. 시스템 RAM 및 CPU/GPU 리소스가 충분한지 확인해주세요.

7. 기여 (Contributing)
프로젝트에 기여하고 싶으시다면, Pull Request를 환영합니다.

이슈 트래커를 통해 버그를 보고하거나 기능을 제안해주세요.
Fork 후 새로운 브랜치에서 작업을 진행해주세요.
코드 스타일을 준수하고, 필요한 경우 테스트 코드를 작성해주세요.

8. 라이선스
(라이선스 정보)
