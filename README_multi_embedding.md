# Multi-Embedding GIST Rules Analyzer

GIST 학칙 및 규정에 대한 다중 임베딩 모델 지원 RAG 시스템입니다.

## 🎯 주요 특징

### 📊 지원하는 임베딩 모델 (MTEB 벤치마크 기준)

| 순위 | 모델명 | 설명 | 차원 | DB 이름 |
|------|--------|------|------|---------|
| 3위 | Qwen/Qwen3-Embedding-0.6B | Qwen3 Embedding 0.6B | 1024 | faiss_qwen3_embedding_0.6b |
| 22위 | jinaai/jina-embeddings-v3 | Jina Embeddings v3 | 1024 | faiss_jina_embeddings_v3 |
| 23위 | BAAI/bge-m3 | BGE M3 | 1024 | faiss_bge_m3 |
| 117위 | sentence-transformers/all-MiniLM-L6-v2 | All MiniLM L6 v2 (기본) | 384 | faiss_all_minilm_l6_v2 |

### 🚀 핵심 기능

- **🤖 멀티 임베딩 모델**: 4개 임베딩 모델 중 선택 가능
- **📊 MTEB 순위 기반**: 성능이 검증된 모델들 사용
- **🔄 동적 전환**: 웹 UI에서 임베딩 모델 실시간 변경
- **🎯 Knee Detection**: 적응형 문서 개수 자동 결정
- **⚡ LiteLLM 통합**: 15+ LLM 프로바이더 지원
- **📈 시각화**: Knee point detection 분석 그래프

## 🛠️ 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

원하는 LLM 프로바이더의 API 키를 설정하세요:

```bash
# 예시: Fireworks AI (기본)
export FIREWORKS_AI_API_KEY="your-fireworks-key"

# 또는 OpenAI
export OPENAI_API_KEY="your-openai-key"

# 또는 Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 3. PDF 문서 준비

`rules/` 디렉토리에 GIST 학칙 및 규정 PDF 파일들을 배치하세요:

```
rules/
├── 학사규정/
│   ├── 학사규정.pdf
│   └── ...
├── 대학원규정/
│   ├── 대학원규정.pdf
│   └── ...
└── ...
```

## 📦 데이터베이스 구축

### 모든 임베딩 모델로 데이터베이스 구축

```bash
python build_multi_embedding_databases.py
```

### 특정 임베딩 모델만 구축

```bash
# 최고 성능 모델 (MTEB 3위)
python build_multi_embedding_databases.py --model "Qwen/Qwen3-Embedding-0.6B"

# 중간 성능 모델 (MTEB 22위)
python build_multi_embedding_databases.py --model "jinaai/jina-embeddings-v3"

# 기본 모델 (MTEB 117위)
python build_multi_embedding_databases.py --model "sentence-transformers/all-MiniLM-L6-v2"
```

### 지원하는 모델 목록 확인

```bash
python build_multi_embedding_databases.py --list-models
```

## 🚀 앱 실행

```bash
python app_gist_rules_analyzer_prebuilt.py
```

웹 브라우저에서 `http://localhost:7860` 접속

## 📊 사용 방법

### 1. 임베딩 모델 선택

- 웹 UI에서 **"📊 임베딩 모델 선택"** 드롭다운 사용
- MTEB 순위가 높을수록 더 정확한 검색 결과
- 모델 변경 시 해당 데이터베이스가 자동으로 로드됨

### 2. 질의응답

- **질의문 입력** 박스에 GIST 규정 관련 질문 입력
- **"🚀 테스트 실행"** 버튼 클릭
- 4개 패널에서 동시에 답변 확인:
  - Dynamic Knee Detection
  - Dynamic + Cross-Encoder (Basic)
  - Dynamic + Cross-Encoder (Advanced)  
  - Dynamic + Multilingual Cross-Encoder

### 3. Knee Detection 시각화

- 질의 실행 후 **"📈 시각화 생성"** 버튼 클릭
- 문서 유사도 분포와 knee point 위치 확인
- 선택된 문서 개수와 선택 근거 분석

## 📈 성능 비교

| 임베딩 모델 | MTEB 순위 | 차원 | 장점 | 단점 |
|------------|----------|------|------|------|
| **Qwen3-0.6B** | **3위** | 1024 | 최고 정확도 | 느린 속도, 큰 용량 |
| **Jina v3** | **22위** | 1024 | 균형잡힌 성능 | 중간 속도 |
| **BGE M3** | **23위** | 1024 | 다국어 지원 | 중간 성능 |
| **MiniLM L6** | **117위** | 384 | 빠른 속도 | 낮은 정확도 |

### 권장 사용 시나리오

- **🎯 정확도 우선**: Qwen3-Embedding-0.6B
- **⚖️ 균형잡힌 사용**: jinaai/jina-embeddings-v3  
- **🌍 다국어 문서**: BAAI/bge-m3
- **⚡ 빠른 테스트**: sentence-transformers/all-MiniLM-L6-v2

## 🗂️ 생성되는 파일 구조

```
프로젝트 루트/
├── faiss_qwen3_embedding_0.6b/          # Qwen3 데이터베이스
│   ├── index.faiss
│   ├── index.pkl
│   ├── database_info.json
│   └── category_mapping.json
├── faiss_jina_embeddings_v3/             # Jina 데이터베이스
├── faiss_bge_m3/                         # BGE 데이터베이스
├── faiss_all_minilm_l6_v2/               # MiniLM 데이터베이스
└── faiss_db/                             # 레거시 데이터베이스 (호환성)
```

## 🔧 고급 설정

### 데이터베이스 크기 최적화

각 임베딩 모델별로 데이터베이스 크기가 다름:

- **384차원 (MiniLM)**: ~50MB
- **1024차원 (Qwen3, Jina, BGE)**: ~120MB

### 메모리 사용량

- **MiniLM**: ~2GB RAM
- **Qwen3/Jina/BGE**: ~4GB RAM

## 🐛 문제 해결

### 데이터베이스 로드 실패

```bash
❌ 데이터베이스를 찾을 수 없습니다
```

**해결책**: 해당 임베딩 모델로 데이터베이스를 구축하세요
```bash
python build_multi_embedding_databases.py --model "모델명"
```

### 메모리 부족 오류

**해결책**: 
1. 작은 차원 모델 사용 (MiniLM L6 v2)
2. 배치 처리 크기 조정
3. 시스템 메모리 증설

### 임베딩 모델 로딩 실패

**해결책**:
1. 인터넷 연결 확인
2. HuggingFace Hub 접근 확인  
3. 디스크 공간 확인

## 📊 벤치마크 결과

### 검색 정확도 (GIST 규정 데이터)

| 모델 | 정확도 | 속도 | 메모리 | 종합 점수 |
|------|--------|------|--------|-----------|
| Qwen3-0.6B | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ | ★★★★☆ |
| Jina v3 | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★★★☆ |
| BGE M3 | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★★★☆ |
| MiniLM L6 | ★★★☆☆ | ★★★★★ | ★★★★★ | ★★★★☆ |

## 🤝 기여 방법

1. 새로운 임베딩 모델 추가
2. 성능 벤치마크 결과 공유
3. 버그 리포트 및 개선 제안
4. 문서화 개선

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

---

**💡 팁**: 처음 사용할 때는 MiniLM L6 v2 모델로 시작해서 시스템을 익힌 후, 더 높은 성능이 필요하면 Qwen3 모델로 업그레이드하는 것을 권장합니다!
