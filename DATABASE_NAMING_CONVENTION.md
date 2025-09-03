# 데이터베이스 파일 네이밍 컨벤션

## 🎯 설계 원칙

1. **완전한 추적성**: 파일명만으로 모든 실험 조건 식별 가능
2. **계층적 구조**: 디렉토리 구조로 주요 분류, 파일명으로 세부 사항
3. **가독성과 간결성**: 길지만 의미있는 구조
4. **자동화 친화적**: 프로그래밍으로 파싱 및 분석 용이

## 📁 디렉토리 구조

```
experiments/outputs/
├── chunking_results/                    # 청킹 결과 저장소
│   ├── token_gpt2-512-100_a1b2c3d4/
│   ├── semantic_qwen3-0.7-1024-3_e5f6g7h8/
│   └── ...
├── embeddings/                         # 임베딩별 분류
│   ├── qwen3_8b/                       # 임베딩 모델명
│   │   ├── token_gpt2-512-100_a1b2c3d4/
│   │   ├── semantic_minilm-0.7-1024-3_e5f6g7h8/
│   │   └── ...
│   ├── jina_v3/
│   ├── bge_m3/
│   └── ...
└── analysis/                          # 분석 결과
    ├── parameter_impact_analysis.json
    ├── optimal_combinations.csv
    └── benchmark_results/
```

## 🏗️ 네이밍 패턴

### 기본 구조
```
{CHUNKER_TYPE}_{KEY_PARAMS}_{PARAM_HASH}
```

### 청킹 결과 파일명
```
{chunker_type}_{key_parameters}_{8char_hash}/
├── chunks.pkl                    # 청킹된 문서들
├── metadata.json                 # 전체 파라미터 메타데이터
└── stats.json                    # 청킹 통계
```

### 임베딩 데이터베이스 파일명
```
{embedding_model}_{chunker_type}_{key_parameters}_{8char_hash}/
├── index.faiss                   # FAISS 인덱스
├── index.pkl                     # FAISS 메타데이터  
├── embedding_metadata.json       # 임베딩 설정
└── experiment_info.json          # 실험 전체 정보
```

## 📝 청커별 핵심 파라미터 정의

### 1. Token Chunker
**파라미터**: `tokenizer-chunksize-overlap`
```bash
# 예시들
token_gpt2-512-100_a1b2c3d4/          # GPT2, 512 토큰, 100 중첩
token_tiktoken-1024-0_e5f6g7h8/        # Tiktoken, 1024 토큰, 중첩 없음
token_char-256-50_d4c3b2a1/            # Character, 256 토큰, 50 중첩
```

### 2. Sentence Chunker  
**파라미터**: `tokenizer-chunksize-minSent-approx`
```bash
# 예시들
sentence_gpt2-1024-2-F_b2c3d4e5/       # GPT2, 1024, 최소2문장, 정확처리
sentence_char-512-1-T_f6g7h8i9/        # Char, 512, 최소1문장, 근사처리
sentence_tiktoken-2048-3-F_a9b8c7d6/   # Tiktoken, 2048, 최소3문장, 정확처리
```

### 3. Late Chunker
**파라미터**: `embeddingmodel-chunksize-minchar`
```bash  
# 예시들
late_minilm-1024-50_c3d4e5f6/          # MiniLM, 1024 크기, 최소50자
late_bge-2048-100_g7h8i9j0/            # BGE, 2048 크기, 최소100자
late_e5-512-24_k1l2m3n4/               # E5, 512 크기, 최소24자
```

### 4. Neural Chunker
**파라미터**: `model-minchar-stride`
```bash
# 예시들  
neural_bert-50-null_d4e5f6g7/          # BERT, 최소50자, 스트라이드없음
neural_distilbert-24-64_h8i9j0k1/      # DistilBERT, 최소24자, 64스트라이드
neural_deberta-100-128_l2m3n4o5/       # DeBERTa, 최소100자, 128스트라이드
```

### 5. Recursive Chunker
**파라미터**: `tokenizer-chunksize-minchar`
```bash
# 예시들
recursive_char-1024-50_e5f6g7h8/       # Character, 1024, 최소50자
recursive_gpt2-2048-100_i9j0k1l2/      # GPT2, 2048, 최소100자  
recursive_tiktoken-512-24_m3n4o5p6/    # Tiktoken, 512, 최소24자
```

### 6. Semantic Chunker
**파라미터**: `embeddingmodel-threshold-window-chunksize`
```bash
# 예시들
semantic_minilm-0.7-3-1024_f6g7h8i9/  # MiniLM, 0.7임계값, 3윈도우, 1024크기
semantic_bge-0.8-5-2048_j0k1l2m3/      # BGE, 0.8임계값, 5윈도우, 2048크기
semantic_e5-0.6-1-512_n4o5p6q7/        # E5, 0.6임계값, 1윈도우, 512크기
```

## 🔍 파라미터 해시 생성 규칙

### 해시 생성 로직
```python
import hashlib
import json

def generate_param_hash(chunker_params: dict) -> str:
    """파라미터 딕셔너리를 8자 해시로 변환"""
    # 1. _target_ 제거
    clean_params = {k: v for k, v in chunker_params.items() if k != "_target_"}
    
    # 2. 정렬된 JSON으로 직렬화  
    param_string = json.dumps(clean_params, sort_keys=True, separators=(',', ':'))
    
    # 3. MD5 해시의 앞 8자리 사용
    return hashlib.md5(param_string.encode()).hexdigest()[:8]

# 예시
params = {
    "tokenizer": "gpt2", 
    "chunk_size": 512, 
    "chunk_overlap": 100
}
hash_value = generate_param_hash(params)  # "a1b2c3d4"
```

### 약어 매핑 테이블
```python
ABBREVIATION_MAP = {
    # 토크나이저
    "character": "char",
    "cl100k_base": "tiktoken", 
    "gpt2": "gpt2",
    "p50k_base": "p50k",
    "r50k_base": "r50k",
    
    # 임베딩 모델 (Late/Semantic용)
    "sentence-transformers/all-MiniLM-L6-v2": "minilm",
    "BAAI/bge-small-en-v1.5": "bge",
    "intfloat/multilingual-e5-small": "e5",
    "Qwen/Qwen3-Embedding-0.6B": "qwen3",
    "jinaai/jina-embeddings-v3": "jina",
    
    # 신경망 모델 (Neural용)  
    "bert-base-uncased": "bert",
    "distilbert-base-uncased": "distilbert",
    "microsoft/deberta-v3-base": "deberta",
    "roberta-base": "roberta",
    
    # Boolean 값
    True: "T",
    False: "F",
    None: "null"
}
```

## 📊 실제 파일명 예시

### 전체 실험 결과 구조
```
experiments/outputs/
├── chunking_results/
│   ├── token_gpt2-512-100_a1b2c3d4/
│   │   ├── chunks.pkl
│   │   ├── metadata.json
│   │   └── stats.json
│   ├── semantic_minilm-0.7-3-1024_f6g7h8i9/
│   │   ├── chunks.pkl  
│   │   ├── metadata.json
│   │   └── stats.json
│   └── ...
│
├── qwen3_8b/                              # 임베딩 모델별 분류
│   ├── token_gpt2-512-100_a1b2c3d4/       # 청킹 결과와 매칭
│   │   ├── index.faiss
│   │   ├── index.pkl
│   │   └── embedding_metadata.json
│   ├── semantic_minilm-0.7-3-1024_f6g7h8i9/
│   │   ├── index.faiss
│   │   ├── index.pkl  
│   │   └── embedding_metadata.json
│   └── ...
│
├── jina_v3/
│   ├── token_gpt2-512-100_a1b2c3d4/
│   ├── sentence_char-1024-2-F_b2c3d4e5/
│   └── ...
│
└── analysis/
    ├── experiment_results.csv             # 전체 결과 매트릭스
    ├── naming_convention_map.json          # 파일명↔실험조건 매핑  
    └── performance_by_naming_pattern.json  # 네이밍 패턴별 성능 분석
```

## 🛠️ 구현 클래스

```python
class DatabaseNamingConvention:
    """데이터베이스 네이밍 컨벤션 관리자"""
    
    def __init__(self):
        self.abbreviations = ABBREVIATION_MAP
    
    def generate_chunker_identifier(self, chunker_type: str, 
                                   params: Dict[str, Any]) -> str:
        """청커 타입과 파라미터로 식별자 생성"""
        
        if chunker_type == "token":
            return self._token_identifier(params)
        elif chunker_type == "sentence":
            return self._sentence_identifier(params)
        elif chunker_type == "late":
            return self._late_identifier(params)
        elif chunker_type == "neural":
            return self._neural_identifier(params)
        elif chunker_type == "recursive":
            return self._recursive_identifier(params)
        elif chunker_type == "semantic":
            return self._semantic_identifier(params)
            
    def _token_identifier(self, params: Dict[str, Any]) -> str:
        tokenizer = self._abbreviate(params.get("tokenizer", "char"))
        chunk_size = params.get("chunk_size", 512)
        overlap = params.get("chunk_overlap", 0)
        param_hash = self._generate_hash(params)
        
        return f"token_{tokenizer}-{chunk_size}-{overlap}_{param_hash}"
    
    def _semantic_identifier(self, params: Dict[str, Any]) -> str:
        model = self._abbreviate(params.get("embedding_model", "minilm"))
        threshold = params.get("threshold", 0.7)
        window = params.get("similarity_window", 3)
        chunk_size = params.get("chunk_size", 1024)
        param_hash = self._generate_hash(params)
        
        return f"semantic_{model}-{threshold}-{window}-{chunk_size}_{param_hash}"
    
    def generate_database_path(self, embedding_model: str, 
                              chunker_identifier: str) -> Path:
        """최종 데이터베이스 경로 생성"""
        embedding_abbrev = self._abbreviate(embedding_model)
        return Path(f"experiments/outputs/{embedding_abbrev}/{chunker_identifier}")
    
    def parse_identifier(self, identifier: str) -> Dict[str, Any]:
        """식별자를 파싱하여 파라미터 복원"""
        # 역방향 파싱 로직 구현
        pass
```

## 📈 네이밍 컨벤션 분석 도구

### 패턴 분석
```python
def analyze_naming_patterns(database_paths: List[Path]) -> Dict[str, Any]:
    """네이밍 패턴별 성능 분석"""
    
    pattern_analysis = {
        "by_chunker_type": defaultdict(list),
        "by_embedding_model": defaultdict(list),
        "by_key_parameters": defaultdict(list),
        "hash_collisions": [],
        "naming_efficiency": {}
    }
    
    for path in database_paths:
        # 파일명 파싱 후 분석
        pass
    
    return pattern_analysis
```

## 🎯 최종 권장사항

### 1. 2단계 계층 구조
- **Level 1**: 임베딩 모델별 분류 (`qwen3_8b/`, `jina_v3/`)
- **Level 2**: 청커 식별자 (`token_gpt2-512-100_a1b2c3d4/`)

### 2. 메타데이터 풍부화
- `experiment_info.json`: 전체 실험 조건
- `naming_convention_map.json`: 파일명↔파라미터 매핑

### 3. 자동 검증 시스템
- 중복 해시 체크
- 파일명 유효성 검증  
- 역방향 파싱 테스트

이 네이밍 컨벤션으로 **5,288개 조합**을 체계적으로 관리하고, 파일명만으로도 실험 조건을 완벽히 추적할 수 있습니다.
