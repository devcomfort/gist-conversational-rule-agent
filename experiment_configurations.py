#!/usr/bin/env python3
"""
실험 설정 통합 관리
==================

모든 토크나이저, 청커, 임베딩 모델 설정을 한 곳에서 관리합니다.
실험 실행기는 이 설정들을 순회하며 모든 조합을 실험합니다.

구조:
- TOKENIZERS: 토크나이저 설정 및 파라미터
- CHUNKERS: 청커 클래스와 파라미터 조합
- EMBEDDING_MODELS: 임베딩 모델과 설정
"""

from chonkie import (
    TokenChunker,
    SentenceChunker,
    LateChunker,
    NeuralChunker,
    RecursiveChunker,
    SemanticChunker,
)

# ===================================================================
# 토크나이저 설정
# ===================================================================

TOKENIZERS = {
    "character": {
        "name": "character",
        "description": "문자 기반 토크나이저 (기본)",
        "implementation": "character",  # 실제 구현에서 사용할 키
    },
    "gpt2": {
        "name": "gpt2",
        "description": "GPT-2 BPE 토크나이저",
        "implementation": "gpt2",
        "requires_tiktoken": True,
    },
    "tiktoken": {
        "name": "cl100k_base",
        "description": "GPT-4 Tiktoken 토크나이저 (최신)",
        "implementation": "cl100k_base",
        "requires_tiktoken": True,
    },
}

# ===================================================================
# 임베딩 모델 설정 (embedding_models.yaml → 코드)
# ===================================================================

EMBEDDING_MODELS = {
    "qwen3_8b": {
        "model_name": "Qwen/Qwen3-Embedding-8B",
        "dimension": 1024,
        "mteb_rank": 2,
        "db_name": "faiss_qwen3_embedding_8b",
        "description": "Qwen3 Embedding 8B - MTEB 2위 (최고 성능)",
        "model_kwargs": {
            "device": "auto",  # 자동 CUDA/CPU 감지
            "trust_remote_code": True,
            "torch_dtype": "float16",
        },
        "encode_kwargs": {
            "normalize_embeddings": True,
            "batch_size": 16,
        },
    },
    "qwen3_0_6b": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "dimension": 1024,
        "mteb_rank": 4,
        "db_name": "faiss_qwen3_embedding_0_6b",
        "description": "Qwen3 Embedding 0.6B - MTEB 4위 (효율적 고성능)",
        "model_kwargs": {
            "device": "auto",
            "trust_remote_code": True,
            "torch_dtype": "float16",
        },
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": 32},
    },
    "jina_v3": {
        "model_name": "jinaai/jina-embeddings-v3",
        "dimension": 1024,
        "mteb_rank": 22,
        "db_name": "faiss_jina_embeddings_v3",
        "description": "Jina Embeddings v3 - MTEB 22위 (균형잡힌 성능)",
        "model_kwargs": {"device": "auto", "trust_remote_code": True},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": 32},
    },
    "bge_m3": {
        "model_name": "BAAI/bge-m3",
        "dimension": 1024,
        "mteb_rank": 23,
        "db_name": "faiss_bge_m3",
        "description": "BGE-M3 - MTEB 23위 (다국어 지원)",
        "model_kwargs": {"device": "auto"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": 32},
    },
    "all_minilm_l6": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "mteb_rank": 117,
        "db_name": "faiss_all_minilm_l6_v2",
        "description": "All-MiniLM-L6-v2 - MTEB 117위 (경량 베이스라인)",
        "model_kwargs": {"device": "auto"},
        "encode_kwargs": {
            "normalize_embeddings": True,
            "batch_size": 64,  # 경량 모델 큰 배치
        },
    },
    "multilingual_e5": {
        "model_name": "intfloat/multilingual-e5-small",
        "dimension": 384,
        "mteb_rank": 45,
        "db_name": "faiss_multilingual_e5_small",
        "description": "Multilingual E5 Small - MTEB 45위 (다국어 특화)",
        "model_kwargs": {"device": "auto"},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": 64},
    },
}

# ===================================================================
# 청커 설정 (chunker_params.yaml → 코드)
# ===================================================================

CHUNKERS = {
    "token": {
        "class": TokenChunker,
        "description": "토큰 기반 청커 - 고정 토큰 수로 분할 (법적문서/기술문서 적합)",
        "parameters": {
            "base": {
                "chunk_overlap": 0,  # 기본 겹침 없음
            },
            "variations": {
                "tokenizer": ["character", "gpt2", "cl100k_base"],
                "chunk_size": [128, 256, 512, 1024, 2048],  # 단문에서 장문까지
                "chunk_overlap": [0, 64, 128, 256],  # 2의 배수, 문맥 보존용
            },
        },
    },
    "sentence": {
        "class": SentenceChunker,
        "description": "문장 기반 청커 - 자연스러운 문장 단위 분할 (일반 문서 적합)",
        "parameters": {
            "base": {
                "chunk_overlap": 0,
                "approximate": False,
                "delim": [". ", "! ", "? ", "\n"],
                "include_delim": "prev",
            },
            "variations": {
                "tokenizer_or_token_counter": ["character", "gpt2", "cl100k_base"],
                "chunk_size": [256, 512, 1024, 2048],  # 다양한 문서 길이 대응
                "min_sentences_per_chunk": [1, 2, 4, 8],  # 2의 배수, 단문~장문
                "min_characters_per_sentence": [32, 64, 128],  # 2의 배수, 실용적 범위
            },
        },
    },
    "late": {
        "class": LateChunker,
        "description": "지연 청커 - 임베딩 기반 의미적 분할 (의미 단위 보존)",
        "parameters": {
            "base": {},
            "variations": {
                "embedding_model": "ALL_EMBEDDINGS",
                "chunk_size": [256, 512, 1024, 2048, 4096],  # 광범위 실험
                "min_characters_per_chunk": [64, 128, 256, 512],  # 2의 배수, 실용성
            },
        },
    },
    "neural": {
        "class": NeuralChunker,
        "description": "신경망 청커 - 딥러닝 모델 기반 지능적 분할",
        "parameters": {
            "base": {
                "device_map": "auto",  # GPU/CPU 자동 선택
            },
            "variations": {
                "model": [
                    "mirth/chonky_distilbert_base_uncased_1",
                    "bert-base-uncased",
                    "distilbert-base-uncased",
                    "bert-base-multilingual-cased",
                    "roberta-base",
                ],
                "min_characters_per_chunk": [64, 128, 256, 512],  # 2의 배수
                "stride": [None, 0.1, 0.2, 0.5],  # 겹침 비율 다양화
            },
        },
    },
    "recursive": {
        "class": RecursiveChunker,
        "description": "재귀 청커 - 계층적 분할로 구조 보존 (기술문서 적합)",
        "parameters": {
            "base": {},
            "variations": {
                "tokenizer_or_token_counter": ["character", "gpt2", "cl100k_base"],
                "chunk_size": [256, 512, 1024, 2048],
                "min_characters_per_chunk": [64, 128, 256],  # 2의 배수
            },
        },
    },
    "semantic": {
        "class": SemanticChunker,
        "description": "의미적 청커 - 의미 유사성 기반 동적 분할 (주제별 분할)",
        "parameters": {
            "base": {
                "delim": [". ", "! ", "? ", "\n"],
                "include_delim": "prev",
                "skip_window": 0,  # 고급 파라미터 복원
                "filter_window": 5,
                "filter_polyorder": 3,
                "filter_tolerance": 0.2,
            },
            "variations": {
                "embedding_model": "ALL_EMBEDDINGS",
                "threshold": [0.6, 0.7, 0.8, 0.9],  # 0.5 제거 (너무 관대함)
                "similarity_window": [2, 4, 8],  # 2의 배수, 문맥 창 크기
                "chunk_size": [256, 512, 1024, 2048, 4096],  # 광범위 실험
                "min_sentences_per_chunk": [1, 2, 4],  # 2의 배수로 확장
                "min_characters_per_sentence": [32, 64, 128, 256],  # 2의 배수
            },
        },
    },
}

# ===================================================================
# 기본 실험 설정
# ===================================================================

EXPERIMENT_CONFIG = {
    "data": {
        "rules_directory": "rules",
        "supported_extensions": [".pdf", ".hwp", ".docx", ".pptx"],
        "max_file_size_mb": 50,
        "exclude_patterns": ["**/.*", "**/__pycache__/**"],
    },
    "output": {
        "base_dir": "experiments/outputs",
        "tokenization_subdir": "tokenization_results",
        "chunking_subdir": "chunking_results",
        "embedding_subdir": "",  # 루트에 임베딩별 폴더
    },
    "execution": {
        "enable_tokenization_phase": True,  # 토큰화 단계 활성화 여부
        "save_intermediate_results": True,  # 중간 결과 저장 여부
        "cleanup_memory_after_each": True,  # 각 실험 후 메모리 정리
    },
    "storage": {
        # 기본 저장 설정만
    },
    "filters": {
        # 실험 범위 제한 (개발/테스트용)
        "tokenizers": None,  # None = 전체, 또는 ["character", "gpt2"]
        "chunkers": None,  # None = 전체, 또는 ["token", "sentence"]
        "embedding_models": None,  # None = 전체, 또는 ["all_minilm_l6"]
    },
    "performance": {
        # 제한 없음 - 시스템 최대 성능 활용
    },
}

# ===================================================================
# 조합 계산 함수들
# ===================================================================


def count_tokenizer_combinations() -> int:
    """토크나이저 조합 수 계산"""
    return len(TOKENIZERS)


def count_chunker_combinations() -> int:
    """청커 파라미터 조합 수 계산"""
    total = 0

    for chunker_name, config in CHUNKERS.items():
        variations = config["parameters"]["variations"]

        # 각 파라미터별 조합 수 계산
        combo_count = 1
        for param_name, param_values in variations.items():
            if param_name == "embedding_model" and param_values == "ALL_EMBEDDINGS":
                # 임베딩 모델 수만큼 곱함
                combo_count *= len(EMBEDDING_MODELS)
            elif isinstance(param_values, list):
                combo_count *= len(param_values)

        total += combo_count

    return total


def count_embedding_combinations() -> int:
    """임베딩 모델 수"""
    return len(EMBEDDING_MODELS)


def calculate_total_experiments() -> dict:
    """전체 실험 조합 수 계산"""
    tokenizer_count = count_tokenizer_combinations()
    chunker_count = count_chunker_combinations()
    embedding_count = count_embedding_combinations()

    # 실제로는 청킹 결과 × 임베딩 모델 조합
    # (토크나이저는 선택적 단계)
    total_experiments = chunker_count * embedding_count

    return {
        "tokenizers": tokenizer_count,
        "chunker_combinations": chunker_count,
        "embedding_models": embedding_count,
        "total_experiments": total_experiments,
    }


# ===================================================================
# 설정 검증 함수들
# ===================================================================


def validate_configurations() -> dict:
    """설정 검증 및 요약"""
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "summary": {},
    }

    # 토크나이저 검증
    tokenizer_names = list(TOKENIZERS.keys())
    validation_results["summary"]["tokenizers"] = tokenizer_names

    # 청커 검증
    chunker_names = list(CHUNKERS.keys())
    validation_results["summary"]["chunkers"] = chunker_names

    # 임베딩 모델 검증
    embedding_names = list(EMBEDDING_MODELS.keys())
    validation_results["summary"]["embedding_models"] = embedding_names

    # 조합 수 계산
    experiment_stats = calculate_total_experiments()
    validation_results["summary"]["experiment_statistics"] = experiment_stats

    # 대형 모델 경고 (배치 크기 기반)
    large_models = [
        name
        for name, config in EMBEDDING_MODELS.items()
        if config.get("encode_kwargs", {}).get("batch_size", 32) <= 16
    ]

    if large_models:
        validation_results["warnings"].append(
            f"대형 모델 감지: {large_models} (작은 배치 크기로 GPU 메모리 절약 중)"
        )

    # 실험 수 경고
    if experiment_stats["total_experiments"] > 5000:
        validation_results["warnings"].append(
            f"대규모 실험: {experiment_stats['total_experiments']}개 실험 예상 (충분한 컴퓨팅 리소스 필요)"
        )

    return validation_results


# ===================================================================
# 설정 표시 함수
# ===================================================================


def print_configuration_summary():
    """설정 요약 출력"""
    print("=" * 80)
    print("🔧 실험 설정 요약")
    print("=" * 80)

    # 토크나이저
    print(f"\n🔤 토크나이저 ({len(TOKENIZERS)}개):")
    for name, config in TOKENIZERS.items():
        print(f"  • {name}: {config['description']}")

    # 임베딩 모델
    print(f"\n🤖 임베딩 모델 ({len(EMBEDDING_MODELS)}개):")
    for name, config in EMBEDDING_MODELS.items():
        print(
            f"  • {name}: {config['description']} (dim={config['dimension']}, rank={config['mteb_rank']}위)"
        )

    # 청커
    print(f"\n🧩 청커 ({len(CHUNKERS)}개):")
    for name, config in CHUNKERS.items():
        variations = config["parameters"]["variations"]
        param_count = 1
        for param_values in variations.values():
            if isinstance(param_values, list):
                param_count *= len(param_values)
            elif param_values == "ALL_EMBEDDINGS":
                param_count *= len(EMBEDDING_MODELS)
        print(f"  • {name}: {config['description']} (~{param_count}개 조합)")

    # 실험 통계
    stats = calculate_total_experiments()
    print(f"\n📊 실험 통계:")
    print(f"  • 청킹 조합: {stats['chunker_combinations']:,}개")
    print(f"  • 임베딩 모델: {stats['embedding_models']}개")
    print(f"  • 총 실험: {stats['total_experiments']:,}개")

    # 검증 결과
    validation = validate_configurations()
    if validation["warnings"]:
        print(f"\n⚠️ 주의사항:")
        for warning in validation["warnings"]:
            print(f"  • {warning}")

    print("=" * 80)


# ===================================================================
# 학술 연구용 추가 함수들
# ===================================================================


def generate_experiment_metadata():
    """실험 메타데이터 생성"""
    import datetime
    import platform

    try:
        import torch

        torch_available = True
    except ImportError:
        torch_available = False

    metadata = {
        "experiment_info": {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_experiments": calculate_total_experiments()["total_experiments"],
            "mode": "unrestricted",
        },
        "system_info": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__ if torch_available else "N/A",
            "cuda_available": torch.cuda.is_available() if torch_available else False,
        },
    }

    return metadata


def estimate_storage_requirements():
    """스토리지 요구량 계산"""
    stats = calculate_total_experiments()

    # 기본 추정치 (제약 없음)
    base_size_mb = 17

    total_size_mb = base_size_mb * stats["total_experiments"]
    total_size_gb = total_size_mb / 1024

    return {
        "total_experiments": stats["total_experiments"],
        "size_per_experiment_mb": base_size_mb,
        "total_size_gb": round(total_size_gb, 2),
    }


def print_experiment_summary():
    """실험 설정 요약"""
    print("\n" + "=" * 80)
    print("🔧 실험 설정 요약 (제약 없음)")
    print("=" * 80)

    # 기본 구성 요약
    print_configuration_summary()

    print(f"\n✅ 실험 원칙:")
    print(f"  • 모든 실험이 제약 없이 수행됩니다")
    print(f"  • 시스템 최대 성능을 활용합니다")
    print(f"  • 모든 데이터가 원본 그대로 보존됩니다")


if __name__ == "__main__":
    # 실험 설정 요약 출력
    print_experiment_summary()

    # 검증 실행
    validation = validate_configurations()

    if validation["valid"]:
        print("✅ 모든 설정이 유효합니다!")
    else:
        print("❌ 설정 오류:")
        for error in validation["errors"]:
            print(f"  • {error}")
