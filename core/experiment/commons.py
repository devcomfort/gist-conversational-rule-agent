#!/usr/bin/env python3
"""
Experiment Commons
=================

실험 시스템의 공통 컴포넌트들을 제공합니다.
모든 실험 모듈(청킹, QA 생성, 임베딩)에서 공통으로 사용되는 클래스와 유틸리티들을 포함합니다.

공통 컴포넌트:
- ExperimentState: 실험 진행 상태 추적
- ExperimentResult: 실험 결과 데이터 구조
- ExperimentNaming: 일관된 네이밍 컨벤션
- DocumentManager: 문서 로딩 및 관리
- TokenizationManager: 토큰화 관리 (선택적)

사용법:
    from experiment_commons import DocumentManager, ExperimentNaming, ExperimentState
"""

import os
import json
import time
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from loguru import logger
from llama_index.core import Document

# 로컬 모듈
from experiment_configurations import EXPERIMENT_CONFIG
from core.loaders import load_document, collect_document_paths

# ===================================================================
# 데이터클래스
# ===================================================================


@dataclass
class ExperimentState:
    """실험 진행 상태"""

    total_documents: int = 0
    tokenization_results: int = 0
    chunking_results: int = 0
    embedding_results: int = 0
    successful_experiments: int = 0
    failed_experiments: int = 0
    start_time: float = 0
    current_phase: str = "초기화"


@dataclass
class ExperimentResult:
    """개별 실험 결과"""

    experiment_id: str
    tokenizer_name: Optional[str]
    chunker_name: str
    chunker_params: Dict[str, Any]
    embedding_name: str
    embedding_model: str
    total_chunks: int
    embedding_time: float
    database_size_mb: float
    database_path: str
    success: bool
    error_message: Optional[str] = None
    timestamp: str = ""


# ===================================================================
# 네이밍 컨벤션
# ===================================================================


class ExperimentNaming:
    """실험 결과 네이밍 컨벤션"""

    ABBREVIATIONS = {
        # 토크나이저
        "character": "char",
        "gpt2": "gpt2",
        "cl100k_base": "tiktoken",
        # 임베딩 모델 (키 사용)
        "qwen3_8b": "qwen8b",
        "qwen3_0_6b": "qwen06b",
        "jina_v3": "jina",
        "bge_m3": "bgem3",
        "all_minilm_l6": "minilm",
        "multilingual_e5": "e5",
        # Boolean
        True: "T",
        False: "F",
        None: "null",
    }

    @classmethod
    def abbreviate(cls, value: Any) -> str:
        """값을 약어로 변환"""
        if value in cls.ABBREVIATIONS:
            return cls.ABBREVIATIONS[value]
        elif isinstance(value, str) and len(value) > 12:
            return value.replace("-", "").replace("_", "")[:8]
        return str(value).replace("-", "_")

    @classmethod
    def generate_chunk_id(cls, chunker_name: str, params: Dict[str, Any]) -> str:
        """청킹 ID 생성"""
        # 파라미터 해시
        param_hash = cls._hash_params(params)

        # 청커별 특화 ID
        if chunker_name == "token":
            tokenizer = cls.abbreviate(params.get("tokenizer", "char"))
            size = params.get("chunk_size", 512)
            overlap = params.get("chunk_overlap", 0)
            return f"token_{tokenizer}_s{size}_o{overlap}_{param_hash}"

        elif chunker_name == "sentence":
            tokenizer = cls.abbreviate(params.get("tokenizer_or_token_counter", "char"))
            size = params.get("chunk_size", 1024)
            min_sent = params.get("min_sentences_per_chunk", 2)
            return f"sent_{tokenizer}_s{size}_m{min_sent}_{param_hash}"

        elif chunker_name in ["late", "semantic"]:
            model = cls.abbreviate(params.get("embedding_model", "unknown"))
            size = params.get("chunk_size", 1024)
            return f"{chunker_name}_{model}_s{size}_{param_hash}"

        elif chunker_name == "neural":
            model_name = params.get("model", "bert")
            model = cls.abbreviate(model_name.split("/")[-1])
            min_char = params.get("min_characters_per_chunk", 50)
            return f"neural_{model}_c{min_char}_{param_hash}"

        elif chunker_name == "recursive":
            tokenizer = cls.abbreviate(params.get("tokenizer_or_token_counter", "char"))
            size = params.get("chunk_size", 1024)
            return f"recur_{tokenizer}_s{size}_{param_hash}"

        else:
            return f"{chunker_name}_{param_hash}"

    @classmethod
    def generate_database_path(cls, embedding_name: str, chunk_id: str) -> str:
        """데이터베이스 경로 생성"""
        embedding_abbrev = cls.abbreviate(embedding_name)
        return f"{embedding_abbrev}/{chunk_id}"

    @classmethod
    def _hash_params(cls, params: Dict[str, Any]) -> str:
        """파라미터 8자 해시 생성"""
        clean_params = {
            k: v for k, v in params.items() if k not in ["_target_", "class"]
        }
        param_str = json.dumps(clean_params, sort_keys=True, separators=(",", ":"))
        return hashlib.md5(param_str.encode()).hexdigest()[:8]


# ===================================================================
# 문서 및 토큰화 관리자
# ===================================================================


class DocumentManager:
    """문서 로딩 관리"""

    def __init__(self):
        self.config = EXPERIMENT_CONFIG["data"]

    def load_all_documents(self) -> List[Document]:
        """모든 지원 문서 로드"""
        logger.info("📖 문서 로딩 시작...")

        # 문서 경로 수집
        document_paths = list(
            collect_document_paths(self.config["rules_directory"], lazy=False)
        )

        # 크기 및 패턴 필터링
        filtered_paths = []
        for path in document_paths:
            try:
                # 크기 체크
                size_mb = os.path.getsize(path) / (1024 * 1024)
                if size_mb > self.config["max_file_size_mb"]:
                    logger.debug(f"⚠️ 크기 초과: {path} ({size_mb:.1f}MB)")
                    continue

                # 패턴 체크 (간단한 구현)
                path_str = str(path)
                skip = False
                for pattern in self.config.get("exclude_patterns", []):
                    if pattern.replace("**/", "").replace("/**", "") in path_str:
                        skip = True
                        break

                if not skip:
                    filtered_paths.append(path)

            except OSError as e:
                logger.debug(f"⚠️ 접근 실패: {path} - {e}")
                continue

        logger.info(f"📄 필터링된 경로: {len(filtered_paths)}개")

        # 문서 로드
        all_documents = []
        for i, path in enumerate(filtered_paths):
            logger.info(f"📄 로딩 [{i + 1}/{len(filtered_paths)}]: {path.name}")

            try:
                docs = load_document(path)

                # 메타데이터 보강 (LlamaIndex Document 그대로 사용)
                for doc in docs:
                    doc.metadata.update(
                        {
                            "file_path": str(path),
                            "file_name": path.name,
                            "file_extension": path.suffix.lower(),
                            "load_timestamp": datetime.now().isoformat(),
                            "doc_id": f"{path.stem}_{hashlib.md5(str(path).encode()).hexdigest()[:8]}",
                        }
                    )

                all_documents.extend(docs)

            except Exception as e:
                logger.error(f"❌ 로드 실패: {path} - {e}")
                continue

        logger.info(f"✅ 문서 로딩 완료: {len(all_documents)}개")
        return all_documents


class TokenizationManager:
    """토큰화 관리 (선택적)"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.token_dir = (
            self.output_dir / EXPERIMENT_CONFIG["output"]["tokenization_subdir"]
        )
        self.token_dir.mkdir(parents=True, exist_ok=True)

    def tokenize_documents(self, documents: List[Document]) -> Dict[str, str]:
        """문서들을 모든 토크나이저로 토큰화"""
        if not EXPERIMENT_CONFIG["execution"]["enable_tokenization_phase"]:
            logger.info("🔤 토큰화 단계 건너뛰기 (비활성화됨)")
            return {}

        logger.info("🔤 토큰화 단계 시작...")

        # 문서 텍스트 결합
        combined_text = "\n\n".join([doc.text for doc in documents])
        doc_hash = hashlib.md5(combined_text.encode()).hexdigest()[:8]

        tokenization_results = {}

        # 필터 적용
        from experiment_configurations import TOKENIZERS

        tokenizers = TOKENIZERS
        if EXPERIMENT_CONFIG["filters"]["tokenizers"]:
            tokenizers = {
                k: v
                for k, v in tokenizers.items()
                if k in EXPERIMENT_CONFIG["filters"]["tokenizers"]
            }

        for tokenizer_name, config in tokenizers.items():
            logger.info(f"🔤 토큰화: {tokenizer_name}")

            try:
                start_time = time.time()

                # 토크나이저별 구현
                impl = config["implementation"]
                if impl == "character":
                    tokens = list(combined_text)
                elif impl in ["gpt2", "cl100k_base"]:
                    import tiktoken

                    tokenizer = tiktoken.get_encoding(impl)
                    tokens = tokenizer.encode(combined_text)
                    tokens = [str(t) for t in tokens]
                else:
                    logger.warning(f"⚠️ 알 수 없는 토크나이저: {impl}")
                    continue

                processing_time = time.time() - start_time

                # 결과 저장
                result_id = self._save_tokenization(
                    tokenizer_name, doc_hash, tokens, processing_time
                )
                tokenization_results[tokenizer_name] = result_id

                logger.info(
                    f"✅ 토큰화 완료: {tokenizer_name} ({len(tokens):,}개, {processing_time:.1f}초)"
                )

            except Exception as e:
                logger.error(f"❌ 토큰화 실패: {tokenizer_name} - {e}")
                continue

        logger.info(f"🎉 토큰화 완료: {len(tokenization_results)}개 결과")
        return tokenization_results

    def _save_tokenization(
        self,
        tokenizer_name: str,
        doc_hash: str,
        tokens: List[str],
        processing_time: float,
    ) -> str:
        """토큰화 결과 저장"""
        result_id = f"{ExperimentNaming.abbreviate(tokenizer_name)}_{doc_hash}"

        # 토큰 저장
        token_file = self.token_dir / f"{result_id}_tokens.pkl"
        with open(token_file, "wb") as f:
            pickle.dump(tokens, f)

        # 메타데이터 저장
        metadata = {
            "result_id": result_id,
            "tokenizer_name": tokenizer_name,
            "doc_hash": doc_hash,
            "token_count": len(tokens),
            "unique_tokens": len(set(tokens)),
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "file_size_mb": token_file.stat().st_size / (1024 * 1024),
        }

        metadata_file = self.token_dir / f"{result_id}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return result_id


# ===================================================================
# 유틸리티 함수
# ===================================================================


def log_experiment_start(phase: str, total_combinations: int):
    """실험 시작 로그"""
    logger.info(f"\n🚀 === {phase} 시작 ===")
    logger.info(f"📊 총 조합 수: {total_combinations:,}개")
    logger.info(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def log_experiment_end(phase: str, successful: int, failed: int, start_time: float):
    """실험 종료 로그"""
    duration = time.time() - start_time
    total = successful + failed
    success_rate = (successful / total * 100) if total > 0 else 0

    logger.info(f"\n🎉 === {phase} 완료 ===")
    logger.info(f"✅ 성공: {successful:,}개")
    logger.info(f"❌ 실패: {failed:,}개")
    logger.info(f"📈 성공률: {success_rate:.1f}%")
    logger.info(f"⏱️ 총 소요시간: {duration:.1f}초")
