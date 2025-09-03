#!/usr/bin/env python3
"""
Multi-Embedding Database Builder
===============================

experiment_configurations.py의 설정을 사용하여
모든 토크나이저, 청커, 임베딩 모델 조합으로 벡터 데이터베이스를 구축합니다.

실행 순서:
1. 문서 로딩
2. 토크나이저별 토큰화 (선택적)
3. 청커별 파라미터 조합으로 청킹
4. 임베딩 모델별로 벡터스토어 생성 및 저장

사용법:
    python multi_embedding_database_builder.py
    python multi_embedding_database_builder.py --dry-run  # 설정 확인만
"""

import os
import sys
import json
import time
import pickle
import hashlib
import itertools
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass

import torch
import gc
import numpy as np
import pandas as pd
from loguru import logger

# 로컬 모듈
from experiment_configurations import (
    TOKENIZERS,
    CHUNKERS,
    EMBEDDING_MODELS,
    EXPERIMENT_CONFIG,
    validate_configurations,
    print_configuration_summary,
)
from loaders import load_document, collect_document_paths

# Vector store and embeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

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
# 네이밍 컨벤션 (간소화)
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
# 실험 관리자들
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

                # 메타데이터 보강
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
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        doc_hash = hashlib.md5(combined_text.encode()).hexdigest()[:8]

        tokenization_results = {}

        # 필터 적용
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


class ChunkingManager:
    """청킹 관리"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.chunk_dir = (
            self.output_dir / EXPERIMENT_CONFIG["output"]["chunking_subdir"]
        )
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

    def generate_chunking_combinations(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """청킹 조합 생성기"""
        logger.info("🧩 청킹 조합 생성...")

        # 필터 적용
        chunkers = CHUNKERS
        if EXPERIMENT_CONFIG["filters"]["chunkers"]:
            chunkers = {
                k: v
                for k, v in chunkers.items()
                if k in EXPERIMENT_CONFIG["filters"]["chunkers"]
            }

        total_combinations = 0

        for chunker_name, config in chunkers.items():
            logger.info(f"🧩 청커: {chunker_name}")

            # 기본 파라미터
            base_params = config["parameters"]["base"].copy()
            variations = config["parameters"]["variations"]

            # 파라미터 조합 생성
            param_lists = {}
            static_params = base_params.copy()

            for param_name, param_values in variations.items():
                if param_name == "embedding_model" and param_values == "ALL_EMBEDDINGS":
                    # 임베딩 모델 필터 적용
                    embedding_models = EMBEDDING_MODELS
                    if EXPERIMENT_CONFIG["filters"]["embedding_models"]:
                        embedding_models = {
                            k: v
                            for k, v in embedding_models.items()
                            if k in EXPERIMENT_CONFIG["filters"]["embedding_models"]
                        }
                    param_lists[param_name] = [
                        v["model_name"] for v in embedding_models.values()
                    ]
                elif isinstance(param_values, list):
                    param_lists[param_name] = param_values
                else:
                    static_params[param_name] = param_values

            # 조합 생성
            if param_lists:
                keys, values = zip(*param_lists.items())
                for combination in itertools.product(*values):
                    params = static_params.copy()
                    params.update(dict(zip(keys, combination)))
                    total_combinations += 1
                    yield chunker_name, params
            else:
                total_combinations += 1
                yield chunker_name, static_params

        logger.info(f"📊 총 청킹 조합: {total_combinations}개")

    def chunk_documents(
        self, chunker_name: str, params: Dict[str, Any], documents: List[Document]
    ) -> Optional[str]:
        """문서 청킹 수행"""
        try:
            # 청커 생성
            chunker_class = CHUNKERS[chunker_name]["class"]
            clean_params = {k: v for k, v in params.items() if v is not None}

            logger.debug(f"  📄 청킹: {chunker_name} with {clean_params}")

            start_time = time.time()
            chunker = chunker_class(**clean_params)

            # 문서들 청킹
            all_chunks = []
            for doc in documents:
                try:
                    chunks = chunker.chunk(doc.page_content)

                    for i, chunk in enumerate(chunks):
                        # 청크 텍스트 추출
                        chunk_text = (
                            chunk.text if hasattr(chunk, "text") else str(chunk)
                        )

                        # Document 객체 생성
                        chunk_doc = Document(
                            page_content=chunk_text,
                            metadata={
                                **doc.metadata,
                                "chunk_index": i,
                                "chunk_size": len(chunk_text),
                                "parent_doc_id": doc.metadata.get("doc_id", "unknown"),
                                "chunker_name": chunker_name,
                                "chunker_params": clean_params,
                            },
                        )
                        all_chunks.append(chunk_doc)

                except Exception as e:
                    logger.debug(f"    ⚠️ 문서 청킹 실패: {e}")
                    continue

            if not all_chunks:
                logger.warning(f"  ❌ 청킹 결과 없음: {chunker_name}")
                return None

            processing_time = time.time() - start_time

            # 결과 저장
            chunk_id = self._save_chunking_result(
                chunker_name, params, all_chunks, processing_time
            )

            logger.debug(
                f"  ✅ 청킹 완료: {chunk_id} ({len(all_chunks)}개, {processing_time:.1f}초)"
            )
            return chunk_id

        except Exception as e:
            logger.error(f"  ❌ 청킹 실패: {chunker_name} - {e}")
            return None

    def _save_chunking_result(
        self,
        chunker_name: str,
        params: Dict[str, Any],
        chunks: List[Document],
        processing_time: float,
    ) -> str:
        """청킹 결과 저장"""
        chunk_id = ExperimentNaming.generate_chunk_id(chunker_name, params)

        # 청크 데이터 저장
        chunks_file = self.chunk_dir / f"{chunk_id}_chunks.pkl"
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)

        # 메타데이터 저장
        metadata = {
            "chunk_id": chunk_id,
            "chunker_name": chunker_name,
            "chunker_params": params,
            "total_chunks": len(chunks),
            "avg_chunk_size": float(np.mean([len(doc.page_content) for doc in chunks])),
            "chunk_size_variance": float(
                np.var([len(doc.page_content) for doc in chunks])
            ),
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "file_size_mb": chunks_file.stat().st_size / (1024 * 1024),
        }

        metadata_file = self.chunk_dir / f"{chunk_id}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return chunk_id

    def load_chunking_result(
        self, chunk_id: str
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """청킹 결과 로드"""
        chunks_file = self.chunk_dir / f"{chunk_id}_chunks.pkl"
        metadata_file = self.chunk_dir / f"{chunk_id}_metadata.json"

        if not chunks_file.exists():
            raise FileNotFoundError(f"청킹 결과 없음: {chunk_id}")

        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)

        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        return chunks, metadata


class EmbeddingManager:
    """임베딩 및 FAISS 저장 관리"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def generate_embedding_combinations(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """임베딩 모델 조합 생성기"""
        # 필터 적용
        embedding_models = EMBEDDING_MODELS
        if EXPERIMENT_CONFIG["filters"]["embedding_models"]:
            embedding_models = {
                k: v
                for k, v in embedding_models.items()
                if k in EXPERIMENT_CONFIG["filters"]["embedding_models"]
            }

        for name, config in embedding_models.items():
            yield name, config

    def create_vectorstore_and_save(
        self,
        embedding_name: str,
        embedding_config: Dict[str, Any],
        chunk_id: str,
        chunks: List[Document],
        chunk_metadata: Dict[str, Any],
    ) -> ExperimentResult:
        """벡터스토어 생성 및 저장"""
        start_time = time.time()

        try:
            # 임베딩 모델 생성
            embedding_model = self._create_embedding_model(embedding_config)

            # 벡터스토어 생성
            logger.debug(f"    🔮 벡터스토어 생성: {embedding_name}")
            vectorstore = FAISS.from_documents(chunks, embedding_model)
            embedding_time = time.time() - start_time

            # 저장 경로 생성
            db_relative_path = ExperimentNaming.generate_database_path(
                embedding_name, chunk_id
            )
            db_path = self.output_dir / db_relative_path
            db_path.mkdir(parents=True, exist_ok=True)

            # FAISS 저장
            vectorstore.save_local(str(db_path))

            # 메타데이터 저장
            full_metadata = {
                "experiment_id": f"{embedding_name}_{chunk_id}",
                "embedding_name": embedding_name,
                "embedding_model": embedding_config["model_name"],
                "embedding_dimension": embedding_config["dimension"],
                "chunk_id": chunk_id,
                "database_path": str(db_path),
                "timestamp": datetime.now().isoformat(),
                **chunk_metadata,
            }

            with open(db_path / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(full_metadata, f, ensure_ascii=False, indent=2)

            # DB 크기 계산
            db_size_mb = sum(
                f.stat().st_size for f in db_path.rglob("*") if f.is_file()
            ) / (1024 * 1024)

            # Multi-GPU 메모리 정리
            if EXPERIMENT_CONFIG["execution"]["cleanup_memory_after_each"]:
                if torch.cuda.is_available():
                    # 모든 GPU 메모리 정리
                    for i in range(torch.cuda.device_count()):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                gc.collect()

            return ExperimentResult(
                experiment_id=full_metadata["experiment_id"],
                tokenizer_name=None,  # 현재는 미사용
                chunker_name=chunk_metadata.get("chunker_name", "unknown"),
                chunker_params=chunk_metadata.get("chunker_params", {}),
                embedding_name=embedding_name,
                embedding_model=embedding_config["model_name"],
                total_chunks=len(chunks),
                embedding_time=embedding_time,
                database_size_mb=db_size_mb,
                database_path=str(db_path),
                success=True,
                timestamp=full_metadata["timestamp"],
            )

        except Exception as e:
            embedding_time = time.time() - start_time
            error_msg = f"임베딩 실패: {embedding_name} × {chunk_id} - {str(e)}"

            return ExperimentResult(
                experiment_id=f"{embedding_name}_{chunk_id}",
                tokenizer_name=None,
                chunker_name=chunk_metadata.get("chunker_name", "unknown"),
                chunker_params=chunk_metadata.get("chunker_params", {}),
                embedding_name=embedding_name,
                embedding_model=embedding_config["model_name"],
                total_chunks=0,
                embedding_time=embedding_time,
                database_size_mb=0,
                database_path="",
                success=False,
                error_message=error_msg,
                timestamp=datetime.now().isoformat(),
            )

    def _create_embedding_model(self, config: Dict[str, Any]) -> HuggingFaceEmbeddings:
        """임베딩 모델 생성 (Multi-GPU 지원)"""
        model_kwargs = config["model_kwargs"].copy()
        encode_kwargs = config["encode_kwargs"].copy()

        # Multi-GPU 자동 설정
        if model_kwargs.get("device") == "auto":
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    # Multi-GPU 환경
                    logger.info(f"🎮 Multi-GPU 감지: {gpu_count}개 GPU 활용")
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["torch_dtype"] = model_kwargs.get(
                        "torch_dtype", "float16"
                    )

                    # Multi-GPU에서 배치 크기 최적화
                    if "batch_size" in encode_kwargs:
                        original_batch_size = encode_kwargs["batch_size"]
                        # GPU 수만큼 배치 크기 증가 (최대 128까지)
                        optimized_batch_size = min(original_batch_size * gpu_count, 128)
                        encode_kwargs["batch_size"] = optimized_batch_size
                        logger.info(
                            f"  📈 배치 크기 최적화: {original_batch_size} → {optimized_batch_size}"
                        )
                else:
                    # Single GPU
                    model_kwargs["device"] = "cuda"
                    logger.info("🎮 Single GPU 사용")
            else:
                # CPU 모드
                model_kwargs["device"] = "cpu"
                logger.info("💻 CPU 모드 사용")

        return HuggingFaceEmbeddings(
            model_name=config["model_name"],
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )


# ===================================================================
# 메인 실험 실행기
# ===================================================================


class MultiEmbeddingExperimentRunner:
    """메인 실험 실행기 - 설정 기반 순회 실험"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or EXPERIMENT_CONFIG["output"]["base_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 관리자들 초기화
        self.doc_manager = DocumentManager()
        self.tokenization_manager = TokenizationManager(self.output_dir)
        self.chunking_manager = ChunkingManager(self.output_dir)
        self.embedding_manager = EmbeddingManager(self.output_dir)

        # 상태 및 결과
        self.state = ExperimentState()
        self.results: List[ExperimentResult] = []

    def _log_gpu_info(self):
        """GPU 환경 정보 로깅"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"🎮 GPU 환경: {gpu_count}개 GPU 감지됨")

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory // (
                    1024**3
                )
                logger.info(f"  • GPU {i}: {gpu_name} ({gpu_memory}GB VRAM)")

            # 현재 메모리 사용량
            logger.info("📊 GPU 메모리 상태:")
            for i in range(gpu_count):
                allocated = torch.cuda.memory_allocated(i) // (1024**2)
                reserved = torch.cuda.memory_reserved(i) // (1024**2)
                total = torch.cuda.get_device_properties(i).total_memory // (1024**2)
                logger.info(
                    f"  • GPU {i}: {allocated}MB 사용 중, {reserved}MB 예약됨 / {total}MB"
                )

            if gpu_count > 1:
                logger.info("🚀 Multi-GPU 최적화 활성화 - 자원 낭비 최소화")
        else:
            logger.info("💻 CPU 모드 - GPU 없음")

    def run_all_experiments(self) -> Dict[str, Any]:
        """모든 실험 실행 - 설정 기반 순회"""
        logger.info("🚀 Multi-Embedding Database Builder 시작!")
        logger.info("📋 설정 기반 순회 실험")

        # GPU 환경 정보
        self._log_gpu_info()
        logger.info("=" * 80)

        self.state.start_time = time.time()

        try:
            # 단계 1: 문서 로딩
            self.state.current_phase = "문서 로딩"
            logger.info(f"\n📖 === {self.state.current_phase} ===")
            documents = self.doc_manager.load_all_documents()

            if not documents:
                return {"success": False, "error": "문서 없음"}

            self.state.total_documents = len(documents)

            # 단계 2: 토큰화 (선택적)
            self.state.current_phase = "토큰화"
            logger.info(f"\n🔤 === {self.state.current_phase} ===")
            tokenization_results = self.tokenization_manager.tokenize_documents(
                documents
            )
            self.state.tokenization_results = len(tokenization_results)

            # 단계 3: 청킹 단계
            self.state.current_phase = "청킹"
            logger.info(f"\n🧩 === {self.state.current_phase} ===")
            chunk_results = self._run_chunking_experiments(documents)
            self.state.chunking_results = len(chunk_results)

            if not chunk_results:
                return {"success": False, "error": "청킹 결과 없음"}

            # 단계 4: 임베딩 단계
            self.state.current_phase = "임베딩"
            logger.info(f"\n🔮 === {self.state.current_phase} ===")

            # 모든 실험 수행 (제약 없음)

            embedding_results = self._run_embedding_experiments(chunk_results)
            self.results = embedding_results
            self.state.embedding_results = len(embedding_results)

            # 결과 저장 및 요약
            summary = self._save_final_results()

            return {
                "success": True,
                "state": self.state,
                "summary": summary,
                "output_dir": str(self.output_dir),
            }

        except KeyboardInterrupt:
            logger.warning("\n❌ 사용자 중단")
            return {"success": False, "error": "중단됨"}

        except Exception as e:
            logger.error(f"\n❌ 실행 오류: {e}")
            return {"success": False, "error": str(e)}

    def _run_chunking_experiments(self, documents: List[Document]) -> List[str]:
        """청킹 실험 실행"""
        chunk_results = []

        combination_count = 0
        for (
            chunker_name,
            params,
        ) in self.chunking_manager.generate_chunking_combinations():
            combination_count += 1
            logger.info(f"🧩 청킹 [{combination_count}]: {chunker_name}")

            chunk_id = self.chunking_manager.chunk_documents(
                chunker_name, params, documents
            )
            if chunk_id:
                chunk_results.append(chunk_id)

        logger.info(f"🎉 청킹 단계 완료: {len(chunk_results)}개 결과")
        return chunk_results

    def _run_embedding_experiments(
        self, chunk_ids: List[str]
    ) -> List[ExperimentResult]:
        """임베딩 실험 실행"""
        results = []

        # 전체 조합 계산
        embedding_combinations = list(
            self.embedding_manager.generate_embedding_combinations()
        )
        total_combinations = len(chunk_ids) * len(embedding_combinations)

        current_combination = 0

        for chunk_id in chunk_ids:
            try:
                # 청킹 결과 로드
                chunks, chunk_metadata = self.chunking_manager.load_chunking_result(
                    chunk_id
                )

                # 각 임베딩 모델로 실험
                for embedding_name, embedding_config in embedding_combinations:
                    current_combination += 1
                    logger.info(
                        f"🔮 임베딩 [{current_combination}/{total_combinations}]: "
                        f"{embedding_name} × {chunk_id}"
                    )

                    result = self.embedding_manager.create_vectorstore_and_save(
                        embedding_name,
                        embedding_config,
                        chunk_id,
                        chunks,
                        chunk_metadata,
                    )

                    results.append(result)

                    if result.success:
                        self.state.successful_experiments += 1
                        logger.info(
                            f"✅ 성공: {embedding_name} ({result.embedding_time:.1f}초, {result.database_size_mb:.1f}MB)"
                        )
                    else:
                        self.state.failed_experiments += 1
                        logger.error(f"❌ 실패: {result.error_message}")

            except Exception as e:
                logger.error(f"❌ 청킹 로드 실패: {chunk_id} - {e}")
                continue

        logger.info(f"🎉 임베딩 단계 완료: {self.state.successful_experiments}개 성공")
        return results

    def _save_final_results(self) -> Dict[str, Any]:
        """최종 결과 저장"""
        # CSV 저장
        results_df = pd.DataFrame(
            [
                {
                    "experiment_id": r.experiment_id,
                    "tokenizer_name": r.tokenizer_name or "",
                    "chunker_name": r.chunker_name,
                    "chunker_params": json.dumps(r.chunker_params),
                    "embedding_name": r.embedding_name,
                    "embedding_model": r.embedding_model,
                    "total_chunks": r.total_chunks,
                    "embedding_time": r.embedding_time,
                    "database_size_mb": r.database_size_mb,
                    "database_path": r.database_path,
                    "success": r.success,
                    "error_message": r.error_message or "",
                    "timestamp": r.timestamp,
                }
                for r in self.results
            ]
        )

        csv_path = self.output_dir / "experiment_results.csv"
        results_df.to_csv(csv_path, index=False)

        # 요약 통계
        successful_results = [r for r in self.results if r.success]
        total_time = time.time() - self.state.start_time

        summary = {
            "execution_time": total_time,
            "total_experiments": len(self.results),
            "successful_experiments": len(successful_results),
            "failed_experiments": len(self.results) - len(successful_results),
            "success_rate": len(successful_results) / len(self.results)
            if self.results
            else 0,
            "total_database_size_mb": sum(
                r.database_size_mb for r in successful_results
            ),
            "avg_embedding_time": np.mean(
                [r.embedding_time for r in successful_results]
            )
            if successful_results
            else 0,
            "state": {
                "total_documents": self.state.total_documents,
                "tokenization_results": self.state.tokenization_results,
                "chunking_results": self.state.chunking_results,
                "embedding_results": self.state.embedding_results,
            },
        }

        # JSON 저장
        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 로그 출력
        logger.info(f"\n" + "=" * 80)
        logger.info("🎉 Multi-Embedding Database Builder 완료!")
        logger.info(f"⏱️ 총 소요시간: {total_time:.1f}초")
        logger.info(f"📄 처리된 문서: {self.state.total_documents}개")
        logger.info(f"🔤 토큰화 결과: {self.state.tokenization_results}개")
        logger.info(f"🧩 청킹 결과: {self.state.chunking_results}개")
        logger.info(f"🔮 임베딩 결과: {self.state.embedding_results}개")
        logger.info(f"✅ 성공: {len(successful_results)}개 벡터스토어")
        logger.info(f"❌ 실패: {len(self.results) - len(successful_results)}개")
        logger.info(f"📈 성공률: {summary['success_rate']:.1%}")
        logger.info(f"💾 총 DB 크기: {summary['total_database_size_mb']:.1f}MB")
        logger.info(f"\n📁 결과 위치: {self.output_dir}")

        return summary


# ===================================================================
# CLI 및 메인 실행
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="Multi-Embedding Database Builder")
    parser.add_argument("--dry-run", action="store_true", help="설정 검증만 실행")
    parser.add_argument("--output-dir", type=str, help="결과 저장 디렉토리")
    parser.add_argument("--config-summary", action="store_true", help="설정 요약 출력")

    args = parser.parse_args()

    # 설정 요약 출력
    if args.config_summary or args.dry_run:
        print_configuration_summary()

        # 설정 검증
        validation = validate_configurations()
        if not validation["valid"]:
            logger.error("❌ 설정 오류:")
            for error in validation["errors"]:
                logger.error(f"  • {error}")
            return 1

        if args.dry_run:
            logger.info("✅ 설정 검증 완료 - 실행하지 않음")
            return 0

    # 실제 실험 실행
    try:
        runner = MultiEmbeddingExperimentRunner(args.output_dir)
        result = runner.run_all_experiments()

        if result["success"]:
            logger.info("🎉 모든 실험 완료!")
            return 0
        else:
            logger.error(f"❌ 실험 실패: {result.get('error')}")
            return 1

    except Exception as e:
        logger.error(f"❌ 실행 오류: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
