#!/usr/bin/env python3
"""
Embedding Database Builder
=========================

Phase 2-3: 임베딩 데이터셋 생성 전용 모듈
저장된 청킹 결과를 기반으로 다양한 임베딩 모델로 벡터 데이터베이스를 구축합니다.

주요 기능:
- 28,560개 조합 임베딩 처리 (청킹 × 임베딩 모델)
- 청킹 결과 파일 기반 처리
- FAISS 벡터 데이터베이스 생성 및 저장
- Multi-GPU 자동 감지 및 최적화

데이터 흐름:
    chunks/*.pkl → Embedding → vectorstores/*/

사용법:
    python embedding_database_builder.py
    python embedding_database_builder.py --chunk-dir experiments/outputs/chunks
    python embedding_database_builder.py --max-combinations 1000
    python embedding_database_builder.py --embedding-models qwen3_8b,bge_m3
    python embedding_database_builder.py --dry-run
"""

import os
import sys
import json
import time
import pickle
import argparse
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Iterator

import torch
import numpy as np
import pandas as pd
from loguru import logger
from llama_index.core import Document

# Vector store and embeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 로컬 모듈
from core.experiment.commons import (
    ExperimentState,
    ExperimentResult,
    ExperimentNaming,
    log_experiment_start,
    log_experiment_end,
)
from experiment_configurations import (
    EMBEDDING_MODELS,
    EXPERIMENT_CONFIG,
    validate_configurations,
    print_configuration_summary,
)

# ===================================================================
# 임베딩 관리자
# ===================================================================


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
# 청킹 결과 로더
# ===================================================================


class ChunkingResultLoader:
    """저장된 청킹 결과 로드 관리"""

    def __init__(self, chunk_dir: Path):
        self.chunk_dir = Path(chunk_dir)

    def scan_available_chunks(self) -> List[str]:
        """사용 가능한 청크 ID 스캔"""
        if not self.chunk_dir.exists():
            logger.warning(f"청크 디렉토리 없음: {self.chunk_dir}")
            return []

        chunk_ids = []
        for chunks_file in self.chunk_dir.glob("*_chunks.pkl"):
            chunk_id = chunks_file.stem.replace("_chunks", "")

            # 메타데이터 파일 존재 확인
            metadata_file = self.chunk_dir / f"{chunk_id}_metadata.json"
            if metadata_file.exists():
                chunk_ids.append(chunk_id)

        logger.info(f"📊 사용 가능한 청크: {len(chunk_ids)}개")
        return sorted(chunk_ids)

    def load_chunk_data(self, chunk_id: str) -> Tuple[List[Document], Dict[str, Any]]:
        """청크 데이터 로드"""
        chunks_file = self.chunk_dir / f"{chunk_id}_chunks.pkl"
        metadata_file = self.chunk_dir / f"{chunk_id}_metadata.json"

        if not chunks_file.exists():
            raise FileNotFoundError(f"청크 파일 없음: {chunks_file}")

        # 청크 데이터 로드
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)

        # 메타데이터 로드
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        return chunks, metadata


# ===================================================================
# 임베딩 실험 실행기
# ===================================================================


class EmbeddingExperimentRunner:
    """임베딩 실험 전용 실행기 - Phase 2-3"""

    def __init__(
        self, output_dir: Optional[str] = None, chunk_dir: Optional[str] = None
    ):
        self.output_dir = Path(output_dir or EXPERIMENT_CONFIG["output"]["base_dir"])
        self.chunk_dir = Path(
            chunk_dir
            or (self.output_dir / EXPERIMENT_CONFIG["output"]["chunking_subdir"])
        )

        # 관리자들 초기화
        self.chunk_loader = ChunkingResultLoader(self.chunk_dir)
        self.embedding_manager = EmbeddingManager(self.output_dir)

        # 상태 추적
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

    def run_all_embedding_experiments(
        self,
        max_combinations: Optional[int] = None,
        chunk_filter: Optional[List[str]] = None,
        embedding_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """모든 임베딩 실험 실행"""

        # GPU 정보 로깅
        self._log_gpu_info()

        log_experiment_start("임베딩 데이터셋 생성", 0)  # 조합 수는 나중에 계산
        self.state.start_time = time.time()
        self.state.current_phase = "임베딩 데이터셋 생성"

        try:
            # 단계 1: 청크 파일 스캔
            self.state.current_phase = "청킹 결과 스캔"
            logger.info(f"\n📊 === {self.state.current_phase} ===")

            available_chunks = self.chunk_loader.scan_available_chunks()
            if not available_chunks:
                return {"success": False, "error": "청킹 결과 없음"}

            # 청크 필터 적용
            if chunk_filter:
                filtered_chunks = []
                for chunk_id in available_chunks:
                    if any(pattern in chunk_id for pattern in chunk_filter):
                        filtered_chunks.append(chunk_id)
                available_chunks = filtered_chunks
                logger.info(f"🔍 청크 필터 적용: {len(available_chunks)}개 선택")

            # 임베딩 모델 필터 적용
            if embedding_filter:
                original_filter = EXPERIMENT_CONFIG["filters"]["embedding_models"]
                EXPERIMENT_CONFIG["filters"]["embedding_models"] = embedding_filter
                logger.info(f"🔍 임베딩 모델 필터 적용: {embedding_filter}")

            # 단계 2: 임베딩 실험 실행
            self.state.current_phase = "임베딩 실험"
            logger.info(f"\n🔮 === {self.state.current_phase} ===")

            embedding_results = self._run_embedding_experiments(
                available_chunks, max_combinations
            )

            self.results = embedding_results
            self.state.embedding_results = len(embedding_results)

            # 필터 복원
            if embedding_filter:
                EXPERIMENT_CONFIG["filters"]["embedding_models"] = original_filter

            # 결과 저장
            summary = self._save_embedding_summary()

            # 성공/실패 분리
            successful_results = [r for r in self.results if r.success]
            failed_results = [r for r in self.results if not r.success]

            log_experiment_end(
                "임베딩 데이터셋 생성",
                len(successful_results),
                len(failed_results),
                self.state.start_time,
            )

            return {
                "success": True,
                "state": self.state,
                "summary": summary,
                "output_dir": str(self.output_dir),
                "embedding_results": embedding_results,
            }

        except KeyboardInterrupt:
            logger.warning("\n❌ 사용자 중단")
            return {"success": False, "error": "중단됨"}

        except Exception as e:
            logger.error(f"\n❌ 임베딩 실행 오류: {e}")
            return {"success": False, "error": str(e)}

    def _run_embedding_experiments(
        self, chunk_ids: List[str], max_combinations: Optional[int] = None
    ) -> List[ExperimentResult]:
        """임베딩 실험 실행"""
        results = []

        # 전체 조합 계산
        embedding_combinations = list(
            self.embedding_manager.generate_embedding_combinations()
        )
        total_combinations = len(chunk_ids) * len(embedding_combinations)

        logger.info(f"📊 총 조합 수: {total_combinations:,}개")

        if max_combinations:
            total_combinations = min(total_combinations, max_combinations)
            logger.info(f"🔍 최대 조합 수로 제한: {max_combinations:,}개")

        current_combination = 0

        for chunk_id in chunk_ids:
            try:
                # 청킹 결과 로드
                chunks, chunk_metadata = self.chunk_loader.load_chunk_data(chunk_id)
                logger.info(f"📦 청크 로드: {chunk_id} ({len(chunks)}개 청크)")

                # 각 임베딩 모델로 실험
                for embedding_name, embedding_config in embedding_combinations:
                    current_combination += 1

                    # 최대 조합 수 제한
                    if max_combinations and current_combination > max_combinations:
                        logger.info(f"⏹️ 최대 조합 수 도달: {max_combinations}개")
                        return results

                    logger.info(
                        f"🔮 임베딩 [{current_combination:,}/{total_combinations:,}]: "
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
                logger.error(f"❌ 청크 로드 실패: {chunk_id} - {e}")
                continue

        logger.info(f"🎉 임베딩 단계 완료: {self.state.successful_experiments}개 성공")
        return results

    def _save_embedding_summary(self) -> Dict[str, Any]:
        """임베딩 결과 요약 저장"""
        # CSV 형태로 결과 저장
        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "experiment_id": result.experiment_id,
                    "tokenizer_name": result.tokenizer_name or "",
                    "chunker_name": result.chunker_name,
                    "chunker_params": json.dumps(result.chunker_params),
                    "embedding_name": result.embedding_name,
                    "embedding_model": result.embedding_model,
                    "total_chunks": result.total_chunks,
                    "embedding_time": result.embedding_time,
                    "database_size_mb": result.database_size_mb,
                    "database_path": result.database_path,
                    "success": result.success,
                    "error_message": result.error_message or "",
                    "timestamp": result.timestamp,
                }
            )

        results_df = pd.DataFrame(results_data)
        results_csv = self.output_dir / "embedding_experiment_results.csv"
        results_df.to_csv(results_csv, index=False)

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
            "results_file": str(results_csv),
        }

        # JSON 요약 저장
        summary_file = self.output_dir / "embedding_experiment_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"\n📊 === 임베딩 결과 요약 ===")
        logger.info(f"⏱️ 총 소요시간: {total_time:.1f}초")
        logger.info(f"🔮 총 실험 수: {len(self.results):,}개")
        logger.info(f"✅ 성공: {len(successful_results):,}개")
        logger.info(f"❌ 실패: {len(self.results) - len(successful_results):,}개")
        logger.info(f"📈 성공률: {summary['success_rate']:.1%}")
        logger.info(f"💾 총 DB 크기: {summary['total_database_size_mb']:.1f}MB")
        logger.info(f"📁 결과 저장 위치: {self.output_dir}")

        return summary


# ===================================================================
# CLI 및 메인 실행
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Embedding Database Builder - Phase 2-3"
    )
    parser.add_argument("--output-dir", type=str, help="결과 저장 디렉토리")
    parser.add_argument("--chunk-dir", type=str, help="청킹 결과 디렉토리")
    parser.add_argument(
        "--max-combinations", type=int, help="최대 처리할 임베딩 조합 수"
    )
    parser.add_argument(
        "--chunk-filter",
        type=str,
        help="청크 ID 필터 (comma-separated: token,sentence)",
    )
    parser.add_argument(
        "--embedding-models",
        type=str,
        help="사용할 임베딩 모델 (comma-separated: qwen3_8b,bge_m3)",
    )
    parser.add_argument("--config-summary", action="store_true", help="설정 요약 출력")
    parser.add_argument("--dry-run", action="store_true", help="설정 확인만 실행")
    parser.add_argument(
        "--scan-chunks", action="store_true", help="사용 가능한 청크 파일 스캔"
    )

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

    # 임베딩 빌더 초기화
    runner = EmbeddingExperimentRunner(args.output_dir, args.chunk_dir)

    # 청크 파일 스캔
    if args.scan_chunks:
        available_chunks = runner.chunk_loader.scan_available_chunks()
        logger.info(f"📊 사용 가능한 청크 파일: {len(available_chunks)}개")
        for chunk_id in available_chunks[:10]:  # 처음 10개만 표시
            logger.info(f"  • {chunk_id}")
        if len(available_chunks) > 10:
            logger.info(f"  • ... 외 {len(available_chunks) - 10}개")
        return 0

    # 필터 처리
    chunk_filter = None
    if args.chunk_filter:
        chunk_filter = [c.strip() for c in args.chunk_filter.split(",")]
        logger.info(f"🔍 청크 필터: {chunk_filter}")

    embedding_filter = None
    if args.embedding_models:
        embedding_filter = [e.strip() for e in args.embedding_models.split(",")]
        logger.info(f"🔍 임베딩 모델 필터: {embedding_filter}")

    # 실제 임베딩 실험 실행
    try:
        result = runner.run_all_embedding_experiments(
            max_combinations=args.max_combinations,
            chunk_filter=chunk_filter,
            embedding_filter=embedding_filter,
        )

        if result["success"]:
            logger.info("🎉 임베딩 실험 완료!")
            return 0
        else:
            logger.error(f"❌ 임베딩 실험 실패: {result.get('error')}")
            return 1

    except Exception as e:
        logger.error(f"❌ 실행 오류: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
