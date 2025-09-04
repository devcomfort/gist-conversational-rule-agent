#!/usr/bin/env python3
"""
Chunking Database Builder
========================

Phase 2-1: 청킹 데이터 생성 전용 모듈
법률 문서를 다양한 청킹 전략으로 분할하고 결과를 저장합니다.

주요 기능:
- 4,760개 청킹 조합 처리
- 토큰, 문장, 의미, 신경망, 재귀 청킹 지원
- 청킹 결과 pickle 형태로 저장
- 청킹 품질 메트릭 수집

데이터 흐름:
    Documents → Chunking → chunks/*.pkl + metadata

사용법:
    python chunking_database_builder.py
    python chunking_database_builder.py --max-chunks 100
    python chunking_database_builder.py --chunkers token,sentence
    python chunking_database_builder.py --dry-run
"""

import os
import sys
import json
import time
import pickle
import itertools
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Iterator

import numpy as np
import pandas as pd
from loguru import logger
from llama_index.core import Document

# 로컬 모듈
from core.experiment.commons import (
    DocumentManager,
    TokenizationManager,
    ExperimentState,
    ExperimentNaming,
    log_experiment_start,
    log_experiment_end,
)
from experiment_configurations import (
    CHUNKERS,
    EMBEDDING_MODELS,
    EXPERIMENT_CONFIG,
    validate_configurations,
    print_configuration_summary,
)

# ===================================================================
# 청킹 관리자
# ===================================================================


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
                    chunks = chunker.chunk(doc.text)

                    for i, chunk in enumerate(chunks):
                        # 청크 텍스트 추출
                        chunk_text = (
                            chunk.text if hasattr(chunk, "text") else str(chunk)
                        )

                        # Document 객체 생성
                        chunk_doc = Document(
                            text=chunk_text,
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
            "avg_chunk_size": float(np.mean([len(doc.text) for doc in chunks])),
            "chunk_size_variance": float(np.var([len(doc.text) for doc in chunks])),
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

    def scan_existing_chunks(self) -> List[str]:
        """기존에 생성된 청크 ID 스캔"""
        if not self.chunk_dir.exists():
            return []

        chunk_ids = []
        for chunks_file in self.chunk_dir.glob("*_chunks.pkl"):
            chunk_id = chunks_file.stem.replace("_chunks", "")

            # 메타데이터 파일 존재 확인
            metadata_file = self.chunk_dir / f"{chunk_id}_metadata.json"
            if metadata_file.exists():
                chunk_ids.append(chunk_id)

        return sorted(chunk_ids)


# ===================================================================
# 청킹 실험 실행기
# ===================================================================


class ChunkingExperimentRunner:
    """청킹 실험 전용 실행기 - Phase 2-1"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or EXPERIMENT_CONFIG["output"]["base_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 관리자들 초기화
        self.doc_manager = DocumentManager()
        self.tokenization_manager = TokenizationManager(self.output_dir)
        self.chunking_manager = ChunkingManager(self.output_dir)

        # 상태 추적
        self.state = ExperimentState()

    def run_all_chunking_experiments(
        self,
        max_chunks: Optional[int] = None,
        chunker_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """모든 청킹 실험 실행"""

        log_experiment_start("청킹 데이터 생성", 0)  # 조합 수는 나중에 계산
        self.state.start_time = time.time()
        self.state.current_phase = "청킹 데이터 생성"

        try:
            # 단계 1: 문서 로딩
            self.state.current_phase = "문서 로딩"
            logger.info(f"\n📖 === {self.state.current_phase} ===")
            documents = self.doc_manager.load_all_documents()

            if not documents:
                return {"success": False, "error": "문서 없음"}

            self.state.total_documents = len(documents)
            logger.info(f"✅ 문서 로딩 완료: {len(documents)}개")

            # 단계 2: 토큰화 (선택적)
            self.state.current_phase = "토큰화"
            logger.info(f"\n🔤 === {self.state.current_phase} ===")
            tokenization_results = self.tokenization_manager.tokenize_documents(
                documents
            )
            self.state.tokenization_results = len(tokenization_results)

            if tokenization_results:
                logger.info(f"✅ 토큰화 완료: {len(tokenization_results)}개 결과")

            # 단계 3: 청킹 실험
            self.state.current_phase = "청킹 실험"
            logger.info(f"\n🧩 === {self.state.current_phase} ===")

            # 필터 적용
            if chunker_filter:
                original_filter = EXPERIMENT_CONFIG["filters"]["chunkers"]
                EXPERIMENT_CONFIG["filters"]["chunkers"] = chunker_filter
                logger.info(f"🔍 청커 필터 적용: {chunker_filter}")

            chunk_results = self._run_chunking_experiments(documents, max_chunks)
            self.state.chunking_results = len(chunk_results)

            # 필터 복원
            if chunker_filter:
                EXPERIMENT_CONFIG["filters"]["chunkers"] = original_filter

            if not chunk_results:
                return {"success": False, "error": "청킹 결과 없음"}

            # 결과 저장
            summary = self._save_chunking_summary(chunk_results)

            log_experiment_end(
                "청킹 데이터 생성",
                len(chunk_results),
                0,  # 실패는 별도 계산 안함
                self.state.start_time,
            )

            return {
                "success": True,
                "state": self.state,
                "summary": summary,
                "output_dir": str(self.output_dir),
                "chunk_results": chunk_results,
            }

        except KeyboardInterrupt:
            logger.warning("\n❌ 사용자 중단")
            return {"success": False, "error": "중단됨"}

        except Exception as e:
            logger.error(f"\n❌ 청킹 실행 오류: {e}")
            return {"success": False, "error": str(e)}

    def _run_chunking_experiments(
        self,
        documents: List[Document],
        max_chunks: Optional[int] = None,
    ) -> List[str]:
        """청킹 실험 실행"""
        chunk_results = []
        combination_count = 0

        for (
            chunker_name,
            params,
        ) in self.chunking_manager.generate_chunking_combinations():
            combination_count += 1

            # 최대 조합 수 제한
            if max_chunks and combination_count > max_chunks:
                logger.info(f"⏹️ 최대 조합 수 도달: {max_chunks}개")
                break

            logger.info(f"🧩 청킹 [{combination_count}]: {chunker_name}")

            chunk_id = self.chunking_manager.chunk_documents(
                chunker_name, params, documents
            )
            if chunk_id:
                chunk_results.append(chunk_id)
                logger.info(f"✅ 완료: {chunk_id}")
            else:
                logger.error(f"❌ 실패: {chunker_name}")

        logger.info(f"🎉 청킹 단계 완료: {len(chunk_results)}개 결과")
        return chunk_results

    def _save_chunking_summary(self, chunk_results: List[str]) -> Dict[str, Any]:
        """청킹 결과 요약 저장"""

        # 청킹 결과 메타데이터 수집
        chunk_metadata = []
        for chunk_id in chunk_results:
            try:
                _, metadata = self.chunking_manager.load_chunking_result(chunk_id)
                chunk_metadata.append(
                    {
                        "chunk_id": chunk_id,
                        "chunker_name": metadata.get("chunker_name", "unknown"),
                        "total_chunks": metadata.get("total_chunks", 0),
                        "avg_chunk_size": metadata.get("avg_chunk_size", 0),
                        "processing_time": metadata.get("processing_time", 0),
                        "file_size_mb": metadata.get("file_size_mb", 0),
                        "timestamp": metadata.get("timestamp", ""),
                    }
                )
            except Exception as e:
                logger.warning(f"메타데이터 로드 실패: {chunk_id} - {e}")

        # CSV 저장
        if chunk_metadata:
            results_df = pd.DataFrame(chunk_metadata)
            results_csv = self.output_dir / "chunking_results.csv"
            results_df.to_csv(results_csv, index=False)
        else:
            results_csv = None

        # 요약 통계
        total_time = time.time() - self.state.start_time
        summary = {
            "execution_time": total_time,
            "total_documents": self.state.total_documents,
            "tokenization_results": self.state.tokenization_results,
            "chunking_results": len(chunk_results),
            "successful_chunks": len(chunk_results),
            "total_chunk_files": sum(
                md.get("total_chunks", 0) for md in chunk_metadata
            ),
            "avg_processing_time": np.mean(
                [md.get("processing_time", 0) for md in chunk_metadata]
            )
            if chunk_metadata
            else 0,
            "total_storage_mb": sum(md.get("file_size_mb", 0) for md in chunk_metadata),
            "results_file": str(results_csv) if results_csv else None,
        }

        # JSON 요약 저장
        summary_file = self.output_dir / "chunking_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"\n📊 === 청킹 결과 요약 ===")
        logger.info(f"⏱️ 총 소요시간: {total_time:.1f}초")
        logger.info(f"📄 처리된 문서: {self.state.total_documents}개")
        logger.info(f"🧩 생성된 청킹 조합: {len(chunk_results)}개")
        logger.info(f"📦 총 청크 개수: {summary['total_chunk_files']:,}개")
        logger.info(f"💾 총 저장 용량: {summary['total_storage_mb']:.1f}MB")
        logger.info(f"📁 결과 저장 위치: {self.output_dir}")

        return summary


# ===================================================================
# CLI 및 메인 실행
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Chunking Database Builder - Phase 2-1"
    )
    parser.add_argument("--output-dir", type=str, help="결과 저장 디렉토리")
    parser.add_argument("--max-chunks", type=int, help="최대 처리할 청킹 조합 수")
    parser.add_argument(
        "--chunkers",
        type=str,
        help="사용할 청커 (comma-separated: token,sentence,recursive)",
    )
    parser.add_argument("--config-summary", action="store_true", help="설정 요약 출력")
    parser.add_argument("--dry-run", action="store_true", help="설정 확인만 실행")
    parser.add_argument(
        "--scan-existing", action="store_true", help="기존 청크 파일 스캔"
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

    # 청킹 빌더 초기화
    runner = ChunkingExperimentRunner(args.output_dir)

    # 기존 청크 스캔
    if args.scan_existing:
        existing_chunks = runner.chunking_manager.scan_existing_chunks()
        logger.info(f"📊 기존 청크 파일: {len(existing_chunks)}개")
        for chunk_id in existing_chunks[:10]:  # 처음 10개만 표시
            logger.info(f"  • {chunk_id}")
        if len(existing_chunks) > 10:
            logger.info(f"  • ... 외 {len(existing_chunks) - 10}개")
        return 0

    # 청커 필터 처리
    chunker_filter = None
    if args.chunkers:
        chunker_filter = [c.strip() for c in args.chunkers.split(",")]
        logger.info(f"🔍 청커 필터: {chunker_filter}")

    # 실제 청킹 실험 실행
    try:
        result = runner.run_all_chunking_experiments(
            max_chunks=args.max_chunks,
            chunker_filter=chunker_filter,
        )

        if result["success"]:
            logger.info("🎉 청킹 실험 완료!")
            return 0
        else:
            logger.error(f"❌ 청킹 실험 실패: {result.get('error')}")
            return 1

    except Exception as e:
        logger.error(f"❌ 실행 오류: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
