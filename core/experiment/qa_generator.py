#!/usr/bin/env python3
"""
QA Dataset Generator
===================

연구 신뢰성을 보장하는 객관적 QA 데이터셋 생성 시스템.
청킹된 법률 문서로부터 표준화된 질의응답 쌍을 생성하고 품질을 평가합니다.

주요 특징:
- Mustache 템플릿 기반 일관된 프롬프트 생성
- 객관적 품질 평가 시스템
- 완전한 추적성과 재현성 보장
- 연구 목적에 최적화된 데이터 구조

사용법:
    python qa_dataset_generator.py
    python qa_dataset_generator.py --chunk-dir experiments/outputs/chunks
    python qa_dataset_generator.py --dry-run
"""

import os
import sys
import json
import time
import pickle
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict

import pystache
import numpy as np
import pandas as pd
from loguru import logger

# LLM 클라이언트 (litellm 사용)
try:
    import litellm
    from litellm import completion
except ImportError:
    logger.error("litellm이 설치되지 않았습니다: pip install litellm")
    sys.exit(1)

# 로컬 모듈
from experiment_configurations import EXPERIMENT_CONFIG

# ===================================================================
# 데이터 구조
# ===================================================================


@dataclass
class QAPair:
    """개별 QA 쌍 데이터"""

    question: str
    answer: str
    qa_type: str  # factual, definitional, procedural, conditional, exception
    difficulty: str  # basic, intermediate, advanced
    answer_source: str
    confidence: str  # high, medium, low


@dataclass
class QADataset:
    """QA 데이터셋 전체 구조"""

    qa_pairs: List[QAPair]
    metadata: Dict[str, Any]


@dataclass
class QAGenerationResult:
    """QA 생성 결과"""

    chunk_id: str
    success: bool
    qa_dataset: Optional[QADataset] = None
    generation_time: float = 0
    error_message: Optional[str] = None
    quality_score: Optional[float] = None
    quality_grade: Optional[str] = None


@dataclass
class QAEvaluationResult:
    """QA 평가 결과"""

    total_score: float
    grade: str  # A, B, C, D, F
    recommendation: str  # accept, revise, reject
    dimension_scores: Dict[str, Any]
    detailed_feedback: Dict[str, Any]
    qa_level_analysis: List[Dict[str, Any]]


# ===================================================================
# 템플릿 렌더링
# ===================================================================


class TemplateRenderer:
    """Mustache 템플릿 렌더링 관리"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.system_prompts_dir = self.base_dir / "system_prompts"
        self.templates_dir = self.base_dir / "templates"

        # 템플릿 캐시
        self._template_cache = {}

    def load_template(self, template_path: Path) -> str:
        """템플릿 파일 로드 (캐시 사용)"""
        if template_path not in self._template_cache:
            if not template_path.exists():
                raise FileNotFoundError(f"템플릿 파일 없음: {template_path}")

            with open(template_path, "r", encoding="utf-8") as f:
                self._template_cache[template_path] = f.read()

        return self._template_cache[template_path]

    def render_qa_generation_prompt(
        self,
        chunk_text: str,
        chunk_id: str,
        source_document: str,
        chunker_name: Optional[str] = None,
        generation_timestamp: Optional[str] = None,
    ) -> Tuple[str, str]:
        """QA 생성용 프롬프트 렌더링 (시스템 + 사용자)"""

        # 시스템 프롬프트
        system_template_path = (
            self.system_prompts_dir / "qa_generation_system_prompt.mustache"
        )
        system_prompt = self.load_template(system_template_path)

        # 사용자 프롬프트 데이터
        user_data = {
            "chunk_text": chunk_text,
            "chunk_id": chunk_id,
            "source_document": source_document,
            "chunker_name": chunker_name,
            "generation_timestamp": generation_timestamp or datetime.now().isoformat(),
        }

        # 사용자 프롬프트
        user_template_path = (
            self.templates_dir / "qa_generation_query_template.mustache"
        )
        user_template = self.load_template(user_template_path)
        user_prompt = pystache.render(user_template, user_data)

        return system_prompt, user_prompt

    def render_qa_evaluation_prompt(
        self,
        original_text: str,
        qa_data: str,  # JSON string
        chunk_id: str,
        source_document: str,
        qa_count: int,
        evaluation_timestamp: Optional[str] = None,
    ) -> Tuple[str, str]:
        """QA 평가용 프롬프트 렌더링 (시스템 + 사용자)"""

        # 시스템 프롬프트
        system_template_path = (
            self.system_prompts_dir / "qa_evaluation_system_prompt.mustache"
        )
        system_prompt = self.load_template(system_template_path)

        # 사용자 프롬프트 데이터
        user_data = {
            "original_text": original_text,
            "qa_data": qa_data,
            "chunk_id": chunk_id,
            "source_document": source_document,
            "qa_count": qa_count,
            "evaluation_timestamp": evaluation_timestamp or datetime.now().isoformat(),
        }

        # 사용자 프롬프트
        user_template_path = (
            self.templates_dir / "qa_evaluation_query_template.mustache"
        )
        user_template = self.load_template(user_template_path)
        user_prompt = pystache.render(user_template, user_data)

        return system_prompt, user_prompt


# ===================================================================
# LLM 클라이언트
# ===================================================================


class LLMClient:
    """LLM API 호출 관리"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "model": "gpt-oss-120b",
            "temperature": 0.1,
            "max_tokens": 2000,
            "timeout": 60,
        }

        # API 키 설정 확인
        if not os.getenv("OPENAI_API_KEY") and self.config["model"].startswith("gpt-"):
            logger.warning("OPENAI_API_KEY 환경변수가 설정되지 않았습니다")

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[str] = "json",
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        LLM API 호출

        Returns:
            (success, response_text, metadata)
        """
        try:
            start_time = time.time()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # litellm을 통한 API 호출
            response = completion(
                model=self.config["model"],
                messages=messages,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                timeout=self.config["timeout"],
            )

            response_time = time.time() - start_time
            response_text = response.choices[0].message.content

            metadata = {
                "model": self.config["model"],
                "response_time": response_time,
                "usage": response.usage.__dict__ if hasattr(response, "usage") else {},
                "timestamp": datetime.now().isoformat(),
            }

            return True, response_text, metadata

        except Exception as e:
            logger.error(f"LLM API 호출 실패: {e}")
            return False, str(e), {}


# ===================================================================
# QA 생성 관리자
# ===================================================================


class QAGenerationManager:
    """QA 데이터셋 생성 관리"""

    def __init__(self, output_dir: Path, llm_config: Optional[Dict[str, Any]] = None):
        self.output_dir = Path(output_dir)
        self.qa_dir = self.output_dir / EXPERIMENT_CONFIG["output"].get(
            "qa_subdir", "qa"
        )
        self.qa_dir.mkdir(parents=True, exist_ok=True)

        # 컴포넌트 초기화
        self.template_renderer = TemplateRenderer()
        self.llm_client = LLMClient(llm_config)

    def generate_qa_for_chunk(
        self, chunk_text: str, chunk_id: str, chunk_metadata: Dict[str, Any]
    ) -> QAGenerationResult:
        """개별 청크에 대한 QA 생성"""

        start_time = time.time()

        try:
            logger.info(f"🎯 QA 생성 시작: {chunk_id}")

            # 프롬프트 렌더링
            system_prompt, user_prompt = (
                self.template_renderer.render_qa_generation_prompt(
                    chunk_text=chunk_text,
                    chunk_id=chunk_id,
                    source_document=chunk_metadata.get("source_document", "unknown"),
                    chunker_name=chunk_metadata.get("chunker_name"),
                    generation_timestamp=datetime.now().isoformat(),
                )
            )

            # LLM 호출
            success, response, llm_metadata = self.llm_client.call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format="json",
            )

            if not success:
                return QAGenerationResult(
                    chunk_id=chunk_id,
                    success=False,
                    error_message=f"LLM 호출 실패: {response}",
                    generation_time=time.time() - start_time,
                )

            # JSON 파싱
            try:
                qa_json = json.loads(response)
                qa_dataset = self._parse_qa_json(qa_json, chunk_id)
            except json.JSONDecodeError as e:
                return QAGenerationResult(
                    chunk_id=chunk_id,
                    success=False,
                    error_message=f"JSON 파싱 실패: {e}",
                    generation_time=time.time() - start_time,
                )

            # 결과 저장
            result_id = self._save_qa_dataset(qa_dataset, chunk_metadata, llm_metadata)
            generation_time = time.time() - start_time

            logger.info(
                f"✅ QA 생성 완료: {chunk_id} ({len(qa_dataset.qa_pairs)}개, {generation_time:.1f}초)"
            )

            return QAGenerationResult(
                chunk_id=chunk_id,
                success=True,
                qa_dataset=qa_dataset,
                generation_time=generation_time,
            )

        except Exception as e:
            logger.error(f"❌ QA 생성 실패: {chunk_id} - {e}")
            return QAGenerationResult(
                chunk_id=chunk_id,
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time,
            )

    def _parse_qa_json(self, qa_json: Dict[str, Any], chunk_id: str) -> QADataset:
        """JSON 데이터를 QADataset 객체로 변환"""

        qa_pairs = []
        for qa_data in qa_json.get("qa_pairs", []):
            qa_pair = QAPair(
                question=qa_data.get("question", ""),
                answer=qa_data.get("answer", ""),
                qa_type=qa_data.get("qa_type", "factual"),
                difficulty=qa_data.get("difficulty", "basic"),
                answer_source=qa_data.get("answer_source", ""),
                confidence=qa_data.get("confidence", "medium"),
            )
            qa_pairs.append(qa_pair)

        metadata = qa_json.get("metadata", {})
        metadata["chunk_id"] = chunk_id
        metadata["generation_timestamp"] = datetime.now().isoformat()

        return QADataset(qa_pairs=qa_pairs, metadata=metadata)

    def _save_qa_dataset(
        self,
        qa_dataset: QADataset,
        chunk_metadata: Dict[str, Any],
        llm_metadata: Dict[str, Any],
    ) -> str:
        """QA 데이터셋 저장"""

        chunk_id = qa_dataset.metadata.get("chunk_id")

        # QA 데이터 저장 (JSON)
        qa_file = self.qa_dir / f"{chunk_id}_qa.json"
        qa_data = {
            "qa_pairs": [asdict(qa) for qa in qa_dataset.qa_pairs],
            "metadata": qa_dataset.metadata,
        }

        with open(qa_file, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, ensure_ascii=False, indent=2)

        # 생성 메타데이터 저장
        generation_metadata = {
            "chunk_id": chunk_id,
            "chunk_metadata": chunk_metadata,
            "llm_metadata": llm_metadata,
            "qa_count": len(qa_dataset.qa_pairs),
            "file_size_mb": qa_file.stat().st_size / (1024 * 1024),
            "generation_timestamp": datetime.now().isoformat(),
        }

        metadata_file = self.qa_dir / f"{chunk_id}_generation_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(generation_metadata, f, ensure_ascii=False, indent=2)

        return chunk_id


# ===================================================================
# QA 품질 검증자
# ===================================================================


class QualityValidator:
    """QA 품질 검증 시스템"""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        self.template_renderer = TemplateRenderer()
        self.llm_client = LLMClient(llm_config)

    def evaluate_qa_quality(
        self,
        original_text: str,
        qa_dataset: QADataset,
        chunk_id: str,
        source_document: str,
    ) -> QAEvaluationResult:
        """QA 품질 평가 수행"""

        try:
            logger.info(f"🔍 품질 평가 시작: {chunk_id}")

            # QA 데이터를 JSON 문자열로 변환
            qa_json_str = json.dumps(
                {
                    "qa_pairs": [asdict(qa) for qa in qa_dataset.qa_pairs],
                    "metadata": qa_dataset.metadata,
                },
                ensure_ascii=False,
                indent=2,
            )

            # 평가 프롬프트 렌더링
            system_prompt, user_prompt = (
                self.template_renderer.render_qa_evaluation_prompt(
                    original_text=original_text,
                    qa_data=qa_json_str,
                    chunk_id=chunk_id,
                    source_document=source_document,
                    qa_count=len(qa_dataset.qa_pairs),
                    evaluation_timestamp=datetime.now().isoformat(),
                )
            )

            # LLM 호출
            success, response, llm_metadata = self.llm_client.call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format="json",
            )

            if not success:
                logger.error(f"평가 LLM 호출 실패: {response}")
                return self._create_default_evaluation(chunk_id, error=response)

            # 평가 결과 파싱
            try:
                eval_json = json.loads(response)
                evaluation = self._parse_evaluation_json(eval_json)

                logger.info(
                    f"✅ 품질 평가 완료: {chunk_id} (점수: {evaluation.total_score:.1f}, 등급: {evaluation.grade})"
                )
                return evaluation

            except json.JSONDecodeError as e:
                logger.error(f"평가 결과 JSON 파싱 실패: {e}")
                return self._create_default_evaluation(chunk_id, error=str(e))

        except Exception as e:
            logger.error(f"품질 평가 실패: {chunk_id} - {e}")
            return self._create_default_evaluation(chunk_id, error=str(e))

    def _parse_evaluation_json(self, eval_json: Dict[str, Any]) -> QAEvaluationResult:
        """평가 JSON을 결과 객체로 변환"""

        overall = eval_json.get("overall_evaluation", {})

        return QAEvaluationResult(
            total_score=float(overall.get("total_score", 0)),
            grade=overall.get("grade", "F"),
            recommendation=overall.get("recommendation", "reject"),
            dimension_scores=eval_json.get("dimension_scores", {}),
            detailed_feedback=eval_json.get("detailed_feedback", {}),
            qa_level_analysis=eval_json.get("qa_level_analysis", []),
        )

    def _create_default_evaluation(
        self, chunk_id: str, error: str
    ) -> QAEvaluationResult:
        """기본/에러 평가 결과 생성"""

        return QAEvaluationResult(
            total_score=0.0,
            grade="F",
            recommendation="reject",
            dimension_scores={},
            detailed_feedback={"error": error},
            qa_level_analysis=[],
        )


# ===================================================================
# 메인 QA 데이터셋 빌더
# ===================================================================


class QADatasetBuilder:
    """QA 데이터셋 빌더 - 전체 파이프라인 관리"""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir or EXPERIMENT_CONFIG["output"]["base_dir"])
        self.chunk_dir = self.output_dir / EXPERIMENT_CONFIG["output"].get(
            "chunking_subdir", "chunks"
        )

        # 관리자들 초기화
        self.qa_generator = QAGenerationManager(self.output_dir, llm_config)
        self.quality_validator = QualityValidator(llm_config)

        # 결과 추적
        self.results: List[QAGenerationResult] = []

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

    def load_chunk_data(self, chunk_id: str) -> Tuple[List[Any], Dict[str, Any]]:
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

    def generate_qa_for_all_chunks(
        self,
        chunk_ids: Optional[List[str]] = None,
        max_chunks: Optional[int] = None,
        include_quality_evaluation: bool = True,
    ) -> Dict[str, Any]:
        """모든 청크에 대한 QA 생성"""

        logger.info("🚀 QA 데이터셋 생성 시작!")

        # 청크 ID 결정
        if chunk_ids is None:
            chunk_ids = self.scan_available_chunks()

        if max_chunks:
            chunk_ids = chunk_ids[:max_chunks]

        logger.info(f"📋 처리할 청크: {len(chunk_ids)}개")

        # 청크별 QA 생성
        successful_results = []
        failed_results = []

        for i, chunk_id in enumerate(chunk_ids, 1):
            logger.info(f"🎯 처리 중 [{i}/{len(chunk_ids)}]: {chunk_id}")

            try:
                # 청크 데이터 로드
                chunks, chunk_metadata = self.load_chunk_data(chunk_id)

                # 청크 텍스트 결합
                chunk_text = "\n\n".join(
                    [
                        chunk.page_content
                        if hasattr(chunk, "page_content")
                        else str(chunk)
                        for chunk in chunks
                    ]
                )

                # QA 생성
                result = self.qa_generator.generate_qa_for_chunk(
                    chunk_text=chunk_text,
                    chunk_id=chunk_id,
                    chunk_metadata=chunk_metadata,
                )

                # 품질 평가 (선택적)
                if result.success and include_quality_evaluation and result.qa_dataset:
                    evaluation = self.quality_validator.evaluate_qa_quality(
                        original_text=chunk_text,
                        qa_dataset=result.qa_dataset,
                        chunk_id=chunk_id,
                        source_document=chunk_metadata.get(
                            "source_document", "unknown"
                        ),
                    )
                    result.quality_score = evaluation.total_score
                    result.quality_grade = evaluation.grade

                self.results.append(result)

                if result.success:
                    successful_results.append(result)
                    logger.info(f"✅ 완료: {chunk_id}")
                else:
                    failed_results.append(result)
                    logger.error(f"❌ 실패: {chunk_id} - {result.error_message}")

            except Exception as e:
                logger.error(f"❌ 청크 처리 실패: {chunk_id} - {e}")
                failed_results.append(
                    QAGenerationResult(
                        chunk_id=chunk_id, success=False, error_message=str(e)
                    )
                )

        # 결과 요약 저장
        summary = self._save_generation_summary(successful_results, failed_results)

        logger.info(f"\n🎉 QA 데이터셋 생성 완료!")
        logger.info(f"✅ 성공: {len(successful_results)}개")
        logger.info(f"❌ 실패: {len(failed_results)}개")
        if include_quality_evaluation and successful_results:
            avg_quality = np.mean(
                [r.quality_score for r in successful_results if r.quality_score]
            )
            logger.info(f"📊 평균 품질 점수: {avg_quality:.1f}점")

        return summary

    def _save_generation_summary(
        self,
        successful_results: List[QAGenerationResult],
        failed_results: List[QAGenerationResult],
    ) -> Dict[str, Any]:
        """생성 결과 요약 저장"""

        # CSV 형태로 결과 저장
        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "chunk_id": result.chunk_id,
                    "success": result.success,
                    "qa_count": len(result.qa_dataset.qa_pairs)
                    if result.qa_dataset
                    else 0,
                    "generation_time": result.generation_time,
                    "quality_score": result.quality_score,
                    "quality_grade": result.quality_grade,
                    "error_message": result.error_message or "",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        results_df = pd.DataFrame(results_data)
        results_csv = self.output_dir / "qa_generation_results.csv"
        results_df.to_csv(results_csv, index=False)

        # 요약 통계
        summary = {
            "execution_time": time.time(),
            "total_chunks": len(self.results),
            "successful_generations": len(successful_results),
            "failed_generations": len(failed_results),
            "success_rate": len(successful_results) / len(self.results)
            if self.results
            else 0,
            "total_qa_pairs": sum(
                len(r.qa_dataset.qa_pairs) for r in successful_results if r.qa_dataset
            ),
            "average_quality_score": np.mean(
                [
                    r.quality_score
                    for r in successful_results
                    if r.quality_score is not None
                ]
            )
            if successful_results
            else None,
            "results_file": str(results_csv),
        }

        # JSON 요약 저장
        summary_file = self.output_dir / "qa_generation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary


# ===================================================================
# CLI 및 메인 실행
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="QA Dataset Generator")
    parser.add_argument("--chunk-dir", type=str, help="청크 데이터 디렉토리 경로")
    parser.add_argument("--output-dir", type=str, help="결과 저장 디렉토리")
    parser.add_argument("--max-chunks", type=int, help="최대 처리할 청크 수")
    parser.add_argument("--chunk-ids", type=str, nargs="+", help="특정 청크 ID만 처리")
    parser.add_argument("--no-evaluation", action="store_true", help="품질 평가 생략")
    parser.add_argument("--dry-run", action="store_true", help="설정 확인만 실행")
    parser.add_argument(
        "--model", type=str, default="gpt-oss-120b", help="사용할 LLM 모델"
    )

    args = parser.parse_args()

    # LLM 설정
    llm_config = {
        "model": args.model,
        "temperature": 0.1,
        "max_tokens": 2000,
        "timeout": 60,
    }

    # QA 빌더 초기화
    builder = QADatasetBuilder(output_dir=args.output_dir, llm_config=llm_config)

    # 청크 디렉토리 업데이트
    if args.chunk_dir:
        builder.chunk_dir = Path(args.chunk_dir)

    # Dry run
    if args.dry_run:
        chunk_ids = builder.scan_available_chunks()
        logger.info(f"✅ 설정 확인 완료")
        logger.info(f"📁 청크 디렉토리: {builder.chunk_dir}")
        logger.info(f"📁 출력 디렉토리: {builder.output_dir}")
        logger.info(f"📊 사용 가능한 청크: {len(chunk_ids)}개")
        logger.info(f"🤖 LLM 모델: {llm_config['model']}")
        return 0

    # 실제 QA 생성 실행
    try:
        result = builder.generate_qa_for_all_chunks(
            chunk_ids=args.chunk_ids,
            max_chunks=args.max_chunks,
            include_quality_evaluation=not args.no_evaluation,
        )

        if result["success_rate"] > 0.8:
            logger.info("🎉 QA 생성 성공!")
            return 0
        else:
            logger.warning(
                f"⚠️ QA 생성 부분 성공 (성공률: {result['success_rate']:.1%})"
            )
            return 1

    except Exception as e:
        logger.error(f"❌ QA 생성 실패: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
