from llama_index.core.schema import Document
from typing import List, Tuple, Dict, Any, Optional, cast
from pathlib import Path
from joblib import Parallel, delayed
from loguru import logger

from loaders import load_document, collect_document_paths
from utils import log_duration
from refiners.refine_as_markdown import refine_as_markdown


@log_duration("경로 수집")
def collect_paths(root_dir: Path | str) -> List[Path]:
    paths: List[Path] = list(collect_document_paths(root_dir, lazy=False))
    logger.info("감지 경로: {}개", len(paths))
    return paths


def load_one(
    document_path: Path,
) -> Tuple[Optional[List[Document]], Optional[Dict[str, Any]]]:
    """단일 파일 로드. 성공 시 (documents, None), 실패 시 (None, {path, reason})."""
    try:
        documents = load_document(document_path)
        return documents, None
    except Exception as exception:
        return None, {"path": str(document_path), "reason": str(exception)}


@log_duration("문서 로드")
def load_documents(
    paths: List[Path],
    *,
    num_workers: int,
    backend: str,
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    if not paths:
        logger.info("로드 대상 경로가 없습니다.")
        return [], []

    tasks = [delayed(load_one)(p) for p in paths]
    results = cast(
        List[Tuple[Optional[List[Document]], Optional[Dict[str, Any]]]],
        Parallel(n_jobs=num_workers, backend=backend)(tasks),
    )

    loaded_documents: List[Document] = []
    failed_details: List[Dict[str, Any]] = []
    for (documents, error_detail), path in zip(results, paths):
        if documents is not None:
            loaded_documents.extend(documents)
        elif error_detail is not None:
            failed_details.append(error_detail)

    logger.info(
        "로드 결과: 성공 {}개, 실패 {}개", len(loaded_documents), len(failed_details)
    )
    return loaded_documents, failed_details


def refine_one(document: Document) -> Document:
    """문서 텍스트를 마크다운으로 정제하여 새 Document로 반환."""
    refined_text = refine_as_markdown(document)
    return Document(text=refined_text, metadata=document.metadata)


@log_duration("정규화")
def normalize_documents(
    documents: List[Document],
    *,
    num_workers: int,
    backend: str,
) -> List[Document]:
    if not documents:
        logger.info("정규화 대상 문서가 없습니다.")
        return []

    tasks = [delayed(refine_one)(d) for d in documents]
    normalized = cast(
        List[Document],
        Parallel(n_jobs=num_workers, backend=backend)(tasks),
    )
    logger.info("정규화 결과: {}개", len(normalized))
    return normalized


def run_normalization_pipeline(
    root_dir: Path | str,
    *,
    num_workers: int = 4,
    prefer_threads: bool = True,
    log_level: str = "INFO",
) -> Dict[str, Any]:
    """경로 수집 → 문서 로드 → 마크다운 정규화 파이프라인.

    - num_workers로 병렬화. 기본 backend="threading" 사용.
    - 반환: {loaded_count, failed_loads, normalized_count, normalized_documents}

    Args:
        root_dir: 스캔 시작 디렉터리.
        num_workers: 병렬 워커 수.
        prefer_threads: True면 threading, False면 loky 백엔드 사용.
        log_level: 콘솔(stderr) 로그 레벨. 지원 값은
            "TRACE"|"DEBUG"|"INFO"|"SUCCESS"|"WARNING"|"ERROR"|"CRITICAL".
            내부에서 logger.remove() 후 stderr 싱크를 지정 레벨로 재등록합니다.
    """
    backend = "threading" if prefer_threads else "loky"
    # 로깅 레벨 구성
    try:
        import sys as _sys

        logger.remove()
        logger.add(_sys.stderr, level=str(log_level).upper())
    except Exception:
        pass

    logger.info(
        "파이프라인 시작: root_dir={}, workers={}, backend={}, level={}",
        root_dir,
        num_workers,
        backend,
        log_level,
    )

    paths = collect_paths(root_dir)
    loaded_documents, failed_loads = load_documents(
        paths, num_workers=num_workers, backend=backend
    )
    normalized_documents = normalize_documents(
        loaded_documents, num_workers=num_workers, backend=backend
    )

    return {
        "loaded_count": len(loaded_documents),
        "failed_loads": failed_loads,
        "normalized_count": len(normalized_documents),
        "normalized_documents": normalized_documents,
    }


if __name__ == "__main__":
    # 예시 실행: 환경변수 또는 고정 경로로 루트 지정 가능
    import os

    root = os.environ.get("DATASET_ROOT", "./rules")
    result = run_normalization_pipeline(root, num_workers=4, prefer_threads=True)
    print(
        {
            "loaded_count": result["loaded_count"],
            "failed_count": len(result["failed_loads"]),
            "normalized_count": result["normalized_count"],
        }
    )
