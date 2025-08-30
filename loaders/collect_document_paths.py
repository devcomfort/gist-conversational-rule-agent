from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, List, Literal, overload

from loguru import logger
from utils.is_document import is_document


def _iter_document_paths(root_path: Path) -> Iterator[Path]:
    """내부 제너레이터: 문서 파일 경로를 지연 평가로 반환합니다.

    실패 케이스는 DEBUG 레벨로 로깅합니다.
    """
    if not root_path.exists():
        logger.debug("collect_document_paths: root not found: {}", root_path)
        return

    try:
        for dirpath, _dirnames, filenames in os.walk(root_path):
            for name in filenames:
                file_path = Path(dirpath) / name
                try:
                    kind = is_document(str(file_path))
                    if kind is not None:
                        logger.debug(
                            "collect_document_paths: detected kind={} for {}",
                            kind,
                            file_path,
                        )
                        yield file_path
                except Exception as e:
                    # MIME 추측 실패 등 모든 예외를 DEBUG로 기록하고 계속 진행
                    logger.debug(
                        "collect_document_paths: detection failed for {}: {}",
                        file_path,
                        e,
                    )
                    continue
    except Exception as e:
        logger.debug("collect_document_paths: walk failed for {}: {}", root_path, e)
        return


@overload
def collect_document_paths(root: Path | str) -> Iterable[Path]: ...


@overload
def collect_document_paths(
    root: Path | str, *, lazy: Literal[True]
) -> Iterable[Path]: ...


@overload
def collect_document_paths(
    root: Path | str, *, lazy: Literal[False] = ...
) -> List[Path]: ...


def collect_document_paths(
    root: Path | str,
    *,
    lazy: bool = True,
) -> Iterable[Path] | List[Path]:
    """루트 경로 하위에서 로드 가능한 문서 파일 경로를 수집합니다.

    - 내부적으로 `utils.is_document`을 사용하여 문서 여부를 판별합니다.
    - 실패 케이스(권한/파싱/탐색 오류 등)는 `loguru`의 DEBUG 레벨로 로깅합니다.
    - `lazy=True`이면 지연 평가 가능한 Iterable을 반환하고, `lazy=False`이면 정렬된 리스트를 반환합니다.

    Args:
        root: 스캔을 시작할 디렉토리 경로
        lazy: True 시 Iterable[Path] (지연 평가), False 시 List[Path] (정렬)

    Returns:
        Iterable[Path] | List[Path]: 문서 파일 경로 컬렉션
    """
    root_path = Path(root)

    document_paths = _iter_document_paths(root_path)

    return document_paths if lazy else list(document_paths)


__all__ = ["collect_document_paths"]
