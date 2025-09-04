from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, List, Literal, overload

from core.utils.get_file_type import get_file_type
from .supported_document import SUPPORTED_DOCUMENT_KIND


def write_detection_log(
    root: Path | str,
    *,
    output_file: Path | str,
) -> int:
    """지정한 루트에서 감지된 모든 경로와 분류 결과를 파일로 저장합니다.

    포맷: TSV
      DETECTED\t{kind}\t{path}
      IGNORED\t{path}
      ERROR\t{path}\t{message}

    Returns:
      기록된 총 레코드 수(int)
    """
    root_path = Path(root)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as fp:
        if not root_path.exists():
            return written
        for dirpath, _dirnames, filenames in os.walk(root_path):
            for name in filenames:
                file_path = Path(dirpath) / name
                try:
                    file_type: tuple[str | None, str | None] | None = get_file_type(
                        str(file_path)
                    )
                    ext = file_type[1] if file_type else None
                    if ext in SUPPORTED_DOCUMENT_KIND:
                        fp.write(f"DETECTED\t{ext}\t{file_path}\n")
                    else:
                        fp.write(f"IGNORED\t{file_path}\n")
                except Exception as e:
                    fp.write(f"ERROR\t{file_path}\t{e}\n")
                finally:
                    written += 1
    return written


def _iter_document_paths(root_path: Path) -> Iterator[Path]:
    """내부 제너레이터: 문서 파일 경로를 지연 평가로 반환합니다.

    이 함수는 로깅이나 집계를 수행하지 않습니다. 파일 타입은
    `core.utils.get_file_type.get_file_type`으로 추정하며,
    `SUPPORTED_DOCUMENT_KIND`에 포함된 확장자만 대상으로 합니다.
    """
    # Guard clause: 존재하지 않으면 빈 이터레이터 반환
    if not root_path.exists():
        return iter(())

    def _is_supported_document(path: Path) -> bool:
        try:
            file_type = get_file_type(str(path))
            ext = file_type[1] if file_type else None
            return ext in SUPPORTED_DOCUMENT_KIND
        except Exception:
            return False

    # Generator expression: 안전한 판별 함수를 이용해 지연 필터링
    return (
        path
        for dirpath, _dirnames, filenames in os.walk(root_path)
        for name in filenames
        for path in [Path(dirpath) / name]
        if _is_supported_document(path)
    )


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

    - 내부적으로 `get_file_type()`과 `SUPPORTED_DOCUMENT_KIND`를 사용하여 지원 확장자만 필터링합니다.
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
