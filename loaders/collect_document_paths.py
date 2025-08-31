from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, List, Literal, overload

from utils.detection import is_document


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
                    kind = is_document(str(file_path))
                    if kind is not None:
                        fp.write(f"DETECTED\t{kind}\t{file_path}\n")
                    else:
                        fp.write(f"IGNORED\t{file_path}\n")
                except Exception as e:
                    fp.write(f"ERROR\t{file_path}\t{e}\n")
                finally:
                    written += 1
    return written


def _iter_document_paths(root_path: Path) -> Iterator[Path]:
    """내부 제너레이터: 문서 파일 경로를 지연 평가로 반환합니다.

    이 함수는 로깅이나 집계를 수행하지 않습니다. 문서 판별은
    `utils.is_document` 결과가 존재하는 파일만 대상으로 합니다.
    """
    if not root_path.exists():
        return

    try:
        for dirpath, _dirnames, filenames in os.walk(root_path):
            for name in filenames:
                file_path = Path(dirpath) / name
                try:
                    kind = is_document(str(file_path))
                    if kind is not None:
                        yield file_path
                except Exception:
                    continue
    except Exception:
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
