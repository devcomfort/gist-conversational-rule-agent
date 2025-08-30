from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

from utils.is_document import is_document


def collect_document_paths(root: Path | str) -> List[Path]:
    """루트 경로 하위에서 로드 가능한 문서 파일 경로를 재귀적으로 수집합니다.

    - 내부적으로 utils.is_document 을 사용하여 문서 여부를 판별합니다.
    - 반환 값은 정렬된 `Path` 리스트입니다.

    Args:
        root: 스캔을 시작할 디렉토리 경로

    Returns:
        List[Path]: 로드 가능한 문서 파일 경로 목록 (정렬됨)
    """
    root_path = Path(root)
    if not root_path.exists():
        return []

    results: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root_path):
        for name in filenames:
            file_path = Path(dirpath) / name
            try:
                if is_document(str(file_path)):
                    results.append(file_path)
            except Exception:
                # MIME 추측 실패 등은 무시하고 계속 진행
                continue

    results.sort()
    return results


__all__ = ["collect_document_paths"]

