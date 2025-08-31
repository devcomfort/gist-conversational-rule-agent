#!/usr/bin/env python3
"""
GIST Rules Database - Loader-Compatible File Scanner
====================================================

1) 이 스크립트는 rules 디렉토리 하위에서 loaders 패키지를 통해 로드 가능한
   파일들(PDF/HWP/DOCX/PPTX)의 경로만을 선별하여 출력합니다.

2) 현재 단계에서는 로드 가능한 파일 경로 수집까지만 수행합니다.

사용법:
    uv run python3 build_rule_database.py

참고:
    - 형식 판별 유틸: utils.is_pdf / is_hwp / is_docx / is_pptx
    - 로더 패키지: loaders.load_pdf/load_hwp/load_docx/load_pptx
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

# 형식 판별 유틸리티
from utils.validators import is_pdf, is_hwp, is_docx, is_pptx


RULES_DIR = Path("rules")


def iter_rule_files(root: Path) -> List[Path]:
    """rules 디렉토리 하위 모든 파일 경로 리스트를 반환합니다."""
    if not root.exists():
        return []
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            files.append(Path(dirpath) / name)
    return files


def is_loader_compatible(path: Path) -> bool:
    """loaders 패키지를 통해 로드 가능한 형식인지 검사합니다."""
    path_str = str(path)
    try:
        return (
            is_pdf(path_str)
            or is_hwp(path_str)
            or is_docx(path_str)
            or is_pptx(path_str)
        )
    except Exception:
        return False


def collect_loader_compatible_paths(root: Path) -> List[Path]:
    """rules 디렉토리에서 로드 가능한 파일 경로를 모두 수집합니다."""
    results: List[Path] = []
    for file_path in iter_rule_files(root):
        if is_loader_compatible(file_path):
            results.append(file_path)
    results.sort()
    return results


def main() -> None:
    print("🚀 Loader-Compatible 파일 스캐너 시작")
    print(f"📁 스캔 경로: {RULES_DIR.resolve()}")

    compatible_paths = collect_loader_compatible_paths(RULES_DIR)
    print(f"🔎 로드 가능한 파일: {len(compatible_paths)}개")

    for p in compatible_paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()
