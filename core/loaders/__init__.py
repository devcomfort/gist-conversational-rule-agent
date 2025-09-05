"""문서 로더 시스템.

이 모듈은 다양한 문서 형식을 로드하고 분석하는 통합 시스템을 제공합니다.
llama_index 기반의 문서 리더들을 활용하여 PDF, HWP, DOCX, PPTX 등의
문서를 처리할 수 있습니다.

주요 기능:
    • 문서 로딩: 다양한 형식의 문서를 자동으로 감지하고 로드
    • 경로 수집: 지원되는 문서 파일들의 경로를 효율적으로 수집
    • 문서 분석: 디렉토리 내 문서들의 통계 및 분류
    • CSV 내보내기: 분석 결과를 구조화된 데이터로 저장

지원되는 문서 형식:
    • PDF (.pdf): Portable Document Format
    • HWP (.hwp): 한글 워드프로세서 파일
    • DOCX (.docx): Microsoft Word 문서
    • PPTX (.pptx): Microsoft PowerPoint 프레젠테이션

사용 예시:
    >>> from core.loaders import load_document, collect_supported_document_paths
    >>> 
    >>> # 문서 로드
    >>> documents = load_document("example.pdf")
    >>> 
    >>> # 지원되는 문서 경로 수집
    >>> paths = collect_supported_document_paths("./documents")
    >>> 
    >>> # 문서 지원 여부 확인
    >>> from core.loaders import is_supported_document
    >>> is_supported = is_supported_document("example.pdf")

CLI 도구:
    이 모듈은 명령줄 인터페이스도 제공합니다:
    
    ```bash
    # 문서 분석 (기본: 모든 파일 포함)
    python -m core.loaders.inspect_document_paths ./documents
    
    # 지원 문서만 분석
    python -m core.loaders.inspect_document_paths ./documents --supported-only
    
    # CSV로 결과 저장
    python -m core.loaders.inspect_document_paths ./documents --save-csv
    ```
"""

from __future__ import annotations

# 핵심 문서 로딩 함수
from .load_document import load_document

# 문서 경로 수집 및 필터링
from .collect_supported_document_paths import collect_supported_document_paths
from .is_supported_document import is_supported_document

# 문서 분석 및 통계
from .inspect_document_paths import (
    inspect_supported_document_paths,
    inspect_all_document_paths,
    collect_all_document_file_info,
    save_inspection_to_csv,
)

# 타입 정의 (타입 별칭)
from .inspect_document_paths import DocumentExtensionCounts, DocumentCountMetric

# 설정 및 매핑
from .supported_document import SUPPORTED_DOCUMENT_KIND
from .loader_map import LOADER_MAP

# 공개 API 정의
__all__ = [
    # 문서 로딩
    "load_document",
    
    # 경로 수집 및 필터링
    "collect_supported_document_paths", 
    "is_supported_document",
    
    # 문서 분석 및 통계
    "inspect_supported_document_paths",
    "inspect_all_document_paths", 
    "collect_all_document_file_info",
    "save_inspection_to_csv",
    
    # 타입 정의
    "DocumentExtensionCounts",
    "DocumentCountMetric",
    
    # 설정 및 매핑
    "SUPPORTED_DOCUMENT_KIND",
    "LOADER_MAP",
]
