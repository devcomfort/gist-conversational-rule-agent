"""일반 문서 형식(DOCX/HWP/PDF/PPTX) 검증 유틸리티

여러 문서 형식을 하나의 헬퍼로 판별하기 위한 편의 함수입니다.
내부적으로 개별 형식 판별기(utils.is_*)들을 호출하여
유효한 경우 해당 형식을 나타내는 리터럴 문자열을, 유효하지 않으면 None을 반환합니다.

경고(중요): 문서 종류는 RAG 시스템이 "처리 가능한가/처리해야 하는가"를 기준으로 결정됩니다.
- 본 로직은 2025-08-30 패치 기준으로 유지되는 문서 형식만 포함합니다.
- 처리 정책/지원 형식 변경 시 이 모듈과 `utils/types.py`의 `DocumentKind`를 반드시 업데이트하세요.
"""

from core.utils.document_types import DocumentKind
from core.utils.file_validators import (
    is_docx,
    is_hwp,
    is_pdf,
    is_pptx,
)

__all__ = ["is_document"]


def is_document(file_path: str) -> DocumentKind | None:
    """파일이 지원되는 일반 문서 형식인지 확인합니다.

    지원 형식(2025-08-30 기준): PDF, HWP, DOCX, PPTX

    참고:
        - DOC/PPT 구형 형식은 llama-hub 기준 기본 리더 미지원으로 "문서"로 취급하지 않습니다.
          (필요 시 전용 리더 도입 후 정책 변경 가능)

    Args:
        file_path: 검사할 파일 경로

    Returns:
        DocumentKind | None: 형식 문자열("pdf" 등) 또는 None

    None이 반환되는 경우:
        - 지원 형식에 해당하지 않는 경우
        - 파일이 존재하지 않거나, 파일이 아닌 경로(예: 디렉토리)인 경우
        - 확장자/MIME을 인식할 수 없거나 감지에 실패한 경우
        - 파일 손상 등으로 하위 판별기들이 모두 False를 반환한 경우
    """
    if is_pdf(file_path):
        return "pdf"
    if is_hwp(file_path):
        return "hwp"
    if is_docx(file_path):
        return "docx"
    # DOC 구형 형식은 현재 미지원 처리 (llama-hub 리더 없음)
    # if is_doc(file_path):
    #     return "doc"
    if is_pptx(file_path):
        return "pptx"
    # PPT 구형 형식은 현재 미지원 처리 (llama-hub 리더 없음)
    # if is_ppt(file_path):
    #     return "ppt"
    return None
