"""일반 문서 형식(DOC/DOCX/HWP/PDF/PPT/PPTX/TXT/JSON) 검증 유틸리티

여러 문서 형식을 하나의 헬퍼로 판별하기 위한 편의 함수입니다.
내부적으로 개별 형식 판별기(utils.is_*)들을 호출하여 True/False를 반환합니다.
"""

from .is_doc import is_doc
from .is_docx import is_docx
from .is_hwp import is_hwp
from .is_pdf import is_pdf
from .is_ppt import is_ppt
from .is_pptx import is_pptx
from .is_txt import is_txt
from .is_json import is_json

__all__ = ["is_document"]


def is_document(file_path: str) -> bool:
    """파일이 지원되는 일반 문서 형식인지 확인합니다.

    지원 형식: DOC, DOCX, HWP, PDF, PPT, PPTX, TXT, JSON

    Args:
        file_path: 검사할 파일 경로

    Returns:
        True: 지원 형식 중 하나에 해당
        False: 그 외 형식
    """
    try:
        return (
            is_pdf(file_path)
            or is_hwp(file_path)
            or is_docx(file_path)
            or is_doc(file_path)
            or is_pptx(file_path)
            or is_ppt(file_path)
            or is_txt(file_path)
            or is_json(file_path)
        )
    except Exception:
        return False


