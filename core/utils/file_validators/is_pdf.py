"""PDF 파일 형식 검증 유틸리티

이 모듈은 주어진 파일이 PDF(Portable Document Format) 형식인지 확인하는 기능을 제공합니다.
MIME 타입을 기반으로 파일 형식을 정확하게 판별합니다.
"""

from ..get_mime_type import get_mime_type

__all__ = ["is_pdf"]


def is_pdf(file_path: str) -> bool:
    """파일이 PDF 형식인지 확인합니다.

    Args:
        file_path: 검사할 파일의 경로

    Returns:
        PDF 형식이면 True, 그렇지 않으면 False

    Examples:
        >>> is_pdf("document.pdf")
        True
        >>> is_pdf("document.docx")
        False

    Note:
        내부적으로 get_mime_type 함수를 사용하여 MIME 타입을 확인합니다.
        PDF 파일의 MIME 타입은 'application/pdf'입니다.
    """
    # MIME 타입을 확인하여 PDF 파일 여부 판별
    return get_mime_type(file_path) == "application/pdf"
