"""Microsoft Word DOCX 파일 형식 검증 유틸리티

이 모듈은 주어진 파일이 Microsoft Word DOCX 형식인지 확인하는 기능을 제공합니다.
MIME 타입을 기반으로 파일 형식을 정확하게 판별합니다.
"""

from utils.get_mime_type import get_mime_type

__all__ = ["is_docx"]


def is_docx(file_path: str) -> bool:
    """파일이 Microsoft Word DOCX 형식인지 확인합니다.

    Args:
        file_path: 검사할 파일의 경로

    Returns:
        DOCX 형식이면 True, 그렇지 않으면 False

    Examples:
        >>> is_docx("document.docx")
        True
        >>> is_docx("document.pdf")
        False

    Note:
        내부적으로 get_mime_type 함수를 사용하여 MIME 타입을 확인합니다.
        DOCX 파일의 MIME 타입은 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'입니다.
    """
    # MIME 타입을 확인하여 DOCX 파일 여부 판별
    return (
        get_mime_type(file_path)
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
