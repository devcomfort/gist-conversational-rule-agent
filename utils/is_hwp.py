"""한글 파일(HWP) 형식 검증 유틸리티

이 모듈은 주어진 파일이 한글 워드프로세서(HWP) 형식인지 확인하는 기능을 제공합니다.
MIME 타입을 기반으로 파일 형식을 정확하게 판별합니다.
"""

from utils.get_mime_type import get_mime_type

__all__ = ["is_hwp"]


def is_hwp(file_path: str) -> bool:
    """파일이 한글(HWP) 형식인지 확인합니다.

    Args:
        file_path: 검사할 파일의 경로

    Returns:
        HWP 형식이면 True, 그렇지 않으면 False

    Examples:
        >>> is_hwp("document.hwp")
        True
        >>> is_hwp("document.docx")
        False

    Note:
        내부적으로 get_mime_type 함수를 사용하여 MIME 타입을 확인합니다.
        한글 파일의 MIME 타입은 'application/x-hwp' 또는 'application/vnd.hancom-hwp'입니다.
    """
    # MIME 타입을 확인하여 한글 파일 여부 판별
    mime_type = get_mime_type(file_path)
    return mime_type in ["application/x-hwp", "application/vnd.hancom-hwp"]
