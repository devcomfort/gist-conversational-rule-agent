"""JSON 파일 감지 유틸리티.

이 모듈은 파일이 JSON 파일인지 확인하는 기능을 제공합니다.
MIME 타입을 기반으로 파일의 JSON 형식 여부를 판단합니다.
"""

from utils.mime import get_mime_type

__all__ = ["is_json"]


def is_json(file_path: str) -> bool:
    """파일이 JSON 파일인지 확인합니다.

    주어진 파일 경로의 파일이 JSON 형식인지 MIME 타입을 통해 확인합니다.

    Args:
        file_path (str): 확인할 파일의 경로

    Returns:
        bool: JSON 파일이면 True, 아니면 False

    Example:
        >>> is_json("data.json")
        True
        >>> is_json("document.pdf")
        False
    """
    return get_mime_type(file_path) == "application/json"
