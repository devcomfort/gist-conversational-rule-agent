from mimetypes import guess_type
from pathlib import Path
from typing import Union

from toolz import pipe


def get_mime_type(path: Union[str, Path]) -> Union[str, None]:
    """파일 경로의 확장자를 기반으로 MIME 타입을 추측합니다.

    문자열 또는 Path 객체를 입력받아 Path로 정규화한 후 MIME 타입을 추측합니다.
    toolz.pipe를 사용하여 함수형 프로그래밍 방식으로 구현되었습니다.

    Args:
        path (Union[str, Path]): 파일의 경로 또는 파일명

    Returns:
        Union[str, None]: MIME 타입 문자열 또는 None (확장자를 인식할 수 없는 경우)

    Examples:
        >>> get_mime_type('document.pdf')
        'application/pdf'
        >>> get_mime_type('교원인사규정.hwp')
        'application/x-hwp'
        >>> get_mime_type('image.jpg')
        'image/jpeg'
        >>> get_mime_type('text.txt')
        'text/plain'
        >>> get_mime_type('unknown.xyz')
        None

    Note:
        확장자 기반 추측이므로 파일의 실제 내용과 다를 수 있습니다.
    """

    def _guess_mime_type(path: Path) -> Union[str, None]:
        mime_type, _ = guess_type(str(path))
        return mime_type

    return pipe(
        path,
        Path,
        _guess_mime_type,
    )
