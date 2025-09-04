from mimetypes import guess_type, guess_extension
from pathlib import Path
from typing import Union, Tuple, Optional


def get_file_type(
    path: Union[str, Path],
) -> Tuple[Optional[str], Optional[str]]:
    """파일의 MIME 타입과 대표 확장자를 추정합니다.

    - 표준 라이브러리 `mimetypes.guess_type`과 `mimetypes.guess_extension`을 사용합니다.
    - 입력은 `str | Path`를 허용하며 내부적으로 문자열로 정규화합니다.

    Parameters
    ----------
    path : str | Path
        파일의 경로 또는 파일명

    Returns
    -------
    (mime_type, ext)
        - mime_type: MIME 타입 문자열 또는 None
        - ext: 대표 확장자(점 포함) 또는 None

    Examples
    --------
    >>> get_file_type('document.pdf')
    ('application/pdf', '.pdf')
    >>> get_file_type('교원인사규정.hwp')
    ('application/x-hwp', '.hwp')
    >>> get_file_type('unknown.xyz')
    (None, None)

    Notes
    -----
    - 확장자 기반 추정이므로 파일의 실제 콘텐츠와 다를 수 있습니다.
    - 일부 MIME 문자열은 세미콜론(;)과 인코딩 정보가 포함될 수 있습니다. 필요 시 후처리 하세요.
    """

    mime, _encoding = guess_type(str(path))
    ext = guess_extension(mime or "") if mime else None
    return mime, ext


if __name__ == "__main__":
    print(get_file_type("document.pdf"))
    print(get_file_type("document.hwp"))
    print(get_file_type("document.kor"))
