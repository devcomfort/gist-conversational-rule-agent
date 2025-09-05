from pathlib import Path
from typing import Union
from core.utils.get_file_type import get_file_type
from .supported_document import SUPPORTED_DOCUMENT_KIND


def is_supported_document(path: Union[Path, str]) -> bool:
    """파일이 지원되는 문서 타입인지 확인합니다.

    이 함수는 파일의 확장자를 기반으로 현재 문서 로더 시스템에서
    지원하는 문서 형식인지 여부를 판단합니다.

    Parameters
    ----------
    path : Path | str
        확인할 파일의 경로

    Returns
    -------
    bool
        지원되는 문서 타입이면 True, 아니면 False

    Examples
    --------
    >>> is_supported_document("document.pdf")
    True
    >>> is_supported_document("document.hwp")
    True
    >>> is_supported_document("document.txt")
    False
    >>> is_supported_document("presentation.pptx")
    True

    Notes
    -----
    지원되는 문서 형식:

    - **PDF (.pdf)**: Portable Document Format
    - **HWP (.hwp)**: 한글 워드프로세서 파일
    - **DOCX (.docx)**: Microsoft Word 문서
    - **PPTX (.pptx)**: Microsoft PowerPoint 프레젠테이션

    파일 형식 판별은 다음과 같이 수행됩니다:

    1. `get_file_type()` 함수를 사용하여 MIME 타입과 확장자를 감지
    2. 감지된 확장자가 `SUPPORTED_DOCUMENT_KIND`에 포함되는지 확인
    3. 확장자 기반 검사를 통해 지원 여부 결정

    이 함수는 `collect_supported_document_paths()` 등에서
    파일 필터링 용도로 사용됩니다.

    See Also
    --------
    get_file_type : 파일의 MIME 타입과 확장자를 감지하는 함수
    SUPPORTED_DOCUMENT_KIND : 지원되는 문서 확장자 집합
    collect_supported_document_paths : 지원되는 문서 경로를 수집하는 함수
    """
    _, ext = get_file_type(path)
    return ext in SUPPORTED_DOCUMENT_KIND


if __name__ == "__main__":
    # 다양한 파일 타입 테스트
    test_files = [
        "document.pdf",
        "document.hwp",
        "document.docx",
        "presentation.pptx",
        "document.txt",
        "image.jpg",
        "data.csv",
        "script.py",
    ]

    print("=== is_supported_document 함수 테스트 ===")
    for file_path in test_files:
        result = is_supported_document(file_path)
        status = "✅ 지원됨" if result else "❌ 지원안됨"
        print(f"{file_path:>15} → {status}")

    print("\n=== 함수 docstring 확인 ===")
    print(f"함수명: {is_supported_document.__name__}")
    docstring = is_supported_document.__doc__
    if docstring:
        print(f"docstring 길이: {len(docstring)} 문자")
        print(f"docstring 첫 줄: {docstring.split(chr(10))[0]}")
    else:
        print("docstring이 없습니다.")
