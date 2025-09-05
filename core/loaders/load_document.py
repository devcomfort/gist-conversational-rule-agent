from pathlib import Path
from typing import List, Union, Dict, Optional, Tuple

from llama_index.core import Document

from core.utils.get_file_type import get_file_type
from .loader_map import LOADER_MAP


# MIME 타입과 확장자 매핑 (중앙 집중식 관리)
MIME_TO_EXTENSION: Dict[str, str] = {
    "application/pdf": ".pdf",
    "application/x-hwp": ".hwp",
    "application/vnd.hancom-hwp": ".hwp",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}


def _get_loader_for_file(
    file_type: Optional[Tuple[Optional[str], Optional[str]]],
) -> Optional[str]:
    """파일 타입 정보를 기반으로 적절한 로더 확장자를 반환합니다.

    Parameters
    ----------
    file_type : tuple[str | None, str | None] | None
        get_file_type()의 반환값 (mime_type, extension)

    Returns
    -------
    str | None
        LOADER_MAP에서 사용할 확장자 키, 지원하지 않는 형식이면 None
    """
    # Early return으로 None 케이스 처리
    if not file_type:
        return None

    # 구조분해할당으로 튜플 언패킹
    mime_type, ext = file_type

    # 1. MIME 타입 기반 매핑 (우선순위) - 단축 평가 활용
    if mime_type and (loader_ext := MIME_TO_EXTENSION.get(mime_type)):
        return loader_ext

    # 2. 확장자 기반 매핑 (보조) - 정규화된 확장자로 직접 검색
    if ext and (normalized_ext := ext.lower()) in LOADER_MAP:
        return normalized_ext

    return None


def load_document(path: Union[str, Path]) -> List[Document]:
    """파일 형식을 자동으로 감지하여 적절한 로더로 문서를 로드합니다.

    MIME 타입과 확장자를 기반으로 파일 형식을 판단하고, 해당하는 전용 로더를 호출하여
    문서를 Document 리스트로 변환합니다. 다양한 문서 형식을 지원합니다.

    Parameters
    ----------
    path : str | Path
        로드할 문서 파일의 경로

    Returns
    -------
    List[Document]
        로드된 문서 내용을 담은 Document 객체들의 리스트

    Raises
    ------
    FileNotFoundError
        파일이 존재하지 않는 경우
    ValueError
        지원하지 않는 파일 형식인 경우
    Exception
        파일 로딩 중 오류가 발생한 경우

    Examples
    --------
    >>> # PDF 파일 로드
    >>> documents = load_document('report.pdf')
    >>> len(documents) >= 1
    True

    >>> # HWP 파일 로드
    >>> documents = load_document('교원인사규정.hwp')
    >>> len(documents) >= 1
    True

    >>> # DOCX 파일 로드
    >>> documents = load_document('문서.docx')
    >>> len(documents) >= 1
    True

    >>> # PPTX 파일 로드
    >>> documents = load_document('프레젠테이션.pptx')
    >>> len(documents) >= 1
    True

    >>> # 지원하지 않는 형식
    >>> load_document('unknown.txt')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: 지원하지 않는 파일 형식...

    Notes
    -----
    지원되는 파일 형식:

    - **PDF (.pdf)**: Portable Document Format
    - **HWP (.hwp)**: 한글 워드프로세서 파일
    - **DOCX (.docx)**: Microsoft Word 문서
    - **PPTX (.pptx)**: Microsoft PowerPoint 프레젠테이션

    파일 형식 판별은 다음 순서로 수행됩니다:

    1. MIME 타입 기반 판별 (우선순위)
    2. 파일 확장자 기반 판별 (보조)

    로더는 `LOADER_MAP`에 정의된 LlamaIndex 리더 클래스를 사용합니다.
    """
    # Path 객체로 변환
    path_obj = Path(path)

    # 파일 존재 여부 확인
    if not path_obj.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    # 파일 타입 감지 및 구조분해할당
    file_type = get_file_type(path)
    loader_ext = _get_loader_for_file(file_type)

    if not loader_ext:
        # 지원하지 않는 형식인 경우 - 구조분해할당으로 오류 정보 추출
        mime_type, detected_ext = file_type if file_type else (None, None)
        display_mime = mime_type or "알 수 없음"
        display_ext = detected_ext or path_obj.suffix
        supported_formats = ", ".join(sorted(LOADER_MAP.keys()))

        raise ValueError(
            f"지원하지 않는 파일 형식입니다.\n"
            f"  파일: {path}\n"
            f"  MIME 타입: {display_mime}\n"
            f"  확장자: {display_ext}\n"
            f"  지원되는 형식: {supported_formats}"
        )

    # 로더 클래스 가져오기 및 인스턴스 생성 - 구조분해할당 활용
    loader_class = LOADER_MAP[loader_ext]
    loader = loader_class()

    try:
        # 문서 로드 실행 및 결과 검증을 한 번에
        if not (documents := loader.load_data(path_obj)):
            raise ValueError(f"문서 로드 결과가 비어있습니다: {path}")

        return documents

    except Exception as e:
        # 로딩 중 발생한 오류를 더 명확한 메시지와 함께 재발생
        raise Exception(
            f"문서 로드 중 오류가 발생했습니다.\n"
            f"  파일: {path}\n"
            f"  로더: {loader_class.__name__}\n"
            f"  오류: {str(e)}"
        ) from e
