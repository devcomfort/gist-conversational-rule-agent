from pathlib import Path
from typing import List, Union

from llama_index.core import Document

from utils.is_docx import is_docx
from utils.is_hwp import is_hwp
from utils.is_pdf import is_pdf
from utils.is_pptx import is_pptx
from utils.get_mime_type import get_mime_type

from .load_docx import load_docx
from .load_hwp import load_hwp
from .load_pdf import load_pdf
from .load_pptx import load_pptx


def load_document(path: Union[str, Path]) -> List[Document]:
    """파일 형식을 자동으로 감지하여 적절한 로더로 문서를 로드합니다.

    MIME 타입을 기반으로 파일 형식을 판단하고, 해당하는 전용 로더를 호출하여
    문서를 Document 리스트로 변환합니다. 다양한 문서 형식을 지원합니다.

    Args:
        path (Union[str, Path]): 로드할 문서 파일의 경로

    Returns:
        List[Document]: 로드된 문서 내용을 담은 Document 객체들의 리스트

    Raises:
        ValueError: 지원하지 않는 파일 형식인 경우
        FileNotFoundError: 파일이 존재하지 않는 경우

    Examples:
        >>> # PDF 파일 로드
        >>> documents = load_document('report.pdf')
        >>> len(documents)
        1
        >>> # HWP 파일 로드
        >>> documents = load_document('교원인사규정.hwp')
        >>> len(documents)
        1
        >>> # DOCX 파일 로드
        >>> documents = load_document('문서.docx')
        >>> len(documents)
        1
        >>> # PPTX 파일 로드
        >>> documents = load_document('프레젠테이션.pptx')
        >>> len(documents)
        1
        >>> # 지원하지 않는 형식
        >>> load_document('unknown.txt')
        ValueError: 지원하지 않는 파일 형식: text/plain

    Note:
        지원되는 파일 형식:
        - PDF (.pdf): Portable Document Format
        - HWP (.hwp): 한글 워드프로세서 파일
        - DOCX (.docx): Microsoft Word 문서
        - PPTX (.pptx): Microsoft PowerPoint 프레젠테이션
    """
    # Path 객체로 변환 (str도 지원하기 위해)
    path_str = str(path)

    # 타입 가드를 사용한 파일 형식 검증 및 로더 호출
    if is_pdf(path_str):
        return load_pdf(path)
    elif is_hwp(path_str):
        return load_hwp(path)
    elif is_docx(path_str):
        return load_docx(path)
    elif is_pptx(path_str):
        return load_pptx(path)
    else:
        # 지원하지 않는 형식인 경우 MIME 타입 정보와 함께 오류 발생
        mime_type = get_mime_type(path)
        supported_formats = "PDF (.pdf), HWP (.hwp), DOCX (.docx), PPTX (.pptx)"
        raise ValueError(
            f"지원하지 않는 파일 형식: {mime_type}\n지원되는 형식: {supported_formats}"
        )
