from pathlib import Path
from typing import List, Union

from llama_index.readers.file import PDFReader
from llama_index.core import Document

from toolz import pipe
from utils.is_pdf import is_pdf


def load_pdf(
    path: Union[str, Path], *, return_full_document: bool = True
) -> List[Document]:
    """PDF 파일을 로드하여 Document 리스트로 반환합니다.

    문자열 또는 Path 객체를 입력받아 Path로 정규화한 후 PDF 파일을 로드합니다.
    toolz.pipe를 사용하여 함수형 프로그래밍 방식으로 구현되었습니다.

    Args:
        path (Union[str, Path]): PDF 파일의 경로
        return_full_document (bool, optional): 전체 문서를 하나의 Document로 반환할지 여부.
            True인 경우 전체 PDF를 단일 Document로, False인 경우 페이지별로 분할하여 반환.
            기본값은 True.

    Returns:
        List[Document]: 로드된 PDF 내용을 담은 Document 객체들의 리스트

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        ValueError: PDF 파일이 아니거나 읽을 수 없는 경우

    Examples:
        >>> # 전체 문서를 하나의 Document로 로드
        >>> documents = load_pdf('document.pdf')
        >>> len(documents)
        1
        >>> # 페이지별로 분할하여 로드
        >>> documents = load_pdf('document.pdf', return_full_document=False)
        >>> len(documents)
        5  # 5페이지 PDF인 경우
        >>> type(documents[0])
        <class 'llama_index.core.schema.Document'>
    """

    def _validate_pdf(path: Path) -> Path:
        if not is_pdf(str(path)):
            raise ValueError(f"파일이 PDF 형식이 아닙니다: {path}")
        return path

    def _load_data(path: Path) -> List[Document]:
        reader = PDFReader(return_full_document=return_full_document)
        loaded_data = reader.load_data(path)
        return loaded_data

    return pipe(
        path,  # 데이터 입력
        Path,  # 경로 정규화
        _validate_pdf,  # 타입 가드 (PDF 검증)
        _load_data,  # 데이터 처리
    )
