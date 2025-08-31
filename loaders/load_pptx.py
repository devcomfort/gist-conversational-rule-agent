from pathlib import Path
from typing import List, Union

from llama_index.core.schema import Document
from llama_index.readers.file import PptxReader

from toolz import pipe
from utils.validators import is_pptx


def load_pptx(path: Union[str, Path]) -> List[Document]:
    """PPTX 파일을 로드하여 Document 리스트로 반환합니다.

    문자열 또는 Path 객체를 입력받아 Path로 정규화한 후 PPTX 파일을 로드합니다.
    toolz.pipe를 사용하여 함수형 프로그래밍 방식으로 구현되었습니다.

    Args:
        path (Union[str, Path]): PPTX 파일의 경로

    Returns:
        List[Document]: 로드된 PPTX 내용을 담은 Document 객체들의 리스트

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        ValueError: PPTX 파일이 아니거나 읽을 수 없는 경우

    Examples:
        >>> documents = load_pptx('프레젠테이션.pptx')
        >>> len(documents)
        1
        >>> type(documents[0])
        <class 'llama_index.core.schema.Document'>
    """

    def _validate_pptx(path: Path) -> Path:
        if not is_pptx(str(path)):
            raise ValueError(f"파일이 PPTX 형식이 아닙니다: {path}")
        return path

    def _load_data(path: Path) -> List[Document]:
        reader = PptxReader()
        loaded_data = reader.load_data(path)
        return loaded_data

    return pipe(
        path,  # 데이터 입력
        Path,  # 경로 정규화
        _validate_pptx,  # 타입 가드 (PPTX 검증)
        _load_data,  # 데이터 처리
    )
