"""LateChunker 생성 유틸리티.

문서 기반 설정으로 Chonkie의 LateChunker를 생성합니다. LateChunker는 문서 수준 임베딩을
이용하여 각 청크의 표현을 구성하는 지연 청킹 전략(late chunking)을 구현합니다. 기본 청킹은
`RecursiveRules`를 따릅니다. 참고: [Late Chunker 문서](https://docs.chonkie.ai/python-sdk/chunkers/late-chunker).
"""

from typing import Optional
from chonkie import LateChunker, RecursiveRules


def create_late_chunker(
    embedding_model: str,
    chunk_size: int,
    rules: Optional[RecursiveRules] = None,
    min_characters_per_chunk: int = 24,
) -> LateChunker:
    """LateChunker를 생성합니다.

    Parameters
    ----------
    embedding_model : str
        SentenceTransformer 모델 식별자. LateChunker는 현재 SentenceTransformer 기반을 지원합니다.
        사용 가능한 모델은 Hugging Face Models에서 확인하세요: https://huggingface.co/models
        (예: sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-base-en-v1.5 등)
    chunk_size : int
        각 청크의 최대 토큰 수(상한). 모델/토크나이저 토큰화 기준으로 해석됩니다.
    rules : RecursiveRules | None, default None
        재귀 분할 규칙. None이면 기본 `RecursiveRules()`가 사용됩니다.
    min_characters_per_chunk : int, default 24
        청크 최소 문자 수(너무 작은 청크 방지).

    Returns
    -------
    LateChunker
        문서 임베딩을 활용한 LateChunker 인스턴스.
    """
    if rules is None:
        rules = RecursiveRules()

    new_late_chunker = LateChunker(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        rules=rules,
        min_characters_per_chunk=min_characters_per_chunk,
    )

    return new_late_chunker
