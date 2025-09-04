"""
토큰 기반 텍스트 청킹 모듈

Hugging Face AutoTokenizer를 사용하여 텍스트를 토큰 단위로 분할하는 기능을 제공합니다.
다양한 사전 훈련 모델 토크나이저를 지원하며, 지정된 크기와 오버랩으로 텍스트를 청크로 나눕니다.

주요 기능:
- 다양한 Hugging Face 모델 토크나이저 지원 (BERT, GPT, LLaMA, Qwen 등)
- 유연한 청크 크기 및 오버랩 설정
- 효율적인 토큰 기반 텍스트 분할

Example (팩토리 함수 사용):
    >>> from core.chunkers.create_token_chunker import create_token_chunker
    >>> token_chunker = create_token_chunker(
    ...     "bert-base-uncased", chunk_size=1000, chunk_overlap=100
    ... )
    >>> chunks = token_chunker.chunk("긴 텍스트 문서...")
"""

from attrs import define, field
from typing import TypeAlias
from chonkie import TokenChunker
from .create_tokenizer import create_tokenizer


@define
class TokenChunkerParameters:
    """TokenChunker 생성을 위한 파라미터 컨테이너.

    Attributes:
        tokenizer_type: 사용할 Hugging Face 모델명 또는 로컬 경로.
        chunk_size: 각 청크의 최대 토큰 수.
        chunk_overlap: 오버랩(overlap). 정수(토큰 수) 또는 비율(float, 0~1).
    """

    tokenizer_type: str
    chunk_size: int
    chunk_overlap: int | float = field(default=0.1)


# 타입 별칭: 오버랩(overlap) — 정수(토큰 수) 또는 비율(float, 0~1)
ChunkOverlap: TypeAlias = int | float


def create_token_chunker(
    tokenizer_type: str,
    chunk_size: int,
    chunk_overlap: ChunkOverlap = 0,
) -> TokenChunker:
    """토큰 기반 청커를 생성하는 팩토리 함수.

    - `chunk_overlap`이 `float`이면 실제 오버랩 수치는 `chunk_size * chunk_overlap`로 계산됩니다.
      예: `chunk_size=100`, `chunk_overlap=0.1` → 실제 오버랩=10

    Args:
        tokenizer_type: 사용할 Hugging Face 모델명 또는 경로.
        chunk_size: 각 청크의 최대 토큰 수.
        chunk_overlap: 청크 간 오버랩 크기. 정수(토큰 수) 또는 비율(float, 0~1).

    Returns:
        TokenChunker: 구성된 토큰 청커 인스턴스.

    Raises:
        ValueError: 토크나이저 초기화 실패 또는 지원하지 않는 모델명인 경우.

    Example:
        >>> token_chunker = create_token_chunker("gpt2", 512, 0.1)
        >>> chunks = token_chunker.chunk("긴 텍스트 문서...")
    """
    # Hugging Face 토크나이저 생성/초기화
    tokenizer = create_tokenizer(tokenizer_type)
    # chonkie.TokenChunker 인스턴스 구성
    new_token_chunker = TokenChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return new_token_chunker
