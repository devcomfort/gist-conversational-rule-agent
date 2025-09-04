"""
토큰 기반 텍스트 청킹 모듈

이 모듈은 Hugging Face AutoTokenizer를 사용하여 텍스트를 토큰 단위로 분할하는 기능을 제공합니다.
다양한 사전 훈련된 모델들의 토크나이저를 지원하며, 지정된 크기와 오버랩으로 텍스트를 청크로 나눕니다.

주요 기능:
- 다양한 Hugging Face 모델 토크나이저 지원 (BERT, GPT, LLaMA, Qwen 등)
- 유연한 청크 크기 및 오버랩 설정
- 효율적인 토큰 기반 텍스트 분할

Example:
    >>> from token_chunker import TokenChunker
    >>> chunker = TokenChunker()
    >>> token_chunker = chunker.create_tokenizer_from_encoding(
    ...     "bert-base-uncased", chunk_size=1000, chunk_overlap=100
    ... )
    >>> chunks = token_chunker.chunk("긴 텍스트 문서...")
"""

from typing import Optional
from chonkie import TokenChunker as _TokenChunker
from transformers import AutoTokenizer

from .create_tokenizer import create_tokenizer


class TokenChunker:
    """
    텍스트를 토큰 기반으로 청크로 분할하는 클래스입니다.

    이 클래스는 Hugging Face AutoTokenizer를 사용하여 다양한 사전 훈련된 모델의
    토크나이저로 텍스트를 토큰화하고, 지정된 크기와 오버랩으로 청크를 생성합니다.

    Attributes:
        _tokenizer (Optional[AutoTokenizer]): Hugging Face 토크나이저 인스턴스
        _token_chunker (Optional[_TokenChunker]): 실제 청킹을 수행하는 객체
    """

    _tokenizer: Optional[AutoTokenizer] = None
    _token_chunker: Optional[_TokenChunker] = None

    def __init__(self):
        """
        TokenChunker 클래스를 초기화합니다.

        초기화 시에는 토크나이저와 청커를 생성하지 않으며,
        create_tokenizer_from_encoding 메소드를 통해 Hugging Face 모델을 지정하여 생성해야 합니다.
        """
        pass

    def create_token_chunker(
        self,
        tokenizer_type: str,
        chunk_size: int,
        chunk_overlap: int  # int인 경우: 토큰 개수로 오버랩 크기 결정
        | float,  # float인 경우: chunk_size * chunk_overlap으로 오버랩 크기 결정 (0-1 범위)
    ) -> _TokenChunker:
        """
        지정된 모델명으로 토크나이저와 청커를 생성합니다.

        Args:
            tokenizer_type (str): 사용할 Hugging Face 모델명 또는 경로
            chunk_size (int): 각 청크의 최대 토큰 수
            chunk_overlap (int | float): 청크 간 오버랩 크기
                - int: 토큰 개수로 직접 지정
                - float: chunk_size에 대한 비율 (0-1)

        Returns:
            _TokenChunker: 생성된 토큰 청커 인스턴스

        Example:
            >>> chunker = TokenChunker()
            >>> token_chunker = chunker.create_token_chunker(
            ...     "bert-base-uncased", chunk_size=1000, chunk_overlap=100
            ... )
        """
        # 지정된 모델명으로 토크나이저 생성
        self.tokenizer = create_tokenizer(tokenizer_type)

        # 토크나이저를 사용하여 청커 인스턴스 생성
        self.new_token_chunker = _TokenChunker(
            tokenizer=self.tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # 인스턴스 변수에 저장
        self._token_chunker = self.new_token_chunker
        self._tokenizer = self.tokenizer

        return self.new_token_chunker

    @property
    def token_chunker(self):
        """
        생성된 토큰 청커에 대한 접근자 프로퍼티입니다.

        Returns:
            _TokenChunker: 생성된 토큰 청커 인스턴스

        Raises:
            ValueError: 토큰 청커가 아직 생성되지 않은 경우
        """
        if self._token_chunker is None:
            raise ValueError(
                "토큰 청커가 생성되지 않았습니다. "
                "먼저 create_tokenizer_from_encoding() 메소드를 호출하세요."
            )
        return self._token_chunker
