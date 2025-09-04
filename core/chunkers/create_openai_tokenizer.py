"""
OpenAI tiktoken 인코딩 생성 유틸리티 모듈

tiktoken을 사용하여 텍스트를 토큰 단위로 인코딩/디코딩하는 `Encoding` 인스턴스를 생성합니다.

주요 기능:
- OpenAI tiktoken 인코딩 이름 지원 (예: "cl100k_base", "o200k_base", "gpt2")
- 빠른 토큰화/디토큰화 퍼포먼스
- 인코딩 이름 기반의 간결한 사용성

Example:
    >>> from core.chunkers.create_openai_tokenizer import create_openai_tokenizer
    >>> enc = create_openai_tokenizer("cl100k_base")
    >>> ids = enc.encode("안녕하세요")
    >>> text = enc.decode(ids)
"""

import tiktoken

Tokenizer = tiktoken.Encoding  # alias for tiktoken


def create_openai_tokenizer(tokenizer_type: str) -> Tokenizer:
    """tiktoken `Encoding`을 생성합니다.

    Args:
        tokenizer_type (str): 사용할 tiktoken 인코딩 이름.
            - 예: "cl100k_base"(GPT-4 계열), "o200k_base"(GPT-4o 계열), "gpt2"(BPE)

    Returns:
        Tokenizer: tiktoken.Encoding 인스턴스

    Raises:
        ValueError: 지원하지 않는 인코딩 이름이거나 초기화에 실패한 경우

    Example:
        >>> enc = create_openai_tokenizer("cl100k_base")
        >>> enc.encode("hello world")
    """
    try:
        tokenizer = tiktoken.get_encoding(tokenizer_type)
        return tokenizer
    except Exception as e:
        # 상세한 에러 메시지 제공 (tiktoken 인코딩 기준)
        error_msg = (
            f"지원하지 않는 tiktoken 인코딩이거나 초기화에 실패했습니다: {tokenizer_type}\n\n"
            f"가능한 확인 사항:\n"
            f"1. 인코딩 이름 확인: 'cl100k_base', 'o200k_base', 'gpt2' 등\n"
            f"2. tiktoken 설치/버전 확인: https://github.com/openai/tiktoken\n\n"
            f"원본 에러: {str(e)}"
        )
        raise ValueError(error_msg) from e
