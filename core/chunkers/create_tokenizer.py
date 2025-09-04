"""
토크나이저 생성 유틸리티 모듈

이 모듈은 Hugging Face AutoTokenizer를 사용하여 다양한 사전 훈련된 모델의
토크나이저를 생성하는 기능을 제공합니다.

주요 기능:
- Hugging Face Hub의 모든 사용 가능한 모델 지원
- 자동 토크나이저 다운로드 및 초기화
- 상세한 에러 메시지 및 가이드 제공

Example:
    >>> from create_tokenizer import create_tokenizer
    >>> tokenizer = create_tokenizer("bert-base-uncased")
    >>> tokenizer = create_tokenizer("gpt2")
    >>> tokenizer = create_tokenizer("Qwen/Qwen2-7B")
"""

from transformers import AutoTokenizer

Tokenizer = AutoTokenizer  # alias for AutoTokenizer


def create_tokenizer(tokenizer_type: str) -> Tokenizer:
    """
    지정된 모델명에 대한 Hugging Face Tokenizer를 생성합니다.

    이 함수는 Hugging Face Hub에서 사용 가능한 모든 사전 훈련된 모델의
    토크나이저를 자동으로 다운로드하고 초기화합니다.

    Args:
        tokenizer_type (str): 생성할 토크나이저의 모델명 또는 경로
            - 공식 Hugging Face 모델: "bert-base-uncased", "gpt2"
            - 조직/사용자 모델: "microsoft/DialoGPT-medium", "Qwen/Qwen2-7B"
            - 로컬 경로: "/path/to/model"

    Returns:
        Tokenizer: 지정된 모델의 Hugging Face 토크나이저 인스턴스

    Raises:
        ValueError: 지원하지 않는 모델명이거나 다운로드에 실패한 경우

    Example:
        >>> # 기본 BERT 토크나이저
        >>> bert_tokenizer = create_tokenizer("bert-base-uncased")

        >>> # GPT-2 토크나이저
        >>> gpt_tokenizer = create_tokenizer("gpt2")

        >>> # 다국어 모델
        >>> multilingual_tokenizer = create_tokenizer("bert-base-multilingual-cased")

        >>> # 한국어 특화 모델
        >>> korean_tokenizer = create_tokenizer("klue/bert-base")

        >>> # 최신 대화형 모델
        >>> qwen_tokenizer = create_tokenizer("Qwen/Qwen2-7B")
    """
    try:
        tokenizer = Tokenizer.from_pretrained(tokenizer_type)
        return tokenizer
    except Exception as e:
        # 상세한 에러 메시지 제공
        error_msg = (
            f"지원하지 않는 모델명이거나 다운로드에 실패했습니다: {tokenizer_type}\n\n"
            f"가능한 해결책:\n"
            f"1. 모델명 확인: https://huggingface.co/models\n"
            f"2. 네트워크 연결 상태 확인\n"
            f"3. Hugging Face 계정 로그인 필요 여부 확인 (private 모델의 경우)\n"
            f"4. 올바른 모델명 형식 사용:\n"
            f"   - 공식 모델: 'bert-base-uncased', 'gpt2'\n"
            f"   - 조직 모델: 'microsoft/DialoGPT-medium'\n"
            f"   - 사용자 모델: 'username/model-name'\n\n"
            f"원본 에러: {str(e)}"
        )
        raise ValueError(error_msg) from e
