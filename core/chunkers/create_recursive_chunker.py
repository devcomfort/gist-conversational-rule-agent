from chonkie import RecursiveChunker, RecursiveRules
from attrs import define, field

from .create_tokenizer import Tokenizer, create_tokenizer
from typing import Union, Literal, TypeAlias


@define
class RecursiveChunkerParameters:
    """Recursive Chunker 생성을 위한 파라미터 컨테이너.

    Attributes:
        tokenizer_or_token_counter: `"character"`, 토크나이저 이름(`str`),
            또는 `Tokenizer` 인스턴스 중 하나. 입력은 아래 정규화 함수로
            `Tokenizer` 또는 `"character"`로 변환되어 사용됩니다.
        chunk_size: 청크 단위의 최대 토큰 수(또는 문자 기반일 경우 문자 수).
        rules: 재귀 청킹 규칙 세트.
        min_characters_per_chunk: 너무 작은 청크 생성을 방지하기 위한 최소 문자 수.
    """

    tokenizer_or_token_counter: Union[Literal["character"], str, Tokenizer]
    chunk_size: int
    rules: RecursiveRules = field(default=RecursiveRules())
    min_characters_per_chunk: int = 24


# 타입 별칭: 가독성과 재사용성을 위해 입력/출력 타입을 명확히 선언합니다.
TokenizerInput: TypeAlias = Union[Literal["character"], str, Tokenizer]
TokenizerOrCharacter: TypeAlias = Union[Literal["character"], Tokenizer]


# Recursive Chunker에서 사용할 토크나이저 입력을 정규화합니다.
def normalize_tokenizer_for_recursive_chunker(
    tokenizer_or_token_counter: TokenizerInput,
) -> TokenizerOrCharacter:
    """Recursive Chunker를 위한 토크나이저 입력을 정규화합니다.

    - `"character"` 입력은 그대로 반환합니다.
    - `Tokenizer` 인스턴스는 그대로 반환합니다.
    - 문자열(`str`) 입력은 토크나이저 이름으로 간주하여 `Tokenizer`를 생성합니다.

    Args:
        tokenizer_or_token_counter: `"character"` | 토크나이저 이름(`str`) | `Tokenizer`.

    Returns:
        TokenizerOrCharacter: `Tokenizer` 인스턴스 또는 `"character"` 문자열.

    Raises:
        ValueError: 허용되지 않은 타입이 주어진 경우.
    """
    if tokenizer_or_token_counter == "character":
        return "character"
    if isinstance(tokenizer_or_token_counter, Tokenizer):
        return tokenizer_or_token_counter
    if isinstance(tokenizer_or_token_counter, str):
        return create_tokenizer(tokenizer_or_token_counter)
    raise ValueError(f"Invalid tokenizer: {tokenizer_or_token_counter}")


def create_recursive_chunker(
    tokenizer_or_token_counter: Union[Literal["character"], str, Tokenizer],
    chunk_size: int,
    rules: RecursiveRules = RecursiveRules(),
    min_characters_per_chunk: int = 24,
) -> RecursiveChunker:
    """Recursive Chunker 인스턴스를 생성합니다.

    입력 타입에 따른 동작:
    - `"character"`: 문자 기반 카운터를 사용합니다. 토크나이저 없이 문자 수 기준으로 분할합니다.
    - `str`(토크나이저 이름): `create_tokenizer()`로 `Tokenizer`를 생성해 사용합니다.
    - `Tokenizer` 인스턴스: 전달된 인스턴스를 그대로 사용합니다.

    Overlap 정책:
    - 모든 청커는 `chunk_overlap`을 정수(int) 또는 비율(float)로 받아들입니다.
    - `int`: 실제 오버랩 토큰 수로 사용됩니다.
    - `float`: 비율로 해석되어 실제 오버랩은 \(chunk\_size \times chunk\_overlap\)로 계산됩니다.
      예: \(100 \times 0.1 = 10\)

    Args:
        tokenizer_or_token_counter: `"character"` | 토크나이저 이름(`str`) | `Tokenizer`.
        chunk_size: 청크 최대 크기(토큰 기준). 문자 기반일 경우 문자 수로 간주.
        rules: 재귀 청킹에 사용할 규칙.
        min_characters_per_chunk: 최소 문자 수 하한선.

    Returns:
        RecursiveChunker: 구성된 재귀 청커 인스턴스.
    """
    tokenizer = normalize_tokenizer_for_recursive_chunker(tokenizer_or_token_counter)
    new_recursive_chunker = RecursiveChunker(
        tokenizer_or_token_counter=tokenizer,
        chunk_size=chunk_size,
        rules=rules,
        min_characters_per_chunk=min_characters_per_chunk,
    )
    return new_recursive_chunker
