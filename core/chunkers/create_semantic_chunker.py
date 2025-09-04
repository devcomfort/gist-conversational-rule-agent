"""SemanticChunker 생성 유틸리티.

Chonkie의 SemanticChunker를 위한 팩토리와 파라미터 컨테이너를 제공합니다.
유사도 기반 분할, Savitzky-Golay 필터링, skip-window 병합 등의 고급 기능을 지원합니다.

참고: SemanticChunker 파라미터 설명은 문서의 Parameters 섹션을 따릅니다.
문서: https://docs.chonkie.ai/python-sdk/chunkers/semantic-chunker#param-similarity-window
"""

from attrs import define, field
from typing import Optional, TypeAlias, Union, List, Literal
from chonkie import SemanticChunker
from chonkie.embeddings import BaseEmbeddings


EmbeddingModelLike: TypeAlias = Union[str, BaseEmbeddings]


@define
class SemanticChunkerParameters:
    """SemanticChunker 생성을 위한 파라미터 컨테이너.

    Attributes:
        embedding_model: 모델 식별자 또는 임베딩 모델 인스턴스.
        threshold: 유사도 임계값(0-1). 낮을수록 더 큰 그룹이 생성됩니다. 기본 0.8.
        chunk_size: 각 청크의 최대 토큰 수. 기본 2048.
        similarity_window: 유사도 계산에 고려할 문장 수. 기본 3.
        skip_window: 비연속 유사 그룹 병합 윈도우. 0=비활성화, 1 이상 활성화.
        min_sentences_per_chunk: 청크당 최소 문장 수. 기본 1.
        min_characters_per_sentence: 문장당 최소 문자 수. 기본 24.
        filter_window: Savitzky-Golay 필터 윈도우 길이. 기본 5.
        filter_polyorder: Savitzky-Golay 폴리노미얼 차수. 기본 3.
        filter_tolerance: 경계 검출 관용치. 기본 0.2.
        delim: 문장 분할 구분자(문자열 또는 문자열 리스트).
        include_delim: 구분자를 이전(prev) 또는 다음(next) 문장에 포함.
    """

    embedding_model: EmbeddingModelLike = "minishlab/potion-base-8M"
    threshold: float = 0.8
    chunk_size: int = 2048
    similarity_window: int = 3
    skip_window: int = 0
    min_sentences_per_chunk: int = 1
    min_characters_per_sentence: int = 24
    filter_window: int = 5
    filter_polyorder: int = 3
    filter_tolerance: float = 0.2
    delim: Union[str, List[str]] = field(factory=lambda: [". ", "! ", "? ", "\n"])
    include_delim: Optional[Literal["prev", "next"]] = "prev"


def create_semantic_chunker(
    embedding_model: EmbeddingModelLike = "minishlab/potion-base-8M",
    threshold: float = 0.8,
    chunk_size: int = 2048,
    similarity_window: int = 3,
    skip_window: int = 0,
    min_sentences_per_chunk: int = 1,
    min_characters_per_sentence: int = 24,
    filter_window: int = 5,
    filter_polyorder: int = 3,
    filter_tolerance: float = 0.2,
    delim: Optional[Union[str, List[str]]] = None,
    include_delim: Optional[Literal["prev", "next"]] = "prev",
) -> SemanticChunker:
    """SemanticChunker를 생성합니다.

    - 유사도 기반 문장 그룹화 및 skip-window 병합을 지원합니다.
    - Savitzky-Golay 필터로 의미 경계 검출을 보조합니다.

    Args:
        embedding_model: 모델 식별자 또는 임베딩 모델 인스턴스.
        threshold: 유사도 임계값(0-1). 낮을수록 더 큰 그룹이 생성.
        chunk_size: 각 청크의 최대 토큰 수.
        similarity_window: 유사도 계산에 고려할 문장 수.
        skip_window: 비연속 유사 그룹 병합 윈도우. 0=비활성화, 1 이상 활성화.
        min_sentences_per_chunk: 청크당 최소 문장 수.
        min_characters_per_sentence: 문장당 최소 문자 수.
        filter_window: Savitzky-Golay 필터 윈도우 길이.
        filter_polyorder: Savitzky-Golay 폴리노미얼 차수.
        filter_tolerance: 경계 검출 관용치.
        delim: 문장 분할 구분자(문자열 또는 문자열 리스트). 기본값: [". ", "! ", "? ", "\n"]
        include_delim: 구분자를 이전(prev) 또는 다음(next) 문장에 포함.

    Returns:
        SemanticChunker: 구성된 의미 기반 청커.
    """
    # 기본 구분자 설정 (문서 권장값)
    if delim is None:
        delim = [". ", "! ", "? ", "\n"]

    # 문서 기본값과 일치하는 설정으로 생성
    chunker = SemanticChunker(
        embedding_model=embedding_model,
        threshold=threshold,
        chunk_size=chunk_size,
        similarity_window=similarity_window,
        skip_window=skip_window,
        min_sentences_per_chunk=min_sentences_per_chunk,
        min_characters_per_sentence=min_characters_per_sentence,
        filter_window=filter_window,
        filter_polyorder=filter_polyorder,
        filter_tolerance=filter_tolerance,
        delim=delim,
        include_delim=include_delim,
    )
    return chunker
