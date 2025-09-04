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
    """SemanticChunker 파라미터.

    Parameters
    ----------
    embedding_model : str | BaseEmbeddings
        모델 식별자 또는 임베딩 모델 인스턴스. 임베딩 품질/도메인 적합도가 경계 품질에 직접 영향.
        사용 가능한 모델은 Hugging Face Models에서 확인하세요: https://huggingface.co/models
        (예: sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-m3, jinaai/jina-embeddings-v3 등)
    threshold : float, default 0.8
        유사도 임계값 [0, 1]. 낮을수록 더 큰 그룹, 높을수록 더 촘촘한 분할.
    chunk_size : int, default 2048
        각 청크의 최대 토큰 수(상한). 유사도 그룹이 커도 초과 시 추가 분할될 수 있음.
    similarity_window : int, default 3
        유사도 계산에서 참조하는 문장 개수(윈도우). 값이 클수록 경계가 매끄럽고 민감도는 낮아짐.
    skip_window : int, default 0
        비연속 유사 그룹 병합 범위. 0=비활성화, 1 이상이면 최대 해당 범위 내 병합.
    min_sentences_per_chunk : int, default 1
        청크당 최소 문장 수(하한)로 과도한 분할 방지.
    min_characters_per_sentence : int, default 24
        문장당 최소 문자 수(하한)로 짧은 단편 노이즈 억제.
    filter_window : int, default 5
        Savitzky–Golay 필터 창 길이(보통 홀수 권장).
    filter_polyorder : int, default 3
        Savitzky–Golay 다항 차수. 일반적으로 0 <= filter_polyorder < filter_window 권장.
    filter_tolerance : float, default 0.2
        경계 검출 관용치. 작을수록 경계에 민감(분할 증가).
    delim : str | list[str], default [". ", "! ", "? ", "\n"]
        문장 분할 구분자. 조밀할수록 문장 경계가 세분화됨.
    include_delim : {"prev", "next"} | None, default "prev"
        구분자를 이전/다음 문장에 포함할지 지정.

    Notes
    -----
    - 유사도는 보통 정규화 임베딩의 코사인 유사도로 해석됩니다.
    - 파라미터 의미는 공식 문서와 합치합니다.
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
    embedding_model: EmbeddingModelLike,
    threshold: float,
    chunk_size: int,
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

    Parameters
    ----------
    embedding_model : str | BaseEmbeddings
        임베딩 모델 식별자 또는 인스턴스. 선택된 임베딩이 유사도 계산의 품질을 좌우합니다.
        사용 가능한 모델은 Hugging Face Models에서 확인하세요: https://huggingface.co/models
    threshold : float
        유사도 임계값 \(\tau\). 낮을수록 그룹이 커지고, 높을수록 더 세분화됩니다.
    chunk_size : int
        각 청크의 최대 토큰 수(상한).
    similarity_window : int, default 3
        유사도 계산에 고려할 문장 수 \(w\). \(w\)가 클수록 경계가 안정적(저민감).
    skip_window : int, default 0
        비연속 유사 그룹 병합 창 \(k\). \(k>0\)이면 떨어져 있는 유사 그룹을 최대 \(k\) 범위 내 병합.
    min_sentences_per_chunk : int, default 1
        청크당 최소 문장 수(하한).
    min_characters_per_sentence : int, default 24
        문장당 최소 문자 수(하한)로 단편적 문장을 억제.
    filter_window : int, default 5
        Savitzky–Golay 필터 창 \(m\). 보통 홀수 권장.
    filter_polyorder : int, default 3
        Savitzky–Golay 다항 차수 \(p\)로 \(0 \le p < m\) 권장.
    filter_tolerance : float, default 0.2
        경계 검출 관용치 \(\epsilon\). 작을수록 경계에 민감.
    delim : str | list[str], optional
        문장 구분자. 미지정 시 [". ", "! ", "? ", "\n"].
    include_delim : {"prev", "next"} | None, default "prev"
        구분자를 이전 또는 다음 문장에 포함.

    Returns
    -------
    SemanticChunker
        구성된 의미 기반 청커.

    Notes
    -----
    - 유사도는 보통 정규화 임베딩의 코사인 유사도로 해석됩니다.
    - 파라미터 의미는 공식 문서와 합치합니다.
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
