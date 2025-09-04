"""Chunker factories and related types.

이 모듈은 청커 팩토리 함수와 관련 타입들을 한곳에서 노출합니다.
런타임 의존성을 최소화하기 위해 지연 임포트를 사용합니다.
"""

from importlib import import_module
from typing import Any

__all__ = [
    # token
    "create_token_chunker",
    "TokenChunkerParameters",
    "ChunkOverlap",
    # recursive
    "create_recursive_chunker",
    "RecursiveChunkerParameters",
    "normalize_tokenizer_for_recursive_chunker",
    "TokenizerInput",
    "TokenizerOrCharacter",
    # embeddings
    "create_embedding_model",
]


_EXPORT_MAP = {
    # token
    "create_token_chunker": ("core.chunkers.create_token_chunker", "create_token_chunker"),
    "TokenChunkerParameters": ("core.chunkers.create_token_chunker", "TokenChunkerParameters"),
    "ChunkOverlap": ("core.chunkers.create_token_chunker", "ChunkOverlap"),
    # recursive
    "create_recursive_chunker": ("core.chunkers.create_recursive_chunker", "create_recursive_chunker"),
    "RecursiveChunkerParameters": ("core.chunkers.create_recursive_chunker", "RecursiveChunkerParameters"),
    "normalize_tokenizer_for_recursive_chunker": (
        "core.chunkers.create_recursive_chunker",
        "normalize_tokenizer_for_recursive_chunker",
    ),
    "TokenizerInput": ("core.chunkers.create_recursive_chunker", "TokenizerInput"),
    "TokenizerOrCharacter": ("core.chunkers.create_recursive_chunker", "TokenizerOrCharacter"),
    # embeddings
    "create_embedding_model": ("core.chunkers.create_embedding_model", "create_embedding_model"),
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _EXPORT_MAP:
        module_path, attr_name = _EXPORT_MAP[name]
        module = import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'core.chunkers' has no attribute {name!r}")


