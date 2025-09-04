from chonkie import RecursiveChunker

from .create_tokenizer import Tokenizer
from typing import Union, List


def create_recursive_chunker(
    tokenizer_or_token_counter: Union[Literal["character"], Tokenizer],
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveChunker: ...
