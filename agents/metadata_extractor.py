# REFERENCE: https://docs.llamaindex.ai/en/stable/api_reference/llms/litellm/

from llama_index.llms.litellm import LiteLLM
from config.metadata_extractor_config import create_llm_kwargs


def create_metadata_extractor() -> LiteLLM:
    """
    Hydra 설정을 사용하여 MetadataExtractor LLM 인스턴스 생성

    Returns:
        LiteLLM: 설정된 LiteLLM 인스턴스
    """
    llm_kwargs = create_llm_kwargs()
    return LiteLLM(**llm_kwargs)


# Initialize LiteLLM with Hydra configuration
MetadataExtractor = create_metadata_extractor()
