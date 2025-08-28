"""
Metadata Extractor 설정 - Hydra Structured Config
"""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    name: str = "fireworks_ai/accounts/fireworks/models/gpt-oss-20b"
    max_tokens: int = 10000
    temperature: float = 0.1
    top_p: float = 0.9
    stream: bool = False


@dataclass
class MetadataExtractorConfig:
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class Config:
    metadata_extractor: MetadataExtractorConfig = field(
        default_factory=MetadataExtractorConfig
    )


# ConfigStore에 스키마 등록 (Hydra 1.2 호환)
cs = ConfigStore.instance()
cs.store(name="metadata_extractor_schema", node=Config)
