"""
Cleaner 설정 - Hydra Structured Config
"""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    name: str = "fireworks_ai/accounts/fireworks/models/gpt-oss-20b"
    max_tokens: int = 100000
    temperature: float = 0.1
    top_p: float = 0.9
    stream: bool = True


@dataclass
class TemplatesConfig:
    base_path: str = "templates"
    markdown_template: str = "refine_markdown.mustache"


@dataclass
class CleanerConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    templates: TemplatesConfig = field(default_factory=TemplatesConfig)


@dataclass
class Config:
    cleaner: CleanerConfig = field(default_factory=CleanerConfig)


# ConfigStore에 스키마 등록 (Hydra 1.2 호환)
cs = ConfigStore.instance()
cs.store(name="cleaner_schema", node=Config)
