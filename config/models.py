"""설정 모델 정의 - Hydra Structured Config 지원"""

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
    """AI 모델 설정"""

    name: str = "fireworks_ai/accounts/fireworks/models/gpt-oss-20b"
    max_tokens: int = 100000
    temperature: float = 0.1
    top_p: float = 0.9
    stream: bool = True


@dataclass
class TemplatesConfig:
    """템플릿 파일 설정"""

    base_path: str = "agentic_cleaners/prompts"
    text_template: str = "refine_text.mustache"
    markdown_template: str = "refine_markdown.mustache"


@dataclass
class ModelOverridesConfig:
    """모델 파라미터 오버라이드 설정"""

    temperature: float = 0.1
    top_p: float = 0.9


@dataclass
class TextRefinementConfig:
    """텍스트 정제 설정"""

    model_overrides: ModelOverridesConfig = field(default_factory=ModelOverridesConfig)


@dataclass
class MarkdownRefinementConfig:
    """마크다운 정제 설정"""

    model_overrides: ModelOverridesConfig = field(default_factory=ModelOverridesConfig)


@dataclass
class CleanerConfig:
    """Agentic Cleaners 메인 설정"""

    model: ModelConfig = field(default_factory=ModelConfig)
    templates: TemplatesConfig = field(default_factory=TemplatesConfig)
    text_refinement: TextRefinementConfig = field(default_factory=TextRefinementConfig)
    markdown_refinement: MarkdownRefinementConfig = field(
        default_factory=MarkdownRefinementConfig
    )


def register_configs():
    """Hydra ConfigStore에 설정 모델들을 등록합니다."""
    cs = ConfigStore.instance()
    cs.store(name="cleaner", node=CleanerConfig)
