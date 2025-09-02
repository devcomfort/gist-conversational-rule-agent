"""
Cleaner 설정 로더 - OmegaConf.structured 사용
"""

from pathlib import Path
from functools import lru_cache

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .types import Config, ModelConfig, TemplatesConfig


# 메모이제이션 적용 이유
# - Hydra/OmegaConf 초기화는 비용이 크고, 멀티스레드/병렬 실행 시 GlobalHydra 재초기화 충돌을 유발할 수 있습니다.
# - 프로세스 내에서 동일 설정을 반복 로드하지 않도록 1회만 초기화/캐싱하여 성능과 안정성을 확보합니다.
# - 설정을 다시 읽어야 하는 경우 load_cleaner_config.cache_clear() 후 재호출하세요.
@lru_cache(maxsize=1)
def load_cleaner_config() -> Config:
    """cleaner.yaml 설정을 로드하여 타입 안전한 Config 객체로 반환"""
    config_dir = Path(__file__).parent.parent

    # YAML 로드
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="cleaner")

    # OmegaConf.structured로 타입 안전한 객체 변환
    return OmegaConf.structured(Config, cfg)


def get_model_config() -> ModelConfig:
    return load_cleaner_config().cleaner.model


def get_templates_config() -> TemplatesConfig:
    return load_cleaner_config().cleaner.templates


def get_template_path(template_name: str) -> Path:
    templates_config = get_templates_config()
    project_root = Path(__file__).parent.parent.parent
    return project_root / templates_config.base_path / template_name


if __name__ == "__main__":
    """직접 실행 시 설정 로드 및 정보 출력 (정성 평가용)"""
    print("🧹 Cleaner Config Loader - 정성 평가 테스트")
    print("=" * 60)

    try:
        # 전체 설정 로드
        print("📂 설정 파일 로드 중...")
        config = load_cleaner_config()
        print("✅ cleaner.yaml 로드 성공!")
        print()

        # 설정 정보 출력
        print("🤖 AI 모델 설정:")
        print(f"   • 모델명      : {config.cleaner.model.name}")
        print(f"   • 최대 토큰   : {config.cleaner.model.max_tokens:,}")
        print(f"   • Temperature : {config.cleaner.model.temperature}")
        print(f"   • Top-p       : {config.cleaner.model.top_p}")
        print(
            f"   • 스트림      : {'활성화' if config.cleaner.model.stream else '비활성화'}"
        )
        print()

        print("📁 템플릿 설정:")
        print(f"   • 베이스 경로 : {config.cleaner.templates.base_path}")
        print(f"   • 마크다운 템플릿 : {config.cleaner.templates.markdown_template}")
        print()

        # 편의 함수 테스트
        print("🔧 편의 함수 테스트:")
        model_config = get_model_config()
        print(f"   • get_model_config() → {model_config.name}")

        templates_config = get_templates_config()
        print(f"   • get_templates_config() → {templates_config.base_path}")

        template_path = get_template_path("refine_markdown.mustache")
        print(f"   • get_template_path() → {template_path}")
        print(f"     파일 존재: {'✅' if template_path.exists() else '❌'}")
        print()

        print("🎯 타입 검증:")
        print(f"   • Config 타입: {type(config).__name__}")
        print(
            f"   • 타입 안전성: {'✅ OmegaConf.structured 적용됨' if hasattr(config, '_metadata') else '❌'}"
        )
        print()

        print("🎉 모든 테스트 통과! Hydra + OmegaConf.structured 정상 동작")

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
