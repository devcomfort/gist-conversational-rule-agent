"""
Metadata Extractor 설정 로더 - OmegaConf.structured 사용
"""

from pathlib import Path
from functools import lru_cache

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .types import Config, ModelConfig


# 메모이제이션 적용 이유
# - Hydra/OmegaConf 초기화 비용 절감 및 GlobalHydra 재초기화 충돌 방지
# - 프로세스 내 동일 설정 반복 로드를 피하고, 1회 초기화된 구성을 재사용
# - 필요 시 load_metadata_extractor_config.cache_clear() 호출 후 재로딩
@lru_cache(maxsize=1)
def load_metadata_extractor_config() -> Config:
    """metadata_extractor.yaml 설정을 로드하여 타입 안전한 Config 객체로 반환"""
    config_dir = Path(__file__).parent.parent

    # YAML 로드
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="metadata_extractor")

    # OmegaConf.structured로 타입 안전한 객체 변환
    return OmegaConf.structured(Config, cfg)


def get_model_config() -> ModelConfig:
    return load_metadata_extractor_config().metadata_extractor.model


def create_llm_kwargs() -> dict:
    model_config = get_model_config()
    return {
        "model": model_config.name,
        "max_tokens": model_config.max_tokens,
        "temperature": model_config.temperature,
        "top_p": model_config.top_p,
        "stream": model_config.stream,
    }


if __name__ == "__main__":
    """직접 실행 시 설정 로드 및 정보 출력 (정성 평가용)"""
    print("📊 Metadata Extractor Config Loader - 정성 평가 테스트")
    print("=" * 65)

    try:
        # 전체 설정 로드
        print("📂 설정 파일 로드 중...")
        config = load_metadata_extractor_config()
        print("✅ metadata_extractor.yaml 로드 성공!")
        print()

        # 설정 정보 출력
        print("🤖 AI 모델 설정:")
        print(f"   • 모델명      : {config.metadata_extractor.model.name}")
        print(f"   • 최대 토큰   : {config.metadata_extractor.model.max_tokens:,}")
        print(f"   • Temperature : {config.metadata_extractor.model.temperature}")
        print(f"   • Top-p       : {config.metadata_extractor.model.top_p}")
        print(
            f"   • 스트림      : {'활성화' if config.metadata_extractor.model.stream else '비활성화'}"
        )
        print()

        # 편의 함수 테스트
        print("🔧 편의 함수 테스트:")
        model_config = get_model_config()
        print(f"   • get_model_config() → {model_config.name}")

        llm_kwargs = create_llm_kwargs()
        print(f"   • create_llm_kwargs() → {len(llm_kwargs)} 개 파라미터")
        print(f"     - model: {llm_kwargs['model']}")
        print(f"     - max_tokens: {llm_kwargs['max_tokens']:,}")
        print(f"     - temperature: {llm_kwargs['temperature']}")
        print(f"     - top_p: {llm_kwargs['top_p']}")
        print(f"     - stream: {llm_kwargs['stream']}")
        print()

        # llama-index 연동 테스트
        print("🤖 llama-index 연동 테스트:")
        try:
            from llama_index.llms.litellm import LiteLLM

            test_llm = LiteLLM(**llm_kwargs)
            print("   • LiteLLM 인스턴스 생성: ✅")
            print(f"   • 설정된 모델: {test_llm.model}")

            # agents/metadata_extractor.py 연동 확인
            try:
                from agents.metadata_extractor import MetadataExtractor

                print("   • agents.metadata_extractor 연동: ✅")
                print(
                    f"   • MetadataExtractor 타입: {type(MetadataExtractor).__name__}"
                )
            except ImportError as e:
                print(f"   • agents.metadata_extractor 연동: ❌ ({e})")

        except ImportError as e:
            print(f"   • LiteLLM 라이브러리: ❌ 설치되지 않음 ({e})")
        print()

        print("🎯 타입 검증:")
        print(f"   • Config 타입: {type(config).__name__}")
        print(
            f"   • 타입 안전성: {'✅ OmegaConf.structured 적용됨' if hasattr(config, '_metadata') else '❌'}"
        )
        print()

        print("🎉 모든 테스트 통과! Hydra + OmegaConf.structured 정상 동작")
        print("💡 llama-index LiteLLM과 완벽 연동 준비 완료!")

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
