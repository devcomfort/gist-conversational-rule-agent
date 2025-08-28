import logging
from typing import Iterator, Union, Any, Optional
import pystache
from llama_index.core import Document
from litellm import completion

from config.load_cleaner_config import load_cleaner_config, get_template_path

logger = logging.getLogger(__name__)


def refine_as_markdown(
    document: Union[Document, str],
    *,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    **kwargs: Any,
) -> Iterator[str]:
    """문서를 마크다운 형식으로 정제합니다.

    AI 모델을 사용하여 문서의 내용을 정제하고 구조화하여
    마크다운 형식으로 변환합니다. 계층적 헤더 구조를 적용하며,
    Hydra 설정과 Mustache 템플릿을 사용합니다.

    Args:
        document (Union[Document, str]): 정제할 문서 또는 텍스트 내용
        model (str, optional): 사용할 AI 모델 (설정값 오버라이드)
        max_tokens (int, optional): 최대 토큰 수 (설정값 오버라이드)
        temperature (float, optional): 생성 온도 (설정값 오버라이드)
        top_p (float, optional): Top-p 샘플링 값 (설정값 오버라이드)
        stream (bool, optional): 스트림 응답 여부 (설정값 오버라이드)
        **kwargs: litellm completion에 전달할 추가 파라미터

    Yields:
        str: 정제된 마크다운 텍스트 청크들

    Raises:
        Exception: AI 모델 호출 또는 응답 처리 중 오류 발생 시

    Examples:
        >>> document = Document(text="공식 문서 내용...")
        >>> for chunk in refine_as_markdown(document):
        ...     print(chunk, end='', flush=True)

    Note:
        - Hydra 설정 파일에서 기본값을 로드하며, 매개변수로 오버라이드 가능
        - Mustache 템플릿을 사용하여 프롬프트를 동적으로 생성
        - 문서 타입과 형식을 템플릿 변수로 전달하여 유연한 처리 지원
    """
    # 설정 로드
    cfg = load_cleaner_config()

    # 매개변수 설정 (오버라이드)
    model_name = model or cfg.cleaner.model.name
    max_tokens_val = max_tokens or cfg.cleaner.model.max_tokens
    temperature_val = temperature or cfg.cleaner.model.temperature
    top_p_val = top_p or cfg.cleaner.model.top_p
    stream_val = stream if stream is not None else cfg.cleaner.model.stream

    # 문서 내용 추출
    if isinstance(document, Document):
        content = document.get_content()
    else:
        content = str(document)

    # Mustache 템플릿 로드 및 렌더링
    template_path = get_template_path(cfg.cleaner.templates.markdown_template)
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    render_data = {"content": content}
    prompt = pystache.render(template, render_data)  # type: ignore[misc]

    try:
        logger.info(f"마크다운 정제 시작 - 모델: {model_name}")

        response = completion(  # type: ignore
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens_val,
            temperature=temperature_val,
            top_p=top_p_val,
            stream=stream_val,
            **kwargs,
        )

        if stream_val:
            logger.info("스트림 응답 수신 시작...")
            for chunk in response:
                if (
                    hasattr(chunk, "choices")
                    and chunk.choices
                    and len(chunk.choices) > 0
                ):
                    choice = chunk.choices[0]
                    if (
                        hasattr(choice, "delta")
                        and choice.delta
                        and hasattr(choice.delta, "content")
                    ):
                        content_chunk = choice.delta.content
                        if content_chunk:
                            yield content_chunk
        else:
            yield response.choices[0].message.content

    except Exception as e:
        logger.error(f"마크다운 정제 중 오류 발생: {str(e)}")
        raise


def estimate_tokens(text: str) -> int:
    """텍스트의 대략적인 토큰 수를 추정합니다."""
    # 간단한 토큰 추정: 단어 수 * 1.3 (평균적으로 1단어 = 1.3토큰)
    words = len(text.split())
    return int(words * 1.3)


if __name__ == "__main__":
    """직접 실행 시 간이 테스트 수행"""
    import time
    import os

    print("📝 Refine as Markdown - 간이 테스트")
    print("=" * 60)

    # 테스트 데이터 (수 줄 수준)
    test_document = """
    프로젝트 개요
    이 프로젝트는 AI 기반 문서 정제 시스템입니다.
    
    주요 기능:
    1. 문서 내용 분석 및 구조화
    2. 마크다운 형식으로 변환
    3. 계층적 헤더 구조 적용
    4. 불필요한 내용 제거
    
    기술 스택:
    - Python 3.11+
    - Hydra 설정 관리
    - LiteLLM API 연동
    - Mustache 템플릿 엔진
    
    사용 방법:
    함수를 호출하여 Document 객체나 문자열을 입력하면
    정제된 마크다운 결과를 스트림으로 반환합니다.
    
    주의사항:
    API 키가 필요하며, 모델에 따라 비용이 발생할 수 있습니다.
    """

    print("📂 입력 문서:")
    print("-" * 40)
    print(test_document.strip())
    print("-" * 40)
    print(f"📊 입력 통계:")
    print(f"   • 글자 수: {len(test_document):,}")
    print(f"   • 단어 수: {len(test_document.split()):,}")
    print(f"   • 라인 수: {len(test_document.strip().split(chr(10))):,}")
    print(f"   • 예상 토큰: {estimate_tokens(test_document):,}")
    print()

    # API 키 확인
    api_key_available = bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("FIREWORKS_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )

    if not api_key_available:
        print("⚠️  API 키가 설정되지 않았습니다.")
        print("   다음 환경변수 중 하나를 설정해주세요:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - FIREWORKS_API_KEY")
        print("   - GROQ_API_KEY")
        print()
        print("🔧 설정 시스템 테스트만 수행합니다...")

        # 설정 로드 테스트
        try:
            cfg = load_cleaner_config()
            template_path = get_template_path(cfg.cleaner.templates.markdown_template)

            print(f"✅ 설정 로드 성공:")
            print(f"   • 모델: {cfg.cleaner.model.name}")
            print(f"   • 최대 토큰: {cfg.cleaner.model.max_tokens:,}")
            print(f"   • Temperature: {cfg.cleaner.model.temperature}")
            print(f"   • 템플릿 파일: {template_path}")
            print(f"   • 템플릿 존재: {'✅' if template_path.exists() else '❌'}")

            if template_path.exists():
                with open(template_path, "r", encoding="utf-8") as f:
                    template_content = f.read()
                print(f"   • 템플릿 길이: {len(template_content)} 문자")

        except Exception as e:
            print(f"❌ 설정 로드 실패: {e}")

        print("\n💡 실제 LLM 호출을 위해서는 API 키를 설정해주세요!")
        exit(0)

    print("🚀 LLM 호출 시작...")
    print()

    try:
        # 시작 시간 기록
        start_time = time.time()

        # 정제 실행 (스트림 모드)
        print("📄 정제된 결과:")
        print("=" * 60)

        result_chunks: list[str] = []
        for chunk in refine_as_markdown(test_document, stream=True):
            print(chunk, end="", flush=True)
            result_chunks.append(chunk)

        # 완료 시간 계산
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 결과 통계
        full_result = "".join(result_chunks)

        print("\n" + "=" * 60)
        print("📊 처리 결과 통계:")
        print(f"   • 소요 시간: {elapsed_time:.2f}초")
        print(f"   • 출력 글자 수: {len(full_result):,}")
        print(f"   • 출력 단어 수: {len(full_result.split()):,}")
        print(f"   • 출력 라인 수: {len(full_result.strip().split(chr(10))):,}")
        print(f"   • 예상 출력 토큰: {estimate_tokens(full_result):,}")
        print(f"   • 처리 속도: {len(full_result) / elapsed_time:.1f} 문자/초")

        # 압축률 계산
        compression_ratio = len(full_result) / len(test_document)
        print(
            f"   • 압축률: {compression_ratio:.2f}x ({'압축됨' if compression_ratio < 1 else '확장됨'})"
        )

        print()
        print("🎉 마크다운 정제 완료!")

    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류 발생:")
        print(f"   {str(e)}")
        print()
        print("💡 가능한 원인:")
        print("   - API 키가 올바르지 않음")
        print("   - 네트워크 연결 문제")
        print("   - 모델 접근 권한 문제")
        print("   - 일일 사용량 초과")

        import traceback

        print(f"\n🔍 상세 오류 정보:")
        traceback.print_exc()
