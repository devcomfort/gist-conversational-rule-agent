import logging
from typing import Iterator, Union, Any, Optional
from pathlib import Path
import pystache
from llama_index.core import Document
from litellm import completion
from .config_utils import load_cleaner_config_with_fallback

logger = logging.getLogger(__name__)


def refine_as_markdown(
    document: Union[Document, str],
    *,
    config_path: Optional[str] = None,
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
        config_path (str, optional): Hydra 설정 파일 경로
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
    cfg = load_cleaner_config_with_fallback(config_path)

    # 매개변수 설정 (오버라이드)
    model_name = model or cfg.model.name
    max_tokens_val = max_tokens or cfg.model.max_tokens
    temperature_val = temperature or cfg.markdown_refinement.model_overrides.temperature
    top_p_val = top_p or cfg.markdown_refinement.model_overrides.top_p
    stream_val = stream if stream is not None else cfg.model.stream

    # 문서 내용 추출
    if isinstance(document, Document):
        content = document.get_content()
    else:
        content = str(document)

    # Mustache 템플릿 로드 및 렌더링
    project_root = Path(__file__).parent.parent  # agentic_cleaners의 상위 디렉토리
    template_path = (
        project_root / cfg.templates.base_path / cfg.templates.markdown_template
    )
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        render_data = {"content": content}
        # mypy가 pystache.render를 인식하지 못하는 문제로 type ignore 사용
        prompt = pystache.render(template, render_data)  # type: ignore[misc]
    except Exception as e:
        logger.warning(f"템플릿 로드 실패, 기본 프롬프트 사용: {e}")
        prompt = f"""다음 문서를 전문적으로 정제하여 완전히 정리된 마크다운 문서로 변환해주세요.

<source_document>
{content}
</source_document>

위 문서를 마크다운 형식으로 완벽하게 정제하여 출력해주세요."""

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
