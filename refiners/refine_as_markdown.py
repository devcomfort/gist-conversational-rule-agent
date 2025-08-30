import logging
from typing import Iterator, Union, Any, Optional, Callable, cast
import pystache
from llama_index.core import Document
import litellm

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
    prompt = pystache.render(template, render_data)

    try:
        logger.info(f"마크다운 정제 시작 - 모델: {model_name}")

        completion_fn: Callable[..., Any] = cast(
            Callable[..., Any], getattr(litellm, "completion")
        )

        response = completion_fn(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens_val,
            temperature=temperature_val,
            top_p=top_p_val,
            stream=stream_val,
            **kwargs,
        )

        if stream_val:

            def _extract_text(evt: Any) -> Optional[str]:
                # OpenAI 호환 객체
                choices_obj = getattr(evt, "choices", None)
                if choices_obj:
                    ch0 = choices_obj[0]
                    delta_obj = getattr(ch0, "delta", None)
                    if delta_obj is not None:
                        content_val = getattr(delta_obj, "content", None)
                        if isinstance(content_val, str) and content_val:
                            return content_val
                    text_val = getattr(ch0, "text", None)
                    if isinstance(text_val, str) and text_val:
                        return text_val

                # dict 형태
                if isinstance(evt, dict):
                    choices = evt.get("choices")
                    if isinstance(choices, list) and choices:
                        ch0 = choices[0]
                        if isinstance(ch0, dict):
                            delta = ch0.get("delta")
                            if isinstance(delta, dict):
                                content_val = delta.get("content")
                                if isinstance(content_val, str) and content_val:
                                    return content_val
                            text_val = ch0.get("text")
                            if isinstance(text_val, str) and text_val:
                                return text_val
                    # fallback: content / message.content
                    content_val = evt.get("content")
                    if isinstance(content_val, str) and content_val:
                        return content_val
                    msg = evt.get("message")
                    if isinstance(msg, dict):
                        content_val = msg.get("content")
                        if isinstance(content_val, str) and content_val:
                            return content_val

                content_attr = getattr(evt, "content", None)
                if isinstance(content_attr, str) and content_attr:
                    return content_attr

                return None

            for event in response:
                text_piece = _extract_text(event)
                if text_piece:
                    yield text_piece
        else:
            # 비스트림 응답 처리
            try:
                if isinstance(response, dict):
                    choices = response.get("choices")
                    if isinstance(choices, list) and choices:
                        ch0 = choices[0]
                        if isinstance(ch0, dict):
                            msg = ch0.get("message")
                            if isinstance(msg, dict):
                                content_val = msg.get("content")
                                if isinstance(content_val, str):
                                    yield content_val
                                    return
                            text_val = ch0.get("text")
                            if isinstance(text_val, str):
                                yield text_val
                                return
                    content_val = response.get("content")
                    if isinstance(content_val, str):
                        yield content_val
                        return

                choices = getattr(response, "choices", None)
                if choices:
                    ch0 = choices[0]
                    message_obj = getattr(ch0, "message", None)
                    if isinstance(message_obj, dict):
                        content_val = message_obj.get("content")
                        if isinstance(content_val, str):
                            yield content_val
                            return
                    elif message_obj is not None:
                        content_val = getattr(message_obj, "content", None)
                        if isinstance(content_val, str):
                            yield content_val
                            return
                    text_val = getattr(ch0, "text", None)
                    if isinstance(text_val, str):
                        yield text_val
                        return

                content_attr = getattr(response, "content", None)
                if isinstance(content_attr, str):
                    yield content_attr
                    return

                yield str(response)
            except Exception:
                yield str(response)

    except Exception as e:
        logger.error(f"마크다운 정제 중 오류 발생: {str(e)}")
        raise
