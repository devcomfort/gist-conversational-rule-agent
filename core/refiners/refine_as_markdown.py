"""
로깅 안내 (loguru)

- 이 모듈은 loguru를 사용합니다. 실행 시 로그 레벨을 명령만으로 지정하려면:

  uv run python3 -c 'import sys, runpy; from loguru import logger; logger.remove(); logger.add(sys.stderr, level="DEBUG"); runpy.run_module("refiners.refine_as_markdown", run_name="__main__")'

- 코드 내에서 설정하려면 (예: INFO 레벨):

  from loguru import logger
  import sys
  logger.remove()
  logger.add(sys.stderr, level="INFO")

"""

from typing import Union, Any, Optional, Callable, cast, Iterator
from dataclasses import asdict
from loguru import logger
import pystache
from llama_index.core import Document
import litellm
import threading
from hydra.core.global_hydra import GlobalHydra
import time

from config.load_cleaner_config import load_cleaner_config, get_template_path
from config.load_cleaner_config.types import ModelInput, ModelConfig

# NOTE: 디버그 로그를 출력하고 싶다면 주석 해제
import sys

logger.add(sys.stderr, level="INFO")


_HYDRA_LOCK = threading.Lock()
_GLOBAL_CLEANER_CFG: Optional[Any] = None

# 임시 스텁 모드: 외부 모델 호출 비활성화 후 원문을 그대로 반환합니다.
# PoC 안정화/테스트 목적. 실제 정제 재개 시 False로 전환하거나 환경변수로 제어하세요.
STUB_REFINER: bool = True


# TODO: 텍스트가 너무 큰 경우, sentence chunking 한 후에 병합하는 것도 고려해야함.


def _get_cleaner_config_threadsafe() -> Any:
    """Hydra GlobalHydra 충돌을 피하기 위해 설정 로드를 1회로 보장합니다.

    - 멀티스레드 환경에서 동시 initialize를 방지하기 위해 Lock 사용
    - 이미 초기화된 경우 재사용
    - 필요 시 GlobalHydra를 안전하게 clear 후 initialize
    """
    global _GLOBAL_CLEANER_CFG
    if _GLOBAL_CLEANER_CFG is not None:
        return _GLOBAL_CLEANER_CFG
    with _HYDRA_LOCK:
        if _GLOBAL_CLEANER_CFG is None:
            try:
                # 이미 초기화된 Hydra 컨텍스트가 있으면 정리
                GlobalHydra.instance().clear()
            except Exception:
                pass
            _GLOBAL_CLEANER_CFG = load_cleaner_config()
    return _GLOBAL_CLEANER_CFG


def _extract_text_from_stream_event(evt: Any) -> Optional[str]:
    """스트림 청크에서 텍스트를 추출합니다.

    - 단일 경로(속성 접근)로 통일: choices[0].delta.content
    - 참조: https://docs.litellm.ai/docs/completion/output
    """
    return evt.choices[0].delta.content


def _handle_stream_response(
    response: Any,
    on_stream: Optional[Callable[[str], None]] = None,
    *,
    metrics: Optional[dict[str, Any]] = None,
) -> Iterator[str]:
    """스트림 응답을 순회하며 텍스트 청크를 생성합니다.

    on_stream 콜백이 제공되면 각 청크마다 콜백을 호출합니다.
    """
    for event in response:
        # usage 청크 수집 (LiteLLM stream_options.include_usage=True 설정 시 전달)
        if metrics is not None:
            usage = getattr(event, "usage", None)
            if usage:
                try:
                    # usage는 dict 호환으로 가정
                    metrics["usage"] = {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    }
                except Exception:
                    pass

        text_piece = _extract_text_from_stream_event(event)
        if text_piece:
            if on_stream is not None:
                try:
                    on_stream(text_piece)
                except Exception:
                    # 콜백 예외는 전체 흐름을 중단하지 않습니다.
                    pass
            yield text_piece


def _extract_text_from_non_stream_response(response: Any) -> str:
    """비-스트림 응답에서 최종 텍스트를 추출합니다.

    - 단일 경로(속성 접근)로 통일: choices[0].message.content
    - 참조: https://docs.litellm.ai/docs/completion/output
    """
    return response.choices[0].message.content


def _collect_stream_text(
    response: Any,
    on_stream: Optional[Callable[[str], None]] = None,
    *,
    metrics: Optional[dict[str, Any]] = None,
) -> str:
    """스트림 응답 청크를 모두 결합하여 하나의 문자열로 반환합니다."""
    return "".join(
        _handle_stream_response(response, on_stream=on_stream, metrics=metrics)
    )


# 파라미터 병합 함수 제거 (인라인 적용)


def refine_as_markdown(
    document: Union[Document, str],
    *,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    on_stream: Optional[Callable[[str], None]] = None,
) -> str:
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
        stream (bool, optional): 스트림 응답 여부 (설정값 오버라이드). True여도 내부에서 합쳐 전체 텍스트를 반환합니다.
        on_stream (Callable[[str], None], optional):
            - stream=True: 모델이 보내는 델타 청크마다 호출됩니다.
            - stream=False: 최종 텍스트가 완성된 후 한 번 호출됩니다.

    Returns:
        str: 정제된 마크다운 전체 텍스트

    Raises:
        Exception: AI 모델 호출 또는 응답 처리 중 오류 발생 시

    Examples:
        >>> document = Document(text="공식 문서 내용...")
        >>> full_text = refine_as_markdown(document)
        >>> print(full_text)

    Note:
        - Hydra 설정 파일에서 기본값을 로드하며, 매개변수로 오버라이드 가능
        - Mustache 템플릿을 사용하여 프롬프트를 동적으로 생성
        - 문서 타입과 형식을 템플릿 변수로 전달하여 유연한 처리 지원
        - litellm 파라미터는 명시된 인자만 사용합니다. 추가 파라미터는 지원하지 않습니다. 참고: https://docs.litellm.ai/docs/completion/usage

    TODO:
        - 대용량 문서 처리: 입력 길이 상한 초과 시 chunking/windowing → 부분 정제 → 병합 전략 적용
        - 호출 동시성 제어 및 rate limit 대응(큐/세마포어, 지수 백오프, 지터)
        - 스트리밍/비-스트리밍 정책 정리 및 부분 실패 복구/재시도 설계
    """
    # 설정 로드
    cfg = _get_cleaner_config_threadsafe()

    # dataclass 기반 파라미터 구성/적용 (인라인 병합)
    input_dc = ModelInput(
        name=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
    )
    # OmegaConf DictConfig → dataclass 강제 변환 (asdict 사용 가능하도록)
    loaded_dc = ModelConfig(
        name=cfg.cleaner.model.name,
        max_tokens=cfg.cleaner.model.max_tokens,
        temperature=cfg.cleaner.model.temperature,
        top_p=cfg.cleaner.model.top_p,
        stream=cfg.cleaner.model.stream,
    )
    applied_dc = ModelConfig(
        name=input_dc.name if input_dc.name is not None else loaded_dc.name,
        max_tokens=(
            input_dc.max_tokens
            if input_dc.max_tokens is not None
            else loaded_dc.max_tokens
        ),
        temperature=(
            input_dc.temperature
            if input_dc.temperature is not None
            else loaded_dc.temperature
        ),
        top_p=(input_dc.top_p if input_dc.top_p is not None else loaded_dc.top_p),
        stream=(input_dc.stream if input_dc.stream is not None else loaded_dc.stream),
    )

    model_name = applied_dc.name
    max_tokens_val = applied_dc.max_tokens
    temperature_val = applied_dc.temperature
    top_p_val = applied_dc.top_p
    stream_val = applied_dc.stream

    # 로깅: 입력/로드/적용 + 변경 항목
    input_params = asdict(input_dc)
    loaded_params = asdict(loaded_dc)
    applied_params = asdict(applied_dc)

    changes: dict[str, dict[str, Any]] = {}
    for key in applied_params.keys():
        if (
            input_params.get(key) is not None
            and input_params[key] != loaded_params[key]
        ):
            changes[key] = {
                "loaded": loaded_params[key],
                "input": input_params[key],
                "applied": applied_params[key],
                "reason": "input override",
            }

    # 항목별 유지/변경 로그
    def _render_change_line(key: str, old_val: Any, new_val: Any) -> str:
        if old_val == new_val:
            return f"- {key}: {new_val} (유지)"
        return f"- {key}: {old_val} -> {new_val} (변경)"

    # 최소 로그: 파라미터 핵심만 요약
    logger.info(
        "정제 파라미터: model={}, max_tokens={}, temperature={}, top_p={}, stream={}",
        applied_params["name"],
        applied_params["max_tokens"],
        applied_params["temperature"],
        applied_params["top_p"],
        applied_params["stream"],
    )

    # 문서 내용 추출
    if isinstance(document, Document):
        content = document.get_content()
    else:
        content = str(document)

    # 임시 스텁: 외부 호출을 수행하지 않고 원문을 그대로 반환
    if STUB_REFINER:
        logger.info("정제 스텁 모드 활성화 - 외부 호출 없이 원문 반환")
        return content

    # Mustache 템플릿 로드 및 렌더링
    template_path = get_template_path(cfg.cleaner.templates.markdown_template)
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    render_data = {"content": content}
    prompt = pystache.render(template, render_data)

    try:
        t0 = time.perf_counter()
        logger.info("마크다운 정제 시작 - 모델: {}", model_name)

        completion_fn: Callable[..., Any] = cast(
            Callable[..., Any], getattr(litellm, "completion")
        )

        # 메트릭 준비 (LiteLLM 제공 항목만 대상)
        metrics: dict[str, Any] = {
            "model": model_name,
            "stream": bool(stream_val),
            "response_ms": None,
            "usage": None,
            # 비용 추적: LiteLLM token usage/cost
            # 참조: https://docs.litellm.ai/docs/completion/token_usage
            "response_cost": None,
        }

        # 참고: LiteLLM output 포맷/속성 참조
        # https://docs.litellm.ai/docs/completion/output

        # 네트워크 일시 오류 완화를 위한 단순 재시도/백오프
        max_attempts = 3
        base_delay_s = 0.5
        last_err: Optional[Exception] = None
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = completion_fn(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens_val,
                    temperature=temperature_val,
                    top_p=top_p_val,
                    stream=stream_val,
                    # 스트리밍 사용 시 usage 청크 포함
                    **(
                        {"stream_options": {"include_usage": True}}
                        if stream_val
                        else {}
                    ),
                )
                break
            except Exception as e:  # APIConnectionError 등 일시 오류 위주 재시도
                last_err = e
                msg = str(e).lower()
                is_transient = (
                    isinstance(e, litellm.APIConnectionError)
                    or "bad file descriptor" in msg
                    or "timeout" in msg
                    or "temporarily unavailable" in msg
                )
                if attempt < max_attempts and is_transient:
                    delay = base_delay_s * (2 ** (attempt - 1))
                    logger.warning(
                        "모델 호출 실패(일시 오류로 판단) - 재시도 {}/{}: {} ({}s 대기)",
                        attempt,
                        max_attempts,
                        str(e),
                        f"{delay:.1f}",
                    )
                    time.sleep(delay)
                    continue
                # 재시도 불가/실패: 즉시 전파
                raise
        assert response is not None, last_err

        # 최종 결과 문자열 생성
        if stream_val:
            final_text = _collect_stream_text(
                response, on_stream=on_stream, metrics=metrics
            )
            # 스트림: 최종 텍스트 결합 후 비용 계산 (helper 사용)
            # 참조: https://docs.litellm.ai/docs/completion/token_usage
            try:
                from litellm import completion_cost as _completion_cost

                cost_val = _completion_cost(
                    model=model_name, prompt=prompt, completion=final_text
                )
                try:
                    metrics["response_cost"] = float(cost_val)
                except Exception:
                    metrics["response_cost"] = cost_val
            except Exception:
                pass
        else:
            final_text = _extract_text_from_non_stream_response(response)
            if on_stream is not None:
                try:
                    on_stream(final_text)
                except Exception:
                    pass

            # 비-스트림: usage/response_ms 수집
            try:
                usage = getattr(response, "usage", None)
                if usage:
                    metrics["usage"] = {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    }
            except Exception:
                pass
            try:
                metrics["response_ms"] = getattr(response, "response_ms", None)
            except Exception:
                pass
            # 비-스트림: response_cost 읽기 또는 helper 계산
            # 참조: https://docs.litellm.ai/docs/completion/token_usage
            try:
                hidden = getattr(response, "_hidden_params", None)
                if isinstance(hidden, dict) and "response_cost" in hidden:
                    metrics["response_cost"] = hidden.get("response_cost")
            except Exception:
                pass
            if metrics.get("response_cost") is None:
                try:
                    from litellm import completion_cost as _completion_cost

                    cost_val = _completion_cost(completion_response=response)
                    try:
                        metrics["response_cost"] = float(cost_val)
                    except Exception:
                        metrics["response_cost"] = cost_val
                except Exception:
                    pass

        # 모듈 전역 메트릭 저장 (메인 가드에서 출력)
        globals()["_LAST_RUN_METRICS"] = metrics

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "마크다운 정제 완료 - 길이: {}, elapsed_ms={:.1f}",
            len(final_text or ""),
            elapsed_ms,
        )
        return final_text

    except Exception as e:
        logger.error("마크다운 정제 중 오류 발생: {}", str(e))
        raise


if __name__ == "__main__":
    # 정성적 테스트: 장문 텍스트로 스트림/비-스트림 경로 모두 점검
    # 기본 로그 레벨 설정 (INFO). 필요 시 상단 주석의 명령으로 DEBUG 실행 가능
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # 테스트용 원문 텍스트
    sample_text = """
        GIST   학칙   (가칭)   —   일부   개정 (예시)
 
        제  목 :  학생 관련  기본  규정 (초안)   
        (본 문서는  예시 데이터 입니다 ,  실제 효력 없음)
 
        제1편   총칫
 
        제1장   일반 규정
        제1조(목적)  이    학칙은   학생의 학사  운영에 관한  기본 사항을  정리하고  한국 공문서 표준 서식에  맞추어 가독성을  높이는 것을 목적으로   한다.
        제2조(정의)  본  학칙에서 사용하는 용어의 정의는  다음과 같다 .
            1.  "학생" : GIST 에   재학(휴학 포함) 중인 자
            2)  "교원" -  대학의  교육 및 연구를 담당 하는  자
            ③  "학사" — 학부 및 대학원  과정 전반을 포함한  제도
 
        제2장  학사 구성 &&  운영
        제3조(적용범위)  이 학칙은 특별히 정한 경우 를 제외하고 모든 학생에게 적용한다. 단 ,   별도의 세칙이 존재하는 경우에는 세칙을 우선  적용할 수 있다.
        제4조(수업연한)
           1. 학부:  8학기 (조기졸업 요건 충족 시 단축 가능)
           2. 석사 : 4학기
           3. 박사  : 6학기
         비고) 아래 표는 과정별 권장 이수학점을   예시로  제시 한다.
 
         과정 | 권장 이수학점 |  비고
         --- | --- | ---
         학부 | 130 | 교양   포함
         석사| 30 | 학위 논문 제외
         박사 | 36| 수료 요건 별도
 
         제5조(선수과목)  과목 간 선수요건은 학과   내규에  따른다. 예:  "자료구조"  ⟶  "알고리즘"
         제6조(성적평가)  절대 / 상대 평가 기준은 학사위원회가   정한다. (A,B,C,D,F)
 
         제3장   휴학·복학 및 학적 변동
         제7조(휴학)  신병 , 창업 , 군입대 등의  사유가  인정되는 경우  휴학을  허가할 수 있다.
           - 일반휴학:   2개 학기 연속   가능
           - 군휴학:   관련 증빙 제출 시 승인
           - 창업휴학:  사업자 등록 등 요건 충족 시 가능
 
         제8조(복학)  휴학 사유가 소멸한 경우 복학하여야   하며,  복학 시점 과 절차는 교무처가 공지한다.
         제9조(제적)  학사경고 누적, 등록금 미납,  기타 중대한 사유 발생 시 제적 할  수  있다.
 
         [부칙]  본 학칙은 2025 . 03 . 01 .  부터  시행한다. 개정사항은 공지 후 즉시 적용한다.
        """

    # 사용되는 mustache 템플릿 경로 확인 (정성적 검증 보조)
    try:
        cfg_dbg = load_cleaner_config()
        tpl_path_dbg = get_template_path(cfg_dbg.cleaner.templates.markdown_template)
        print(f"사용 템플릿: {tpl_path_dbg}")
    except Exception:
        pass

    non_stream_result: Optional[str] = None
    full_stream_text: Optional[str] = None
    ns_err: Optional[str] = None
    st_err: Optional[str] = None

    try:
        print("\n===== 비-스트림 테스트 =====\n")

        non_stream_result = refine_as_markdown(
            Document(text=sample_text),
            stream=False,
        )

    except Exception as e:
        ns_err = str(e)

    try:
        print("\n===== 스트림 테스트 =====\n")

        full_stream_text = refine_as_markdown(
            Document(text=sample_text),
            stream=True,
        )

    except Exception as e:
        st_err = str(e)

    # 모든 출력 완료 후, 원본/AFTER REFINE 전체 본문 + 메트릭 출력
    try:
        print("\n===== 원본 =====\n")
        print(sample_text)

        print("\n===== AFTER REFINE (비-스트림) =====\n")
        if non_stream_result is not None:
            print(non_stream_result)
        elif ns_err is not None:
            print(f"(실패) {ns_err}")
        else:
            print("(비어 있음)")

        print("\n===== AFTER REFINE (스트림) =====\n")
        if full_stream_text is not None:
            print(full_stream_text)
        elif st_err is not None:
            print(f"(실패) {st_err}")
        else:
            print("(비어 있음)")

        # 메트릭 블록: LiteLLM 출력 포맷/속성 참조
        # 참조: https://docs.litellm.ai/docs/completion/output
        try:
            metrics = globals().get("_LAST_RUN_METRICS", None)
            print("\n===== 메트릭 =====\n")
            if not metrics:
                print("(메트릭 없음)")
            else:
                # 간단 요약
                print(
                    "모델:",
                    metrics.get("model"),
                    "\n스트림:",
                    metrics.get("stream"),
                    "\nAPI 응답(ms):",
                    metrics.get("response_ms")
                    if metrics.get("response_ms") is not None
                    else "-",
                    sep=" ",
                )
                usage = metrics.get("usage") or {}
                print(
                    "\n토큰 - prompt:",
                    usage.get("prompt_tokens", "-"),
                    "completion:",
                    usage.get("completion_tokens", "-"),
                    "total:",
                    usage.get("total_tokens", "-"),
                )
                # 비용(USD): LiteLLM 제공 response_cost 또는 helper 계산값
                # 참조: https://docs.litellm.ai/docs/completion/token_usage
                rcost = metrics.get("response_cost")
                print("비용(USD):", rcost if rcost is not None else "-")
        except Exception as e:
            print(f"메트릭 출력 실패: {e}")
    except Exception as e:
        print(f"before/after 출력 실패: {e}", file=sys.stderr)
