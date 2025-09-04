from __future__ import annotations

import time
from functools import wraps
from typing import Callable, TypeVar, ParamSpec
from loguru import logger

P = ParamSpec("P")
"""ParamSpec P: 래핑 대상 함수의 매개변수 시그니처를 보존하기 위한 타입 파라미터."""

R = TypeVar("R")
"""TypeVar R: 래핑 대상 함수의 반환 타입을 나타내는 타입 파라미터."""


def log_elapsed_time(stage_name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """[Deprecated] log_elapsed_time로 대체 예정.

    단계 함수 데코레이터: 시작/완료와 경과 시간(ms)을 로깅합니다.
    새 코드에서는 `log_elapsed_time`을 사용하세요.

    Parameters
    ----------
    stage_name : str
        로그에 표시할 단계 이름

    Returns
    -------
    Callable[[Callable[P, R]], Callable[P, R]]
        원래 함수 시그니처를 보존한 데코레이터

    Type Variables
    --------------
    P : ParamSpec
        데코레이터가 감싸는 원본 함수의 매개변수 시그니처를 보존하기 위한 타입 파라미터
    R : TypeVar
        원본 함수의 반환 타입을 나타내는 타입 파라미터
    """

    def _decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.info("{} 시작", stage_name)
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info("{} 완료: elapsed_ms={:.1f}", stage_name, elapsed_ms)
            return result

        return _wrapper

    return _decorator
