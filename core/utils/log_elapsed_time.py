from __future__ import annotations

import time
from functools import wraps
from typing import Callable, TypeVar, ParamSpec
from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


def log_duration(stage_name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """단계 함수 데코레이터: 시작/완료와 경과시간(ms) 로깅.

    Args:
        stage_name: 로그에 표시할 단계 이름

    Returns:
        데코레이터: 원래 함수 시그니처를 보존한 래퍼 반환
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
