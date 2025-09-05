from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Union, Optional, List
import numpy as np
from toolz import pipe

from .is_supported_document import is_supported_document


def _iter_supported_document_paths(root_path: Union[Path, str]) -> Iterator[Path]:
    """내부 제너레이터: 문서 파일 경로를 지연 평가(lazy)로 반환합니다.

    Parameters
    ----------
    root_path : Path
        스캔을 시작할 디렉토리 경로.

    Returns
    -------
    Iterator[Path]
        조건을 만족하는 파일 경로들을 순회 가능한 제너레이터로 반환합니다.

    Notes
    -----
    - 파일 타입 추정은 `core.utils.get_file_type.get_file_type`을 사용합니다.
    - 필터 기준: `is_supported_document(path)`가 적용됩니다.
    - 예외는 내부에서 삼켜지고(False 처리), 순회는 계속됩니다.
    - 대용량 디렉토리에서도 메모리 효율적으로 동작합니다(지연 평가).
    """
    # Path 객체로 변환 (str도 지원하기 위해)
    root_path = Path(root_path)

    # Guard clause: 존재하지 않으면 빈 이터레이터 반환
    if not root_path.exists():
        return iter(())

    # os.walk는 (dirpath, dirnames, filenames)를 생성합니다.
    # 아래 제너레이터는 파일 단위 Path를 구성한 뒤, 지원 문서인지 판별하여 통과시킵니다.
    paths = (
        Path(dirpath) / name
        for dirpath, _dirnames, filenames in os.walk(root_path)
        for name in filenames
    )
    # is_supported_document(path): MIME/확장자 기반 정책을 캡슐화한 필터 함수(별도 모듈)
    return (path for path in paths if is_supported_document(path))


def _sample_supported_document_paths(
    paths: Iterable[Union[Path, str]], n: int
) -> Iterable[str]:
    """지원 문서 경로 컬렉션에서 무작위 샘플 n개를 반환합니다.

    Parameters
    ----------
    paths : Iterable[Path | str]
        샘플링 대상 경로 이터러블(제너레이터/리스트 모두 가능).
    n : int
        추출할 샘플 개수(비복원 추출). 0 이상이어야 합니다.

    Returns
    -------
    Iterable[Path]
        샘플링된 경로들의 이터러블(Path)을 반환합니다.

    Notes
    -----
    - 현재 구현은 전체 경로를 메모리에 적재(list)한 뒤 샘플링합니다.
      매우 큰 디렉토리에서는 스트리밍 샘플링으로 교체를 고려하세요.
    - 무작위성/재현성을 위해 외부에서 시드 고정을 권장합니다.
    - 의존성: numpy 필요(추후 표준 라이브러리 기반으로 대체 가능).
    - TODO: n이 전체 개수를 초과하는 경우의 동작 정의(자동 min 적용/예외 등).
    """
    # 전체 후보 경로 수집(지연 평가 → 물질화)
    paths = list(paths)
    # 비복원 샘플링: numpy 사용(향후 대체 가능)
    samples = np.random.choice(np.array(paths), n, replace=False)
    return iter(samples)


def collect_supported_document_paths(
    root: Union[Path, str],
    lazy: bool = True,
    n: Optional[int] = None,
) -> Iterable[Path]:
    """지원 문서 경로를 수집합니다.

    Parameters
    ----------
    root : Path | str
        스캔을 시작할 루트 경로.
    lazy : bool, default True
        True이면 지연 평가 가능한 iterable을, False이면 즉시 평가된 list를 반환합니다.
    n : int | None, default None
        지정 시 결과에서 정확히 n개를 무작위 샘플링해 반환합니다.

    Returns
    -------
    Iterable[Path] | List[Path]
        lazy=True → Iterable[Path], lazy=False → List[Path].

    Notes
    -----
    - 내부적으로 `_iter_supported_document_paths`를 사용해 지원 문서만 필터링합니다.
    - 반환 타입은 `lazy` 인자에 따라 결정되며, 타입 힌트와 일치합니다.
    """
    return pipe(
        _iter_supported_document_paths(root),
        lambda paths: paths if lazy else list(paths),
        lambda paths: paths
        if n is None
        else _sample_supported_document_paths(paths, n),
    )


if __name__ == "__main__":
    # 로컬 실행용 간단 데모
    try:
        from tabulate import tabulate  # type: ignore
    except Exception:  # pragma: no cover
        tabulate = None  # fallback

    print("\n=== _iter_supported_document_paths (first 10) ===")
    it_paths = _iter_supported_document_paths(Path("."))
    first10 = [str(p) for _, p in zip(range(10), it_paths)]
    for p in first10:
        print(p)

    print("\n=== collect_supported_document_paths (lazy=True, first 10) ===")
    it2 = collect_supported_document_paths(Path("."), lazy=True)
    print(f"type(lazy=True): {type(it2).__name__}")
    first10_lazy = [str(p) for _, p in zip(range(10), it2)]
    for p in first10_lazy:
        print(p)

    print("\n=== collect_supported_document_paths (lazy=False, count & first 10) ===")
    all_paths = list(collect_supported_document_paths(Path("."), lazy=False))
    print(f"type(lazy=False): {type(all_paths).__name__}")
    print(f"count={len(all_paths)}")
    for p in all_paths[:10]:
        print(str(p))

    print("\n🎲 Test 4: _sample_supported_document_paths (n=10), repeat=10")
    print("-" * 60)

    if not all_paths:
        print("  ❌ Cannot perform sampling test: no documents found")
    elif len(all_paths) < 10:
        print(
            f"  ⚠️  Only {len(all_paths)} documents found, using n={len(all_paths)} for sampling"
        )
        N_SAMPLES = len(all_paths)
    else:
        N_SAMPLES = 10

    if all_paths:
        # 수집: 각 run 별로 샘플 N_SAMPLES개씩
        samples_matrix: List[List[str]] = []  # shape: [runs][samples]
        for run_idx in range(10):  # 10번 반복
            try:
                sampled = list(_sample_supported_document_paths(all_paths, N_SAMPLES))
                samples_matrix.append([str(x) for x in sampled])
            except ValueError as e:
                print(f"  ❌ Sampling failed on run {run_idx + 1}: {e}")
                break

        if samples_matrix:
            # 전치: 행=sample_index, 열=run
            sample_rows: List[List[str]] = list(
                map(list, zip(*samples_matrix))
            )  # shape: [samples][runs]

            # 파일명만 추출 (경로가 너무 길 경우)
            sample_rows_short = [
                [Path(path).name for path in row] for row in sample_rows
            ]

            headers = ["Sample #"] + [
                f"Run {i}" for i in range(1, len(samples_matrix) + 1)
            ]
            rows = [
                [f"sample_{idx}"] + row
                for idx, row in enumerate(sample_rows_short, start=1)
            ]

            print(
                f"  📊 Sampling Results: {N_SAMPLES} samples × {len(samples_matrix)} runs"
            )
            if tabulate:
                print(tabulate(rows, headers=headers, tablefmt="grid", maxcolwidths=15))
            else:
                # Fallback formatting
                print("  " + " | ".join(f"{h:>15}" for h in headers))
                print("  " + "-" * (len(headers) * 17 - 1))
                for row in rows:
                    print("  " + " | ".join(f"{str(cell):>15}" for cell in row))

    print("\n🔍 Test 5: collect_supported_document_paths lazy 타입 검증")
    print("-" * 60)

    # lazy=True 테스트
    lazy_result = collect_supported_document_paths(Path("."), lazy=True)
    lazy_type = type(lazy_result).__name__
    lazy_module = type(lazy_result).__module__
    print("  📊 lazy=True:")
    print(f"    타입: {lazy_type}")
    print(f"    모듈: {lazy_module}")
    print("    설명: 지연 평가 가능한 제너레이터")

    # lazy=False 테스트
    eager_result = collect_supported_document_paths(Path("."), lazy=False)
    eager_type = type(eager_result).__name__
    eager_module = type(eager_result).__module__
    print("  📊 lazy=False:")
    print(f"    타입: {eager_type}")
    print(f"    모듈: {eager_module}")
    print("    설명: 즉시 평가된 리스트")
    if isinstance(eager_result, list):
        print(f"    길이: {len(eager_result)} 개 문서")
    else:
        print("    길이: 제너레이터이므로 길이 측정 불가")

    # 타입 검증
    from collections.abc import Iterable

    print(f"  ✅ lazy=True는 Iterable: {isinstance(lazy_result, Iterable)}")
    print(f"  ✅ lazy=False는 list: {isinstance(eager_result, list)}")
    print(
        f"  ✅ 둘 다 Iterable: {isinstance(lazy_result, Iterable) and isinstance(eager_result, Iterable)}"
    )
