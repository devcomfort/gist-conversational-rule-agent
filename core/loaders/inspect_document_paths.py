"""지원되는 문서 경로 분석 도구.

특정 디렉토리의 지원되는 문서 파일들을 확장자별로 카운트하여 분석하는 모듈입니다.
기존 모듈들(`get_file_type`, `collect_supported_document_paths` 등)을 활용하여 구현되었습니다.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Literal, List, Optional, Annotated
from operator import itemgetter
import numpy as np

from toolz import pipe
import polars as pl
import typer

from core.utils.get_file_type import get_file_type
from .collect_supported_document_paths import collect_supported_document_paths
from .supported_document import SUPPORTED_DOCUMENT_KIND

# 더 직관적인 타입명으로 개선
DocumentExtensionCounts = Dict[str, int]
"""문서 확장자별 개수를 나타내는 딕셔너리 타입.

Key: 파일 확장자 (예: '.pdf', '.hwp', '.docx')
Value: 해당 확장자를 가진 파일의 개수
"""

DocumentCountMetric = Dict[Literal["supported", "unsupported"], DocumentExtensionCounts]
"""지원/비지원 문서별 확장자 카운트를 나타내는 딕셔너리 타입.

Key: "supported" 또는 "unsupported"
Value: 해당 카테고리의 확장자별 개수 딕셔너리
"""


def inspect_supported_document_paths(root: Union[Path, str]) -> DocumentExtensionCounts:
    """지원되는 문서 파일들을 확장자별로 카운트합니다.

    Parameters
    ----------
    root : Path | str
        분석할 루트 디렉토리 경로

    Returns
    -------
    DocumentExtensionCounts
        확장자별 파일 개수를 담은 딕셔너리
        예: {'.pdf': 150, '.hwp': 80, '.docx': 45, '.pptx': 25}

    Examples
    --------
    >>> counts = inspect_supported_document_paths("./documents")
    >>> counts['.pdf']
    150
    >>> sum(counts.values())  # 총 지원 문서 수
    300

    Notes
    -----
    - 지원되는 문서만 카운트됩니다 (PDF, HWP, DOCX, PPTX)
    - MIME 타입 감지는 `get_file_type()` 함수를 사용합니다
    - 내부적으로 `collect_supported_document_paths()`를 활용합니다
    """
    # numpy의 unique_counts 결과를 딕셔너리로 변환
    unique_extensions, counts = pipe(
        root,
        collect_supported_document_paths,
        lambda paths: map(get_file_type, paths),
        lambda file_types: map(itemgetter(1), file_types),  # 확장자만 추출
        lambda extensions: filter(None, extensions),  # None 값 제거
        list,
        lambda extensions: np.unique(extensions, return_counts=True),
    )

    # numpy 배열을 순수 Python 딕셔너리로 변환
    return {str(ext): int(count) for ext, count in zip(unique_extensions, counts)}


def inspect_all_document_paths(root: Union[Path, str]) -> DocumentCountMetric:
    """모든 문서 파일들을 지원/비지원별로 확장자별 카운트합니다.

    Parameters
    ----------
    root : Path | str
        분석할 루트 디렉토리 경로

    Returns
    -------
    DocumentCountMetric
        지원/비지원별 확장자별 파일 개수를 담은 딕셔너리
        예: {
            "supported": {'.pdf': 150, '.hwp': 80, '.docx': 45, '.pptx': 25},
            "unsupported": {'.txt': 100, '.jpg': 50, '.py': 30}
        }

    Examples
    --------
    >>> counts = inspect_all_document_paths("./documents")
    >>> counts["supported"]['.pdf']
    150
    >>> sum(counts["supported"].values())  # 총 지원 문서 수
    300
    >>> sum(counts["unsupported"].values())  # 총 비지원 문서 수
    180

    Notes
    -----
    - 모든 파일을 스캔하여 지원/비지원으로 분류합니다
    - MIME 타입 감지는 `get_file_type()` 함수를 사용합니다
    - 지원 여부는 `SUPPORTED_DOCUMENT_KIND`를 기준으로 판단합니다
    """
    root_path = Path(root)

    if not root_path.exists():
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {root}")

    supported_extensions = []
    unsupported_extensions = []

    # 디렉토리 순회하여 모든 파일 검사
    for current_dir, dirs, files in os.walk(root_path):
        for file_name in files:
            file_path = Path(current_dir) / file_name

            # 파일 타입 감지
            file_type = get_file_type(file_path)
            _, ext = file_type if file_type else (None, None)

            # 확장자가 없으면 파일명에서 추출 시도
            if not ext:
                ext = file_path.suffix or "no_extension"

            # 지원 여부에 따라 분류
            if ext.lower() in SUPPORTED_DOCUMENT_KIND:
                supported_extensions.append(ext)
            else:
                unsupported_extensions.append(ext)

    # numpy unique_counts로 카운트
    supported_unique, supported_counts = (
        np.unique(supported_extensions, return_counts=True)
        if supported_extensions
        else ([], [])
    )
    unsupported_unique, unsupported_counts = (
        np.unique(unsupported_extensions, return_counts=True)
        if unsupported_extensions
        else ([], [])
    )

    return {
        "supported": {
            str(ext): int(count)
            for ext, count in zip(supported_unique, supported_counts)
        },
        "unsupported": {
            str(ext): int(count)
            for ext, count in zip(unsupported_unique, unsupported_counts)
        },
    }


def collect_all_document_file_info(root: Union[Path, str]) -> List[Dict[str, str]]:
    """모든 파일의 정보를 수집하여 리스트로 반환합니다.

    Parameters
    ----------
    root : Path | str
        분석할 루트 디렉토리 경로

    Returns
    -------
    List[Dict[str, str]]
        각 파일의 정보를 담은 딕셔너리 리스트
        - is_supported: "True" 또는 "False"
        - path: 파일의 절대 경로
        - extension: 파일 확장자 (없으면 "no_extension")

    Examples
    --------
    >>> file_info = collect_all_document_file_info("./docs")
    >>> print(file_info[0])
    {'is_supported': 'True', 'path': '/abs/path/to/doc.pdf', 'extension': '.pdf'}
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {root}")

    file_info_list = []

    for current_dir, dirs, files in os.walk(root_path):
        for file_name in files:
            file_path = Path(current_dir) / file_name
            absolute_path = file_path.resolve()

            # 파일 타입 감지
            file_type = get_file_type(file_path)
            _, ext = file_type if file_type else (None, None)

            if not ext:
                ext = file_path.suffix or "no_extension"

            # 지원 여부 확인
            is_supported = ext.lower() in SUPPORTED_DOCUMENT_KIND

            file_info_list.append(
                {
                    "is_supported": str(is_supported),
                    "path": str(absolute_path),
                    "extension": ext,
                }
            )

    return file_info_list


def save_inspection_to_csv(
    file_info_list: List[Dict[str, str]], output_dir: Optional[Union[Path, str]] = None
) -> Path:
    """파일 정보를 CSV로 저장합니다.

    Parameters
    ----------
    file_info_list : List[Dict[str, str]]
        저장할 파일 정보 리스트
    output_dir : Path | str, optional
        출력 디렉토리 경로. None이면 프로젝트 루트의 /log 디렉토리 사용

    Returns
    -------
    Path
        저장된 CSV 파일의 경로

    Examples
    --------
    >>> file_info = collect_all_document_file_info("./docs")
    >>> csv_path = save_inspection_to_csv(file_info)
    >>> print(f"CSV 저장됨: {csv_path}")
    """
    # 출력 디렉토리 설정
    if output_dir is None:
        # 프로젝트 루트 디렉토리 찾기
        current_path = Path(__file__).resolve()
        project_root = current_path
        while project_root.parent != project_root:
            if (project_root / "pyproject.toml").exists() or (
                project_root / ".git"
            ).exists():
                break
            project_root = project_root.parent
        output_dir = project_root / "log"
    else:
        output_dir = Path(output_dir)

    # 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    # 타임스탬프 기반 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"document_inspection_{timestamp}.csv"
    csv_path = output_dir / csv_filename

    # Polars DataFrame 생성 및 저장
    df = pl.DataFrame(file_info_list)
    df.write_csv(csv_path)

    return csv_path


def _format_inspection_table(
    counts: DocumentExtensionCounts,
    root_path: Union[Path, str],
    sort_by_count: bool = True,
) -> str:
    """분석 결과를 표 형태로 포맷팅합니다.

    Parameters
    ----------
    counts : DocumentExtensionCounts
        확장자별 파일 개수 딕셔너리
    root_path : Path | str
        분석한 루트 경로
    sort_by_count : bool, default True
        개수 기준으로 정렬할지 여부 (False면 확장자명 기준)

    Returns
    -------
    str
        포맷팅된 표 문자열

    Examples
    --------
    >>> counts = {'.pdf': 150, '.hwp': 80}
    >>> table = format_inspection_table(counts, "./docs")
    >>> print(table)
    """
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    lines = []
    lines.append("=" * 70)
    lines.append("📊 Supported Document Path Inspection")
    lines.append("=" * 70)
    lines.append(f"📁 Analyzed Path: {Path(root_path).resolve()}")

    if not counts:
        lines.append("\n❌ No supported documents found in the specified path.")
        lines.append("=" * 70)
        return "\n".join(lines)

    # 총 파일 수 계산
    total_files = sum(counts.values())
    lines.append(f"📋 Total Supported Documents: {total_files:,}")
    lines.append("")

    # 정렬
    if sort_by_count:
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_items = sorted(counts.items())

    # 표 생성
    if tabulate:
        table_data = []
        for ext, count in sorted_items:
            percentage = (count / total_files * 100) if total_files > 0 else 0
            table_data.append([ext, count, f"{percentage:.1f}%"])

        # Total 행 추가
        table_data.append(["**Total**", total_files, "100.0%"])

        table = tabulate(
            table_data,
            headers=["Extension", "Count", "Percentage"],
            tablefmt="grid",
            numalign="right",
            stralign="center",
        )
        lines.append(table)
    else:
        # tabulate가 없는 경우 fallback
        lines.append("Extension | Count | Percentage")
        lines.append("-" * 30)
        for ext, count in sorted_items:
            percentage = (count / total_files * 100) if total_files > 0 else 0
            lines.append(f"{ext:>9} | {count:>5} | {percentage:>6.1f}%")
        lines.append("-" * 30)
        lines.append(f"{'**Total**':>9} | {total_files:>5} | {'100.0%':>6}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("✅ Inspection Complete")
    lines.append("=" * 70)

    return "\n".join(lines)


def _format_all_documents_table(
    metric: DocumentCountMetric,
    root_path: Union[Path, str],
    sort_by_count: bool = True,
) -> str:
    """지원/비지원 문서 분석 결과를 표 형태로 포맷팅합니다.

    Parameters
    ----------
    metric : DocumentCountMetric
        지원/비지원별 확장자별 파일 개수 딕셔너리
    root_path : Path | str
        분석한 루트 경로
    sort_by_count : bool, default True
        개수 기준으로 정렬할지 여부 (False면 확장자명 기준)

    Returns
    -------
    str
        포맷팅된 표 문자열
    """
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    lines = []
    lines.append("=" * 80)
    lines.append("📊 All Document Path Inspection (Supported + Unsupported)")
    lines.append("=" * 80)
    lines.append(f"📁 Analyzed Path: {Path(root_path).resolve()}")

    supported_counts = metric.get("supported", {})
    unsupported_counts = metric.get("unsupported", {})

    total_supported = sum(supported_counts.values())
    total_unsupported = sum(unsupported_counts.values())
    total_files = total_supported + total_unsupported

    lines.append(f"📋 Total Files: {total_files:,}")
    lines.append(f"  • Supported Documents: {total_supported:,}")
    lines.append(f"  • Unsupported Documents: {total_unsupported:,}")
    lines.append("")

    if not supported_counts and not unsupported_counts:
        lines.append("❌ No files found in the specified path.")
        lines.append("=" * 80)
        return "\n".join(lines)

    # 지원되는 문서 표
    if supported_counts:
        lines.append("🟢 Supported Documents:")

        # 정렬
        if sort_by_count:
            sorted_supported = sorted(
                supported_counts.items(), key=lambda x: x[1], reverse=True
            )
        else:
            sorted_supported = sorted(supported_counts.items())

        if tabulate:
            table_data = []
            for ext, count in sorted_supported:
                percentage = (
                    (count / total_supported * 100) if total_supported > 0 else 0
                )
                table_data.append([ext, count, f"{percentage:.1f}%"])

            # Total 행 추가
            table_data.append(["**Total**", total_supported, "100.0%"])

            table = tabulate(
                table_data,
                headers=["Extension", "Count", "Percentage"],
                tablefmt="grid",
                numalign="right",
                stralign="center",
            )
            lines.append(table)
        else:
            # tabulate가 없는 경우 fallback
            lines.append("Extension | Count | Percentage")
            lines.append("-" * 30)
            for ext, count in sorted_supported:
                percentage = (
                    (count / total_supported * 100) if total_supported > 0 else 0
                )
                lines.append(f"{ext:>9} | {count:>5} | {percentage:>6.1f}%")
            lines.append("-" * 30)
            lines.append(f"{'**Total**':>9} | {total_supported:>5} | {'100.0%':>6}")

        lines.append("")

    # 지원되지 않는 문서 표 (상위 10개만)
    if unsupported_counts:
        lines.append("🔴 Unsupported Documents (Top 10):")

        # 정렬 (개수 기준으로 상위 10개)
        sorted_unsupported = sorted(
            unsupported_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]

        if tabulate:
            table_data = []
            for ext, count in sorted_unsupported:
                percentage = (
                    (count / total_unsupported * 100) if total_unsupported > 0 else 0
                )
                table_data.append([ext, count, f"{percentage:.1f}%"])

            # 표시된 항목들의 합계
            shown_count = sum(count for _, count in sorted_unsupported)
            if len(unsupported_counts) > 10:
                others_count = total_unsupported - shown_count
                table_data.append(
                    [
                        "**Others**",
                        others_count,
                        f"{(others_count / total_unsupported * 100):.1f}%",
                    ]
                )

            table_data.append(["**Total**", total_unsupported, "100.0%"])

            table = tabulate(
                table_data,
                headers=["Extension", "Count", "Percentage"],
                tablefmt="grid",
                numalign="right",
                stralign="center",
            )
            lines.append(table)
        else:
            # tabulate가 없는 경우 fallback
            lines.append("Extension | Count | Percentage")
            lines.append("-" * 30)
            for ext, count in sorted_unsupported:
                percentage = (
                    (count / total_unsupported * 100) if total_unsupported > 0 else 0
                )
                lines.append(f"{ext:>9} | {count:>5} | {percentage:>6.1f}%")

            # Others 행
            if len(unsupported_counts) > 10:
                shown_count = sum(count for _, count in sorted_unsupported)
                others_count = total_unsupported - shown_count
                lines.append(
                    f"{'**Others**':>9} | {others_count:>5} | {(others_count / total_unsupported * 100):>6.1f}%"
                )

            lines.append("-" * 30)
            lines.append(f"{'**Total**':>9} | {total_unsupported:>5} | {'100.0%':>6}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("✅ Inspection Complete")
    lines.append("=" * 80)

    return "\n".join(lines)


# Typer 앱 생성
app = typer.Typer(
    name="inspect-document-paths",
    help="📊 문서 경로 분석 도구 - 모든 파일의 확장자별 통계를 제공합니다.",
    epilog="""
🔍 예시:
  python -m core.loaders.inspect_document_paths ./documents
  python -m core.loaders.inspect_document_paths ./rules --sort-by-name
  python -m core.loaders.inspect_document_paths ./rules --supported-only --save-csv
    """,
    rich_markup_mode="rich",
)


@app.command()
def main(
    directory: Annotated[
        str,
        typer.Argument(help="📁 분석할 디렉토리 경로", show_default="현재 디렉토리"),
    ] = ".",
    sort_by_name: Annotated[
        bool,
        typer.Option(
            "--sort-by-name",
            "-n",
            help="📝 확장자명 기준으로 정렬 (기본값: 개수 기준 정렬)",
            show_default="개수 기준 정렬",
        ),
    ] = False,
    supported_only: Annotated[
        bool,
        typer.Option(
            "--supported-only",
            "-s",
            help="📝 지원되는 문서만 분석 (기본값: 모든 문서 포함)",
            show_default="모든 문서 포함",
        ),
    ] = False,
    save_csv: Annotated[
        bool,
        typer.Option(
            "--save-csv",
            "-c",
            help="💾 분석 결과를 CSV 파일로 저장 (개별 파일 정보 포함)",
            show_default="CSV 저장 안함",
        ),
    ] = False,
    output_dir: Annotated[
        Optional[str],
        typer.Option(
            "--output-dir",
            "-o",
            help="📂 CSV 파일 저장 디렉토리",
            show_default="프로젝트 루트/log",
        ),
    ] = None,
):
    """
    📊 문서 경로를 분석하여 확장자별 통계를 제공합니다.

    이 도구는 지정된 디렉토리를 재귀적으로 탐색하여 모든 파일을 분석하고
    지원되는 문서 형식(.pdf, .hwp, .docx, .pptx)과 비지원 형식을 구분하여
    확장자별로 분류합니다.

    🎯 주요 기능:
    • 확장자별 파일 개수 통계 (기본: 모든 파일 포함)
    • 지원/비지원 문서 분류 및 통계
    • CSV 형태로 개별 파일 정보 내보내기
    • 다양한 정렬 옵션
    """
    try:
        if supported_only:
            # 지원되는 문서만 분석
            typer.echo("🔍 지원되는 문서만 분석 중...")
            counts = inspect_supported_document_paths(directory)
            table = _format_inspection_table(
                counts, directory, sort_by_count=not sort_by_name
            )
        else:
            # 모든 파일 분석 (지원/비지원 포함) - 기본값
            typer.echo("🔍 모든 파일 분석 중 (지원/비지원 포함)...")
            metric = inspect_all_document_paths(directory)
            table = _format_all_documents_table(
                metric, directory, sort_by_count=not sort_by_name
            )

        typer.echo(table)

        # CSV 저장 처리
        if save_csv:
            typer.echo("\n" + "=" * 70)
            typer.echo("💾 CSV 파일 저장 중...")

            try:
                # 개별 파일 정보 수집
                file_info_list = collect_all_document_file_info(directory)

                # CSV 저장
                csv_path = save_inspection_to_csv(file_info_list, output_dir=output_dir)

                typer.echo(f"✅ CSV 파일이 저장되었습니다: {csv_path}")
                typer.echo(
                    f"📊 총 {len(file_info_list):,}개 파일 정보가 저장되었습니다."
                )

                # 간단한 통계 출력
                supported_count = sum(
                    1 for info in file_info_list if info["is_supported"] == "True"
                )
                unsupported_count = len(file_info_list) - supported_count
                typer.echo(f"   • 지원 문서: {supported_count:,}개")
                typer.echo(f"   • 비지원 문서: {unsupported_count:,}개")

            except Exception as csv_error:
                typer.echo(f"❌ CSV 저장 중 오류 발생: {csv_error}", err=True)

            typer.echo("=" * 70)

    except FileNotFoundError as e:
        typer.echo(f"❌ 오류: 디렉토리를 찾을 수 없습니다 - {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ 예상치 못한 오류: {e}", err=True)
        raise typer.Exit(1)


def _main():
    """CLI 진입점 (하위 호환성을 위해 유지)."""
    app()


if __name__ == "__main__":
    exit(_main())
