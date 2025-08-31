"""공용 문서 형식 타입 정의.

경고(중요): 문서 종류는 RAG 시스템이 "처리 가능한가/처리해야 하는가"를 기준으로 결정됩니다.
- 본 정의는 2025-08-30 패치 기준으로 유지되는 문서 형식만 포함합니다.
- 처리 정책이 변경되면 반드시 이 타입을 업데이트해야 합니다.
"""

from typing import Literal, TypeAlias

DocumentKind: TypeAlias = Literal["pdf", "hwp", "doc", "docx", "ppt", "pptx"]
"""RAG에서 처리 가능한/처리해야 하는 문서 형식 집합.

- 2025-08-30 기준 형식: pdf, hwp, doc, docx, ppt, pptx
- 처리 정책/지원 형식 변경 시 이 타입을 반드시 업데이트할 것
"""

__all__ = ["DocumentKind"]
