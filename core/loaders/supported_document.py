"""지원 문서 형식 정의

로더 시스템(`core/loaders/*`)이 현재 "처리 대상"으로 간주하는 파일 확장자 집합을 정의합니다.
이 목록은 확장자 기반 사전 필터링, 설정 검증, UX 표시 등에 활용됩니다.

주의
- 실제 형식 판별과 로딩은 MIME 기반 추정(`core/utils/get_mime_type.py:get_file_type`)과
  개별 로더(`core/loaders/load_document.py`, `core/loaders/load_pdf.py`,
  `core/loaders/load_docx.py`, `core/loaders/load_pptx.py`, `core/loaders/load_hwp.py`)에서 수행됩니다.
- 지원 형식 추가/제거 시 위 모듈들도 함께 갱신해야 일관성이 유지됩니다.
"""

SUPPORTED_DOCUMENT_KIND: set[str] = {".pdf", ".hwp", ".docx", ".pptx"}
"""지원 문서 확장자 집합.

### Notes
- **사용처**: 사전 필터링, 설정 검증, UX 표시 등.
- **주의**: 실제 형식 판별/로딩은 `core/utils/get_mime_type.py:get_file_type` 및
  개별 로더(`core/loaders/load_document.py`, `core/loaders/load_pdf.py`,
  `core/loaders/load_docx.py`, `core/loaders/load_pptx.py`, `core/loaders/load_hwp.py`)에서 수행됩니다.
- **동기화**: 목록 변경 시 위 모듈들도 함께 갱신해야 일관성이 유지됩니다.

### Todo
- `.ppt`, `.doc` 지원 확정 시 문서 로더 구현 후 테스트해야함. (load_ppt, load_doc 만들어야 됩니다)
"""
