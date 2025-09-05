"""지원 문서 형식 정의

로더 시스템(`core/loaders/*`)이 현재 "처리 대상"으로 간주하는 파일 확장자 집합을 정의합니다.
이 목록은 확장자 기반 사전 필터링, 설정 검증, UX 표시 등에 활용됩니다.

주의
- 실제 형식 판별과 로딩은 MIME 기반 추정(`core/utils/get_mime_type.py:get_file_type`)과
  개별 로더(`core/loaders/load_document.py`, `core/loaders/load_pdf.py`,
  `core/loaders/load_docx.py`, `core/loaders/load_pptx.py`, `core/loaders/load_hwp.py`)에서 수행됩니다.
- 지원 형식 추가/제거 시 위 모듈들도 함께 갱신해야 일관성이 유지됩니다.
"""

from .loader_map import LOADER_MAP


SUPPORTED_DOCUMENT_KIND: set[str] = set(LOADER_MAP.keys())
"""시스템에서 지원하는 문서 확장자 집합.

`LOADER_MAP`에서 자동으로 생성되는 지원 문서 확장자들의 집합입니다.
파일 처리 전 사전 필터링, 설정 검증, 사용자 인터페이스 표시 등에 활용됩니다.

### 현재 지원 확장자
- `.hwp`: 한글 문서
- `.docx`: Microsoft Word 문서  
- `.pdf`: PDF 문서
- `.pptx`: Microsoft PowerPoint 문서

### 주요 사용처
- **사전 필터링**: 파일 수집 시 지원되지 않는 형식 제외
- **설정 검증**: 사용자 설정의 파일 형식 유효성 검사
- **UX 표시**: 지원 형식 목록을 사용자에게 안내
- **타입 체크**: 런타임에서 파일 확장자 유효성 검증

### 아키텍처 관계
- **데이터 소스**: `LOADER_MAP` 딕셔너리의 키 집합
- **형식 판별**: `core/utils/get_mime_type.py:get_file_type`에서 MIME 기반 판별
- **실제 로딩**: 개별 로더 모듈들에서 수행
  - `core/loaders/load_document.py` (통합 로더)
  - `core/loaders/load_pdf.py`
  - `core/loaders/load_docx.py` 
  - `core/loaders/load_pptx.py`
  - `core/loaders/load_hwp.py`

### 확장 시 주의사항
- 새로운 형식 추가 시 `LOADER_MAP`에 먼저 추가하면 자동으로 반영됨
- 개별 로더 모듈과 MIME 타입 판별 로직도 함께 구현해야 함
- 기존 코드에서 하드코딩된 확장자 목록이 있다면 함께 갱신 필요

### TODO:
- `.ppt`, `.doc` 지원 시 해당 로더 구현 후 `LOADER_MAP`에 추가
- 이미지 기반 문서 형식 (스캔 PDF 등) 지원 검토

### 예시
```python
# 파일 확장자 검증
file_ext = ".pdf"
if file_ext in SUPPORTED_DOCUMENT_KIND:
    print(f"{file_ext} 형식은 지원됩니다.")

# 지원 형식 목록 표시
print("지원 형식:", ", ".join(sorted(SUPPORTED_DOCUMENT_KIND)))
```
"""
