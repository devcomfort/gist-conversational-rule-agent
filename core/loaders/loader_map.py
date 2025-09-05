from llama_index.readers.file import HWPReader, DocxReader, PDFReader, PptxReader
from llama_index.core.readers.base import BaseReader

LOADER_MAP: dict[str, type[BaseReader]] = {
    ".hwp": HWPReader,
    ".docx": DocxReader,
    ".pdf": PDFReader,
    ".pptx": PptxReader,
}
"""
## Extension to Loader Map

파일 확장자별 문서 로더 매핑 딕셔너리.

각 지원 문서 형식의 확장자를 키로, 해당 형식을 처리할 수 있는 
LlamaIndex 리더 클래스를 값으로 하는 매핑입니다.

### 지원 형식
- `.hwp`: 한글 문서 (HWPReader)
- `.docx`: Microsoft Word 문서 (DocxReader)  
- `.pdf`: PDF 문서 (PDFReader)
- `.pptx`: Microsoft PowerPoint 문서 (PptxReader)

### 사용처
- `load_document.py`에서 파일 확장자에 따른 적절한 로더 선택
- `supported_document.py`에서 지원 확장자 목록 생성
- 새로운 문서 형식 지원 시 이 딕셔너리에 추가

### 주의사항
- 새로운 형식 추가 시 해당 리더 클래스를 import해야 함
- 확장자는 소문자로 통일 (예: `.PDF` → `.pdf`)
- 실제 파일 형식 판별은 MIME 타입 기반으로 수행됨

### 예시
```python
# 사용 예시
file_ext = ".pdf"
if file_ext in LOADER_MAP:
    loader_class = LOADER_MAP[file_ext]
    loader = loader_class()
```
"""
