"""
File Validators Package
======================

다양한 파일 형식 검증 유틸리티들을 제공하는 패키지

지원하는 파일 형식:
- 문서: PDF, HWP, DOC, DOCX, PPT, PPTX
- 텍스트: TXT, JSON

모듈 구성:
- is_pdf: PDF 파일 검증
- is_hwp: 한글(HWP) 파일 검증  
- is_doc: Microsoft Word DOC 파일 검증
- is_docx: Microsoft Word DOCX 파일 검증
- is_ppt: Microsoft PowerPoint PPT 파일 검증
- is_pptx: Microsoft PowerPoint PPTX 파일 검증
- is_txt: 텍스트(TXT) 파일 검증
- is_json: JSON 파일 검증
"""

from .is_doc import is_doc
from .is_docx import is_docx
from .is_hwp import is_hwp
from .is_json import is_json
from .is_pdf import is_pdf
from .is_ppt import is_ppt
from .is_pptx import is_pptx
from .is_txt import is_txt

__all__ = [
    "is_doc",
    "is_docx",
    "is_hwp",
    "is_json",
    "is_pdf",
    "is_ppt",
    "is_pptx",
    "is_txt",
]