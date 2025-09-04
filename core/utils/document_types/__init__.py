"""
Document Types Package
=====================

RAG 시스템에서 처리 가능한 문서 타입 정의 모듈

모듈 구성:
- document_kind: 문서 형식 타입 정의 (DocumentKind)
"""

from .document_kind import DocumentKind

__all__ = ["DocumentKind"]