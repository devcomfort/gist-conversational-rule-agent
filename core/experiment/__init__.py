"""
Experiment Package
=================

Phase 2 데이터 생성을 위한 모듈화된 실험 시스템

모듈 구성:
- commons: 공통 컴포넌트 (DocumentManager, ExperimentNaming 등)
- chunking_builder: Phase 2-1 청킹 데이터 생성
- qa_generator: Phase 2-2 QA 데이터셋 생성  
- embedding_builder: Phase 2-3 임베딩 데이터셋 생성

아키텍처: Microservice-like 독립 모듈
데이터 흐름: Documents → Chunking → QA/Embedding
"""

# 주요 컴포넌트 export
from .commons import (
    DocumentManager,
    TokenizationManager,
    ExperimentNaming,
    ExperimentState,
    ExperimentResult,
)

__all__ = [
    "DocumentManager",
    "TokenizationManager", 
    "ExperimentNaming",
    "ExperimentState",
    "ExperimentResult",
]
