#!/usr/bin/env python3
"""
Embedding Runner
================

Phase 2-3: 임베딩 데이터셋 생성 실행 스크립트 (편의용 wrapper)

사용법:
    python embedding_runner.py --help
    python embedding_runner.py --max-combinations 100 --embedding-models qwen3_8b,bge_m3
"""

import sys
from core.experiment.embedding_builder import main

if __name__ == "__main__":
    sys.exit(main())
