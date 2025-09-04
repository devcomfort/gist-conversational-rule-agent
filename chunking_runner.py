#!/usr/bin/env python3
"""
Chunking Runner
===============

Phase 2-1: 청킹 데이터 생성 실행 스크립트 (편의용 wrapper)

사용법:
    python chunking_runner.py --help
    python chunking_runner.py --max-chunks 10 --chunkers token
"""

import sys
from core.experiment.chunking_builder import main

if __name__ == "__main__":
    sys.exit(main())
