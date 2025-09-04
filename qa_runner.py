#!/usr/bin/env python3
"""
QA Runner
=========

Phase 2-2: QA 데이터셋 생성 실행 스크립트 (편의용 wrapper)

사용법:
    python qa_runner.py --help
    python qa_runner.py --max-chunks 50 --model gpt-oss-120b
"""

import sys
from core.experiment.qa_generator import main

if __name__ == "__main__":
    sys.exit(main())
