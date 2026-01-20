#!/bin/bash

# 일일 파이프라인 실행 스크립트

# 프로젝트 루트 디렉토리로 이동
cd "/Users/yu_seok/Documents/workspace/nbCamp/Project/Yahoo Finance"

# Python 환경 활성화 (필요한 경우)
# source /path/to/venv/bin/activate

# 파이프라인 실행 (automation 폴더의 스크립트 실행)
/opt/miniconda3/bin/python automation/daily_pipeline.py

# 실행 결과 코드
EXIT_CODE=$?

# 로그 기록
echo "$(date): Pipeline executed with exit code $EXIT_CODE" >> logs/execution.log

exit $EXIT_CODE
