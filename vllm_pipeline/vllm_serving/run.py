"""
vllm_serving 직접 실행 진입점.

사용법:
    cd src/vllm_serving
    python run.py

    # 커스텀 포트
    VS_PORT=17810 python run.py
"""

import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import uvicorn
from vllm_serving.config import settings


def main():
    uvicorn.run(
        "vllm_serving.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
