"""
Gateway 직접 실행 진입점.

사용법:
    cd src/vllm_gateway
    python run.py
    
    # 또는 커스텀 포트
    GW_PORT=8080 python run.py
"""

import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import uvicorn
from core.config import settings


def main():
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        loop="uvloop",
        http="httptools",
        access_log=False,
    )


if __name__ == "__main__":
    main()
