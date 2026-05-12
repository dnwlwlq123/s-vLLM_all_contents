#!/bin/bash
# =============================================================
# Gateway 재시작 래퍼 — guided_choice on/off 토글 포함
# 사용법:
#   bash restart_gateway.sh          # ON (기본, guided_choice 활성)
#   bash restart_gateway.sh on       # ON
#   bash restart_gateway.sh off      # OFF (프롬프팅팀 새 라벨 테스트용)
# =============================================================

MODE="${1:-on}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 종료
pkill -9 -f start_gateway.sh 2>/dev/null
pkill -9 -f "python3 main.py" 2>/dev/null
sleep 2

# 환경변수 설정
case "$MODE" in
  on|ON|true|1)
    export GUIDED_CHOICE_ENABLED=true
    echo "=== Gateway 재시작: guided_choice ON ==="
    ;;
  off|OFF|false|0)
    export GUIDED_CHOICE_ENABLED=false
    echo "=== Gateway 재시작: guided_choice OFF (자유 생성 모드) ==="
    ;;
  *)
    echo "사용법: bash restart_gateway.sh [on|off]"
    exit 1
    ;;
esac

# 시작
cd "$SCRIPT_DIR" || exit 1
nohup bash start_gateway.sh gemma-4-31B-it 6006 >> /tmp/gateway_gemma4.log 2>&1 &
echo "PID: $!"
echo "로그: tail -f /tmp/gateway_gemma4.log"
