#!/usr/bin/env bash
# Full production transcription run.
# Run inside tmux/screen so it survives terminal disconnects:
#   tmux new -s transcribe ./run_production.sh
set -euo pipefail
cd "$(dirname "$0")"

ENGINE="${1:-qwen3}"
BENCH_DIR="benchmarks/production_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCH_DIR"
LOG="$BENCH_DIR/run.log"
GPU_LOG="$BENCH_DIR/gpu.csv"
MONITOR_LOG="$BENCH_DIR/monitor.log"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw \
  --format=csv -l 5 > "$GPU_LOG" 2>&1 &
GPU_PID=$!

(
  sleep 20
  while true; do
    SRTS=$(find transcriptions/ -name "*.srt" 2>/dev/null | wc -l)
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader 2>/dev/null)
    TPID=$(pgrep -f "transcribe.py --asr-engine" 2>/dev/null | head -1)
    MEM="?"
    if [ -n "$TPID" ] && [ -f "/proc/$TPID/status" ]; then
      MEM=$(awk '/VmRSS/{printf "%.1fG", $2/1048576}' /proc/$TPID/status 2>/dev/null)
    fi
    SYSMEM=$(free -g 2>/dev/null | awk '/Mem:/{print $3"/"$2"G"}')
    echo "$(date '+%H:%M:%S') SRTs=$SRTS GPU=$GPU RAM=$MEM SysMem=$SYSMEM"
    sleep 30
  done
) >> "$MONITOR_LOG" 2>&1 &
MON_PID=$!

echo "Engine: $ENGINE" | tee -a "$LOG"
echo "Started at $(date)" | tee -a "$LOG"
echo "Logs: $BENCH_DIR/" | tee -a "$LOG"
echo "GPU monitor PID: $GPU_PID, Progress monitor PID: $MON_PID" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

START=$(date +%s)

.venv/bin/python -u transcribe.py \
  --asr-engine "$ENGINE" \
  --continue-on-error \
  2>&1 | tee -a "$LOG"

RC=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))

echo "" >> "$LOG"
echo "=== FINISHED (exit code $RC) ===" >> "$LOG"
echo "Total wall time: ${ELAPSED}s ($((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m)" >> "$LOG"
echo "Finished at $(date)" >> "$LOG"

FINAL_SRTS=$(find transcriptions/ -name "*.srt" 2>/dev/null | wc -l)
echo "Final SRT count: $FINAL_SRTS" >> "$LOG"

kill $GPU_PID $MON_PID 2>/dev/null
wait $GPU_PID $MON_PID 2>/dev/null || true

echo ""
echo "=== DONE ==="
echo "Wall time: ${ELAPSED}s ($((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m)"
echo "SRT files: $FINAL_SRTS"
echo "Logs: $BENCH_DIR/"
