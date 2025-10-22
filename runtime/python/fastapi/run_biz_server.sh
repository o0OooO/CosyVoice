
#!/usr/bin/env bash
set -euo pipefail

# 启动 runtime/python/fastapi/biz_server.py
# 可通过环境变量覆盖：
#   BIZ_PORT                  默认 50010
#   COSYVOICE_BACKEND_HOST    默认 127.0.0.1
#   COSYVOICE_BACKEND_PORT    默认 50000
#   COSYVOICE_MODEL_DIR       默认 iic/CosyVoice-300M

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${BIZ_PORT:=50010}"
: "${COSYVOICE_BACKEND_HOST:=127.0.0.1}"
: "${COSYVOICE_BACKEND_PORT:=50000}"
: "${COSYVOICE_MODEL_DIR:=iic/CosyVoice-300M}"

echo "[biz_server] port=$BIZ_PORT backend=${COSYVOICE_BACKEND_HOST}:${COSYVOICE_BACKEND_PORT} model_dir=$COSYVOICE_MODEL_DIR"

exec python3 biz_server.py \
  --port "$BIZ_PORT" \
  --backend_host "$COSYVOICE_BACKEND_HOST" \
  --backend_port "$COSYVOICE_BACKEND_PORT" \
  --model_dir "$COSYVOICE_MODEL_DIR"


