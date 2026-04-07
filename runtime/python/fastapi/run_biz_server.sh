
#!/usr/bin/env bash
set -euo pipefail

# 启动 runtime/python/fastapi/biz_server.py
# 可通过环境变量覆盖：
#   BIZ_PORT                  默认 50010
#   COSYVOICE_BACKEND_HOST    默认 127.0.0.1
#   COSYVOICE_BACKEND_PORT    默认 50000
#   COSYVOICE_MODEL_DIR       默认 pretrained_models/CosyVoice-300M
#   COSYVOICE_MODEL_DIR_V2    默认 pretrained_models/CosyVoice2-0.5B
#   COSYVOICE_MODEL_DIR_V3    默认 pretrained_models/Fun-CosyVoice3-0.5B

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$SCRIPT_DIR"

# Matcha-TTS 路径注入
export PYTHONPATH="${ROOT_DIR}/third_party/Matcha-TTS:${PYTHONPATH:-}"

# 修复 Matcha-TTS conformer 导入问题
DECODER_FILE="${ROOT_DIR}/third_party/Matcha-TTS/matcha/models/components/decoder.py"
if [ -f "$DECODER_FILE" ] && grep -q "^from conformer import ConformerBlock" "$DECODER_FILE"; then
    echo "[biz_server] Fixing conformer import in Matcha-TTS decoder.py..."
    sed -i.bak 's/^from conformer import ConformerBlock/from conformer.conformer import ConformerBlock/' "$DECODER_FILE"
fi

: "${BIZ_PORT:=50010}"
: "${COSYVOICE_BACKEND_HOST:=127.0.0.1}"
: "${COSYVOICE_BACKEND_PORT:=50000}"
: "${COSYVOICE_MODEL_DIR:=pretrained_models/CosyVoice-300M}"
: "${COSYVOICE_MODEL_DIR_V2:=pretrained_models/CosyVoice2-0.5B}"
: "${COSYVOICE_MODEL_DIR_V3:=pretrained_models/Fun-CosyVoice3-0.5B}"

# 检查模型目录是否存在，不存在则从 ModelScope 下载
download_if_missing() {
    local dir="$ROOT_DIR/$1"
    local repo_id="$2"
    if [ ! -d "$dir" ]; then
        echo "[biz_server] $dir not found, downloading $repo_id ..."
        python3 -c "from modelscope import snapshot_download; snapshot_download('$repo_id', local_dir='$dir')"
    fi
}

download_if_missing "$COSYVOICE_MODEL_DIR"    "iic/CosyVoice-300M"
download_if_missing "$COSYVOICE_MODEL_DIR_V2" "iic/CosyVoice2-0.5B"
download_if_missing "$COSYVOICE_MODEL_DIR_V3" "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

echo "[biz_server] port=$BIZ_PORT backend=${COSYVOICE_BACKEND_HOST}:${COSYVOICE_BACKEND_PORT}"
echo "[biz_server] model_dir=$COSYVOICE_MODEL_DIR  v2=$COSYVOICE_MODEL_DIR_V2  v3=$COSYVOICE_MODEL_DIR_V3"

exec python3 biz_server.py \
  --port "$BIZ_PORT" \
  --backend_host "$COSYVOICE_BACKEND_HOST" \
  --backend_port "$COSYVOICE_BACKEND_PORT" \
  --model_dir "$COSYVOICE_MODEL_DIR" \
  --model_dir_v2 "$COSYVOICE_MODEL_DIR_V2" \
  --model_dir_v3 "$COSYVOICE_MODEL_DIR_V3"


