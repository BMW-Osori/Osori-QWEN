#!/bin/sh
# 튜닝 결과를 AI 서버의 후보 모델로 올림 (운영 모델은 그대로): ai/deploy_chat.sh <어댑터 폴더> <작업 폴더>
# 어댑터 → GGUF f16 → q4_K_M(임시 Ollama) → 백엔드를 거쳐 AI 서버로 복사 → safety-chat:candidate 생성
# 점검 뒤 교체: ssh AI 서버 'ollama cp safety-chat safety-chat:prev && ollama cp safety-chat:candidate safety-chat'
set -e
ADAPTERS=$(cd "$1" && pwd)
WORK=${2:?작업 폴더}
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
KEY=${SSH_KEY:-$HOME/.ssh/bmw-safety-ec2}
BACKEND=${BACKEND:-ubuntu@43.200.19.220}
AI=${AI:-ubuntu@172.31.8.223}
PROXY="ssh -i $KEY -W %h:%p $BACKEND"
HERE=$(cd "$(dirname "$0")" && pwd)

sh "$HERE/export_gguf.sh" "$WORK" "$ADAPTERS"

# 임시 Ollama 로 양자화. 저장소에는 f16 도 남으므로 매니페스트에서 결과 파일을 찾음
export OLLAMA_MODELS="$WORK/ollama" OLLAMA_HOST=127.0.0.1:11499
ollama serve >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
until ollama list >/dev/null 2>&1; do sleep 1; done
echo "FROM $WORK/safety-chat-f16.gguf" > "$WORK/q.Modelfile"
ollama create tmp --quantize q4_K_M -f "$WORK/q.Modelfile"
kill $SERVER; trap - EXIT
DIGEST=$(python3 -c "import json,sys; print(next(l['digest'] for l in json.load(open(sys.argv[1]))['layers'] if l['mediaType'].endswith('.model')))" "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/tmp/latest")
BLOB="$OLLAMA_MODELS/blobs/$(echo "$DIGEST" | tr ':' '-')"
ls -lh "$BLOB"

scp -i "$KEY" -o ProxyCommand="$PROXY" "$BLOB" "$AI:safety-chat.gguf"
scp -i "$KEY" -o ProxyCommand="$PROXY" "$HERE/chat.Modelfile" "$AI:chat.Modelfile"
ssh -i "$KEY" -o ProxyCommand="$PROXY" "$AI" 'ollama create safety-chat:candidate -f chat.Modelfile && rm -f safety-chat.gguf && ollama list | grep safety-chat'
