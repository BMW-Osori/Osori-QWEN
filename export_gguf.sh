#!/bin/sh
# ai/adapters(MLX LoRA) 를 원본 모델에 합쳐 GGUF f16 으로 변환 (EC2 Ollama 용). Apple Silicon 에서 실행, GPU 불필요.
# 사용: ./ai/export_gguf.sh <작업 폴더> [어댑터 폴더]   → <작업 폴더>/safety-chat-f16.gguf
set -e
cd "$(dirname "$0")"
OUT=${1:?작업 폴더를 지정하세요}
ADAPTERS=${2:-adapters}
MODEL=Qwen/Qwen2.5-3B-Instruct  # 학습에 쓴 4bit 대신 원본: 양자화를 두 번 거치는 손실을 피함
python3 -m mlx_lm fuse --model "$MODEL" --adapter-path "$ADAPTERS" --save-path "$OUT/fused"
[ -d "$OUT/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$OUT/llama.cpp"
PYTHONPATH="$OUT/llama.cpp/gguf-py" python3 "$OUT/llama.cpp/convert_hf_to_gguf.py" "$OUT/fused" --outtype f16 --outfile "$OUT/safety-chat-f16.gguf"
