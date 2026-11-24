#!/bin/sh
# 대화형 Safety AI: Qwen2.5-3B-Instruct(4bit) 에 예시 데이터로 LoRA 파인튜닝 (Apple Silicon, MLX).
# 운영 반영: ai/deploy_chat.sh
set -e
cd "$(dirname "$0")"
MODEL=mlx-community/Qwen2.5-3B-Instruct-4bit

# 인자 없이 실행하면 예시 데이터로, 있으면 그 폴더의 train/valid.jsonl 로 학습 (ai/worker.py): train.sh <데이터> <어댑터> <반복 수>
DATA=${1:-data}
ADAPTERS=${2:-adapters}
ITERS=${3:-390}  # 예시 152개 × 약 2.6회
[ -n "$1" ] || python3 make_chat_dataset.py
python3 -m mlx_lm lora --model "$MODEL" --train --data "$DATA" --adapter-path "$ADAPTERS" \
  --iters "$ITERS" --batch-size 1 --num-layers 16 --grad-checkpoint --learning-rate 1e-4 --mask-prompt \
  --steps-per-report 10 --steps-per-eval 50

# 서빙 (web/server.py 의 /api/chat 이 호출):
#   python3 -m mlx_lm server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 8080
#   (어댑터는 web/server.py 가 요청마다 ai/adapters 를 지정)
