# Osori-QWEN

보행자 촉각 경고 채팅 모델(`safety-chat`)의 튜닝 도구와 결과물입니다. Qwen2.5-3B-Instruct 를 Apple Silicon Mac 에서 MLX LoRA 로 튜닝합니다. **Improved using Qwen.** Qwen Research License 에 따라 연구·평가 목적으로만 쓸 수 있습니다 (`NOTICE`).

- `make_chat_dataset.py`: 학습 데이터 생성 (`data/`). 시스템 프롬프트와 참고 데이터를 서빙 코드와 같은 함수로 만들므로 [Osori-Presentation](https://github.com/BMW-Osori/Osori-Presentation) 을 이 저장소 옆에 받아 두어야 함
- `train.sh`: LoRA 학습
- `export_gguf.sh`: 원본 모델에 합쳐 GGUF 로 변환
- `deploy_chat.sh`: 양자화 후 AI 서버에 `safety-chat:candidate` 로 업로드 (운영 모델은 그대로)
- `adapters/`: 현재 운영 모델의 LoRA 어댑터

```sh
python3 make_chat_dataset.py
sh train.sh data adapters 350                # 약 30분
sh deploy_chat.sh adapters <작업 폴더>        # 약 5분, 환경 변수 SSH_KEY·BACKEND·AI 로 대상 지정
```

후보를 점검한 뒤 AI 서버에서 교체합니다: `ollama cp safety-chat safety-chat:prev && ollama cp safety-chat:candidate safety-chat`

참고 데이터 줄의 문구(`web/server.py` 의 `facts`)를 바꾸면 모델이 값을 잘못 옮길 수 있으니 다시 학습하세요.
