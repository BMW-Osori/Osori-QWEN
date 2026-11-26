# Osori-QWEN

보행자 촉각 경고 채팅의 **AI 분석** 모델(`safety-chat`) 튜닝 도구와 결과물입니다. 채팅 본문(요약·표·가이드)은 서버 코드가 쓰고, 이 모델은 서버가 계산한 짧은 데이터를 보고 끝의 분석 1~2문장만 씁니다 (AI 서버 CPU 가 초당 약 10토큰이라 긴 답은 1분 걸림). Qwen2.5-3B-Instruct 를 Apple Silicon Mac 에서 MLX LoRA 로 튜닝합니다. **Improved using Qwen.** Qwen Research License 에 따라 연구·평가 목적으로만 쓸 수 있습니다 (`NOTICE`).

- `make_chat_dataset.py`: 학습 데이터 생성 (`data/`). 질문 → 데이터 → 정답(기준 분석)을 서빙 코드(`web/answers.py` 의 `route`)로 만들므로 [Osori-Presentation](https://github.com/BMW-Osori/Osori-Presentation) 을 이 저장소 옆에 받아 두어야 함
- `train.sh`: LoRA 학습
- `export_gguf.sh`: 원본 모델에 합쳐 GGUF 로 변환
- `deploy_chat.sh`: q4_0 으로 양자화해 AI 서버에 `safety-chat:candidate` 로 업로드 (운영 모델은 그대로). AI 서버(ARM)는 q4_0 을 전용 커널로 돌려 q4_K_M 보다 빠름
- `adapters/`: 현재 운영 모델의 LoRA 어댑터

```sh
python3 make_chat_dataset.py
sh train.sh data adapters 430                # 약 8분 (답이 짧아 빠름)
sh deploy_chat.sh adapters <작업 폴더>        # 약 5분, 환경 변수 SSH_KEY·BACKEND·AI 로 대상 지정
```

후보를 점검한 뒤 AI 서버에서 교체합니다: `ollama cp safety-chat safety-chat:prev && ollama cp safety-chat:candidate safety-chat`

모델에 주는 데이터 형식(Osori-Presentation `web/answers.py` 의 각 `*_reply` 의 `data`)을 바꾸면 모델이 값을 잘못 옮길 수 있으니 다시 학습하세요. 해석(거리·RSSI)·권장 행동은 데이터에 미리 넣고 등급(HIGH 등)은 분석 문장에 넣지 않습니다: 3B 모델이 직접 고르게 하면 자주 틀립니다.

## 함께 쓰는 저장소

| 저장소 | 역할 |
| --- | --- |
| [Osori-BE](https://github.com/BMW-Osori/Osori-BE) | 백엔드 (Spring Boot, PostgreSQL) |
| [Osori-Presentation](https://github.com/BMW-Osori/Osori-Presentation) | 시연 사이트와 AI 서비스 |
| [Osori-FE](https://github.com/BMW-Osori/Osori-FE) | 앱 (Flutter) |
