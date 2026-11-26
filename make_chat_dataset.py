"""공공데이터 사고다발지역 + 가상 감지 시나리오 → AI 분석 모델 튜닝 데이터(ai/data/{train,valid}.jsonl, mlx-lm chat 형식).

채팅 본문(제목·표·가이드)은 서버(web/answers.py)가 바로 쓰고, 모델은 끝의 "AI 분석" 1~2문장만 씀.
질문 → 데이터 → 정답(기준 분석)을 서빙과 같은 answers.route 로 만듦: 형식이 다르면 모델이 값을 잘못 옮김.
여러 지역 시나리오를 섞어 특정 지명·숫자를 외우지 않고 데이터를 읽어 쓰도록 함.

python3 ai/make_chat_dataset.py
"""
import json
import math
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = next(p for p in (ROOT.parent / "web", ROOT.parent / "Osori-Presentation" / "web") if p.exists())  # 같은 폴더 또는 나란히 받은 Osori-Presentation
sys.path.insert(0, str(WEB))
import answers  # noqa: E402
from answers import analysis_messages, events_by_zone, match_zones, place_words, route  # noqa: E402


class Clock(datetime):  # 데이터 기준 시각을 시나리오마다 다르게
    current = datetime(2026, 11, 7, 9, 0)

    @classmethod
    def now(cls, tz=None):
        return cls.current


answers.datetime = Clock
ZONES = [{"riskZoneId": i, **z} for i, z in enumerate(json.loads((WEB.parent / "deploy" / "risk-zones.json").read_text()), 1)]
CITIES = ["경기도 성남시", "서울특별시 강남구", "부산광역시", "대구광역시", "경기도 수원시"]
RECENT_QS = ["최근 마지막 데이터 알려줘", "마지막 감지 기록 알려줘", "가장 최근 감지된 곳 어디야?", "최근 데이터 알려줘", "최신 감지 보여줘", "마지막으로 감지된 거 뭐야?"]
examples = []


def add(q, kind, events, here=None):
    """서빙과 같은 route 로 데이터·정답을 만듦. 질문이 기대한 유형으로 가는지도 확인."""
    r = route(q, ZONES, events, here)
    assert r.kind == kind and r.data, (q, kind, r.kind)
    examples.append({"messages": [*analysis_messages(q, r.data), {"role": "assistant", "content": r.reference}]})


def virtual_events(zones, rng, base):
    """구역 주변 가상 감지. 앞의 두 구역은 동시 다발이 되게 몰아서 (deploy/make_zones.py 의 sample_events 와 같은 규칙)."""
    events = []
    for rank, zone in enumerate(zones):
        at = base + timedelta(hours=rank * 1.7)
        for _ in range(4 if rank < 2 else rng.randint(1, 3)):
            at += timedelta(minutes=rng.randint(2, 4) if rank < 2 else rng.randint(20, 80))
            distance = round(rng.uniform(1.5, 9.5), 1)
            events.append({"eventId": len(events) + 1, "latitude": zone["latitude"] + rng.uniform(-.0004, .0004), "longitude": zone["longitude"] + rng.uniform(-.0004, .0004),
                           "maxRssi": round(-45 - 24 * math.log10(distance)), "minEstimatedDistance": distance, "alertLevel": "DANGER" if distance <= 3 else "CAUTION",
                           "consecutiveCount": rng.randint(2, 6), "startedAt": at.isoformat()})
    return sorted(events, key=lambda e: e["startedAt"])


def spot_of(zone):
    """질문에 쓸 지점 이름: 이 구역만 찾히는 말."""
    return next((w for w in reversed(place_words(zone["name"])[0]) if match_zones(w, ZONES) == [zone]), None)


def here_event(here, seed, minutes):
    """내 위치(또는 구역)에서 난 감지 (사고다발지역 밖일 수 있음)."""
    d = round(random.Random(seed).uniform(1.5, 9.5), 1)
    return {"eventId": 900 + seed, "latitude": here[0], "longitude": here[1], "maxRssi": round(-45 - 24 * math.log10(d)), "minEstimatedDistance": d,
            "alertLevel": "DANGER" if d <= 3 else "CAUTION", "consecutiveCount": 2 + seed % 5, "startedAt": (Clock.current - timedelta(minutes=minutes)).isoformat()}


rng = random.Random(11)
for n, city in enumerate(CITIES):
    local = [z for z in ZONES if z["name"].startswith(city)]
    assert len(local) >= 10, (city, len(local))
    rng.shuffle(local)
    Clock.current = datetime(2026, 11, 1 + n * 3, rng.randint(7, 22), rng.randint(0, 59))
    active = local[:rng.randint(4, 7)]
    events = virtual_events(active, rng, Clock.current - timedelta(hours=14))
    if n == 3:  # 동시 다발이 없는 경우도 학습: 몰린 구역의 감지를 줄임
        events = [e for i, e in enumerate(events) if i % 4 == 0]
    if n == 1:  # 사고다발지역 밖의 동시 다발 (내 위치 시연에서 생김)
        out = Clock.current - timedelta(hours=1)
        events = sorted(events + [{"eventId": 800 + i, "latitude": 35.2, "longitude": 128.9, "maxRssi": -55, "minEstimatedDistance": 2.5 + i, "alertLevel": "CAUTION",
                                   "consecutiveCount": 3, "startedAt": (out + timedelta(minutes=2 * i)).isoformat()} for i in range(3)], key=lambda e: e["startedAt"])
    grouped = events_by_zone(ZONES, events)
    hit = [z for z in active if grouped[z["name"]]]
    quiet = [z for z in local if not grouped[z["name"]]]

    for zone in hit[:3]:
        if spot := spot_of(zone):
            add(f"{spot} 위험해?", "zone", events)
            add(f"{spot} 상황 알려줘", "zone", events)
    for zone in [z for z in quiet if spot_of(z)][:2]:
        add(f"{spot_of(zone)} 위험해?", "quiet", events)
    for k, (q, zone, extra) in enumerate((("내 주변 위험해?", hit[0], False), ("여기 근처에 위험한 곳 있어?", quiet[0], False), ("방금 감지된 거 위험해?", hit[-1], True), ("지금 위험해?", quiet[-1], True))):
        here = (zone["latitude"] + .001, zone["longitude"] - .001)
        add(q, "near", sorted(events + [here_event(here, n * 7 + k, 1 + k % 3)] if extra else events, key=lambda e: e["startedAt"]), here)
    sea = (35.0 + n * .01, 128.9)  # 바다 위: 주변에 구역 없음
    add("내 위치 주변 위험구역 알려줘", "near", events, sea)
    add("현재 위치에 감지된 거 있어?", "near", sorted(events + [here_event(sea, n, 2)], key=lambda e: e["startedAt"]), sea)
    # 지역: 도시·구(감지 있는 곳)·동(감지 없는 곳). 유형마다 예시가 적으면 모델이 문장을 지어냄
    words = [city.split()[-1]] + [next((w for w in place_words(z["name"])[1] if w.endswith(suffix)), None) for z, suffix in ((hit[0], "구"), (quiet[0], "동"))]
    for w in dict.fromkeys(w for w in words if w):
        add(f"{w} 어디가 위험해?", "region", events)
    # 요약·순위·시간대·동시 다발: 감지 일부만 써서 수치가 다른 변형 (동시 다발이 없는 경우도 생김)
    for k, part in enumerate((events, events[len(events) // 2:], events[::3])):
        add(("전체 현황 요약해줘", "오늘 현황 알려줘", "감지 통계 요약해줘")[k], "summary", part)
        add(("동시 다발 구역 어디야?", "차량이 몰리는 곳 알려줘", "동시 다발 구역 있어?")[k], "burst", part)
        if any(grouped[z["name"]] for z in ZONES if any(e in grouped[z["name"]] for e in part)):
            add(("제일 위험한 구역 어디야?", "가장 위험한 곳 알려줘", "위험한 구역 순위 보여줘")[k], "top", part)
        if any(e["alertLevel"] == "DANGER" for e in part):
            add(("언제 제일 위험해?", "위험한 시간대 알려줘", "몇 시에 제일 위험해?")[k], "hours", part)
    # 최근 감지: 그대로 / 동시 다발 구역에 다시 감지 / 구역 밖. 위치가 사고다발지역 안팎 모두 나오게
    hot = hit[0]
    for k, extra in enumerate(([], [here_event((hot["latitude"], hot["longitude"]), n * 10 + 1, 20)], [here_event(sea, n * 10 + 2, 4)])):
        add(RECENT_QS[(n * 3 + k) % len(RECENT_QS)], "recent", sorted(events + extra, key=lambda e: e["startedAt"]))
    e = grouped[hit[0]["name"]][0]
    if spot := spot_of(hit[0]):
        add(f"{spot}에서 거리 {e['minEstimatedDistance']}m, RSSI {e['maxRssi']}dBm으로 {e['consecutiveCount']}회 연속 감지됐어. 얼마나 위험해?", "event", events)
    add(f"거리 {round(rng.uniform(1.5, 9.5), 1)}m, RSSI {rng.randint(-82, -50)}dBm으로 {rng.randint(1, 6)}회 연속 감지됐어. 얼마나 위험해?", "event", events)

# 정답의 숫자는 전부 데이터나 질문에 있어야 함 (없으면 모델이 지어내는 법을 배움)
for ex in examples:
    prompt, answer = ex["messages"][1]["content"], ex["messages"][2]["content"]
    for number in re.findall(r"\d+(?:\.\d+)?(?:건|곳|점|명|%)|\d\d:\d\d|\d+\.\d+|-\d+dBm", answer):
        assert number in prompt, f"{prompt[:40]}: 근거 없는 수치 {number}"

rng.shuffle(examples)
out = ROOT / "data"
out.mkdir(exist_ok=True)
valid = examples[::10]  # 10% 검증용 (학습에도 그대로 둠: 데이터가 작아 형식을 익히는 게 목적)
for name, rows in (("train", examples), ("valid", valid)):
    (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"train {len(examples)}, valid {len(valid)} → {out}")
