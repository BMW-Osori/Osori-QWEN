"""공공데이터 사고다발지역 + 가상 감지 시나리오 → 채팅 모델 튜닝 데이터(ai/data/{train,valid}.jsonl, mlx-lm chat 형식).

시스템 프롬프트와 참고 데이터는 서빙(web/server.py)과 같은 함수로 만듦: 형식이 다르면 모델이 값을 잘못 옮김.
여러 지역 시나리오를 섞어 특정 지명·숫자를 외우지 않고 참고 데이터를 읽어 답하도록 함.

python3 ai/make_chat_dataset.py
"""
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = next(p for p in (ROOT.parent / "web", ROOT.parent / "Osori-Presentation" / "web") if p.exists())  # 같은 폴더 또는 나란히 받은 Osori-Presentation
sys.path.insert(0, str(WEB))
import server  # noqa: E402
from server import CHAT_SYSTEM, MIN_COUNT, RADIUS_M, WINDOW_MIN, bursts, events_by_zone, facts, match_zones, place_words  # noqa: E402


class Clock(datetime):  # 참고 데이터의 기준 시각을 시나리오마다 다르게
    current = datetime(2026, 11, 7, 9, 0)

    @classmethod
    def now(cls, tz=None):
        return cls.current


server.datetime = Clock
ZONES = [{"riskZoneId": i, **z} for i, z in enumerate(json.loads((WEB.parent / "deploy" / "risk-zones.json").read_text()), 1)]
CITIES = ["경기도 성남시", "서울특별시 강남구", "부산광역시", "대구광역시", "경기도 수원시"]
RECENT_QS = ["최근 마지막 데이터 알려줘", "마지막 감지 기록 알려줘", "가장 최근 감지된 곳 어디야?", "최근 데이터 알려줘", "최신 감지 보여줘", "마지막으로 감지된 거 뭐야?"]
hhmm = lambda t: t[11:16]
GUIDE = {
    "HIGH": ["즉시 걸음을 멈추고 앞뒤 차량을 확인하세요.", "이어폰을 빼거나 볼륨을 최소로 낮추세요.", "인도 안쪽이나 건물 쪽으로 이동해 차량과 거리를 두세요."],
    "MEDIUM": ["이어폰 볼륨을 낮추고 주변 소리에 귀 기울이세요.", "횡단보도·주차장 출입구에서는 좌우를 한 번 더 확인하세요.", "진동이 반복되면 잠시 멈춰 차량 위치를 확인하세요."],
    "LOW": ["평소처럼 이동하되 차량 진입로 근처에서는 속도를 줄이세요.", "휴대폰 화면만 보며 걷지 마세요.", "진동이 연속으로 오면 바로 주변을 확인하세요."],
}
examples = []


def guide(level):
    return "\n".join(f"{n}. {g}" for n, g in enumerate(GUIDE[level], 1))


def harm_rows(zone):
    d = server.ACCIDENTS[zone["name"]]
    return f"| 사고 유형 | {'·'.join(d['types'])} |\n| 사상자 | {d['casualties']}명 |\n| 사망자 | {d['deaths']}명 |"


def analysis(e, zone):
    d, rssi, c, a = e["minEstimatedDistance"], e["maxRssi"], e["consecutiveCount"], zone["accidentCount"]
    return " ".join([
        f"추정 거리 {d}m는 " + ("3m 이내로 차량이 바로 옆까지 와 있는 상태입니다." if d < 3
                            else "3~6m 사이로 차량이 가까이 접근한 상태입니다." if d < 6 else "6m 이상으로 아직 거리에 여유가 있습니다."),
        f"신호 세기 {rssi}dBm는 " + ("강한 편이라 차량이 매우 가깝다는 뜻입니다." if rssi >= -60
                                else "중간 세기로 차량이 다가오고 있을 수 있습니다." if rssi >= -75 else "약한 편이라 차량이 지나가거나 멀어지는 중일 가능성이 큽니다."),
        f"연속 {c}회 감지되어 차량이 계속 머물거나 다가오고 있습니다." if c >= 5
        else f"{c}회 연속 감지되어 접근이 이어지고 있습니다." if c >= 3 else f"감지가 {c}회에 그쳐 일시적인 접근일 수 있습니다.",
        f"이 구역의 누적 사고는 {a}건입니다.",
    ])


def zone_answer(zone, es, burst):
    name, level = zone["name"], zone["riskLevel"]
    closest = min(es, key=lambda e: e["minEstimatedDistance"])
    z = next((z for z in burst if z["zone"] == name), None)
    span = f"{hhmm(z['start'])}–{hhmm(z['end'])} · {z['count']}건" if z else "해당 없음"
    timeline = "\n".join(f"- **{hhmm(e['startedAt'])}** · 거리 {e['minEstimatedDistance']}m · RSSI {e['maxRssi']}dBm · 연속 {e['consecutiveCount']}회 · {e['alertLevel']}" for e in es)
    return f"""## 📍 {name} 위험도 리포트

> 💡 {name}에서 차량 접근이 **{len(es)}건** 감지됐고, 구역 위험도는 **{zone['riskScore']}점, {level}**입니다.{f" {hhmm(z['start'])}–{hhmm(z['end'])} 사이 {z['count']}건이 몰린 동시 다발 구역이에요." if z else ""}

### 📊 현황

| 항목 | 값 |
| --- | --- |
| 감지 건수 | {len(es)}건 |
| 구역 위험도 | {zone['riskScore']}점 ({level}) |
| 누적 사고 | {zone['accidentCount']}건 |
{harm_rows(zone)}
| 동시 다발 | {span} |

### 🕒 감지 기록

{timeline}

### ⚠️ 위험 분석

가장 가까웠던 {hhmm(closest['startedAt'])} 감지를 기준으로 보면, {analysis(closest, zone)}

### ✅ 행동 가이드

{guide(level)}

---
{{NOTE}}"""


def quiet_answer(zone):
    name, level = zone["name"], zone["riskLevel"]
    return f"""## 📍 {name} 위험도 리포트

> 💡 {name}의 구역 위험도는 **{zone['riskScore']}점, {level}**입니다. 실시간 감지 기록은 아직 없습니다.

### 📊 현황

| 항목 | 값 |
| --- | --- |
| 구역 위험도 | {zone['riskScore']}점 ({level}) |
| 누적 사고 | {zone['accidentCount']}건 |
{harm_rows(zone)}
| 감지 건수 | 0건 |

### ⚠️ 위험 분석

위험도는 한국도로교통공단 사고다발지역의 사고 내역으로 계산한 값입니다. 지금 감지된 차량이 없어도 사고가 반복된 곳이니 주의하세요.

### ✅ 행동 가이드

{guide(level)}

---
{{NOTE}}"""


def zone_table(picked, grouped):
    return "| 구역 | 구역 위험도 | 누적 사고 | 감지 |\n| --- | --- | --- | --- |\n" + "\n".join(
        f"| {z['name']} | {z['riskScore']}점 ({z['riskLevel']}) | {z['accidentCount']}건 | {len(grouped[z['name']])}건 |" for z in picked)


def near_timeline(close):
    """내 위치 근처 감지 기록 섹션 (참고 데이터의 최근 5건)."""
    if not close:
        return "내 위치 1000m 안 감지 기록은 **0건**입니다."
    rows = "\n".join(f"- **{hhmm(e['startedAt'])}** · 거리 {e['minEstimatedDistance']}m · RSSI {e['maxRssi']}dBm · 연속 {e['consecutiveCount']}회 · {e['alertLevel']}" for e in close[-5:])
    return f"내 위치 1000m 안 감지 기록은 **{len(close)}건**입니다.\n\n{rows}"


def near_answer(picked, grouped, close=()):
    if not picked:
        return f"""## 📍 내 주변 위험구역

> 💡 내 위치 1000m 안에 등록된 사고다발지역은 **0곳**입니다.

### 🕒 내 주변 감지

{near_timeline(close)}

### ⚠️ 알아 두세요

사고다발지역이 아니어도 저소음 차량은 다가올 수 있습니다. 진동이 울리면 바로 주변을 확인하세요.

### ✅ 행동 가이드

{guide("LOW")}

---
{{NOTE}}"""
    top = max(picked, key=lambda z: z["riskScore"])
    return f"""## 📍 내 주변 위험구역

> 💡 내 위치 1000m 안 위험구역은 **{len(picked)}곳**이고, 가장 위험한 곳은 **{top['name']}, {top['riskScore']}점**입니다.

### 📊 구역 목록

{zone_table(picked, grouped)}

### 🕒 내 주변 감지

{near_timeline(close)}

### ✅ 행동 가이드

{guide(top['riskLevel'])}

---
{{NOTE}}"""


def region_answer(region, picked, grouped):
    top = picked[0]
    return f"""## 🗺️ {region} 위험구역

> 💡 {region}에서 위험도가 가장 높은 곳은 **{top['name']}, {top['riskScore']}점**입니다.

### 📊 위험도 높은 구역

{zone_table(picked, grouped)}

### ✅ 행동 가이드

{guide(top['riskLevel'])}

---
{{NOTE}}"""


def burst_answer(burst, by_name):
    if not burst:
        return """## 🗺️ 동시 다발 구역

> 💡 현재 동시 다발 구역은 **0곳**입니다.

### ✅ 행동 가이드

1. 평소처럼 이동하되 차량 진입로 근처에서는 속도를 줄이세요.
2. 진동이 연속으로 오면 걸음을 멈추고 차량이 지나갈 때까지 기다리세요.

---
{NOTE}"""
    rows = "\n".join(f"| {z['zone']} | {hhmm(z['start'])}–{hhmm(z['end'])} | {z['count']}건 | " + (f"{by_name[z['zone']]['riskScore']}점" if z["zone"] in by_name else "–") + " |" for z in burst)
    return f"""## 🗺️ 동시 다발 구역

> 💡 짧은 시간에 감지가 몰린 **동시 다발 구역은 {len(burst)}곳**입니다. 이 구역에서는 이어폰 볼륨을 낮추고 전방을 주시하세요.

### 📊 구역 목록

| 구역 | 시간 | 건수 | 구역 위험도 |
| --- | --- | --- | --- |
{rows}

### ⚠️ 왜 위험한가요?

짧은 시간에 저소음 차량이 여러 번 접근한 곳입니다. 진동 경고가 연달아 울려 대응이 늦어지기 쉽습니다.

### ✅ 행동 가이드

1. 해당 시간대에는 가능하면 다른 길로 우회하세요.
2. 구역에 들어서면 이어폰 볼륨을 낮추고 휴대폰 화면에서 눈을 떼세요.
3. 진동이 연속으로 오면 걸음을 멈추고 차량이 지나갈 때까지 기다리세요.

---
{{NOTE}}"""


def top_answer(live, grouped):
    rows = "\n".join(f"| {n} | {z['name']} | {z['riskScore']}점 | {len(grouped[z['name']])}건 | {z['accidentCount']}건 |" for n, z in enumerate(live, 1))
    return f"""## 🔥 지금 가장 위험한 구역

> 💡 감지 기록이 있는 구역 중 위험도가 가장 높은 곳은 **{live[0]['name']}, {live[0]['riskScore']}점**입니다.

### 📊 순위

| 순위 | 구역 | 구역 위험도 | 감지 건수 | 누적 사고 |
| --- | --- | --- | --- | --- |
{rows}

### ✅ 행동 가이드

{guide(live[0]['riskLevel'])}

---
{{NOTE}}"""


def summary_answer(zones, events, burst):
    alerts, risk = Counter(e["alertLevel"] for e in events), Counter(z["riskLevel"] for z in zones)
    bullets = "\n".join(f"- **{z['zone']}** · {hhmm(z['start'])}–{hhmm(z['end'])} · {z['count']}건" for z in burst) or "- 없음"
    return f"""## 📋 보행 안전 현황 요약

> 💡 차량 접근 **{len(events)}건**, 그중 DANGER **{alerts['DANGER']}건**입니다. 위험구역 {len(zones)}곳 중 HIGH는 **{risk['HIGH']}곳**이고, 동시 다발 구역은 **{len(burst)}곳**이에요.

### 📊 전체 지표

| 항목 | 값 |
| --- | --- |
| 총 감지 | {len(events)}건 |
| DANGER | {alerts['DANGER']}건 |
| CAUTION | {alerts['CAUTION']}건 |
| 위험구역 | {len(zones)}곳 |
| HIGH 구역 | {risk['HIGH']}곳 |
| MEDIUM 구역 | {risk['MEDIUM']}곳 |
| LOW 구역 | {risk['LOW']}곳 |

### 🗺️ 동시 다발 구역

{bullets}

### ✅ 안전 수칙

{guide("MEDIUM")}

---
{{NOTE}}"""


def hours_answer(events):
    by_hour = Counter(int(e["startedAt"][11:13]) for e in events if e["alertLevel"] == "DANGER")
    if not by_hour:
        return """## 🕒 위험 시간대 분석

> 💡 지금까지 DANGER 감지가 없어 위험 시간대를 꼽을 수 없습니다.

---
{NOTE}"""
    peak, n = by_hour.most_common(1)[0]
    rows = "\n".join(f"| {h}시 | {c}건 |" for h, c in sorted(by_hour.items()))
    return f"""## 🕒 위험 시간대 분석

> 💡 DANGER가 가장 많았던 시간대는 **{peak}시, {n}건**입니다.

### 📊 시간대별 DANGER 건수

| 시간 | DANGER |
| --- | --- |
{rows}

### ✅ 행동 가이드

{guide("HIGH")}

---
{{NOTE}}"""


def event_answer(e, zone):
    level = "HIGH" if e["alertLevel"] == "DANGER" else "MEDIUM"
    return f"""## 🚨 차량 접근 분석

> 💡 {zone['name']} 구역 위험도는 **{zone['riskScore']}점, {zone['riskLevel']}**입니다.

### 📊 감지 신호

| 항목 | 값 |
| --- | --- |
| 위치 | {zone['name']} |
| 추정 거리 | {e['minEstimatedDistance']}m |
| 신호 세기 (RSSI) | {e['maxRssi']}dBm |
| 연속 감지 | {e['consecutiveCount']}회 |
| 구역 누적 사고 | {zone['accidentCount']}건 |

### ⚠️ 위험 분석

{analysis(e, zone)}

### ✅ 행동 가이드

{guide(level)}

---
{{NOTE}}"""


def recent_answer(events, grouped):
    if not events:
        return f"""## 🕒 최근 감지 기록

> 💡 아직 감지 기록이 **0건**입니다.

### ✅ 행동 가이드

{guide("LOW")}

---
{{NOTE}}"""
    e = events[-1]
    zone = {e["eventId"]: name for name, es in grouped.items() for e in es}.get(e["eventId"])
    at = f"{e['startedAt'][5:10]} {hhmm(e['startedAt'])}"
    return f"""## 🕒 최근 감지 기록

> 💡 가장 최근 감지는 **{at}** {zone or '사고다발지역 밖'}에서 난 **{e['alertLevel']}**입니다.

### 📊 감지 정보

| 항목 | 값 |
| --- | --- |
| 시각 | {at} |
| 위치 | {zone or '구역 밖'} |
| 추정 거리 | {e['minEstimatedDistance']}m |
| 신호 세기 (RSSI) | {e['maxRssi']}dBm |
| 연속 감지 | {e['consecutiveCount']}회 |
| 경고 단계 | {e['alertLevel']} |

### ⚠️ 알아 두세요

'구역 밖'은 사고다발지역으로 등록되지 않은 곳이지만, 저소음 차량은 어디서든 다가올 수 있습니다. 다른 감지 기록은 구역 이름으로 물어보세요.

### ✅ 행동 가이드

{guide("HIGH" if e["alertLevel"] == "DANGER" else "MEDIUM")}

---
{{NOTE}}"""


def no_data(place):
    return f"""## 📍 {place} 데이터 없음

> 💡 {place}은(는) **사고다발지역 데이터에서 찾지 못했습니다**. 사고가 없다는 뜻은 아닙니다.

### ✅ 대신 이렇게 해 보세요

1. "모란시장 위험해?"처럼 지점이나 동네 이름으로 물어보세요.
2. "동시 다발 구역 어디야?"로 차량이 몰린 곳을 확인하세요.
3. 데이터가 없는 곳에서도 진동 경고가 오면 바로 멈추고 주변을 확인하세요.

---
{{NOTE}}"""


def add(q, answer, zones, events, here=None):
    # 서빙과 같은 시스템 프롬프트(+참고 데이터)로 학습해야 모델이 참고 데이터를 읽어 답하는 법을 배움
    context = facts(q, zones, events, here)
    note = "*데이터 기준: " + re.search(r"- 데이터 기준: (.+)", context)[1] + "*"
    examples.append({"messages": [
        {"role": "system", "content": f"{CHAT_SYSTEM}\n\n{context}"},
        {"role": "user", "content": q},
        {"role": "assistant", "content": answer.replace("{NOTE}", note)},
    ]})


def virtual_events(zones, rng, base):
    """구역 주변 가상 감지. 앞의 두 구역은 동시 다발이 되게 몰아서 (deploy/make_zones.py 의 sample_events 와 같은 규칙)."""
    events = []
    for rank, zone in enumerate(zones):
        at = base + timedelta(hours=rank * 1.7)
        for _ in range(4 if rank < 2 else rng.randint(1, 3)):
            at += timedelta(minutes=rng.randint(2, 4) if rank < 2 else rng.randint(20, 80))
            distance = round(rng.uniform(1.5, 9.5), 1)
            events.append({"eventId": len(events) + 1, "latitude": zone["latitude"] + rng.uniform(-.0004, .0004), "longitude": zone["longitude"] + rng.uniform(-.0004, .0004),
                           "maxRssi": round(-45 - 24 * __import__("math").log10(distance)), "minEstimatedDistance": distance, "alertLevel": "DANGER" if distance <= 3 else "CAUTION",
                           "consecutiveCount": rng.randint(2, 6), "startedAt": at.isoformat()})
    return sorted(events, key=lambda e: e["startedAt"])


rng = random.Random(11)
scenarios = []
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
    scenarios.append((city, local, active, events))
    grouped, burst = events_by_zone(ZONES, events), bursts(ZONES, events)
    by_name = {z["name"]: z for z in ZONES}
    hit = [z for z in active if grouped[z["name"]]]
    quiet = [z for z in local if not grouped[z["name"]]]

    def spot_of(zone):
        """질문에 쓸 지점 이름: 이 구역만 찾히는 말."""
        spots = place_words(zone["name"])[0]
        return next((w for w in reversed(spots) if match_zones(w, ZONES) == [zone]), None)

    for zone in hit[:3]:
        if spot := spot_of(zone):
            add(f"{spot} 위험해?", zone_answer(zone, grouped[zone["name"]], burst), ZONES, events)
            add(f"{spot} 상황 알려줘", zone_answer(zone, grouped[zone["name"]], burst), ZONES, events)
    asked = 0
    for zone in quiet:
        if asked < 2 and (spot := spot_of(zone)):
            add(f"{spot} 위험해?", quiet_answer(zone), ZONES, events)
            asked += 1
    def here_event(here, seed):
        """내 위치에서 방금 난 감지 (사고다발지역 밖일 수 있음)."""
        d = round(random.Random(seed).uniform(1.5, 9.5), 1)
        return {"eventId": 900 + seed, "latitude": here[0], "longitude": here[1], "maxRssi": round(-45 - 24 * __import__("math").log10(d)), "minEstimatedDistance": d,
                "alertLevel": "DANGER" if d <= 3 else "CAUTION", "consecutiveCount": 2 + seed % 5, "startedAt": (Clock.current - timedelta(minutes=1 + seed % 3)).isoformat()}

    for k, (q, zone, extra) in enumerate((("내 주변 위험해?", hit[0], False), ("여기 근처에 위험한 곳 있어?", quiet[0], False), ("방금 감지된 거 위험해?", hit[-1], True), ("지금 위험해?", quiet[-1], True))):
        here = (zone["latitude"] + .001, zone["longitude"] - .001)
        now_events = events + [here_event(here, n * 7 + k)] if extra else events
        now_grouped = events_by_zone(ZONES, now_events)
        picked = sorted((z for z in ZONES if abs(z["latitude"] - here[0]) < .02), key=lambda z: server.distance_m(*here, z["latitude"], z["longitude"]))
        picked = sorted([z for z in picked if server.distance_m(*here, z["latitude"], z["longitude"]) <= server.NEAR_M][:3], key=lambda z: -z["riskScore"])
        close = [e for e in sorted(now_events, key=lambda e: e["startedAt"]) if server.distance_m(*here, e["latitude"], e["longitude"]) <= server.NEAR_M]
        add(q, near_answer(picked, now_grouped, close), ZONES, sorted(now_events, key=lambda e: e["startedAt"]), here)
    sea = (35.0 + n * .01, 128.9)  # 바다 위: 주변에 구역 없음
    add("내 위치 주변 위험구역 알려줘", near_answer([], grouped), ZONES, events, sea)
    alone = sorted(events + [here_event(sea, n)], key=lambda e: e["startedAt"])
    add("현재 위치에 감지된 거 있어?", near_answer([], events_by_zone(ZONES, alone), [here_event(sea, n)]), ZONES, alone, sea)
    region = city.split()[-1]
    picked = match_zones(f"{region} 어디가 위험해?", ZONES)
    add(f"{region} 어디가 위험해?", region_answer(region, picked, grouped), ZONES, events)
    add("동시 다발 구역 어디야?", burst_answer(burst, by_name), ZONES, events)
    add("차량이 몰리는 곳 알려줘", burst_answer(burst, by_name), ZONES, events)
    live = sorted(hit, key=lambda z: -z["riskScore"])[:3]
    add("제일 위험한 구역 어디야?", top_answer(live, grouped), ZONES, events)
    add("전체 현황 요약해줘", summary_answer(ZONES, events, burst), ZONES, events)
    add("언제 제일 위험해?", hours_answer(events), ZONES, events)
    # 최근 감지: 그대로 / 동시 다발 구역에 다시 감지 / 구역 밖 + 동시 다발 구역. 동시 다발 줄의 시각을 감지 시각으로 옮기지 않게
    hot = next(by_name[b["zone"]] for b in bursts(ZONES, events) if b["zone"] in by_name) if burst else hit[0]
    again = lambda at, i, minutes: {**here_event(at, n * 10 + i), "eventId": 950 + i, "startedAt": (Clock.current - timedelta(minutes=minutes)).isoformat()}
    for k, extra in enumerate(([], [again((hot["latitude"], hot["longitude"]), i, 20 + 25 * i) for i in range(2)],
                               [again(sea, 0, 4), again((hot["latitude"], hot["longitude"]), 1, 30)])):
        recent = sorted(events + extra, key=lambda e: e["startedAt"])
        add(RECENT_QS[(n * 3 + k) % len(RECENT_QS)], recent_answer(recent, events_by_zone(ZONES, recent)), ZONES, recent)
    e = next(e for e in events if grouped[hit[0]["name"]] and e in grouped[hit[0]["name"]])
    if spot := spot_of(hit[0]):
        add(f"{spot}에서 거리 {e['minEstimatedDistance']}m, RSSI {e['maxRssi']}dBm으로 {e['consecutiveCount']}회 연속 감지됐어. 얼마나 위험해?", event_answer(e, hit[0]), ZONES, events)

add("최근 데이터 알려줘", recent_answer([], {}), ZONES, [])  # 감지가 하나도 없을 때

# 데이터 답변의 숫자는 전부 참고 데이터나 질문에 있어야 함 (없으면 모델이 지어냄)
for ex in examples:
    prompt = ex["messages"][0]["content"] + ex["messages"][1]["content"]
    for number in re.findall(r"\d+(?:\.\d+)?(?:건|곳|점|명)|\d\d:\d\d|\d+\.\d+", ex["messages"][2]["content"]):
        assert number in prompt, f"{ex['messages'][1]['content']}: 근거 없는 수치 {number}"

GENERAL = {
    "너는 누구야?": """## 🤖 Safety AI 소개

> 💡 **보행자 촉각 경고** 시스템의 안전 도우미입니다. 저소음 차량 접근 데이터를 바탕으로 보행 위험을 알려드려요.

### 📌 할 수 있는 일

- **구역별 위험도**: "모란시장 위험해?"처럼 물어보면 감지 기록과 위험도를 정리해 드립니다.
- **동시 다발 구역**: 짧은 시간에 차량이 몰린 곳을 알려드립니다.
- **신호 해석**: 거리·RSSI·연속 감지 횟수로 위험 정도를 설명합니다.
- **대처 방법**: 진동 경고가 왔을 때 어떻게 해야 하는지 안내합니다.

### ✅ 이렇게 물어보세요

1. 동시 다발 구역 어디야?
2. 오늘 전체 현황 요약해줘
3. 진동이 울리면 어떻게 해야 해?

---
*데이터 기준: 감지 이벤트*""",
    "저소음 차량이 뭐야?": """## 🔇 저소음 차량이란?

> 💡 전기차·하이브리드처럼 **저속에서 소리가 거의 나지 않는 차량**입니다. 이어폰을 낀 보행자는 접근을 알아채기 어렵습니다.

### 📊 특징

| 항목 | 내용 |
| --- | --- |
| 대표 차종 | 전기차, 하이브리드, 수소차 |
| 위험 구간 | 시속 20km 이하 저속 주행 |
| 취약 보행자 | 이어폰 사용자, 고령자, 시각장애인 |

### ⚠️ 왜 위험한가요?

엔진 소리가 없어 뒤나 옆에서 다가와도 들리지 않습니다. 골목, 주차장 출입구, 횡단보도처럼 차량과 보행자가 가까이 섞이는 곳에서 특히 위험합니다.

### ✅ 우리 시스템의 역할

1. 차량 송신기의 RF 신호를 웨어러블이 감지합니다.
2. 임계 거리 안이면 웨어러블과 휴대폰이 진동으로 경고합니다.
3. 앱이 감지 기록을 서버에 저장하고, AI 가 위험구역의 위험도를 산출합니다.

---
*참고: 보행자 촉각 경고 시스템 구상안*""",
    "진동이 울리면 어떻게 해야 해?": """## 📳 진동 경고 대처법

> 💡 진동은 **저소음 차량이 가까이 왔다는 신호**입니다. 바로 걸음을 멈추고 주변을 확인하세요.

### 📊 진동 패턴

| 패턴 | 의미 |
| --- | --- |
| 짧게 1회 | 임계 거리 내 접근 (CAUTION) |
| 길게 반복 | 매우 가까운 접근 (DANGER) |
| 연속 여러 번 | 동시 다발 구역 진입 |

### ⚠️ 주의할 점

진동이 길게 반복되면 DANGER 단계입니다. 차량이 3m 이내에 있을 수 있으니 휴대폰 화면을 보며 걷는 것을 멈추세요.

### ✅ 행동 가이드

1. 즉시 걸음을 멈추고 앞뒤 차량을 확인하세요.
2. 이어폰을 빼거나 볼륨을 최소로 낮추세요.
3. 인도 안쪽이나 건물 쪽으로 이동해 차량이 지나가길 기다리세요.

---
*경고 단계: CAUTION 임계 거리 내 접근 · DANGER 매우 가까움*""",
    "RSSI가 뭐야?": """## 📡 RSSI (신호 세기)

> 💡 RSSI 는 차량 송신기 신호가 **얼마나 세게 잡히는지**를 dBm 단위로 나타낸 값입니다. 0에 가까울수록 차량이 가깝습니다.

### 📊 해석 기준

| RSSI | 의미 |
| --- | --- |
| -60dBm 이상 | 강함 · 차량이 매우 가까움 |
| -60 ~ -75dBm | 중간 · 차량이 다가오는 중 |
| -75dBm 미만 | 약함 · 지나가거나 멀어지는 중 |

### ⚠️ 함께 보는 값

- **추정 거리**: RSSI 로 계산한 차량과의 거리
- **연속 감지 횟수**: 같은 차량이 몇 번 연달아 잡혔는지
- **누적 사고**: 해당 구역의 사고 이력

### ✅ 활용

1. 신호가 강하고 연속 감지가 많은 구역일수록 구역 위험도가 높아집니다.
2. 앱의 경고 기록에서 RSSI 와 거리를 함께 확인할 수 있습니다.
3. 신호가 약해도 진동이 반복되면 주변을 확인하세요.

---
*판단 기준: 70점 이상 HIGH · 40점 이상 MEDIUM · 그 외 LOW*""",
    "경고 단계가 뭐야?": """## 📳 차량 접근 경고 단계

> 💡 접근 이벤트의 경고 단계는 **SAFE, CAUTION, DANGER, RELEASED** 네 가지입니다. 대기, 경고, 심화, 해제 순서로 바뀌어요.

### 📊 단계표

| 단계 | 값 | 의미 |
| --- | --- | --- |
| 대기 | SAFE | 위험 범위 안에 차량 없음 |
| 경고 | CAUTION | 설정된 임계 거리 안으로 차량 접근 |
| 심화 | DANGER | 차량이 매우 가까운 상태 |
| 해제 | RELEASED | 차량이 위험 범위를 벗어남 |

### ⚠️ 위험구역 단계와 다릅니다

경고 단계는 지금 이 순간의 차량 접근 상황이고, HIGH·MEDIUM·LOW는 그 장소가 얼마나 위험한지를 나타내는 구역 단계입니다.

### ✅ 단계별 행동

1. **DANGER**: 걸음을 멈추고 인도 안쪽으로 이동하세요.
2. **CAUTION**: 이어폰 볼륨을 낮추고 좌우를 확인하세요.
3. **RELEASED**: 주변을 확인한 뒤 이동하세요.

---
*경고 단계: 보행자 디바이스가 판단, 진동은 서버 없이 작동*""",
    "위험 단계 기준 알려줘": """## 🚦 위험구역 단계 기준

> 💡 위험구역의 위험도가 **70점 이상이면 HIGH, 40점 이상이면 MEDIUM, 그 아래는 LOW** 입니다.

### 📊 단계표

| 단계 | 점수 | 의미 |
| --- | --- | --- |
| HIGH | 70~100점 | 사고 및 보행 위험도가 높은 구역 |
| MEDIUM | 40~69점 | 주의가 필요한 구역 |
| LOW | 0~39점 | 일반 구역 |

### ⚠️ 점수를 올리는 요인

- 구역에서 감지된 **최소 거리**가 가까울수록
- 구역의 **최대 RSSI** 가 강할수록
- **연속 감지**와 **DANGER 감지**가 많을수록
- **누적 사고**가 많은 구역일수록

### ✅ 단계별 행동

1. **HIGH**: 걸음을 멈추고 인도 안쪽으로 이동하세요.
2. **MEDIUM**: 이어폰 볼륨을 낮추고 좌우를 확인하세요.
3. **LOW**: 평소처럼 이동하되 진입로에서는 속도를 줄이세요.

---
*판단 기준: safety-ai 위험도 분석 모델*""",
    "동시 다발 구역 기준이 뭐야?": f"""## 🗺️ 동시 다발 구역 기준

> 💡 반경 **{RADIUS_M}m** 안에서 **{WINDOW_MIN}분** 이내에 차량 접근이 **{MIN_COUNT}건 이상** 감지된 곳입니다.

### 📊 판단 조건

| 조건 | 값 |
| --- | --- |
| 거리 | {RADIUS_M}m 이내 |
| 시간 | {WINDOW_MIN}분 이내 |
| 건수 | {MIN_COUNT}건 이상 |

### ⚠️ 왜 따로 보나요?

한 번의 감지보다 짧은 시간에 여러 번 감지되는 곳이 더 위험합니다. 차량이 몰리거나 전방 주시가 어려운 구간이라 진동 경고가 연달아 울립니다.

### ✅ 행동 가이드

1. 지도 탭에서 동시 다발 구역을 미리 확인하세요.
2. 해당 구역에서는 이어폰 볼륨을 낮추세요.
3. 앱 알림 탭에서 '동시 다발 알림'을 켜 두세요.

---
{{NOTE}}""",
    "이어폰 끼고 걸어도 괜찮아?": """## 🎧 이어폰 착용 보행

> 💡 괜찮지만 **위험 구역에서는 볼륨을 낮추세요.** 웨어러블이 차량 접근을 진동으로 알려 드립니다.

### 📊 상황별 권장

| 상황 | 권장 |
| --- | --- |
| 일반 인도 | 평소 볼륨 |
| 횡단보도·골목 | 볼륨 낮추기 |
| 동시 다발 구역 | 한쪽 이어폰 빼기 |

### ⚠️ 주의할 점

저소음 차량은 소리로 알아채기 어렵습니다. 노이즈 캔슬링을 켜면 주변 소리가 더 줄어드니 위험 구역에서는 주변음 모드를 쓰세요.

### ✅ 행동 가이드

1. 웨어러블을 연결해 두고 진동 경고를 켜 두세요.
2. 진동이 오면 바로 볼륨을 낮추고 주변을 확인하세요.
3. 출퇴근 시간 역 주변에서는 특히 주의하세요.

---
*참고: 보행자 촉각 경고 시스템*""",
    "웨어러블 기기는 어떻게 연결해?": """## ⌚ 웨어러블 연결 방법

> 💡 앱 **홈 탭의 '기기 찾기'** 에서 보행자 디바이스를 검색해 페어링하고 연결하면 됩니다.

### 📊 연결 순서

| 단계 | 내용 |
| --- | --- |
| 1 | 웨어러블 전원 켜기 |
| 2 | 홈 탭 → 기기 찾기 → 주변 기기 검색 |
| 3 | 새로 찾은 기기에서 '페어링' 선택 |
| 4 | 연결되면 상단에 '연결됨' 표시 |

### ⚠️ 연결이 안 될 때

- 휴대폰 블루투스가 켜져 있는지 확인하세요.
- 웨어러블과 1m 이내에서 다시 검색하세요.
- iOS 는 일반 블루투스 기기 검색을 지원하지 않아 Android 에서 연결하세요.

### ✅ 연결 후

1. 차량 신호를 받으면 웨어러블과 휴대폰이 함께 진동합니다.
2. 주변 위험구역의 위험도가 경고 기록에 표시됩니다.
3. 알림 탭에서 진동·푸시 설정을 바꿀 수 있습니다.

---
*참고: 보행자 촉각 경고 앱*""",
}

for i, (q, a) in enumerate(list(GENERAL.items()) * 3):  # 안내 답변은 참고 데이터에 근거가 없어 외워야 하므로 여러 번
    _, _, _, events = scenarios[i % len(scenarios)]
    add(q, a, ZONES, events)
# 데이터에 없는 장소: 전체 요약 수치를 그 장소 것처럼 쓰지 않도록
for i, place in enumerate(p for p in ("잠실야구장", "일산호수공원", "인천국제공항", "제주국제공항", "경복궁", "남산타워", "속초해수욕장", "김포공항", "독립기념관", "에버랜드") if not match_zones(p, ZONES)):
    add(f"{place} 위험해?", no_data(place), ZONES, scenarios[i % len(scenarios)][3])
    add(f"{place} 상황 알려줘", no_data(place), ZONES, scenarios[(i + 1) % len(scenarios)][3])

rng.shuffle(examples)
out = ROOT / "data"
out.mkdir(exist_ok=True)
valid = examples[::10]  # 10% 검증용 (학습에도 그대로 둠: 데이터가 작아 형식을 익히는 게 목적)
for name, rows in (("train", examples), ("valid", valid)):
    (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
# 앱에서 굵게가 깨지는 패턴(닫는 ** 앞이 구두점, 뒤가 글자) 방지
word = re.compile(r"[\w가-힣]")
for ex in examples:
    a = ex["messages"][2]["content"]
    for m in re.finditer(r"\*\*([^*\n]+)\*\*", a):
        after = a[m.end():m.end() + 1]
        assert not (not word.match(m[1][-1]) and word.match(after or " ")), f"굵게 깨짐: {m[0]}{after}"
print(f"train {len(examples)}, valid {len(valid)} → {out}")
