# -*- coding: utf-8 -*-
# road_leader.py
"""
Road Leader Engine for Baccarat Predictor AI Engine v10.x

역할:
- Big Road / 중국점 3종 / 본매(비드) 기준으로
  각 로드맵이 최근 구간에서 얼마나 잘 맞는지(적중률)를 추적
- 5개 로드맵별 다음 수 방향(P/B) 신호를 생성
- 최근 적중률 기반으로 리더를 선정하여 Feature에 제공

변경 요약 (2025-12-29)
----------------------------------------------------
1) road_leader는 “리더 신뢰 상태 판단기”로만 동작하도록 역할을 고정
   - PASS/PROBE/NORMAL 진입 판단을 절대 하지 않음(상위 recommend.py 전용)
   - side(방향) 단독 확정 금지 유지: can_override_side는 STRONG에서만 True로 “허용 가능” 플래그만 제공
2) Big 계열(비드/빅로드)과 China 계열(빅아이/스몰/꼬마) “독립 평가” 후 overall leader 선정 규칙을 명문화
   - trust tier 우선 → 동일 tier면 confidence 우선 → 동일이면 Big 계열 우선
3) “리더는 매 판 갈아치우지 않는다” 안정성 규칙 추가
   - 동일 tier에서는 기존 overall leader를 유지
   - 상위 tier가 등장할 때만 leader 교체 허용(그 외엔 유지)
4) 연속 실패 기반 강등 시 ‘뒤집기(즉시 반대 방향 제안)’ 차단
   - 연속 실패로 강등된 직후, 신호가 즉시 반대로 뒤집히는 케이스는 한 템포 NONE으로 해제
5) 출력 계약 보강(호환 유지)
   - road_hit_rates/road_prediction_totals를 leader_state 내부에도 함께 제공(기존 top-level도 유지)
   - leader_ready/leader_not_ready_reason 키를 leader_state에 alias로 추가(기존 ready/reason 의미 유지)
----------------------------------------------------

변경 요약 (2025-12-28)
----------------------------------------------------
1) "3판 이후부터 리더 사용"을 상태 머신으로 도입 (NONE/WEAK/MID/STRONG)
   - 리더를 매 판 갈아치우지 않고, 신뢰 상태(tier)만 승급/강등한다.
   - 0~2판은 무조건 NONE (초기 우연/노이즈 방지)
   - 3판 이후부터 WEAK/MID/STRONG 산정 가능(“리더를 쓴다”는 의미)
2) 리더 연속 실패 시 단계 하락(강등) 규칙 추가
   - 최근 window의 trailing loss(연속 0) 기반으로 1~2회면 한 단계 하락,
     3회 이상이면 NONE으로 해제(반대 제안 금지, 즉시 뒤집기 금지)
3) leader_state에 아래 필드 추가(상위 로직/GPT 프롬프트와 정합)
   - leader_trust_state: NONE|WEAK|MID|STRONG
   - confidence_note: 한 문장 사유
   - can_override_side: STRONG에서만 True (그 외 False)
4) 기존 readiness(ready/reason) 의미는 유지
   - ready는 “통계/신호가 충분히 쌓인 분석 준비도”로 유지(기존 하드가드)
   - 다만 leader_signal은 trust_state가 NONE이 아니면 생성될 수 있다(= 3판 이후 사용)
----------------------------------------------------

변경 요약 (2025-12-24)
----------------------------------------------------
1) 중국점 3종(BigEye/Small/Cockroach) "독립 격상" 적용
   - _select_china_leader()에서 통계 임계값(_CHINA_MIN_TOTAL/_CHINA_MIN_WINDOW/_CHINA_THRESHOLD)과
     2-of-3(_CHINA_MIN_AGREE)로 신호를 "차단"하던 로직 제거
   - 중국점 신호(P/B)가 1개라도 존재하면 즉시 China Leader 후보로 인정(그림 트리거 우선)
   - 적중률/윈도우는 "confidence 참고"로만 사용(진입/신호 생성 차단 금지)
2) leader_state에 china_signals/china_windows 추가
   - recommend.py 등 상위 로직에서 2/3 체크(최근 3회 2회 적중 등)를 할 수 있도록
     중국점별 window(0/1 히트) 리스트를 leader_state로 제공
----------------------------------------------------

변경 요약 (2025-12-23)
----------------------------------------------------
1) 폴백 전면 금지(근본 강화)
   - prev_round_winner가 None이면 자동 추론(pb_seq[-1]) 금지 → 즉시 예외
   - pb_stats.pb_ratio / streak_info.current_streak / pattern_dict.pattern_type 등
     필수 계약 누락 시 기본값으로 때우지 않고 즉시 예외
2) 아이들포턴시(중복 누적 방지) 하드 적용
   - pb_seq 길이가 증가하지 않았으면(타이/중복 호출) stats 누적을 절대 수행하지 않음
3) pb_seq 점프(여러 라운드가 한 번에 증가) 발생 시
   - 통계 누적은 “정확히 복원 불가”하므로 금지
   - leader_state를 리셋하고 round_index를 pb_len으로 동기화(오염 방지)
   - leader_ready=False + 사유 명시로 운영 추적 가능
4) 미사용 코드 정리
   - prev_round_winner None 자동추론 로직 제거
   - adv_features는 시그니처 호환용으로만 유지(검증 후 미사용 명시)
----------------------------------------------------

변경 요약 (2025-12-22)
----------------------------------------------------
1) "리더 준비 미달"은 예외로 엔진을 중단하지 않도록 유지
   - 준비 미달은 leader_ready=False + leader_not_ready_reason 반환
   - 단, pb_seq empty 등 분석 불가 입력은 즉시 예외
2) 리더 로드 선택 기준 강화
   - 5개 로드맵별 적중률(hit_rate) 산출
   - 가장 적중률 높은 로드맵을 리더로 선정하여 leader_signal 제공
3) engine_state 저장 호환성
   - window(deque) → list 변환은 engine_state 쪽에서 처리하도록 유지
----------------------------------------------------

주의/정책:
- road_leader.py는 recommend.py의 PASS/PROBE/NORMAL/side 결정을 절대 침범하지 않는다.
- 여기서 제공하는 leader_* 정보는 “기존 방향 후보를 강화/약화”하는 보조 신호일 뿐이다.
- 입력 계약(Type/필수키/Value) 위반은 즉시 예외(폴백 금지).
- 단, 통계 부족/준비 미달은 예외가 아니라 leader_ready=False로 반환한다.
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 리더 평가 대상 로드
_LEADER_ROADS = ("bead", "bigroad", "bigeye", "small", "cockroach")
_BIG_ROAD_ROADS = ("bead", "bigroad")
_CHINA_ROADS = ("bigeye", "small", "cockroach")

# window/리더 선택 기준
_MIN_TOTAL = 12
_LEADER_THRESHOLD = 0.60
_WINDOW_MAXLEN = 20
_MIN_WINDOW_FOR_READY = 6

# 3판부터 WEAK/MID/STRONG 산정 가능
_TRUST_MIN_PB_LEN = 3

# tier 산정 임계값(보수적으로 유지)
# - STRONG은 기존 기준과 정합(12판 & 0.60)
# - MID/WEAK는 "리더를 참고" 수준으로만(단독 override는 STRONG만)
_BIG_TIER_WEAK_MIN_TOTAL = 3
_BIG_TIER_WEAK_MIN_ACC = 0.55
_BIG_TIER_MID_MIN_TOTAL = 7
_BIG_TIER_MID_MIN_ACC = 0.57
_BIG_TIER_STRONG_MIN_TOTAL = _MIN_TOTAL
_BIG_TIER_STRONG_MIN_ACC = _LEADER_THRESHOLD

_CHINA_TIER_WEAK_MIN_TOTAL = 3
_CHINA_TIER_WEAK_MIN_ACC = 0.52
_CHINA_TIER_MID_MIN_TOTAL = 7
_CHINA_TIER_MID_MIN_ACC = 0.54
_CHINA_TIER_STRONG_MIN_TOTAL = 12
_CHINA_TIER_STRONG_MIN_ACC = 0.56

_leader_state: Dict[str, Any] = {}


class RoadLeaderError(RuntimeError):
    """road_leader 계열 오류(폴백 금지)."""


class RoadLeaderNotReadyError(RoadLeaderError):
    """
    분석 자체가 성립 불가한 입력(pb_seq empty 등)에서만 예외.
    준비 미달(통계 부족/신호 부족)은 예외가 아니라 leader_ready=False로 반환한다.
    """


def reset_leader_state() -> None:
    """새 슈 시작 시 리더 로드 통계 초기화."""
    global _leader_state
    _leader_state = {
        "round_index": 0,  # pb_seq 길이 기준
        "stats": {
            name: {
                "total": 0,
                "correct": 0,
                "window": deque(maxlen=_WINDOW_MAXLEN),
            }
            for name in _LEADER_ROADS
        },
        "last_signals": {name: None for name in _LEADER_ROADS},

        # 안정성 규칙(“매 판 갈아치우지 않음”)을 위한 내부 상태
        # - 동일 tier에서는 기존 overall leader 유지
        # - 상위 tier 등장 시에만 교체 허용
        "active_overall": None,        # {"source": "big"/"china", "road": str, "roads": [str], ...}
        "last_overall_signal": None,   # "P"/"B"/None
    }


def get_state() -> Dict[str, Any]:
    """UNDO용: 현재 리더 상태 스냅샷 반환."""
    return copy.deepcopy(_leader_state)


def _normalize_state_windows(state: Dict[str, Any]) -> Dict[str, Any]:
    """복원된 상태에서 window(list)를 deque(maxlen=...)로 정규화."""
    if not state or "stats" not in state:
        return state

    stats = state.get("stats") or {}
    for name in _LEADER_ROADS:
        s = stats.get(name)
        if not isinstance(s, dict):
            continue
        w = s.get("window")
        if isinstance(w, deque):
            if w.maxlen != _WINDOW_MAXLEN:
                s["window"] = deque(list(w), maxlen=_WINDOW_MAXLEN)
        elif isinstance(w, list):
            s["window"] = deque(w, maxlen=_WINDOW_MAXLEN)
        elif w is None:
            s["window"] = deque(maxlen=_WINDOW_MAXLEN)
        else:
            s["window"] = deque(maxlen=_WINDOW_MAXLEN)
    return state


def set_state(state: Optional[Dict[str, Any]]) -> None:
    """UNDO 복구용: 리더 상태를 통째로 교체."""
    global _leader_state
    if state is None:
        logger.error("[ROAD-LEADER] set_state(None) forbidden")
        raise RoadLeaderError("set_state(None) forbidden (폴백 금지): call reset_leader_state() explicitly")
    _leader_state = _normalize_state_windows(copy.deepcopy(state))


def _require_dict_arg(name: str, obj: Any) -> None:
    if not isinstance(obj, dict):
        raise TypeError(f"[road_leader] {name} must be dict, got {type(obj).__name__}")


def _require_list_str_arg(name: str, obj: Any) -> None:
    if not isinstance(obj, list) or any(not isinstance(x, str) for x in obj):
        raise TypeError(f"[road_leader] {name} must be List[str], got {type(obj).__name__}")


def _require_prev_winner(prev_round_winner: Any) -> None:
    if prev_round_winner is None:
        raise RoadLeaderError("[road_leader] prev_round_winner is None forbidden (폴백 금지)")
    if prev_round_winner not in ("P", "B", "T"):
        raise ValueError(f"[road_leader] prev_round_winner must be 'P'/'B'/'T', got {prev_round_winner}")


def _require_pb_stats(pb_stats: Dict[str, Any]) -> None:
    # pb_stats는 다양한 키가 올 수 있으나, 최소한 pb_ratio는 있어야 한다.
    if "pb_ratio" not in pb_stats:
        raise KeyError("[road_leader] pb_stats missing required key: pb_ratio (폴백 금지)")
    if not isinstance(pb_stats["pb_ratio"], (int, float)):
        raise TypeError("[road_leader] pb_stats.pb_ratio must be number")


def _require_streak_info(streak_info: Dict[str, Any]) -> None:
    if "current_streak" not in streak_info:
        raise KeyError("[road_leader] streak_info missing required key: current_streak (폴백 금지)")
    cs = streak_info["current_streak"]
    if not isinstance(cs, dict):
        raise TypeError("[road_leader] streak_info.current_streak must be dict")
    if "side" not in cs or "length" not in cs:
        raise KeyError("[road_leader] streak_info.current_streak missing keys: side/length")
    if cs["side"] not in ("P", "B", None):
        raise ValueError("[road_leader] streak_info.current_streak.side must be 'P'/'B'/None")
    if not isinstance(cs["length"], int):
        raise TypeError("[road_leader] streak_info.current_streak.length must be int")


def _require_pattern_dict(pattern_dict: Dict[str, Any]) -> None:
    if "pattern_type" not in pattern_dict:
        raise KeyError("[road_leader] pattern_dict missing required key: pattern_type (폴백 금지)")
    if not isinstance(pattern_dict["pattern_type"], str):
        raise TypeError("[road_leader] pattern_dict.pattern_type must be str")


def _compute_bead_signal(pb_seq: List[str], pb_stats: Dict[str, Any]) -> Optional[str]:
    """
    본매(비드) 신호: pb_ratio 기반 단순 추세.
    - tie(T)는 제외
    """
    pb_ratio = float(pb_stats.get("pb_ratio", 0.5))
    if pb_ratio >= 0.55:
        return "P"
    if pb_ratio <= 0.45:
        return "B"
    return None


def _compute_bigroad_signal(pb_seq: List[str], streak_info: Dict[str, Any], pattern_dict: Dict[str, Any]) -> Optional[str]:
    """
    Big Road 신호:
    - current_streak이 충분히 길면 그 방향 추종
    - 특정 패턴(핑퐁 등)에서는 반대 제안 가능
    """
    cs = streak_info["current_streak"]
    side = cs.get("side")
    length = int(cs.get("length") or 0)

    if side in ("P", "B") and length >= 2:
        return side

    pattern_type = pattern_dict.get("pattern_type") or ""
    if isinstance(pattern_type, str) and "PINGPONG" in pattern_type.upper():
        if side == "P":
            return "B"
        if side == "B":
            return "P"

    return None


def _compute_china_road_signal(road_name: str, pb_seq: List[str]) -> Optional[str]:
    """
    중국점 3종 신호: 간단한 룰 기반(기존 로직 유지).
    - bigeye/small/cockroach 각각 다르게 해석할 수 있으나,
      여기서는 최소한 "신호 차단 금지" 원칙을 위해 가능한 한 P/B를 내놓는다.
    """
    if not pb_seq:
        return None

    # tie(T)는 제외한 최근 2개만 본다.
    filtered = [x for x in pb_seq if x in ("P", "B")]
    if len(filtered) < 2:
        return None

    a, b = filtered[-2], filtered[-1]
    if a == b:
        # 연속이면 추세 따라가기
        return b
    # 교대면 다음도 교대(핑퐁)로 예측
    return a


def _compute_signals_for_next(
    pb_seq: List[str],
    pb_stats: Dict[str, Any],
    streak_info: Dict[str, Any],
    pattern_dict: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    """
    5개 로드맵의 "다음 수" 방향 신호 생성.
    - 신호 생성은 차단하지 않는다(가능한 경우 P/B를 제공).
    """
    bead = _compute_bead_signal(pb_seq, pb_stats)
    bigroad = _compute_bigroad_signal(pb_seq, streak_info, pattern_dict)

    bigeye = _compute_china_road_signal("bigeye", pb_seq)
    small = _compute_china_road_signal("small", pb_seq)
    cockroach = _compute_china_road_signal("cockroach", pb_seq)

    return {
        "bead": bead,
        "bigroad": bigroad,
        "bigeye": bigeye,
        "small": small,
        "cockroach": cockroach,
    }


def _compute_hit_rates(stats: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    각 로드별 적중률(hit_rate)과 예측 누적(total) 반환.
    """
    hit_rates: Dict[str, float] = {}
    totals: Dict[str, int] = {}
    for name in _LEADER_ROADS:
        s = stats[name]
        total = int(s["total"])
        correct = int(s["correct"])
        totals[name] = total
        hit_rates[name] = (correct / total) if total > 0 else 0.0
    return hit_rates, totals


def _trailing_losses(window: deque) -> int:
    """
    window 끝에서부터 연속 0(실패) 개수.
    window 원소는 0/1.
    """
    n = 0
    for v in reversed(window):
        if int(v) == 0:
            n += 1
        else:
            break
    return n


def _apply_loss_demote(tier: str, loss_streak: int) -> Tuple[str, Optional[str]]:
    """
    연속 실패 기반 강등 규칙:
    - 1~2회: tier 1단계 강등
    - 3회 이상: 즉시 NONE 해제
    """
    if loss_streak >= 3:
        return "NONE", "loss_streak>=3 → NONE"
    if loss_streak <= 0:
        return tier, None

    order = ["NONE", "WEAK", "MID", "STRONG"]
    if tier not in order:
        return "NONE", "invalid tier"
    idx = order.index(tier)
    if idx <= 0:
        return "NONE", "already NONE"
    # 한 단계 강등
    new_tier = order[idx - 1]
    return new_tier, f"loss_streak={loss_streak} → demote {tier}->{new_tier}"


def _tier_rank(tier: str) -> int:
    return {"NONE": 0, "WEAK": 1, "MID": 2, "STRONG": 3}.get(tier, 0)


def _tier_from_stats(total: int, acc: float, kind: str) -> str:
    """
    누적(total)과 적중률(acc) 기반 tier 산정.
    kind: "big" or "china"
    """
    if kind == "big":
        if total >= _BIG_TIER_STRONG_MIN_TOTAL and acc >= _BIG_TIER_STRONG_MIN_ACC:
            return "STRONG"
        if total >= _BIG_TIER_MID_MIN_TOTAL and acc >= _BIG_TIER_MID_MIN_ACC:
            return "MID"
        if total >= _BIG_TIER_WEAK_MIN_TOTAL and acc >= _BIG_TIER_WEAK_MIN_ACC:
            return "WEAK"
        return "NONE"

    if kind == "china":
        if total >= _CHINA_TIER_STRONG_MIN_TOTAL and acc >= _CHINA_TIER_STRONG_MIN_ACC:
            return "STRONG"
        if total >= _CHINA_TIER_MID_MIN_TOTAL and acc >= _CHINA_TIER_MID_MIN_ACC:
            return "MID"
        if total >= _CHINA_TIER_WEAK_MIN_TOTAL and acc >= _CHINA_TIER_WEAK_MIN_ACC:
            return "WEAK"
        return "NONE"

    return "NONE"


def _select_big_leader_with_tier(
    stats: Dict[str, Dict[str, Any]],
    road_hit_rates: Dict[str, float],
    road_prediction_totals: Dict[str, int],
    new_signals: Dict[str, Optional[str]],
    pb_len: int,
) -> Tuple[Optional[str], Optional[str], float, str, Optional[str]]:
    """
    Big 계열(bead/bigroad) 중 현재 기준 best 후보를 선정하고 tier를 산정.
    Return: (leader_road, leader_signal, leader_conf, tier, note)
    """
    if pb_len < _TRUST_MIN_PB_LEN:
        return None, None, 0.0, "NONE", "pb_len < 3 → NONE"

    best_road = None
    best_conf = -1.0
    best_total = 0

    for road in _BIG_ROAD_ROADS:
        sig = new_signals.get(road)
        if sig not in ("P", "B"):
            continue
        total = int(road_prediction_totals.get(road, 0) or 0)
        conf = float(road_hit_rates.get(road, 0.0) or 0.0)

        if conf > best_conf:
            best_conf = conf
            best_road = road
            best_total = total
        elif conf == best_conf and total > best_total:
            # 동일 적중률이면 표본 큰 쪽
            best_road = road
            best_total = total

    if not best_road:
        return None, None, 0.0, "NONE", "no big signals"

    tier = _tier_from_stats(best_total, float(best_conf), kind="big")

    # 연속 실패 기반 강등
    window = stats[best_road]["window"]
    loss_streak = _trailing_losses(window)
    tier2, note = _apply_loss_demote(tier, loss_streak)
    tier = tier2

    if tier == "NONE":
        return None, None, 0.0, "NONE", note or "big tier NONE"

    return best_road, new_signals.get(best_road), float(best_conf), tier, note


def _select_china_leader_with_tier(
    stats: Dict[str, Dict[str, Any]],
    road_hit_rates: Dict[str, float],
    road_prediction_totals: Dict[str, int],
    new_signals: Dict[str, Optional[str]],
    pb_len: int,
) -> Tuple[List[str], Optional[str], float, str, Optional[str]]:
    """
    China 계열(bigeye/small/cockroach) 중 신호가 존재하면 무조건 후보로 인정.
    Return: (leader_roads, leader_signal, leader_conf, tier, note)
    - leader_roads: 다수결로 선택된 신호에 투표한 로드들(1~3)
    """
    if pb_len < _TRUST_MIN_PB_LEN:
        return [], None, 0.0, "NONE", "pb_len < 3 → NONE"

    active = [r for r in _CHINA_ROADS if new_signals.get(r) in ("P", "B")]
    if not active:
        return [], None, 0.0, "NONE", "no china signals"

    votes_p = [r for r in active if new_signals.get(r) == "P"]
    votes_b = [r for r in active if new_signals.get(r) == "B"]

    if len(votes_p) > len(votes_b):
        leader_signal = "P"
        leader_roads = votes_p
    elif len(votes_b) > len(votes_p):
        leader_signal = "B"
        leader_roads = votes_b
    else:
        # 동률이면 적중률 높은 로드 하나만 채택
        best = active[0]
        best_acc = -1.0
        for r in active:
            acc = float(road_hit_rates.get(r, 0.0) or 0.0)
            if acc > best_acc:
                best_acc = acc
                best = r
        leader_signal = new_signals.get(best)
        leader_roads = [best]

    if leader_signal not in ("P", "B"):
        return [], None, 0.0, "NONE", "china leader_signal invalid"

    # confidence: leader_roads 평균 적중률(참고용)
    conf_sum = 0.0
    totals = []
    for r in leader_roads:
        conf_sum += float(road_hit_rates.get(r, 0.0) or 0.0)
        totals.append(int(road_prediction_totals.get(r, 0) or 0))
    leader_conf = float(round(conf_sum / max(len(leader_roads), 1), 4))
    max_total = max(totals) if totals else 0

    tier = _tier_from_stats(max_total, float(leader_conf), kind="china")

    # 연속 실패 기반 강등(leader_roads 중 worst loss 기준)
    worst_loss = 0
    for r in leader_roads:
        worst_loss = max(worst_loss, _trailing_losses(stats[r]["window"]))
    tier2, note = _apply_loss_demote(tier, worst_loss)
    tier = tier2

    if tier == "NONE":
        return [], None, 0.0, "NONE", note or "china tier NONE"

    return leader_roads, leader_signal, float(leader_conf), tier, note


def _choose_overall_leader(
    big_tuple: Tuple[Optional[str], Optional[str], float, str, Optional[str]],
    china_tuple: Tuple[List[str], Optional[str], float, str, Optional[str]],
) -> Tuple[Optional[str], Optional[str], float, Optional[str], str, Optional[str]]:
    """
    Big/China 독립 평가 결과를 받아 overall leader를 선정한다.
    우선순위:
    1) trust tier 우선
    2) 동일 tier면 confidence 우선
    3) 동일하면 Big 계열 우선
    Return: (leader_road, leader_signal, leader_conf, leader_source, leader_tier, confidence_note)
    """
    big_road, big_sig, big_conf, big_tier, big_note = big_tuple
    china_roads, china_sig, china_conf, china_tier, china_note = china_tuple

    # 기본값
    best = (None, None, 0.0, None, "NONE", "no leader")

    # big 후보
    if big_tier != "NONE" and big_road and big_sig in ("P", "B"):
        best = (big_road, big_sig, float(big_conf), "big", big_tier, big_note or "big leader")

    # china 후보와 비교
    if china_tier != "NONE" and china_sig in ("P", "B") and china_roads:
        candidate = ("+".join(china_roads), china_sig, float(china_conf), "china", china_tier, china_note or "china leader")
        # tier 우선
        if _tier_rank(china_tier) > _tier_rank(best[4]):
            best = candidate
        elif _tier_rank(china_tier) == _tier_rank(best[4]):
            # 동일 tier면 confidence
            if float(china_conf) > float(best[2] or 0.0):
                best = candidate
            elif float(china_conf) == float(best[2] or 0.0):
                # 동일이면 Big 우선 → best 유지
                pass

    return best


def _normalize_active_overall(active: Any) -> Optional[Dict[str, Any]]:
    """저장된 active_overall을 안전하게 정규화(구버전 state 호환)."""
    if not isinstance(active, dict):
        return None
    src = active.get("source")
    if src == "big":
        road_name = active.get("road")
        if road_name not in _BIG_ROAD_ROADS:
            return None
        return {"source": "big", "road": road_name}
    if src == "china":
        roads = active.get("roads")
        if not isinstance(roads, list) or not roads:
            return None
        norm: List[str] = []
        for r in roads:
            if r in _CHINA_ROADS and r not in norm:
                norm.append(r)
        if not norm:
            return None
        return {"source": "china", "roads": norm}
    return None


def _big_candidate_for_road(
    stats: Dict[str, Dict[str, Any]],
    road_hit_rates: Dict[str, float],
    road_prediction_totals: Dict[str, int],
    new_signals: Dict[str, Optional[str]],
    pb_len: int,
    road_name: str,
) -> Tuple[Optional[str], Optional[str], float, str, Optional[str], bool]:
    """
    특정 Big 로드(bead/bigroad)의 현재 신뢰 상태 산정.
    Return: (road, signal, conf, tier, note, demoted_this_round)
    """
    if pb_len < _TRUST_MIN_PB_LEN:
        return None, None, 0.0, "NONE", "pb_len < 3 → NONE", False
    if road_name not in _BIG_ROAD_ROADS:
        return None, None, 0.0, "NONE", "invalid big road", False

    total = int(road_prediction_totals.get(road_name, 0) or 0)
    acc = float(road_hit_rates.get(road_name, 0.0) or 0.0)
    tier = _tier_from_stats(total, acc, kind="big")
    if tier == "NONE":
        return None, None, 0.0, "NONE", "big tier NONE (insufficient stats)", False

    w = stats.get(road_name, {}).get("window")
    window = w if isinstance(w, deque) else deque(list(w) if isinstance(w, list) else [], maxlen=_WINDOW_MAXLEN)
    loss_streak = _trailing_losses(window)

    demoted_tier, note = _apply_loss_demote(tier, loss_streak)
    demoted = bool(loss_streak >= 1 and demoted_tier != tier)
    tier = demoted_tier
    if tier == "NONE":
        return None, None, 0.0, "NONE", note or "demoted to NONE", demoted

    sig = new_signals.get(road_name)
    signal = sig if sig in ("P", "B") else None
    if signal is None:
        return None, None, 0.0, "NONE", "big road has no valid signal", demoted

    return road_name, signal, float(acc), tier, note, demoted


def _china_candidate_for_roads(
    stats: Dict[str, Dict[str, Any]],
    road_hit_rates: Dict[str, float],
    road_prediction_totals: Dict[str, int],
    new_signals: Dict[str, Optional[str]],
    pb_len: int,
    roads: List[str],
) -> Tuple[Optional[str], Optional[str], float, str, Optional[str], bool]:
    """
    특정 China 로드 묶음의 현재 신뢰 상태 산정.
    - roads 중 valid signal( P/B )만으로 다수결 신호를 산정
    Return: (leader_road_name, signal, conf, tier, note, demoted_this_round)
    """
    if pb_len < _TRUST_MIN_PB_LEN:
        return None, None, 0.0, "NONE", "pb_len < 3 → NONE", False

    norm: List[str] = []
    for r in roads:
        if r in _CHINA_ROADS and r not in norm:
            norm.append(r)
    if not norm:
        return None, None, 0.0, "NONE", "no china roads", False

    active = [r for r in norm if new_signals.get(r) in ("P", "B")]
    if not active:
        return None, None, 0.0, "NONE", "china roads have no valid signals", False

    votes_p = [r for r in active if new_signals.get(r) == "P"]
    votes_b = [r for r in active if new_signals.get(r) == "B"]
    if len(votes_p) > len(votes_b):
        signal = "P"
        used = votes_p
    elif len(votes_b) > len(votes_p):
        signal = "B"
        used = votes_b
    else:
        # 동률: 적중률 우선(참고용)
        best = active[-1]
        best_acc = -1.0
        for r in active:
            acc = float(road_hit_rates.get(r, 0.0) or 0.0)
            if acc > best_acc:
                best_acc = acc
                best = r
        signal = new_signals.get(best)
        used = [best] if signal in ("P", "B") else []

    if not used or signal not in ("P", "B"):
        return None, None, 0.0, "NONE", "china signal selection failed", False

    # confidence(참고용): used 평균 적중률
    conf_sum = 0.0
    for r in used:
        conf_sum += float(road_hit_rates.get(r, 0.0) or 0.0)
    conf = float(round(conf_sum / max(len(used), 1), 4))

    totals = [int(road_prediction_totals.get(r, 0) or 0) for r in used]
    total = max(totals) if totals else 0
    tier = _tier_from_stats(total, float(conf), kind="china")
    if tier == "NONE":
        return None, None, 0.0, "NONE", "china tier NONE (insufficient stats)", False

    worst_loss = 0
    for r in used:
        w = stats.get(r, {}).get("window")
        window = w if isinstance(w, deque) else deque(list(w) if isinstance(w, list) else [], maxlen=_WINDOW_MAXLEN)
        worst_loss = max(worst_loss, _trailing_losses(window))

    demoted_tier, note = _apply_loss_demote(tier, worst_loss)
    demoted = bool(worst_loss >= 1 and demoted_tier != tier)
    tier = demoted_tier
    if tier == "NONE":
        return None, None, 0.0, "NONE", note or "demoted to NONE", demoted

    leader_road_name = "+".join(used)
    return leader_road_name, signal, float(conf), tier, note, demoted


def _apply_overall_stability(
    best: Tuple[Optional[str], Optional[str], float, Optional[str], str, Optional[str]],
    stats: Dict[str, Dict[str, Any]],
    road_hit_rates: Dict[str, float],
    road_prediction_totals: Dict[str, int],
    new_signals: Dict[str, Optional[str]],
    pb_len: int,
) -> Tuple[Optional[str], Optional[str], float, Optional[str], str, str]:
    """
    overall leader 안정성 규칙 적용:
    - 동일 tier에서는 기존 overall leader 유지
    - 상위 tier 등장 시에만 교체 허용
    - 연속 실패 강등 직후 즉시 반대 방향(뒤집기) 제안은 한 템포 NONE으로 해제
    Return: (leader_road, leader_signal, leader_conf, leader_source, leader_trust_state, confidence_note)
    """
    best_road, best_sig, best_conf, best_source, best_tier, best_note = best

    # 0~2판: 무조건 NONE + active 초기화
    if pb_len < _TRUST_MIN_PB_LEN:
        _leader_state["active_overall"] = None
        _leader_state["last_overall_signal"] = None
        return None, None, 0.0, None, "NONE", "pb_len < 3 → NONE"

    active = _normalize_active_overall(_leader_state.get("active_overall"))
    prev_sig = _leader_state.get("last_overall_signal") if _leader_state else None

    # 현재 active의 “현재 상태”를 재평가(동일 tier 유지/강등 반영)
    active_road: Optional[str] = None
    active_sig: Optional[str] = None
    active_conf: float = 0.0
    active_source: Optional[str] = None
    active_tier: str = "NONE"
    active_note: Optional[str] = None
    active_demoted: bool = False

    if active:
        if active["source"] == "big":
            road_name = str(active.get("road"))
            cand = _big_candidate_for_road(stats, road_hit_rates, road_prediction_totals, new_signals, pb_len, road_name)
            active_road, active_sig, active_conf, active_tier, active_note, active_demoted = cand
            active_source = "big" if active_tier != "NONE" else None
        elif active["source"] == "china":
            roads = list(active.get("roads") or [])
            cand = _china_candidate_for_roads(stats, road_hit_rates, road_prediction_totals, new_signals, pb_len, roads)
            active_road, active_sig, active_conf, active_tier, active_note, active_demoted = cand
            active_source = "china" if active_tier != "NONE" else None

    # best 후보가 NONE이고 active도 NONE이면 NONE
    if _tier_rank(best_tier) == 0 and _tier_rank(active_tier) == 0:
        _leader_state["active_overall"] = None
        _leader_state["last_overall_signal"] = None
        return None, None, 0.0, None, "NONE", "no leader candidate"

    # active가 유효하면, 동일 tier에서는 무조건 유지(신뢰 안정)
    chosen_road: Optional[str]
    chosen_sig: Optional[str]
    chosen_conf: float
    chosen_source: Optional[str]
    chosen_tier: str
    chosen_note: str

    if _tier_rank(active_tier) > 0:
        # 상위 tier 등장 시에만 교체
        if _tier_rank(best_tier) > _tier_rank(active_tier):
            chosen_road, chosen_sig, chosen_conf, chosen_source, chosen_tier = best_road, best_sig, float(best_conf or 0.0), best_source, best_tier
            chosen_note = (best_note or "").strip() or f"switch to higher tier {chosen_tier}"
        else:
            chosen_road, chosen_sig, chosen_conf, chosen_source, chosen_tier = active_road, active_sig, float(active_conf or 0.0), active_source, active_tier
            chosen_note = (active_note or "").strip() or f"keep prev leader {chosen_tier}"
    else:
        # active 없음/무효 → best 사용
        chosen_road, chosen_sig, chosen_conf, chosen_source, chosen_tier = best_road, best_sig, float(best_conf or 0.0), best_source, best_tier
        chosen_note = (best_note or "").strip() or f"use best leader {chosen_tier}"

    # 뒤집기(즉시 반대 제안) 차단: “연속 실패로 강등된 직후”만 가드
    # - active_demoted가 True인 상태에서 즉시 반대로 신호가 바뀌면, 한 템포 NONE으로 해제
    if active_demoted and prev_sig in ("P", "B") and chosen_sig in ("P", "B") and prev_sig != chosen_sig:
        _leader_state["active_overall"] = None
        _leader_state["last_overall_signal"] = None
        return None, None, 0.0, None, "NONE", "flip blocked after loss demotion"

    # active_overall 저장(안정성 유지용)
    if chosen_tier == "NONE" or chosen_source not in ("big", "china") or chosen_sig not in ("P", "B") or not chosen_road:
        _leader_state["active_overall"] = None
        _leader_state["last_overall_signal"] = None
        return None, None, 0.0, None, "NONE", "no stable leader"

    if chosen_source == "big":
        _leader_state["active_overall"] = {"source": "big", "road": chosen_road}
    else:
        # chosen_road는 "+". 내부 저장은 road list로 유지
        roads = []
        for r in (chosen_road.split("+") if isinstance(chosen_road, str) else []):
            if r in _CHINA_ROADS and r not in roads:
                roads.append(r)
        if not roads:
            roads = [r for r in _CHINA_ROADS if new_signals.get(r) == chosen_sig]
            roads = roads[:1] if roads else []
        _leader_state["active_overall"] = {"source": "china", "roads": roads}

    _leader_state["last_overall_signal"] = chosen_sig

    # confidence_note는 한 문장으로 강제
    note = chosen_note.replace("\n", " ").strip()
    if not note:
        note = f"{chosen_source} leader {chosen_tier}"
    return chosen_road, chosen_sig, float(chosen_conf), chosen_source, chosen_tier, note


def _check_ready(pb_seq: List[str]) -> Tuple[bool, str]:
    """
    리더 준비도(readiness) 체크: 기존 의미 유지.
    - 최소 표본(_MIN_TOTAL)과 window 길이(_MIN_WINDOW_FOR_READY) 기준
    - 준비 미달은 예외가 아니라 (False, reason)으로 반환
    """
    if not pb_seq:
        return False, "pb_seq empty"

    if len(pb_seq) < _MIN_TOTAL:
        return False, f"BigRoad insufficient: {len(pb_seq)} < {_MIN_TOTAL}"

    # 현재 window 최대 len(_WINDOW_MAXLEN) 대비 최소 필요 길이
    # NOTE: 실제 window 누적은 pred total이 늘어야 증가하므로, 단순 pb_seq 길이로 추정한다.
    if len(pb_seq) < _MIN_WINDOW_FOR_READY + 2:
        return False, f"window insufficient: {len(pb_seq)} < {_MIN_WINDOW_FOR_READY + 2}"

    return True, ""


def update_and_get_leader_features(
    prev_round_winner: Optional[str],
    pb_seq: List[str],
    pb_stats: Dict[str, Any],
    streak_info: Dict[str, Any],
    pattern_dict: Dict[str, Any],
    adv_features: Dict[str, Any],  # 시그니처 호환용(검증만 하고 미사용)
) -> Dict[str, Any]:
    """
    한 판 기준 리더 로드 상태 업데이트 + 리더 Feature 반환.

    폴백 금지:
    - 입력 무결성(Type/Value/필수키) 위반은 즉시 예외
    - 준비 미달(통계 부족/신호 부족)은 예외로 죽이지 않고 leader_ready=False로 명시 반환
    """
    _require_prev_winner(prev_round_winner)
    _require_list_str_arg("pb_seq", pb_seq)
    _require_dict_arg("pb_stats", pb_stats)
    _require_dict_arg("streak_info", streak_info)
    _require_dict_arg("pattern_dict", pattern_dict)
    if not isinstance(adv_features, dict):
        raise TypeError(f"[road_leader] adv_features must be dict, got {type(adv_features).__name__}")

    _require_pb_stats(pb_stats)
    _require_streak_info(streak_info)
    _require_pattern_dict(pattern_dict)

    if not _leader_state:
        reset_leader_state()

    stats: Dict[str, Dict[str, Any]] = _leader_state["stats"]
    last_signals: Dict[str, Optional[str]] = _leader_state["last_signals"]

    prev_round_index = int(_leader_state.get("round_index") or 0)
    pb_len = len(pb_seq)

    # -------- pb_seq 점프 감지: 정확히 복원 불가 → 오염 방지 리셋 --------
    if prev_round_index > 0 and pb_len > prev_round_index + 1:
        logger.error(
            "[ROAD-LEADER] pb_seq jumped: prev_round_index=%d -> pb_len=%d. "
            "Cannot reconstruct intermediate rounds. Leader state will be reset to avoid contamination.",
            prev_round_index, pb_len
        )
        reset_leader_state()
        _leader_state["round_index"] = pb_len

        if not pb_seq:
            raise RoadLeaderNotReadyError("pb_seq empty (cannot compute signals)")
        new_signals = _compute_signals_for_next(pb_seq, pb_stats, streak_info, pattern_dict)
        _leader_state["last_signals"] = new_signals

        leader_ready, not_ready_reason = _check_ready(pb_seq)
        leader_ready = False
        not_ready_reason = (not_ready_reason + " | " if not_ready_reason else "") + "pb_seq jumped: leader_state reset"

        china_signals = {r: new_signals.get(r) for r in _CHINA_ROADS}
        china_windows = {r: [] for r in _CHINA_ROADS}

        # trust state는 pb_len 기준으로만(점프 후 초기화 구간은 NONE)
        trust_state = "NONE"
        can_override_side = False
        confidence_note = "pb_seq jumped → leader_state reset → NONE"

        return {
            "leader_state": {
                "ready": False,
                "reason": not_ready_reason,

                "leader_road": None,
                "leader_signal": None,
                "leader_confidence": 0.0,
                "leader_source": None,

                "big_leader_road": None,
                "big_leader_signal": None,
                "big_leader_confidence": 0.0,

                "china_leader_roads": [],
                "china_leader_signal": None,
                "china_leader_confidence": 0.0,

                "china_signals": china_signals,
                "china_windows": china_windows,

                # NEW (trust state)
                "leader_trust_state": trust_state,
                "confidence_note": confidence_note,
                "can_override_side": can_override_side,

                # 호환/계약 유지: alias
                "leader_ready": False,
                "leader_not_ready_reason": not_ready_reason,

                # 출력 책임(leader_state로도 제공)
                "road_hit_rates": {name: 0.0 for name in _LEADER_ROADS},
                "road_prediction_totals": {name: 0 for name in _LEADER_ROADS},
            },

            "road_hit_rates": {name: 0.0 for name in _LEADER_ROADS},
            "road_prediction_totals": {name: 0 for name in _LEADER_ROADS},
        }

    # -------- 1) 적중률 누적(직전 저장된 신호 vs 이번에 확정된 승자) --------
    advanced = pb_len > prev_round_index
    if advanced and prev_round_winner in ("P", "B"):
        for name in _LEADER_ROADS:
            sig = last_signals.get(name)
            if sig in ("P", "B"):
                s = stats[name]
                s["total"] += 1
                hit = 1 if sig == prev_round_winner else 0
                if hit:
                    s["correct"] += 1
                w = s.get("window")
                if not isinstance(w, deque):
                    s["window"] = deque(list(w) if isinstance(w, list) else [], maxlen=_WINDOW_MAXLEN)
                s["window"].append(hit)

    # round index 동기화(중복 호출이면 그대로)
    _leader_state["round_index"] = pb_len

    # -------- 2) 이번 pb_seq 기준 다음 신호 계산 --------
    if not pb_seq:
        raise RoadLeaderNotReadyError("pb_seq empty (cannot compute signals)")
    new_signals = _compute_signals_for_next(pb_seq, pb_stats, streak_info, pattern_dict)
    _leader_state["last_signals"] = new_signals

    # -------- 3) hit rates / totals 산출 --------
    road_hit_rates, road_prediction_totals = _compute_hit_rates(stats)

    # -------- 4) 중국점 window 제공(상위 2/3 판단용) --------
    china_signals = {r: new_signals.get(r) for r in _CHINA_ROADS}
    china_windows: Dict[str, List[int]] = {}
    for r in _CHINA_ROADS:
        w = stats.get(r, {}).get("window")
        if isinstance(w, deque):
            china_windows[r] = list(w)
        elif isinstance(w, list):
            china_windows[r] = w[:]
        else:
            china_windows[r] = []

    # -------- 5) trust state 기반 듀얼 리더 선정 --------
    big_tuple = _select_big_leader_with_tier(stats, road_hit_rates, road_prediction_totals, new_signals, pb_len)
    china_tuple = _select_china_leader_with_tier(stats, road_hit_rates, road_prediction_totals, new_signals, pb_len)

    big_leader_road, big_leader_signal, big_leader_conf, big_tier, _big_note = big_tuple
    china_leader_roads, china_leader_signal, china_leader_conf, china_tier, _china_note = china_tuple

    # -------- 6) overall leader: tier 우선 + confidence (best) --------
    best_overall = _choose_overall_leader(big_tuple, china_tuple)

    # -------- 6-1) stability: 동일 tier에서는 기존 leader 유지, 상위 tier만 교체 --------
    leader_road, leader_signal, leader_conf, leader_source, leader_trust_state, confidence_note = _apply_overall_stability(
        best_overall, stats, road_hit_rates, road_prediction_totals, new_signals, pb_len
    )

    can_override_side = bool(leader_trust_state == "STRONG")

    # -------- 7) readiness 판정(기존 의미 유지) + 사유 기록 --------
    leader_ready, not_ready_reason = _check_ready(pb_seq)

    # readiness는 기존 의미 그대로 유지한다.
    # 단, leader_signal은 trust_state가 NONE이 아니면 존재할 수 있다(= 3판 이후 리더 사용).
    if not leader_ready:
        logger.warning("[ROAD-LEADER] NOT READY (non-fatal): %s", not_ready_reason)

    return {
        "leader_state": {
            "ready": bool(leader_ready),
            "reason": not_ready_reason,

            "leader_road": leader_road,
            "leader_signal": leader_signal if leader_signal in ("P", "B") else None,
            "leader_confidence": float(leader_conf or 0.0),
            "leader_source": leader_source,

            "big_leader_road": big_leader_road,
            "big_leader_signal": big_leader_signal if big_leader_signal in ("P", "B") else None,
            "big_leader_confidence": float(big_leader_conf or 0.0),

            "china_leader_roads": china_leader_roads,
            "china_leader_signal": china_leader_signal if china_leader_signal in ("P", "B") else None,
            "china_leader_confidence": float(china_leader_conf or 0.0),

            "china_signals": china_signals,
            "china_windows": china_windows,

            # NEW: trust state outputs (프롬프트/상위 로직에서 직접 사용)
            "leader_trust_state": leader_trust_state,
            "confidence_note": confidence_note,
            "can_override_side": can_override_side,

            # 호환/계약 유지: alias
            "leader_ready": bool(leader_ready),
            "leader_not_ready_reason": not_ready_reason,

            # 출력 책임(leader_state로도 제공)
            "road_hit_rates": road_hit_rates,
            "road_prediction_totals": road_prediction_totals,
        },

        "road_hit_rates": road_hit_rates,
        "road_prediction_totals": road_prediction_totals,
    }
