# -*- coding: utf-8 -*-
# road_leader.py
"""
Road Leader Engine for Baccarat Predictor AI Engine v12.1
(RULE-ONLY · STRICT · NO-FALLBACK · FAIL-FAST)

역할:
- Big Road / 중국점 3종 / 본매(bead) 기준으로
  각 로드맵이 최근 구간에서 얼마나 잘 맞는지 추적
- 5개 로드맵별 다음 수 방향(P/B) 신호를 생성
- 최근 성과 기반 confidence를 계산해 overall leader를 선정
- recommend.py의 PASS / PROBE / NORMAL / side 결정을 침범하지 않는
  순수 "보조 신뢰 엔진"으로만 동작

변경 요약 (2026-03-14)
----------------------------------------------------
1) RULE-ONLY 구조 반영
   - GPT / 프롬프트 관련 설명 제거
   - leader_state 계약을 v12 rule-only 구조 기준으로 정리
2) confidence 산식 개선
   - confidence =
       0.60 * hit_rate +
       0.25 * window_consistency +
       0.15 * stability
3) 중국점 신호 개선
   - future_simulator 기반으로
     다음 판이 P/B일 때 각 중국점 road가 무엇을 찍는지 비교해
     road별 P/B 방향 신호를 결정
4) STRICT 상태 복원/검증 강화
   - set_state() / _normalize_state_windows() 에서
     stats/window/signal_window/last_signals 구조를 엄격 검증
   - 잘못된 상태를 조용히 비우거나 교정하지 않고 즉시 예외 처리
5) readiness 정합성 개선
   - 단순 pb_len만 보지 않고 실제 total/window/signal_window 축적 상태를 함께 확인
6) 기존 안정성 규칙 유지
   - 동일 tier에서는 기존 overall leader 유지
   - 상위 tier 등장 시에만 교체
   - loss demotion 직후 즉시 flip 차단
----------------------------------------------------

중요 정책:
- 입력 계약(Type/필수키/Value) 위반은 즉시 예외
- 준비 미달(통계 부족/신호 부족)은 예외가 아니라 leader_ready=False로 반환
- road_leader.py는 상위 recommend.py의 진입/베팅 결정을 절대 침범하지 않는다.
"""

from __future__ import annotations

import copy
import logging
import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import future_simulator

logger = logging.getLogger(__name__)

# 리더 평가 대상 로드
_LEADER_ROADS = ("bead", "bigroad", "bigeye", "small", "cockroach")
_BIG_ROAD_ROADS = ("bead", "bigroad")
_CHINA_ROADS = ("bigeye", "small", "cockroach")

_VALID_SIDE = ("P", "B")
_VALID_WINNER = ("P", "B", "T")

# window / readiness 기준
_MIN_TOTAL = 12
_WINDOW_MAXLEN = 20
_MIN_WINDOW_FOR_READY = 6

# 3판부터 WEAK/MID/STRONG 산정 가능
_TRUST_MIN_PB_LEN = 3

# confidence tier 기준(보수적 유지)
_BIG_TIER_WEAK_MIN_TOTAL = 3
_BIG_TIER_WEAK_MIN_CONF = 0.55
_BIG_TIER_MID_MIN_TOTAL = 7
_BIG_TIER_MID_MIN_CONF = 0.57
_BIG_TIER_STRONG_MIN_TOTAL = 12
_BIG_TIER_STRONG_MIN_CONF = 0.60

_CHINA_TIER_WEAK_MIN_TOTAL = 3
_CHINA_TIER_WEAK_MIN_CONF = 0.52
_CHINA_TIER_MID_MIN_TOTAL = 7
_CHINA_TIER_MID_MIN_CONF = 0.54
_CHINA_TIER_STRONG_MIN_TOTAL = 12
_CHINA_TIER_STRONG_MIN_CONF = 0.56

_leader_state: Dict[str, Any] = {}


class RoadLeaderError(RuntimeError):
    """road_leader 계열 오류(폴백 금지)."""


class RoadLeaderNotReadyError(RoadLeaderError):
    """
    분석 자체가 성립 불가한 입력(pb_seq empty 등)에서만 예외.
    준비 미달(통계 부족/신호 부족)은 예외가 아니라 leader_ready=False로 반환한다.
    """


def _as_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"{name} must be int, got {type(value).__name__}")


def _as_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be float, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be float, got {type(value).__name__}")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def _require_dict_arg(name: str, obj: Any) -> None:
    if not isinstance(obj, dict):
        raise TypeError(f"[road_leader] {name} must be dict, got {type(obj).__name__}")


def _require_list_str_arg(name: str, obj: Any) -> None:
    if not isinstance(obj, list):
        raise TypeError(f"[road_leader] {name} must be List[str], got {type(obj).__name__}")
    for i, x in enumerate(obj):
        if not isinstance(x, str):
            raise TypeError(f"[road_leader] {name}[{i}] must be str, got {type(x).__name__}")


def _normalize_side(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"[road_leader] {name} must be str, got {type(value).__name__}")
    s = value.strip().upper()
    if s not in _VALID_SIDE:
        raise ValueError(f"[road_leader] {name} invalid: {value!r} (allowed: {_VALID_SIDE})")
    return s


def _normalize_side_or_none(value: Any, *, name: str) -> Optional[str]:
    if value is None:
        return None
    return _normalize_side(value, name=name)


def _normalize_winner(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"[road_leader] {name} must be str, got {type(value).__name__}")
    s = value.strip().upper()
    if s not in _VALID_WINNER:
        raise ValueError(f"[road_leader] {name} invalid: {value!r} (allowed: {_VALID_WINNER})")
    return s


def _validate_pb_seq(pb_seq: Any, *, name: str) -> List[str]:
    if not isinstance(pb_seq, list):
        raise TypeError(f"[road_leader] {name} must be List[str], got {type(pb_seq).__name__}")
    out: List[str] = []
    for i, item in enumerate(pb_seq):
        out.append(_normalize_side(item, name=f"{name}[{i}]"))
    return out


def _validate_window_values(window: Any, *, name: str) -> deque:
    if isinstance(window, deque):
        raw = list(window)
    elif isinstance(window, list):
        raw = window
    elif window is None:
        raw = []
    else:
        raise TypeError(f"[road_leader] {name} must be deque/list/None, got {type(window).__name__}")

    normalized: List[int] = []
    for i, v in enumerate(raw):
        iv = _as_int(v, name=f"{name}[{i}]")
        if iv not in (0, 1):
            raise ValueError(f"[road_leader] {name}[{i}] must be 0/1, got {iv}")
        normalized.append(iv)
    return deque(normalized[-_WINDOW_MAXLEN:], maxlen=_WINDOW_MAXLEN)


def _validate_signal_window(signal_window: Any, *, name: str) -> List[str]:
    if signal_window is None:
        return []
    if not isinstance(signal_window, list):
        raise TypeError(f"[road_leader] {name} must be list[str] or None, got {type(signal_window).__name__}")
    normalized: List[str] = []
    for i, v in enumerate(signal_window):
        normalized.append(_normalize_side(v, name=f"{name}[{i}]"))
    return normalized[-_WINDOW_MAXLEN:]


def _validate_stats_dict(stats: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(stats, dict):
        raise TypeError(f"[road_leader] stats must be dict, got {type(stats).__name__}")

    normalized: Dict[str, Dict[str, Any]] = {}
    for name in _LEADER_ROADS:
        if name not in stats:
            raise KeyError(f"[road_leader] stats missing road key: {name}")
        s = stats[name]
        if not isinstance(s, dict):
            raise TypeError(f"[road_leader] stats[{name}] must be dict, got {type(s).__name__}")

        if "total" not in s or "correct" not in s or "window" not in s or "signal_window" not in s:
            raise KeyError(
                f"[road_leader] stats[{name}] missing one of required keys: total/correct/window/signal_window"
            )

        total = _as_int(s["total"], name=f"stats[{name}].total")
        correct = _as_int(s["correct"], name=f"stats[{name}].correct")
        if total < 0:
            raise ValueError(f"[road_leader] stats[{name}].total must be >= 0, got {total}")
        if correct < 0:
            raise ValueError(f"[road_leader] stats[{name}].correct must be >= 0, got {correct}")
        if correct > total:
            raise ValueError(
                f"[road_leader] stats[{name}].correct must be <= total, got correct={correct}, total={total}"
            )

        normalized[name] = {
            "total": total,
            "correct": correct,
            "window": _validate_window_values(s["window"], name=f"stats[{name}].window"),
            "signal_window": _validate_signal_window(s["signal_window"], name=f"stats[{name}].signal_window"),
        }
    return normalized


def _validate_last_signals(last_signals: Any) -> Dict[str, Optional[str]]:
    if not isinstance(last_signals, dict):
        raise TypeError(f"[road_leader] last_signals must be dict, got {type(last_signals).__name__}")
    normalized: Dict[str, Optional[str]] = {}
    for name in _LEADER_ROADS:
        if name not in last_signals:
            raise KeyError(f"[road_leader] last_signals missing road key: {name}")
        normalized[name] = _normalize_side_or_none(last_signals[name], name=f"last_signals[{name}]")
    return normalized


def _validate_active_overall(active: Any) -> Optional[Dict[str, Any]]:
    if active is None:
        return None
    if not isinstance(active, dict):
        raise TypeError(f"[road_leader] active_overall must be dict or None, got {type(active).__name__}")

    src = active.get("source")
    if src == "big":
        road_name = active.get("road")
        if road_name not in _BIG_ROAD_ROADS:
            raise ValueError(f"[road_leader] active_overall.big road invalid: {road_name!r}")
        return {"source": "big", "road": road_name}

    if src == "china":
        roads = active.get("roads")
        if not isinstance(roads, list) or not roads:
            raise TypeError("[road_leader] active_overall.china.roads must be non-empty list")
        norm: List[str] = []
        for i, r in enumerate(roads):
            if not isinstance(r, str):
                raise TypeError(f"[road_leader] active_overall.china.roads[{i}] must be str")
            s = r.strip()
            if s not in _CHINA_ROADS:
                raise ValueError(f"[road_leader] active_overall.china.roads[{i}] invalid: {r!r}")
            if s not in norm:
                norm.append(s)
        return {"source": "china", "roads": norm}

    raise ValueError(f"[road_leader] active_overall.source invalid: {src!r}")


def reset_leader_state() -> None:
    """새 슈 시작 시 리더 로드 통계 초기화."""
    global _leader_state
    _leader_state = {
        "round_index": 0,
        "stats": {
            name: {
                "total": 0,
                "correct": 0,
                "window": deque(maxlen=_WINDOW_MAXLEN),
                "signal_window": [],
            }
            for name in _LEADER_ROADS
        },
        "last_signals": {name: None for name in _LEADER_ROADS},
        "active_overall": None,
        "last_overall_signal": None,
    }


def get_state() -> Dict[str, Any]:
    """UNDO용: 현재 리더 상태 스냅샷 반환."""
    return copy.deepcopy(_leader_state)


def _normalize_state_windows(state: Dict[str, Any]) -> Dict[str, Any]:
    """복원된 상태를 STRICT하게 검증하고 정규화한다."""
    if not isinstance(state, dict):
        raise TypeError(f"[road_leader] state must be dict, got {type(state).__name__}")

    if "round_index" not in state or "stats" not in state or "last_signals" not in state:
        raise KeyError("[road_leader] state missing required keys: round_index/stats/last_signals")

    round_index = _as_int(state["round_index"], name="state.round_index")
    if round_index < 0:
        raise ValueError(f"[road_leader] state.round_index must be >= 0, got {round_index}")

    stats = _validate_stats_dict(state["stats"])
    last_signals = _validate_last_signals(state["last_signals"])
    active_overall = _validate_active_overall(state.get("active_overall"))
    last_overall_signal = _normalize_side_or_none(
        state.get("last_overall_signal"),
        name="state.last_overall_signal",
    )

    return {
        "round_index": round_index,
        "stats": stats,
        "last_signals": last_signals,
        "active_overall": active_overall,
        "last_overall_signal": last_overall_signal,
    }


def set_state(state: Optional[Dict[str, Any]]) -> None:
    """UNDO 복구용: 리더 상태를 통째로 교체."""
    global _leader_state
    if state is None:
        logger.error("[ROAD-LEADER] set_state(None) forbidden")
        raise RoadLeaderError("set_state(None) forbidden (폴백 금지): call reset_leader_state() explicitly")
    _leader_state = _normalize_state_windows(copy.deepcopy(state))


def _require_prev_winner(prev_round_winner: Any) -> None:
    if prev_round_winner is None:
        raise RoadLeaderError("[road_leader] prev_round_winner is None forbidden (폴백 금지)")
    _normalize_winner(prev_round_winner, name="prev_round_winner")


def _require_pb_stats(pb_stats: Dict[str, Any]) -> None:
    if "pb_ratio" not in pb_stats:
        raise KeyError("[road_leader] pb_stats missing required key: pb_ratio (폴백 금지)")
    fv = _as_float(pb_stats["pb_ratio"], name="pb_stats.pb_ratio")
    if fv < 0.0 or fv > 1.0:
        raise ValueError(f"[road_leader] pb_stats.pb_ratio must be in [0,1], got {fv}")


def _require_streak_info(streak_info: Dict[str, Any]) -> None:
    if "current_streak" not in streak_info:
        raise KeyError("[road_leader] streak_info missing required key: current_streak (폴백 금지)")
    cs = streak_info["current_streak"]
    if not isinstance(cs, dict):
        raise TypeError("[road_leader] streak_info.current_streak must be dict")
    if "side" not in cs or "length" not in cs:
        raise KeyError("[road_leader] streak_info.current_streak missing keys: side/length")
    _normalize_side_or_none(cs["side"], name="streak_info.current_streak.side")
    length = _as_int(cs["length"], name="streak_info.current_streak.length")
    if length < 0:
        raise ValueError("[road_leader] streak_info.current_streak.length must be >= 0")


def _require_pattern_dict(pattern_dict: Dict[str, Any]) -> None:
    if "pattern_type" not in pattern_dict:
        raise KeyError("[road_leader] pattern_dict missing required key: pattern_type (폴백 금지)")
    if not isinstance(pattern_dict["pattern_type"], str):
        raise TypeError("[road_leader] pattern_dict.pattern_type must be str")


def _compute_bead_signal(pb_seq: List[str], pb_stats: Dict[str, Any]) -> Optional[str]:
    """
    bead 신호:
    - pb_ratio 기반 단순 추세
    """
    pb_ratio = _as_float(pb_stats["pb_ratio"], name="pb_stats.pb_ratio")
    if pb_ratio >= 0.55:
        return "P"
    if pb_ratio <= 0.45:
        return "B"
    return None


def _compute_bigroad_signal(
    pb_seq: List[str],
    streak_info: Dict[str, Any],
    pattern_dict: Dict[str, Any],
) -> Optional[str]:
    """
    Big Road 신호:
    - current_streak 길이>=2면 추세 추종
    - pingpong이면 반대편 제안
    """
    cs = streak_info["current_streak"]
    side = _normalize_side_or_none(cs["side"], name="streak_info.current_streak.side")
    length = _as_int(cs["length"], name="streak_info.current_streak.length")

    if side in _VALID_SIDE and length >= 2:
        return side

    pattern_type = pattern_dict["pattern_type"]
    if isinstance(pattern_type, str) and "PINGPONG" in pattern_type.upper():
        if side == "P":
            return "B"
        if side == "B":
            return "P"

    return None


def _build_future_marks_for_next() -> Dict[str, Dict[str, Optional[str]]]:
    """
    다음 판이 P/B일 때 중국점 3종이 무엇을 찍는지 계산한다.
    """
    sim_p = future_simulator.simulate_future_for_side("P", max_rows=6)
    sim_b = future_simulator.simulate_future_for_side("B", max_rows=6)

    return {
        "P": {
            "bigeye": sim_p.get("big_eye"),
            "small": sim_p.get("small_road"),
            "cockroach": sim_p.get("cockroach"),
        },
        "B": {
            "bigeye": sim_b.get("big_eye"),
            "small": sim_b.get("small_road"),
            "cockroach": sim_b.get("cockroach"),
        },
    }


def _compute_china_signal_from_future(
    road_name: str,
    future_marks: Dict[str, Dict[str, Optional[str]]],
) -> Optional[str]:
    """
    road_name별로
    - if P -> R and if B -> B  => P
    - if P -> B and if B -> R  => B
    - 그 외 => None
    """
    if road_name not in _CHINA_ROADS:
        raise ValueError(f"invalid china road: {road_name}")

    p_mark = future_marks["P"].get(road_name)
    b_mark = future_marks["B"].get(road_name)

    if p_mark == "R" and b_mark == "B":
        return "P"
    if p_mark == "B" and b_mark == "R":
        return "B"
    return None


def _compute_signals_for_next(
    pb_seq: List[str],
    pb_stats: Dict[str, Any],
    streak_info: Dict[str, Any],
    pattern_dict: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    """
    5개 로드맵의 '다음 수' 방향 신호 생성.
    """
    bead = _compute_bead_signal(pb_seq, pb_stats)
    bigroad = _compute_bigroad_signal(pb_seq, streak_info, pattern_dict)

    future_marks = _build_future_marks_for_next()
    bigeye = _compute_china_signal_from_future("bigeye", future_marks)
    small = _compute_china_signal_from_future("small", future_marks)
    cockroach = _compute_china_signal_from_future("cockroach", future_marks)

    return {
        "bead": bead,
        "bigroad": bigroad,
        "bigeye": bigeye,
        "small": small,
        "cockroach": cockroach,
    }


def _compute_hit_rates(stats: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, int]]:
    hit_rates: Dict[str, float] = {}
    totals: Dict[str, int] = {}
    for name in _LEADER_ROADS:
        s = stats[name]
        total = _as_int(s["total"], name=f"stats[{name}].total")
        correct = _as_int(s["correct"], name=f"stats[{name}].correct")
        totals[name] = total
        hit_rates[name] = (correct / total) if total > 0 else 0.0
    return hit_rates, totals


def _trailing_losses(window: deque) -> int:
    n = 0
    for v in reversed(list(window)):
        iv = _as_int(v, name="window_value")
        if iv not in (0, 1):
            raise ValueError(f"window_value must be 0/1, got {iv}")
        if iv == 0:
            n += 1
        else:
            break
    return n


def _get_signal_window(stat: Dict[str, Any]) -> List[str]:
    return _validate_signal_window(stat.get("signal_window"), name="stat.signal_window")


def _append_signal_history(stat: Dict[str, Any], signal: Optional[str]) -> None:
    if signal not in _VALID_SIDE:
        return
    sw = _get_signal_window(stat)
    sw.append(signal)
    stat["signal_window"] = sw[-_WINDOW_MAXLEN:]


def _compute_signal_stability(signal_window: List[str]) -> float:
    """
    방향 flip이 적을수록 높다.
    - 길이 < 2 이면 안정도를 충분히 판단할 근거가 없으므로 0.0
    """
    signals = _validate_signal_window(signal_window, name="signal_window")
    if len(signals) < 2:
        return 0.0

    flips = 0
    for i in range(1, len(signals)):
        if signals[i] != signals[i - 1]:
            flips += 1

    denom = len(signals) - 1
    if denom <= 0:
        return 0.0

    stability = 1.0 - (flips / denom)
    return max(0.0, min(1.0, float(stability)))


def _compute_road_confidences(
    stats: Dict[str, Dict[str, Any]],
    road_hit_rates: Dict[str, float],
) -> Dict[str, float]:
    """
    confidence =
      0.60 * hit_rate +
      0.25 * window_consistency +
      0.15 * stability
    """
    out: Dict[str, float] = {}

    for name in _LEADER_ROADS:
        stat = stats[name]
        hit_rate = _as_float(road_hit_rates.get(name, 0.0), name=f"road_hit_rates[{name}]")
        if hit_rate < 0.0 or hit_rate > 1.0:
            raise ValueError(f"road_hit_rates[{name}] must be in [0,1], got {hit_rate}")

        window_values = list(_validate_window_values(stat.get("window"), name=f"stats[{name}].window"))
        window_consistency = (sum(window_values) / float(len(window_values))) if window_values else 0.0
        stability = _compute_signal_stability(_get_signal_window(stat))

        conf = (
            0.60 * hit_rate +
            0.25 * window_consistency +
            0.15 * stability
        )
        conf = max(0.0, min(1.0, conf))

        out[name] = float(round(conf, 6))

    return out


def _apply_loss_demote(tier: str, loss_streak: int) -> Tuple[str, Optional[str]]:
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
    new_tier = order[idx - 1]
    return new_tier, f"loss_streak={loss_streak} → demote {tier}->{new_tier}"


def _tier_rank(tier: str) -> int:
    return {"NONE": 0, "WEAK": 1, "MID": 2, "STRONG": 3}.get(tier, 0)


def _tier_from_confidence(total: int, confidence: float, kind: str) -> str:
    if kind == "big":
        if total >= _BIG_TIER_STRONG_MIN_TOTAL and confidence >= _BIG_TIER_STRONG_MIN_CONF:
            return "STRONG"
        if total >= _BIG_TIER_MID_MIN_TOTAL and confidence >= _BIG_TIER_MID_MIN_CONF:
            return "MID"
        if total >= _BIG_TIER_WEAK_MIN_TOTAL and confidence >= _BIG_TIER_WEAK_MIN_CONF:
            return "WEAK"
        return "NONE"

    if kind == "china":
        if total >= _CHINA_TIER_STRONG_MIN_TOTAL and confidence >= _CHINA_TIER_STRONG_MIN_CONF:
            return "STRONG"
        if total >= _CHINA_TIER_MID_MIN_TOTAL and confidence >= _CHINA_TIER_MID_MIN_CONF:
            return "MID"
        if total >= _CHINA_TIER_WEAK_MIN_TOTAL and confidence >= _CHINA_TIER_WEAK_MIN_CONF:
            return "WEAK"
        return "NONE"

    return "NONE"


def _select_big_leader_with_tier(
    stats: Dict[str, Dict[str, Any]],
    road_hit_rates: Dict[str, float],
    road_confidences: Dict[str, float],
    road_prediction_totals: Dict[str, int],
    new_signals: Dict[str, Optional[str]],
    pb_len: int,
) -> Tuple[Optional[str], Optional[str], float, str, Optional[str]]:
    if pb_len < _TRUST_MIN_PB_LEN:
        return None, None, 0.0, "NONE", "pb_len < 3 → NONE"

    best_road = None
    best_conf = -1.0
    best_total = 0

    for road_name in _BIG_ROAD_ROADS:
        sig = new_signals.get(road_name)
        if sig not in _VALID_SIDE:
            continue

        total = _as_int(road_prediction_totals.get(road_name, 0), name=f"road_prediction_totals[{road_name}]")
        conf = _as_float(road_confidences.get(road_name, 0.0), name=f"road_confidences[{road_name}]")

        if conf > best_conf:
            best_conf = conf
            best_road = road_name
            best_total = total
        elif conf == best_conf and total > best_total:
            best_road = road_name
            best_total = total

    if not best_road:
        return None, None, 0.0, "NONE", "no big signals"

    tier = _tier_from_confidence(best_total, float(best_conf), kind="big")

    window = _validate_window_values(stats[best_road]["window"], name=f"stats[{best_road}].window")
    loss_streak = _trailing_losses(window)
    tier2, note = _apply_loss_demote(tier, loss_streak)
    tier = tier2

    if tier == "NONE":
        return None, None, 0.0, "NONE", note or "big tier NONE"

    return best_road, new_signals[best_road], float(best_conf), tier, note


def _select_china_leader_with_tier(
    stats: Dict[str, Dict[str, Any]],
    road_confidences: Dict[str, float],
    road_prediction_totals: Dict[str, int],
    new_signals: Dict[str, Optional[str]],
    pb_len: int,
) -> Tuple[List[str], Optional[str], float, str, Optional[str]]:
    if pb_len < _TRUST_MIN_PB_LEN:
        return [], None, 0.0, "NONE", "pb_len < 3 → NONE"

    active = [r for r in _CHINA_ROADS if new_signals.get(r) in _VALID_SIDE]
    if not active:
        return [], None, 0.0, "NONE", "no china signals"

    votes_p = [r for r in active if new_signals[r] == "P"]
    votes_b = [r for r in active if new_signals[r] == "B"]

    if len(votes_p) > len(votes_b):
        leader_signal = "P"
        leader_roads = votes_p
    elif len(votes_b) > len(votes_p):
        leader_signal = "B"
        leader_roads = votes_b
    else:
        best = active[0]
        best_conf = -1.0
        for r in active:
            conf = _as_float(road_confidences.get(r, 0.0), name=f"road_confidences[{r}]")
            if conf > best_conf:
                best_conf = conf
                best = r
        leader_signal = new_signals[best]
        leader_roads = [best]

    if leader_signal not in _VALID_SIDE:
        return [], None, 0.0, "NONE", "china leader_signal invalid"

    conf_sum = 0.0
    totals = []
    for r in leader_roads:
        conf_sum += _as_float(road_confidences.get(r, 0.0), name=f"road_confidences[{r}]")
        totals.append(_as_int(road_prediction_totals.get(r, 0), name=f"road_prediction_totals[{r}]"))
    leader_conf = float(round(conf_sum / max(len(leader_roads), 1), 6))
    max_total = max(totals) if totals else 0

    tier = _tier_from_confidence(max_total, float(leader_conf), kind="china")

    worst_loss = 0
    for r in leader_roads:
        worst_loss = max(
            worst_loss,
            _trailing_losses(_validate_window_values(stats[r]["window"], name=f"stats[{r}].window")),
        )
    tier2, note = _apply_loss_demote(tier, worst_loss)
    tier = tier2

    if tier == "NONE":
        return [], None, 0.0, "NONE", note or "china tier NONE"

    return leader_roads, leader_signal, float(leader_conf), tier, note


def _choose_overall_leader(
    big_tuple: Tuple[Optional[str], Optional[str], float, str, Optional[str]],
    china_tuple: Tuple[List[str], Optional[str], float, str, Optional[str]],
) -> Tuple[Optional[str], Optional[str], float, Optional[str], str, Optional[str]]:
    big_road, big_sig, big_conf, big_tier, big_note = big_tuple
    china_roads, china_sig, china_conf, china_tier, china_note = china_tuple

    best = (None, None, 0.0, None, "NONE", "no leader")

    if big_tier != "NONE" and big_road and big_sig in _VALID_SIDE:
        best = (big_road, big_sig, float(big_conf), "big", big_tier, big_note or "big leader")

    if china_tier != "NONE" and china_sig in _VALID_SIDE and china_roads:
        candidate = (
            "+".join(china_roads),
            china_sig,
            float(china_conf),
            "china",
            china_tier,
            china_note or "china leader",
        )
        if _tier_rank(china_tier) > _tier_rank(best[4]):
            best = candidate
        elif _tier_rank(china_tier) == _tier_rank(best[4]):
            if float(china_conf) > float(best[2] or 0.0):
                best = candidate

    return best


def _normalize_active_overall(active: Any) -> Optional[Dict[str, Any]]:
    return _validate_active_overall(active)


def _big_candidate_for_road(
    stats: Dict[str, Dict[str, Any]],
    road_confidences: Dict[str, float],
    road_prediction_totals: Dict[str, int],
    new_signals: Dict[str, Optional[str]],
    pb_len: int,
    road_name: str,
) -> Tuple[Optional[str], Optional[str], float, str, Optional[str], bool]:
    if pb_len < _TRUST_MIN_PB_LEN:
        return None, None, 0.0, "NONE", "pb_len < 3 → NONE", False
    if road_name not in _BIG_ROAD_ROADS:
        return None, None, 0.0, "NONE", "invalid big road", False

    total = _as_int(road_prediction_totals.get(road_name, 0), name=f"road_prediction_totals[{road_name}]")
    conf = _as_float(road_confidences.get(road_name, 0.0), name=f"road_confidences[{road_name}]")
    tier = _tier_from_confidence(total, conf, kind="big")
    if tier == "NONE":
        return None, None, 0.0, "NONE", "big tier NONE (insufficient stats)", False

    window = _validate_window_values(stats.get(road_name, {}).get("window"), name=f"stats[{road_name}].window")
    loss_streak = _trailing_losses(window)

    demoted_tier, note = _apply_loss_demote(tier, loss_streak)
    demoted = bool(loss_streak >= 1 and demoted_tier != tier)
    tier = demoted_tier
    if tier == "NONE":
        return None, None, 0.0, "NONE", note or "demoted to NONE", demoted

    sig = new_signals.get(road_name)
    signal = sig if sig in _VALID_SIDE else None
    if signal is None:
        return None, None, 0.0, "NONE", "big road has no valid signal", demoted

    return road_name, signal, float(conf), tier, note, demoted


def _china_candidate_for_roads(
    stats: Dict[str, Dict[str, Any]],
    road_confidences: Dict[str, float],
    road_prediction_totals: Dict[str, int],
    new_signals: Dict[str, Optional[str]],
    pb_len: int,
    roads: List[str],
) -> Tuple[Optional[str], Optional[str], float, str, Optional[str], bool]:
    if pb_len < _TRUST_MIN_PB_LEN:
        return None, None, 0.0, "NONE", "pb_len < 3 → NONE", False

    norm: List[str] = []
    for r in roads:
        if r in _CHINA_ROADS and r not in norm:
            norm.append(r)
    if not norm:
        return None, None, 0.0, "NONE", "no china roads", False

    active = [r for r in norm if new_signals.get(r) in _VALID_SIDE]
    if not active:
        return None, None, 0.0, "NONE", "china roads have no valid signals", False

    votes_p = [r for r in active if new_signals[r] == "P"]
    votes_b = [r for r in active if new_signals[r] == "B"]
    if len(votes_p) > len(votes_b):
        signal = "P"
        used = votes_p
    elif len(votes_b) > len(votes_p):
        signal = "B"
        used = votes_b
    else:
        best = active[-1]
        best_conf = -1.0
        for r in active:
            conf = _as_float(road_confidences.get(r, 0.0), name=f"road_confidences[{r}]")
            if conf > best_conf:
                best_conf = conf
                best = r
        signal = new_signals[best]
        used = [best] if signal in _VALID_SIDE else []

    if not used or signal not in _VALID_SIDE:
        return None, None, 0.0, "NONE", "china signal selection failed", False

    conf_sum = 0.0
    for r in used:
        conf_sum += _as_float(road_confidences.get(r, 0.0), name=f"road_confidences[{r}]")
    conf = float(round(conf_sum / max(len(used), 1), 6))

    totals = [_as_int(road_prediction_totals.get(r, 0), name=f"road_prediction_totals[{r}]") for r in used]
    total = max(totals) if totals else 0
    tier = _tier_from_confidence(total, float(conf), kind="china")
    if tier == "NONE":
        return None, None, 0.0, "NONE", "china tier NONE (insufficient stats)", False

    worst_loss = 0
    for r in used:
        worst_loss = max(
            worst_loss,
            _trailing_losses(_validate_window_values(stats.get(r, {}).get("window"), name=f"stats[{r}].window")),
        )

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
    road_confidences: Dict[str, float],
    road_prediction_totals: Dict[str, int],
    new_signals: Dict[str, Optional[str]],
    pb_len: int,
) -> Tuple[Optional[str], Optional[str], float, Optional[str], str, str]:
    best_road, best_sig, best_conf, best_source, best_tier, best_note = best

    if pb_len < _TRUST_MIN_PB_LEN:
        _leader_state["active_overall"] = None
        _leader_state["last_overall_signal"] = None
        return None, None, 0.0, None, "NONE", "pb_len < 3 → NONE"

    active = _normalize_active_overall(_leader_state.get("active_overall"))
    prev_sig = _normalize_side_or_none(
        _leader_state.get("last_overall_signal"),
        name="state.last_overall_signal",
    )

    active_road: Optional[str] = None
    active_sig: Optional[str] = None
    active_conf: float = 0.0
    active_source: Optional[str] = None
    active_tier: str = "NONE"
    active_note: Optional[str] = None
    active_demoted: bool = False

    if active:
        if active["source"] == "big":
            road_name = str(active["road"])
            cand = _big_candidate_for_road(
                stats,
                road_confidences,
                road_prediction_totals,
                new_signals,
                pb_len,
                road_name,
            )
            active_road, active_sig, active_conf, active_tier, active_note, active_demoted = cand
            active_source = "big" if active_tier != "NONE" else None
        elif active["source"] == "china":
            roads = list(active["roads"])
            cand = _china_candidate_for_roads(
                stats,
                road_confidences,
                road_prediction_totals,
                new_signals,
                pb_len,
                roads,
            )
            active_road, active_sig, active_conf, active_tier, active_note, active_demoted = cand
            active_source = "china" if active_tier != "NONE" else None

    if _tier_rank(best_tier) == 0 and _tier_rank(active_tier) == 0:
        _leader_state["active_overall"] = None
        _leader_state["last_overall_signal"] = None
        return None, None, 0.0, None, "NONE", "no leader candidate"

    if _tier_rank(active_tier) > 0:
        if _tier_rank(best_tier) > _tier_rank(active_tier):
            chosen_road, chosen_sig, chosen_conf, chosen_source, chosen_tier = (
                best_road,
                best_sig,
                float(best_conf or 0.0),
                best_source,
                best_tier,
            )
            chosen_note = (best_note or "").strip() or f"switch to higher tier {chosen_tier}"
        else:
            chosen_road, chosen_sig, chosen_conf, chosen_source, chosen_tier = (
                active_road,
                active_sig,
                float(active_conf or 0.0),
                active_source,
                active_tier,
            )
            chosen_note = (active_note or "").strip() or f"keep prev leader {chosen_tier}"
    else:
        chosen_road, chosen_sig, chosen_conf, chosen_source, chosen_tier = (
            best_road,
            best_sig,
            float(best_conf or 0.0),
            best_source,
            best_tier,
        )
        chosen_note = (best_note or "").strip() or f"use best leader {chosen_tier}"

    if active_demoted and prev_sig in _VALID_SIDE and chosen_sig in _VALID_SIDE and prev_sig != chosen_sig:
        _leader_state["active_overall"] = None
        _leader_state["last_overall_signal"] = None
        return None, None, 0.0, None, "NONE", "flip blocked after loss demotion"

    if chosen_tier == "NONE" or chosen_source not in ("big", "china") or chosen_sig not in _VALID_SIDE or not chosen_road:
        _leader_state["active_overall"] = None
        _leader_state["last_overall_signal"] = None
        return None, None, 0.0, None, "NONE", "no stable leader"

    if chosen_source == "big":
        _leader_state["active_overall"] = {"source": "big", "road": chosen_road}
    else:
        roads = []
        for r in (chosen_road.split("+") if isinstance(chosen_road, str) else []):
            if r in _CHINA_ROADS and r not in roads:
                roads.append(r)
        if not roads:
            roads = [r for r in _CHINA_ROADS if new_signals.get(r) == chosen_sig][:1]
        _leader_state["active_overall"] = {"source": "china", "roads": roads}

    _leader_state["last_overall_signal"] = chosen_sig

    note = chosen_note.replace("\n", " ").strip()
    if not note:
        note = f"{chosen_source} leader {chosen_tier}"
    return chosen_road, chosen_sig, float(chosen_conf), chosen_source, chosen_tier, note


def _check_ready(pb_seq: List[str], stats: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
    if not pb_seq:
        return False, "pb_seq empty"

    if len(pb_seq) < _MIN_TOTAL:
        return False, f"BigRoad insufficient: {len(pb_seq)} < {_MIN_TOTAL}"

    max_total = 0
    max_window_len = 0
    has_any_signal_history = False

    for name in _LEADER_ROADS:
        stat = stats[name]
        total = _as_int(stat["total"], name=f"stats[{name}].total")
        window = _validate_window_values(stat["window"], name=f"stats[{name}].window")
        signal_window = _get_signal_window(stat)

        max_total = max(max_total, total)
        max_window_len = max(max_window_len, len(window))
        if signal_window:
            has_any_signal_history = True

    if max_total < _MIN_TOTAL:
        return False, f"leader total insufficient: {max_total} < {_MIN_TOTAL}"
    if max_window_len < _MIN_WINDOW_FOR_READY:
        return False, f"leader window insufficient: {max_window_len} < {_MIN_WINDOW_FOR_READY}"
    if not has_any_signal_history:
        return False, "leader signal_window empty"

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
    """
    _require_prev_winner(prev_round_winner)
    pb_seq_v = _validate_pb_seq(pb_seq, name="pb_seq")
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

    prev_round_index = _as_int(_leader_state.get("round_index", 0), name="_leader_state.round_index")
    pb_len = len(pb_seq_v)

    # pb_seq jump 감지
    if prev_round_index > 0 and pb_len > prev_round_index + 1:
        logger.error(
            "[ROAD-LEADER] pb_seq jumped: prev_round_index=%d -> pb_len=%d. "
            "Cannot reconstruct intermediate rounds. Leader state will be reset to avoid contamination.",
            prev_round_index, pb_len
        )
        reset_leader_state()
        _leader_state["round_index"] = pb_len

        if not pb_seq_v:
            raise RoadLeaderNotReadyError("pb_seq empty (cannot compute signals)")

        new_signals = _compute_signals_for_next(pb_seq_v, pb_stats, streak_info, pattern_dict)
        _leader_state["last_signals"] = new_signals

        leader_ready, not_ready_reason = _check_ready(pb_seq_v, _leader_state["stats"])
        leader_ready = False
        not_ready_reason = (not_ready_reason + " | " if not_ready_reason else "") + "pb_seq jumped: leader_state reset"

        china_signals = {r: new_signals.get(r) for r in _CHINA_ROADS}
        china_windows = {r: [] for r in _CHINA_ROADS}

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
                "leader_trust_state": "NONE",
                "confidence_note": "pb_seq jumped → leader_state reset → NONE",
                "can_override_side": False,
                "leader_ready": False,
                "leader_not_ready_reason": not_ready_reason,
                "road_hit_rates": {name: 0.0 for name in _LEADER_ROADS},
                "road_confidences": {name: 0.0 for name in _LEADER_ROADS},
                "road_prediction_totals": {name: 0 for name in _LEADER_ROADS},
            },
            "road_hit_rates": {name: 0.0 for name in _LEADER_ROADS},
            "road_confidences": {name: 0.0 for name in _LEADER_ROADS},
            "road_prediction_totals": {name: 0 for name in _LEADER_ROADS},
        }

    # 1) 적중률 누적
    advanced = pb_len > prev_round_index
    prev_winner_norm = _normalize_winner(prev_round_winner, name="prev_round_winner")

    if advanced and prev_winner_norm in _VALID_SIDE:
        for name in _LEADER_ROADS:
            sig = last_signals.get(name)
            if sig in _VALID_SIDE:
                s = stats[name]
                s["total"] += 1
                hit = 1 if sig == prev_winner_norm else 0
                if hit:
                    s["correct"] += 1
                s["window"] = _validate_window_values(s.get("window"), name=f"stats[{name}].window")
                s["window"].append(hit)

    _leader_state["round_index"] = pb_len

    # 2) 이번 pb_seq 기준 다음 신호 계산
    if not pb_seq_v:
        raise RoadLeaderNotReadyError("pb_seq empty (cannot compute signals)")
    new_signals = _compute_signals_for_next(pb_seq_v, pb_stats, streak_info, pattern_dict)
    _leader_state["last_signals"] = new_signals

    # signal history는 실제 round advance 때만 1회 적재
    if advanced:
        for name in _LEADER_ROADS:
            _append_signal_history(stats[name], new_signals.get(name))

    # 3) hit rates / totals / confidences
    road_hit_rates, road_prediction_totals = _compute_hit_rates(stats)
    road_confidences = _compute_road_confidences(stats, road_hit_rates)

    # 4) 중국점 windows
    china_signals = {r: new_signals.get(r) for r in _CHINA_ROADS}
    china_windows: Dict[str, List[int]] = {}
    for r in _CHINA_ROADS:
        china_windows[r] = list(_validate_window_values(stats[r]["window"], name=f"stats[{r}].window"))

    # 5) trust state 기반 리더 선정
    big_tuple = _select_big_leader_with_tier(
        stats,
        road_hit_rates,
        road_confidences,
        road_prediction_totals,
        new_signals,
        pb_len,
    )
    china_tuple = _select_china_leader_with_tier(
        stats,
        road_confidences,
        road_prediction_totals,
        new_signals,
        pb_len,
    )

    big_leader_road, big_leader_signal, big_leader_conf, big_tier, _big_note = big_tuple
    china_leader_roads, china_leader_signal, china_leader_conf, china_tier, _china_note = china_tuple

    # 6) overall leader + stability
    best_overall = _choose_overall_leader(big_tuple, china_tuple)
    leader_road, leader_signal, leader_conf, leader_source, leader_trust_state, confidence_note = _apply_overall_stability(
        best_overall,
        stats,
        road_confidences,
        road_prediction_totals,
        new_signals,
        pb_len,
    )

    can_override_side = bool(leader_trust_state == "STRONG")

    # 7) readiness
    leader_ready, not_ready_reason = _check_ready(pb_seq_v, stats)
    if not leader_ready:
        logger.warning("[ROAD-LEADER] NOT READY (non-fatal): %s", not_ready_reason)

    return {
        "leader_state": {
            "ready": bool(leader_ready),
            "reason": not_ready_reason,

            "leader_road": leader_road,
            "leader_signal": leader_signal if leader_signal in _VALID_SIDE else None,
            "leader_confidence": float(leader_conf or 0.0),
            "leader_source": leader_source,

            "big_leader_road": big_leader_road,
            "big_leader_signal": big_leader_signal if big_leader_signal in _VALID_SIDE else None,
            "big_leader_confidence": float(big_leader_conf or 0.0),

            "china_leader_roads": china_leader_roads,
            "china_leader_signal": china_leader_signal if china_leader_signal in _VALID_SIDE else None,
            "china_leader_confidence": float(china_leader_conf or 0.0),

            "china_signals": china_signals,
            "china_windows": china_windows,

            "leader_trust_state": leader_trust_state,
            "confidence_note": confidence_note,
            "can_override_side": can_override_side,

            "leader_ready": bool(leader_ready),
            "leader_not_ready_reason": not_ready_reason,

            "road_hit_rates": road_hit_rates,
            "road_confidences": road_confidences,
            "road_prediction_totals": road_prediction_totals,
        },

        "road_hit_rates": road_hit_rates,
        "road_confidences": road_confidences,
        "road_prediction_totals": road_prediction_totals,
    }