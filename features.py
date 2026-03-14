# -*- coding: utf-8 -*-
# features.py
"""
features.py
====================================================
Feature Builder for Baccarat Predictor AI Engine v12.1
(RULE-ONLY · STRICT · NO-FALLBACK)

역할
- PB / 스트릭 / 패턴 / 흐름 / 고급 Feature 통합
- 슈 레짐 분류 + 미래 구간 레짐 예측
- Road Leader Engine 연동(leader_state / leader_* / road_hit_rates 등)
- recommend.py(rule-only deterministic engine)가 요구하는 feature contract 생성

중요 정책
----------------------------------------------------
- STRICT · NO-FALLBACK · FAIL-FAST
- 누락/비정상/계산 실패는 즉시 예외
- 준비 미달은 엔진 중단이 아니라 상태값(analysis_ready=False)로만 기록
- GPT / LLM 의존 없음
- features["leader_state"] 는 반드시 존재해야 한다
----------------------------------------------------

변경 요약 (2026-03-14)
----------------------------------------------------
1) RULE-ONLY 구조 반영
   - ENGINE_VERSION / 문서 설명 정리
   - GPT/ML 설명 제거
2) leader_state 계약 고정
   - prev_round_winner 없음 / road_leader not ready 상황에서도
     features["leader_state"] 를 항상 생성
   - predictor_adapter v12 / recommend v12가 요구하는
     leader_confidence / leader_trust_state / leader_signal 보장
3) app/excel 호환 유지
   - top-level alias 유지:
     leader_road / leader_signal / leader_confidence / road_hit_rates / road_prediction_totals
4) recommend 호환 alias 추가
   - chaos = flow_chaos_risk
   - stability = flow_stability
5) leader_state 정규화 치명 버그 수정
   - _normalize_leader_bundle() 내부 _require_dict 호출 인자 순서 오류 수정
6) runtime road cache 일원화
   - 부분 계산 금지
   - road.big_road_matrix 가 비어 있으면 road.recompute_all_roads()로 전체 파생 캐시 재계산
7) 범위 검증 강화
   - tie_ratio / entropy / pattern_noise_ratio 를 0~1로 강제
   - flow_direction 은 rule-only 계약상 "neutral"만 허용
----------------------------------------------------

기존 변경 요약 (2026-01-05)
----------------------------------------------------
1) 학습용 핵심 컬럼을 build_feature_payload_v3() 최종 반환에 반드시 포함
   - chaos_index
   - pattern_stability
   - flow_stability
2) STRICT 유지
----------------------------------------------------

기존 변경 요약 (2025-12-30)
----------------------------------------------------
1) entry_momentum feature 정식 복구
2) road PB 통계는 road.compute_pb_stats()만 사용
----------------------------------------------------

기존 변경 요약 (2025-12-29)
----------------------------------------------------
1) pb_ratio 타입 충돌 해결
2) streak_info 계약 충돌 해결
----------------------------------------------------
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import flow
import pattern
import road
import road_leader

from features_bigroad import (
    compute_beauty_score,
    compute_future_scenarios,
    compute_regime_forecast,
    compute_temporal_features,
)
from features_china import compute_advanced_features

logger = logging.getLogger(__name__)

ENGINE_VERSION = "12.1-rule-only-strict"
FEATURE_SCHEMA_VERSION = 4

__all__ = [
    "ENGINE_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "build_feature_payload_v3",
]


# -----------------------------
# Local readiness (non-fatal)
# -----------------------------
class AnalysisNotReadyError(RuntimeError):
    """준비 미달(초반 라운드/필수 입력 부족). 엔진 중단 금지: 상태값으로만 기록."""


def _assert_analysis_ready_local(features: Dict[str, Any]) -> None:
    """
    준비도는 상태 플래그로만 기록한다.
    - 지나치게 엄격하면 초반 500을 유발하므로 최소 조건만 본다.
    """
    rounds_total = features.get("rounds_total")
    if not isinstance(rounds_total, int) or rounds_total <= 0:
        raise AnalysisNotReadyError("rounds_total not ready")

    if "flow_chaos_risk" not in features or "flow_strength" not in features or "flow_stability" not in features:
        raise AnalysisNotReadyError("flow features not ready")

    if "pattern_score" not in features:
        raise AnalysisNotReadyError("pattern_score not ready")

    for k in ("big_eye_recent", "small_road_recent", "cockroach_recent"):
        v = features.get(k)
        if not isinstance(v, list):
            raise AnalysisNotReadyError(f"{k} not ready")


# -----------------------------
# Strict helpers
# -----------------------------
def _require_dict(name: str, obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError(f"{name} must be dict, got {type(obj).__name__}")
    return obj


def _require_key(d: Dict[str, Any], key: str, *, name: str) -> Any:
    if key not in d:
        raise KeyError(f"{name} missing required key: {key}")
    return d[key]


def _require_list(name: str, obj: Any) -> list:
    if not isinstance(obj, list):
        raise TypeError(f"{name} must be list, got {type(obj).__name__}")
    return obj


def _as_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be float, got bool")
    if isinstance(value, (int, float)):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite")
        return v
    raise TypeError(f"{name} must be float, got {type(value).__name__}")


def _as_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"{name} must be int, got {type(value).__name__}")


def _require_unit_interval(value: Any, *, name: str) -> float:
    v = _as_float(value, name=name)
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be in [0,1], got {v}")
    return v


def _require_minus1_plus1(value: Any, *, name: str) -> float:
    v = _as_float(value, name=name)
    if v < -1.0 or v > 1.0:
        raise ValueError(f"{name} must be in [-1,1], got {v}")
    return v


def _require_nonempty_str(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    s = value.strip()
    if not s:
        raise ValueError(f"{name} must be non-empty str")
    return s


def _require_score_0_100(value: Any, *, name: str) -> float:
    v = _as_float(value, name=name)
    if v < 0.0 or v > 100.0:
        raise ValueError(f"{name} must be in [0,100], got {v}")
    return v


def _require_flow_direction_neutral(value: Any, *, name: str) -> str:
    s = _require_nonempty_str(value, name=name).lower()
    if s != "neutral":
        raise ValueError(f"{name} must be 'neutral', got {value!r}")
    return "neutral"


def _validate_runtime_road_state(pb_seq: List[str]) -> None:
    if not isinstance(pb_seq, list):
        raise TypeError(f"pb_seq must be list, got {type(pb_seq).__name__}")
    if len(pb_seq) == 0:
        raise RuntimeError("pb_seq empty")

    if not hasattr(road, "recompute_all_roads"):
        raise AttributeError("road.recompute_all_roads missing")

    if not getattr(road, "big_road_matrix", None):
        road.recompute_all_roads()

    _require_list("road.big_road_matrix", road.big_road_matrix)
    _require_list("road.big_road_positions", road.big_road_positions)
    _require_list("road.big_eye_seq", road.big_eye_seq)
    _require_list("road.small_road_seq", road.small_road_seq)
    _require_list("road.cockroach_seq", road.cockroach_seq)

    if len(road.big_road_positions) != len(pb_seq):
        raise RuntimeError(
            f"road.big_road_positions length mismatch: {len(road.big_road_positions)} != len(pb_seq)({len(pb_seq)})"
        )

    if hasattr(road, "validate_roadmap_integrity"):
        road_ok, road_error = road.validate_roadmap_integrity()
        if not isinstance(road_ok, bool):
            raise TypeError("road.validate_roadmap_integrity() must return (bool, str)")
        if not road_ok:
            raise RuntimeError(f"road.validate_roadmap_integrity failed: {road_error}")


# -----------------------------
# Entry momentum
# -----------------------------
def _compute_entry_momentum(
    *,
    flow_strength: Any,
    flow_stability: Any,
    pattern_energy: Any,
    momentum: Any,
    streak_info: Dict[str, Any],
) -> float:
    """
    entry_momentum: 현재 흐름이 진입 방향으로 가속되고 있는 정도(0~1).
    """
    fs = _require_unit_interval(flow_strength, name="entry_momentum.flow_strength")
    st = _require_unit_interval(flow_stability, name="entry_momentum.flow_stability")
    pe = _require_minus1_plus1(pattern_energy, name="entry_momentum.pattern_energy")
    mom = _require_unit_interval(momentum, name="entry_momentum.momentum")

    si = _require_dict("entry_momentum.streak_info", streak_info)
    cs = _require_dict("entry_momentum.current_streak", _require_key(si, "current_streak", name="streak_info"))
    if "len" not in cs:
        raise KeyError("streak_info.current_streak missing required key: len")
    cur_len = _as_int(cs["len"], name="entry_momentum.current_streak.len")
    if cur_len <= 0:
        raise ValueError(f"entry_momentum.current_streak.len invalid: {cur_len}")

    streaks = _require_list("entry_momentum.streaks", _require_key(si, "streaks", name="streak_info"))
    prev_len = 1
    if len(streaks) >= 2 and isinstance(streaks[-2], dict) and "len" in streaks[-2]:
        prev_len = _as_int(streaks[-2]["len"], name="entry_momentum.prev_streak.len")
        if prev_len <= 0:
            raise ValueError(f"entry_momentum.prev_streak.len invalid: {prev_len}")

    denom = cur_len + prev_len
    if denom <= 0:
        raise ValueError("entry_momentum.streak_change invalid denom")

    streak_change_signed = float(cur_len - prev_len) / float(denom)
    if not math.isfinite(streak_change_signed) or streak_change_signed < -1.0 or streak_change_signed > 1.0:
        raise ValueError("entry_momentum.streak_change must be finite in [-1,1]")
    streak_accel = (streak_change_signed + 1.0) / 2.0

    energy_gate = (pe + 1.0) / 2.0

    value = (fs ** 0.85) * (st ** 0.65) * (mom ** 0.90) * (streak_accel ** 0.80) * (energy_gate ** 1.20)

    if not math.isfinite(value):
        raise ValueError("entry_momentum must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"entry_momentum out of range: {value}")
    return float(value)


# -----------------------------
# PB ratio helpers
# -----------------------------
def _coerce_or_compute_pb_ratio(raw_value: Any, *, p_count: int, b_count: int) -> float:
    if isinstance(raw_value, bool):
        raise TypeError("pb_stats.pb_ratio must be number, got bool")

    if isinstance(raw_value, (int, float)):
        v = float(raw_value)
        if not math.isfinite(v):
            raise ValueError("pb_stats.pb_ratio must be finite")
        return v

    denom = p_count + b_count
    if denom <= 0:
        raise ValueError("pb_stats.pb_ratio invalid: (p_count + b_count) <= 0")
    v = float(p_count) / float(denom)
    if not math.isfinite(v):
        raise ValueError("pb_stats.pb_ratio must be finite")
    return v


def _build_pb_ratio_dict(raw_value: Any, *, ratio: float, p_count: int, b_count: int) -> Dict[str, Any]:
    if not math.isfinite(float(ratio)):
        raise ValueError("pb_ratio(dict).ratio must be finite")

    base: Dict[str, Any] = {}
    if isinstance(raw_value, dict):
        base.update(raw_value)

    base["ratio"] = float(ratio)
    base["value"] = float(ratio)
    base["p_ratio"] = float(ratio)
    base["p"] = int(p_count)
    base["b"] = int(b_count)
    base["denom"] = int(p_count + b_count)
    return base


# -----------------------------
# streak_info adapter for road_leader
# -----------------------------
def _normalize_streak_info_for_road_leader(streak_info: Dict[str, Any]) -> Dict[str, Any]:
    src = _require_dict("streak_info", streak_info)

    cs = _require_dict("streak_info.current_streak", _require_key(src, "current_streak", name="streak_info"))
    if "side" in cs and "length" in cs:
        side = cs.get("side")
        length = cs.get("length")
    elif "who" in cs and "len" in cs:
        side = cs.get("who")
        length = cs.get("len")
    else:
        raise KeyError("streak_info.current_streak missing keys: (who,len) or (side,length)")

    if side is not None and side not in ("P", "B"):
        raise ValueError(f"streak_info.current_streak.side invalid: {side!r}")
    length_i = _as_int(length, name="streak_info.current_streak.length")
    if length_i < 0:
        raise ValueError(f"streak_info.current_streak.length invalid: {length_i}")

    cs_norm = dict(cs)
    cs_norm["side"] = side
    cs_norm["length"] = length_i

    streaks_raw = _require_list("streak_info.streaks", _require_key(src, "streaks", name="streak_info"))
    streaks_norm = []
    for idx, s in enumerate(streaks_raw):
        if not isinstance(s, dict):
            raise TypeError(f"streak_info.streaks[{idx}] must be dict, got {type(s).__name__}")
        if "side" in s and "length" in s:
            s_side = s.get("side")
            s_len = s.get("length")
        elif "who" in s and "len" in s:
            s_side = s.get("who")
            s_len = s.get("len")
        else:
            raise KeyError(f"streak_info.streaks[{idx}] missing keys: (who,len) or (side,length)")

        if s_side is not None and s_side not in ("P", "B"):
            raise ValueError(f"streak_info.streaks[{idx}].side invalid: {s_side!r}")
        s_len_i = _as_int(s_len, name=f"streak_info.streaks[{idx}].length")
        if s_len_i < 0:
            raise ValueError(f"streak_info.streaks[{idx}].length invalid: {s_len_i}")

        s_norm = dict(s)
        s_norm["side"] = s_side
        s_norm["length"] = s_len_i
        streaks_norm.append(s_norm)

    out = dict(src)
    out["current_streak"] = cs_norm
    out["streaks"] = streaks_norm
    return out


# -----------------------------
# Leader helpers
# -----------------------------
def _build_empty_leader_bundle(reason: str) -> Dict[str, Any]:
    leader_state: Dict[str, Any] = {
        "ready": False,
        "reason": reason,
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
        "china_signals": {},
        "china_windows": {},
        "leader_trust_state": "NONE",
        "confidence_note": reason,
        "can_override_side": False,
        "leader_ready": False,
        "leader_not_ready_reason": reason,
        "road_hit_rates": {},
        "road_confidences": {},
        "road_prediction_totals": {},
    }

    return {
        "leader_state": leader_state,
        "leader_ready": False,
        "leader_not_ready_reason": reason,
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
        "leader_trust_state": "NONE",
        "confidence_note": reason,
        "can_override_side": False,
        "road_hit_rates": {},
        "road_confidences": {},
        "road_prediction_totals": {},
    }


def _normalize_leader_bundle(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    road_leader.update_and_get_leader_features() 출력 정규화.
    최종적으로 features["leader_state"] 와 top-level alias를 모두 보장한다.
    """
    obj = _require_dict("leader_info", raw)

    leader_state = _require_dict(
        "leader_info.leader_state",
        _require_key(obj, "leader_state", name="leader_info"),
    )

    required_keys = [
        "leader_confidence",
        "leader_trust_state",
        "leader_signal",
        "leader_road",
        "leader_source",
        "leader_ready",
        "leader_not_ready_reason",
        "road_hit_rates",
        "road_prediction_totals",
    ]
    for k in required_keys:
        if k not in leader_state:
            raise KeyError(f"leader_info.leader_state missing required key: {k}")

    leader_confidence = _require_unit_interval(
        leader_state["leader_confidence"],
        name="leader_info.leader_state.leader_confidence",
    )
    leader_trust_state = leader_state["leader_trust_state"]
    if not isinstance(leader_trust_state, str):
        raise TypeError("leader_info.leader_state.leader_trust_state must be str")
    leader_trust_state = leader_trust_state.strip().upper()
    if leader_trust_state not in ("NONE", "WEAK", "MID", "STRONG"):
        raise ValueError(f"leader_info.leader_state.leader_trust_state invalid: {leader_trust_state!r}")

    leader_signal = leader_state["leader_signal"]
    if leader_signal is not None and leader_signal not in ("P", "B"):
        raise ValueError(f"leader_info.leader_state.leader_signal invalid: {leader_signal!r}")

    road_hit_rates = leader_state["road_hit_rates"]
    road_prediction_totals = leader_state["road_prediction_totals"]

    if not isinstance(road_hit_rates, dict):
        raise TypeError("leader_info.leader_state.road_hit_rates must be dict")
    if not isinstance(road_prediction_totals, dict):
        raise TypeError("leader_info.leader_state.road_prediction_totals must be dict")

    road_confidences = leader_state.get("road_confidences", {})
    if not isinstance(road_confidences, dict):
        raise TypeError("leader_info.leader_state.road_confidences must be dict")

    return {
        "leader_state": leader_state,
        "leader_ready": bool(leader_state["leader_ready"]),
        "leader_not_ready_reason": leader_state["leader_not_ready_reason"],
        "leader_road": leader_state["leader_road"],
        "leader_signal": leader_signal,
        "leader_confidence": float(leader_confidence),
        "leader_source": leader_state["leader_source"],
        "big_leader_road": leader_state.get("big_leader_road"),
        "big_leader_signal": leader_state.get("big_leader_signal"),
        "big_leader_confidence": float(leader_state.get("big_leader_confidence") or 0.0),
        "china_leader_roads": leader_state.get("china_leader_roads", []),
        "china_leader_signal": leader_state.get("china_leader_signal"),
        "china_leader_confidence": float(leader_state.get("china_leader_confidence") or 0.0),
        "leader_trust_state": leader_trust_state,
        "confidence_note": leader_state.get("confidence_note"),
        "can_override_side": bool(leader_state.get("can_override_side", False)),
        "road_hit_rates": road_hit_rates,
        "road_confidences": road_confidences,
        "road_prediction_totals": road_prediction_totals,
    }


# -----------------------------
# Main
# -----------------------------
def build_feature_payload_v3(
    prev_round_winner: Optional[str] = None,
) -> Dict[str, Any]:
    """
    RULE-ONLY deterministic engine용 Feature JSON 생성.

    원칙:
    - 폴백 금지(무결성): 누락/비정상/계산 실패는 즉시 예외
    - 준비 미달(초반 라운드/리더 누적 부족 등): 엔진 중단 금지 → 상태값으로만 표시
    """
    try:
        if prev_round_winner is not None and prev_round_winner not in ("P", "B", "T"):
            raise ValueError(f"prev_round_winner invalid: {prev_round_winner}")

        pb_stats_raw = _require_dict("pb_stats", road.compute_pb_stats())
        pb_seq = road.get_pb_sequence()
        if not isinstance(pb_seq, list):
            raise TypeError(f"pb_seq must be list, got {type(pb_seq).__name__}")
        if len(pb_seq) == 0:
            raise RuntimeError("pb_seq empty")

        _validate_runtime_road_state(pb_seq)

        total_rounds = _as_int(_require_key(pb_stats_raw, "total_rounds", name="pb_stats"), name="pb_stats.total_rounds")
        p_count = _as_int(_require_key(pb_stats_raw, "p_count", name="pb_stats"), name="pb_stats.p_count")
        b_count = _as_int(_require_key(pb_stats_raw, "b_count", name="pb_stats"), name="pb_stats.b_count")
        _require_key(pb_stats_raw, "t_count", name="pb_stats")
        raw_pb_ratio = _require_key(pb_stats_raw, "pb_ratio", name="pb_stats")
        tie_ratio = _require_unit_interval(
            _require_key(pb_stats_raw, "tie_ratio", name="pb_stats"),
            name="pb_stats.tie_ratio",
        )
        entropy = _require_unit_interval(
            _require_key(pb_stats_raw, "entropy", name="pb_stats"),
            name="pb_stats.entropy",
        )

        if total_rounds <= 0:
            raise RuntimeError(f"total_rounds invalid: {total_rounds}")

        pb_ratio_float = _coerce_or_compute_pb_ratio(raw_pb_ratio, p_count=p_count, b_count=b_count)
        pb_ratio_dict = _build_pb_ratio_dict(raw_value=raw_pb_ratio, ratio=pb_ratio_float, p_count=p_count, b_count=b_count)

        pb_stats_num = dict(pb_stats_raw)
        pb_stats_num["pb_ratio"] = float(pb_ratio_float)

        pb_stats_china = dict(pb_stats_raw)
        pb_stats_china["pb_ratio"] = dict(pb_ratio_dict)

        streak_info = _require_dict("streak_info", road.compute_streaks(pb_seq))
        _require_key(streak_info, "streaks", name="streak_info")
        _require_key(streak_info, "current_streak", name="streak_info")
        _require_key(streak_info, "max_streak_p", name="streak_info")
        _require_key(streak_info, "max_streak_b", name="streak_info")
        _require_key(streak_info, "avg_streak_len", name="streak_info")
        _require_key(streak_info, "last_20", name="streak_info")
        _require_key(streak_info, "trend_strength", name="streak_info")
        _require_key(streak_info, "momentum", name="streak_info")

        pattern_dict = _require_dict("pattern_dict", pattern.compute_pattern_features(pb_seq))
        pattern_score = _require_score_0_100(
            _require_key(pattern_dict, "pattern_score", name="pattern_dict"),
            name="pattern_dict.pattern_score",
        )
        pattern_type = _require_nonempty_str(
            _require_key(pattern_dict, "pattern_type", name="pattern_dict"),
            name="pattern_dict.pattern_type",
        )
        pattern_energy = _require_minus1_plus1(
            _require_key(pattern_dict, "pattern_energy", name="pattern_dict"),
            name="pattern_dict.pattern_energy",
        )
        pattern_symmetry = _require_unit_interval(
            _require_key(pattern_dict, "pattern_symmetry", name="pattern_dict"),
            name="pattern_dict.pattern_symmetry",
        )
        pattern_noise_ratio = _require_unit_interval(
            _require_key(pattern_dict, "pattern_noise_ratio", name="pattern_dict"),
            name="pattern_dict.pattern_noise_ratio",
        )
        pattern_reversal_signal = _as_float(
            _require_key(pattern_dict, "pattern_reversal_signal", name="pattern_dict"),
            name="pattern_dict.pattern_reversal_signal",
        )
        pattern_stability = _require_unit_interval(
            _require_key(pattern_dict, "pattern_stability", name="pattern_dict"),
            name="pattern_dict.pattern_stability",
        )

        temporal = _require_dict("temporal", compute_temporal_features(pb_stats_num, streak_info))
        _require_key(temporal, "pattern_drift", name="temporal")
        _require_key(temporal, "run_speed", name="temporal")
        _require_key(temporal, "tie_volatility", name="temporal")

        flow_input = dict(streak_info)
        flow_input["pattern_type"] = pattern_type
        flow_input["pattern_energy"] = pattern_energy
        flow_input["pb_len"] = len(pb_seq)

        flow_dict = _require_dict(
            "flow_dict",
            flow.compute_flow_features(road.big_eye_seq, road.small_road_seq, road.cockroach_seq, flow_input),
        )

        flow_strength = _require_unit_interval(
            _require_key(flow_dict, "flow_strength", name="flow_dict"),
            name="flow_dict.flow_strength",
        )
        flow_stability = _require_unit_interval(
            _require_key(flow_dict, "flow_stability", name="flow_dict"),
            name="flow_dict.flow_stability",
        )
        flow_chaos_risk = _require_unit_interval(
            _require_key(flow_dict, "flow_chaos_risk", name="flow_dict"),
            name="flow_dict.flow_chaos_risk",
        )
        flow_reversal_risk = _require_unit_interval(
            _require_key(flow_dict, "flow_reversal_risk", name="flow_dict"),
            name="flow_dict.flow_reversal_risk",
        )
        flow_direction = _require_flow_direction_neutral(
            _require_key(flow_dict, "flow_direction", name="flow_dict"),
            name="flow_dict.flow_direction",
        )

        tie_volatility = temporal.get("tie_volatility")
        if not isinstance(tie_volatility, (int, float)):
            raise TypeError("tie_volatility invalid")

        if tie_volatility >= 0.7:
            turbulence_rounds = 3
        elif tie_volatility >= 0.4:
            turbulence_rounds = 1
        else:
            turbulence_rounds = 0

        tie_state = {
            "turbulence_rounds": int(turbulence_rounds),
            "tie_volatility": float(tie_volatility),
        }

        chaos_index = float(flow_chaos_risk)

        entry_momentum = _compute_entry_momentum(
            flow_strength=flow_strength,
            flow_stability=flow_stability,
            pattern_energy=pattern_energy,
            momentum=streak_info["momentum"],
            streak_info=streak_info,
        )

        future_scenarios = _require_dict("future_scenarios", compute_future_scenarios(pb_seq))
        for future_key in ("P", "B"):
            _require_dict(f"future_scenarios.{future_key}", _require_key(future_scenarios, future_key, name="future_scenarios"))

        adv = _require_dict(
            "adv",
            compute_advanced_features(pb_seq, pb_stats_china, streak_info, pattern_dict, temporal, flow_dict),
        )

        beauty_score = compute_beauty_score(pb_seq, pattern_dict, flow_dict, adv)
        if not isinstance(beauty_score, (int, float)):
            raise TypeError(f"beauty_score must be number, got {type(beauty_score).__name__}")

        regime_forecast = _require_dict(
            "regime_forecast",
            compute_regime_forecast(pb_seq, pattern_dict, temporal, flow_dict, adv, future_scenarios),
        )

        # -----------------------------
        # Road Leader
        # -----------------------------
        if prev_round_winner is None:
            leader_bundle = _build_empty_leader_bundle("prev_round_winner is None: leader update skipped")
        else:
            try:
                leader_streak_info = _normalize_streak_info_for_road_leader(streak_info)
                leader_raw = _require_dict(
                    "leader_info",
                    road_leader.update_and_get_leader_features(
                        prev_round_winner,
                        pb_seq,
                        pb_stats_num,
                        leader_streak_info,
                        pattern_dict,
                        adv,
                    ),
                )
                leader_bundle = _normalize_leader_bundle(leader_raw)
            except road_leader.RoadLeaderNotReadyError as e:
                logger.warning("[FEATURES] road_leader not ready (non-fatal): %s", e)
                leader_bundle = _build_empty_leader_bundle(str(e))

        streaks = _require_list("streak_info.streaks", streak_info["streaks"])
        trimmed_streaks = streaks[-12:]

        base_features: Dict[str, Any] = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION,

            "rounds_total": int(total_rounds),
            "p_count": int(p_count),
            "b_count": int(b_count),
            "t_count": pb_stats_raw["t_count"],

            "pb_ratio": float(pb_ratio_float),
            "tie_ratio": float(tie_ratio),
            "entropy": float(entropy),

            "streaks": trimmed_streaks,
            "current_streak": streak_info["current_streak"],
            "max_streak_p": streak_info["max_streak_p"],
            "max_streak_b": streak_info["max_streak_b"],
            "avg_streak_len": streak_info["avg_streak_len"],
            "last_20": streak_info["last_20"],
            "trend_strength": streak_info["trend_strength"],
            "momentum": streak_info["momentum"],
            "entry_momentum": float(entry_momentum),

            "pattern_score": float(pattern_score),
            "pattern_type": pattern_type,
            "pattern_energy": float(pattern_energy),
            "pattern_symmetry": float(pattern_symmetry),
            "pattern_noise_ratio": float(pattern_noise_ratio),
            "pattern_reversal_signal": float(pattern_reversal_signal),
            "pattern_stability": float(pattern_stability),

            "pattern_drift": temporal["pattern_drift"],
            "run_speed": temporal["run_speed"],
            "tie_volatility": temporal["tie_volatility"],

            "flow_strength": float(flow_strength),
            "flow_stability": float(flow_stability),
            "flow_chaos_risk": float(flow_chaos_risk),
            "chaos_index": float(chaos_index),
            "flow_reversal_risk": float(flow_reversal_risk),
            "flow_direction": flow_direction,

            # recommend/app 호환 alias
            "chaos": float(flow_chaos_risk),
            "stability": float(flow_stability),

            "big_eye_recent": road.big_eye_seq[-12:],
            "small_road_recent": road.small_road_seq[-12:],
            "cockroach_recent": road.cockroach_seq[-12:],

            "future_scenarios": future_scenarios,
            "beauty_score": float(beauty_score),
            "tie_state": tie_state,
        }

        base_features.update(adv)
        base_features.update(regime_forecast)
        base_features.update(leader_bundle)

        try:
            _assert_analysis_ready_local(base_features)
            base_features["analysis_ready"] = True
            base_features["analysis_not_ready_reason"] = None
        except AnalysisNotReadyError as e:
            logger.warning("[FEATURES] ANALYSIS NOT READY (non-fatal): %s", e)
            base_features["analysis_ready"] = False
            base_features["analysis_not_ready_reason"] = str(e)

        return base_features

    except Exception as e:
        logger.exception("[FEATURES] build_feature_payload_v3 failed: %s", e)
        raise