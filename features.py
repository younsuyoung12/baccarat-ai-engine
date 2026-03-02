# -*- coding: utf-8 -*-
# features.py
"""
features.py
====================================================
Feature Builder for Baccarat Predictor AI Engine v9.3 (STRICT / NO-FALLBACK)

역할
- PB / 스트릭 / 패턴 / 흐름 / 고급 Feature 통합
- 슈 레짐 분류 + 미래 구간 레짐 예측
- Road Leader Engine 연동(leader_* / road_hit_rates 등)

변경 요약 (2026-01-05)
----------------------------------------------------
1) 학습용 핵심 컬럼을 build_feature_payload_v3() 최종 반환에 "반드시" 포함:
   - chaos_index: flow_dict["flow_chaos_risk"] 를 그대로 사용(이름 통일)
   - pattern_stability: pattern.compute_pattern_features(pb_seq) 결과에서 제공(없으면 즉시 예외)
   - flow_stability: 기존 값 그대로 유지(변형 금지)
2) STRICT 유지:
   - chaos_index / pattern_stability 는 0~1 finite float 강제
   - 누락/비정상 시 폴백 없이 즉시 예외

변경 요약 (2025-12-30)
----------------------------------------------------
1) entry_momentum feature 정식 복구
   - engine_state 비의존(무상태) 계산
   - flow_strength/flow_stability/pattern_energy/momentum/최근 streak 변화 조합
   - 0~1 finite float 보장, 계산 불가 시 즉시 ValueError (폴백 금지)
2) road PB 통계는 실제 API(road.compute_pb_stats)만 사용
   - 존재하지 않는 road.get_pb_stats 호출 제거(계약 불일치 방지)
----------------------------------------------------

변경 요약 (2025-12-29)
----------------------------------------------------
1) pb_ratio 타입 충돌(road_leader=float vs features_china=dict) 해결
   - pb_stats_num: pb_ratio=float  (road_leader / temporal 용)
   - pb_stats_china: pb_ratio=dict (features_china 용)
   - 다른 파일 수정 없이 features.py 내부에서만 분리 전달
2) 폴백 금지 유지
   - pb_ratio 계산 불가(표본 0)면 ValueError
   - tie_ratio/entropy 비정상(NaN/Inf)면 ValueError
3) streak_info 계약 충돌(road=who/len vs road_leader=side/length) 해결
   - road_leader 호출 직전에 leader_streak_info로 정규화(side/length 키 추가)
   - features_china는 기존 who/len 그대로 사용
----------------------------------------------------

변경 요약 (2025-12-23)
----------------------------------------------------
1) features_entry.py 삭제 반영
2) 분석 준비도(non-fatal) 로직을 features.py 내부로 내장
3) 사용하지 않는 re-export 정리
----------------------------------------------------
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

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

ENGINE_VERSION = "9.3-bigroad-chinese-official-ai-hybridC-beauty-extended-strict"
FEATURE_SCHEMA_VERSION = 3

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
    features_entry.assert_analysis_ready(...) 대체.
    - 여기서는 엔진을 죽이지 않는다(외부에서 catch하여 상태 플래그로만 기록).
    - 지나치게 엄격하면 초반 500을 유발하므로, "정말 최소 조건"만 체크한다.
    """
    rounds_total = features.get("rounds_total")
    if not isinstance(rounds_total, int) or rounds_total <= 0:
        raise AnalysisNotReadyError("rounds_total not ready")

    # flow 핵심값
    if "flow_chaos_risk" not in features or "flow_strength" not in features or "flow_stability" not in features:
        raise AnalysisNotReadyError("flow features not ready")

    # pattern 핵심값
    if "pattern_score" not in features:
        raise AnalysisNotReadyError("pattern_score not ready")

    # 중국점 시퀀스(최근값)만 있어도 UI는 표시 가능
    for k in ("big_eye_recent", "small_road_recent", "cockroach_recent"):
        v = features.get(k)
        if not isinstance(v, list):
            raise AnalysisNotReadyError(f"{k} not ready")

    # 리더는 non-fatal이므로 필수 아님(없어도 ready 판단에 포함하지 않음)


# -----------------------------
# Strict helpers (폴백 금지)
# -----------------------------
def _require_dict(name: str, obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError(f"{name} must be dict, got {type(obj).__name__}")
    return obj


def _require_key(d: Dict[str, Any], key: str, *, name: str) -> Any:
    if key not in d:
        raise KeyError(f"{name} missing required key: {key}")
    return d[key]


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
    """
    0~1 범위의 finite float 강제(폴백 금지).
    """
    v = _as_float(value, name=name)
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be in [0,1], got {v}")
    return v


def _require_minus1_plus1(value: Any, *, name: str) -> float:
    """
    -1~+1 범위의 finite float 강제(폴백 금지).
    """
    v = _as_float(value, name=name)
    if v < -1.0 or v > 1.0:
        raise ValueError(f"{name} must be in [-1,1], got {v}")
    return v


def _compute_entry_momentum(
    *,
    flow_strength: Any,
    flow_stability: Any,
    pattern_energy: Any,
    momentum: Any,
    streak_info: Dict[str, Any],
) -> float:
    """
    entry_momentum: 현재 흐름이 진입 방향으로 "가속"되고 있는 정도(0~1).

    - 무상태(stateless): engine_state 의존 금지
    - 폴백 금지: 타입/범위/계산 불가 시 즉시 예외
    - 단순 평균 금지: 가중/게이트(곱) 결합으로 가속 여부 반영
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
    # 최근 이전 스트릭(반대편 포함) 길이: 없으면 "최소 스트릭=1" 기준으로 변화량 산출
    prev_len = 1
    if len(streaks) >= 2 and isinstance(streaks[-2], dict) and "len" in streaks[-2]:
        prev_len = _as_int(streaks[-2]["len"], name="entry_momentum.prev_streak.len")
        if prev_len <= 0:
            raise ValueError(f"entry_momentum.prev_streak.len invalid: {prev_len}")

    # 최근 streak 길이 변화(-1~+1) → 0~1 정규화
    denom = cur_len + prev_len
    if denom <= 0:
        raise ValueError("entry_momentum.streak_change invalid denom")
    streak_change_signed = float(cur_len - prev_len) / float(denom)  # [-1, +1]
    if not math.isfinite(streak_change_signed) or streak_change_signed < -1.0 or streak_change_signed > 1.0:
        raise ValueError("entry_momentum.streak_change must be finite in [-1,1]")
    streak_accel = (streak_change_signed + 1.0) / 2.0  # [0,1]

    # 패턴 에너지(-1~+1) → 0~1 (가속/감속 반영)
    energy_gate = (pe + 1.0) / 2.0  # [0,1]

    # ✅ 가중/게이트 결합(단순 평균 금지)
    value = (fs ** 0.85) * (st ** 0.65) * (mom ** 0.90) * (streak_accel ** 0.80) * (energy_gate ** 1.20)

    if not math.isfinite(value):
        raise ValueError("entry_momentum must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"entry_momentum out of range: {value}")
    return float(value)


def _require_list(name: str, obj: Any) -> list:
    if not isinstance(obj, list):
        raise TypeError(f"{name} must be list, got {type(obj).__name__}")
    return obj


def _coerce_or_compute_pb_ratio(raw_value: Any, *, p_count: int, b_count: int) -> float:
    """
    pb_ratio(float) 계산용.
    - raw_value가 숫자면 사용
    - 아니면 p_count/(p_count+b_count)로 계산
    - 표본 0이면 ValueError (폴백 금지)
    """
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
    """
    features_china용 pb_stats.pb_ratio(dict) 생성.
    - dict 요구만 충족하면 되는 상황을 대비해 최소/호환 키를 함께 제공
    """
    if not math.isfinite(float(ratio)):
        raise ValueError("pb_ratio(dict).ratio must be finite")

    base: Dict[str, Any] = {}
    if isinstance(raw_value, dict):
        base.update(raw_value)

    base["ratio"] = float(ratio)
    base["value"] = float(ratio)     # 호환용
    base["p_ratio"] = float(ratio)   # 호환용
    base["p"] = int(p_count)
    base["b"] = int(b_count)
    base["denom"] = int(p_count + b_count)
    return base


def _normalize_streak_info_for_road_leader(streak_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    road.compute_streaks() 계약: current_streak={who,len}, streaks=[{who,len}, ...]
    road_leader 계약:        current_streak={side,length}, streaks=[{side,length}, ...]
    => features.py에서만 어댑터로 변환해 전달한다(원본 streak_info는 변경 금지).
    """
    src = _require_dict("streak_info", streak_info)

    # current_streak
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

    # streaks list
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
# Main
# -----------------------------
def build_feature_payload_v3(
    prev_round_winner: Optional[str] = None,
) -> Dict[str, Any]:
    """
    GPT/ML/규칙 엔진에 넘기는 Feature JSON (STRICT).

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

        # pb_stats 필수 키(폴백 금지)
        total_rounds = _as_int(_require_key(pb_stats_raw, "total_rounds", name="pb_stats"), name="pb_stats.total_rounds")
        p_count = _as_int(_require_key(pb_stats_raw, "p_count", name="pb_stats"), name="pb_stats.p_count")
        b_count = _as_int(_require_key(pb_stats_raw, "b_count", name="pb_stats"), name="pb_stats.b_count")
        _require_key(pb_stats_raw, "t_count", name="pb_stats")
        raw_pb_ratio = _require_key(pb_stats_raw, "pb_ratio", name="pb_stats")
        tie_ratio = _as_float(_require_key(pb_stats_raw, "tie_ratio", name="pb_stats"), name="pb_stats.tie_ratio")
        entropy = _as_float(_require_key(pb_stats_raw, "entropy", name="pb_stats"), name="pb_stats.entropy")

        if total_rounds <= 0:
            raise RuntimeError(f"total_rounds invalid: {total_rounds}")
        if not math.isfinite(tie_ratio):
            raise ValueError("pb_stats.tie_ratio must be finite")
        if not math.isfinite(entropy):
            raise ValueError("pb_stats.entropy must be finite")

        # ✅ pb_ratio는 (1) float 버전과 (2) dict 버전으로 분리해 사용
        pb_ratio_float = _coerce_or_compute_pb_ratio(raw_pb_ratio, p_count=p_count, b_count=b_count)
        pb_ratio_dict = _build_pb_ratio_dict(raw_pb_ratio, ratio=pb_ratio_float, p_count=p_count, b_count=b_count)

        pb_stats_num = dict(pb_stats_raw)
        pb_stats_num["pb_ratio"] = float(pb_ratio_float)      # road_leader/temporal 용(숫자)

        pb_stats_china = dict(pb_stats_raw)
        pb_stats_china["pb_ratio"] = dict(pb_ratio_dict)      # features_china 용(dict)

        streak_info = _require_dict("streak_info", road.compute_streaks(pb_seq))
        _require_key(streak_info, "streaks", name="streak_info")
        _require_key(streak_info, "current_streak", name="streak_info")
        _require_key(streak_info, "max_streak_p", name="streak_info")
        _require_key(streak_info, "max_streak_b", name="streak_info")
        _require_key(streak_info, "avg_streak_len", name="streak_info")
        _require_key(streak_info, "last_20", name="streak_info")
        _require_key(streak_info, "trend_strength", name="streak_info")
        _require_key(streak_info, "momentum", name="streak_info")

        # Big Road / 중국점 3종이 비어 있으면 한 번 초기 계산(필요 시)
        if not road.big_road_matrix:
            road.big_road_matrix, road.big_road_positions = road.build_big_road_structure(pb_seq)
            road.big_eye_seq, road.small_road_seq, road.cockroach_seq = road.compute_chinese_roads(
                road.big_road_matrix, road.big_road_positions, pb_seq
            )

        # 중국점 시퀀스 타입 검증(폴백 금지)
        _require_list("road.big_eye_seq", road.big_eye_seq)
        _require_list("road.small_road_seq", road.small_road_seq)
        _require_list("road.cockroach_seq", road.cockroach_seq)

        pattern_dict = _require_dict("pattern_dict", pattern.compute_pattern_features(pb_seq))
        _require_key(pattern_dict, "pattern_score", name="pattern_dict")
        _require_key(pattern_dict, "pattern_type", name="pattern_dict")
        _require_key(pattern_dict, "pattern_energy", name="pattern_dict")
        _require_key(pattern_dict, "pattern_symmetry", name="pattern_dict")
        _require_key(pattern_dict, "pattern_noise_ratio", name="pattern_dict")
        _require_key(pattern_dict, "pattern_reversal_signal", name="pattern_dict")

        # ✅ (학습 필수) pattern_stability: pattern.py가 제공해야 한다. 없으면 즉시 예외(폴백 금지).
        _require_key(pattern_dict, "pattern_stability", name="pattern_dict")
        pattern_stability = _require_unit_interval(
            pattern_dict["pattern_stability"],
            name="pattern_dict.pattern_stability",
        )

        # ✅ temporal은 숫자 pb_ratio를 쓰는 쪽(기존 의도 유지)
        temporal = _require_dict("temporal", compute_temporal_features(pb_stats_num, streak_info))
        _require_key(temporal, "pattern_drift", name="temporal")
        _require_key(temporal, "run_speed", name="temporal")
        _require_key(temporal, "tie_volatility", name="temporal")

        flow_dict = _require_dict(
            "flow_dict",
            flow.compute_flow_features(road.big_eye_seq, road.small_road_seq, road.cockroach_seq, streak_info),
        )

        # -----------------------------
        # TIE STATE (항상 존재해야 함)
        # -----------------------------
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
        base_tie_state = tie_state

        _require_key(flow_dict, "flow_strength", name="flow_dict")
        _require_key(flow_dict, "flow_stability", name="flow_dict")
        _require_key(flow_dict, "flow_chaos_risk", name="flow_dict")
        _require_key(flow_dict, "flow_reversal_risk", name="flow_dict")
        _require_key(flow_dict, "flow_direction", name="flow_dict")

        # ✅ (학습 필수) chaos_index: flow_chaos_risk를 그대로 사용하되 이름을 통일
        chaos_index = _require_unit_interval(
            flow_dict["flow_chaos_risk"],
            name="flow_dict.flow_chaos_risk",
        )

        # -----------------------------
        # entry_momentum (ML/UI reference only)
        # -----------------------------
        entry_momentum = _compute_entry_momentum(
            flow_strength=flow_dict["flow_strength"],
            flow_stability=flow_dict["flow_stability"],
            pattern_energy=pattern_dict["pattern_energy"],
            momentum=streak_info["momentum"],
            streak_info=streak_info,
        )

        future_scenarios = _require_dict("future_scenarios", compute_future_scenarios(pb_seq))

        # ✅ 중국점 고급 feature는 dict pb_ratio를 사용하는 쪽
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
        # Road Leader (준비 미달은 non-fatal)
        # -----------------------------
        try:
            if prev_round_winner is None:
                leader_info = {
                    "leader_ready": False,
                    "leader_not_ready_reason": "prev_round_winner is None: leader update skipped",
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
                    "road_hit_rates": {},
                    "road_prediction_totals": {},
                }
            else:
                leader_streak_info = _normalize_streak_info_for_road_leader(streak_info)
                leader_info = _require_dict(
                    "leader_info",
                    road_leader.update_and_get_leader_features(
                        prev_round_winner,
                        pb_seq,
                        pb_stats_num,  # ✅ road_leader는 float pb_ratio
                        leader_streak_info,
                        pattern_dict,
                        adv,
                    ),
                )
        except road_leader.RoadLeaderNotReadyError as e:
            logger.warning("[FEATURES] road_leader not ready (non-fatal): %s", e)
            leader_info = {
                "leader_ready": False,
                "leader_not_ready_reason": str(e),
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
                "road_hit_rates": {},
                "road_prediction_totals": {},
            }

        streaks = _require_list("streak_info.streaks", streak_info["streaks"])
        trimmed_streaks = streaks[-12:]

        base_features: Dict[str, Any] = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION,

            "rounds_total": int(total_rounds),
            "p_count": int(p_count),
            "b_count": int(b_count),
            "t_count": pb_stats_raw["t_count"],

            # ✅ 출력은 숫자 유지(추천/UI/기존 코드 호환)
            "pb_ratio": float(pb_ratio_float),
            "tie_ratio": float(tie_ratio),
            "entropy": float(entropy),

            "streaks": trimmed_streaks,
            "current_streak": streak_info["current_streak"],  # road/features_china 호환(who/len 유지)
            "max_streak_p": streak_info["max_streak_p"],
            "max_streak_b": streak_info["max_streak_b"],
            "avg_streak_len": streak_info["avg_streak_len"],
            "last_20": streak_info["last_20"],
            "trend_strength": streak_info["trend_strength"],
            "momentum": streak_info["momentum"],
            "entry_momentum": float(entry_momentum),

            "pattern_score": pattern_dict["pattern_score"],
            "pattern_type": pattern_dict["pattern_type"],
            "pattern_energy": pattern_dict["pattern_energy"],
            "pattern_symmetry": pattern_dict["pattern_symmetry"],
            "pattern_noise_ratio": pattern_dict["pattern_noise_ratio"],
            "pattern_reversal_signal": pattern_dict["pattern_reversal_signal"],

            # ✅ (학습 필수) pattern_stability
            "pattern_stability": float(pattern_stability),

            "pattern_drift": temporal["pattern_drift"],
            "run_speed": temporal["run_speed"],
            "tie_volatility": temporal["tie_volatility"],

            "flow_strength": flow_dict["flow_strength"],
            "flow_stability": flow_dict["flow_stability"],

            # 기존 키 유지(호환) + 학습용 통일 키 추가
            "flow_chaos_risk": flow_dict["flow_chaos_risk"],
            "chaos_index": float(chaos_index),

            "flow_reversal_risk": flow_dict["flow_reversal_risk"],
            "flow_direction": flow_dict["flow_direction"],

            "big_eye_recent": road.big_eye_seq[-12:],
            "small_road_recent": road.small_road_seq[-12:],
            "cockroach_recent": road.cockroach_seq[-12:],

            "future_scenarios": future_scenarios,
            "beauty_score": float(beauty_score),
            "tie_state": base_tie_state,
        }

        base_features.update(adv)
        base_features.update(regime_forecast)
        base_features.update(leader_info)

        # -----------------------------
        # ✅ 최종 분석 준비도: "상태"로만 기록, 엔진 중단 금지
        # -----------------------------
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
