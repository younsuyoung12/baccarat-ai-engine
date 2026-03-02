# -*- coding: utf-8 -*-
# features_bigroad.py
"""
features_bigroad.py
====================================================
BigRoad/시계열/미래 시나리오/Regime Forecast/Beauty Score 모듈

역할:
- pattern_score soft-cap, temporal features 계산
- 미래 1~2판 시나리오 평가
- 레짐 예측(regime forecast)
- beauty_score(그림 예쁨) 계산

변경 요약 (2025-12-22)
----------------------------------------------------
1) features.py에서 BigRoad/Forecast/Beauty 관련 로직을 분리
2) 외부 입력/계약 위반을 숨기지 않도록 불필요한 예외 삼키기 로직은 두지 않음(상위에서 그대로 예외 전파)
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Tuple

import flow
import pattern
import road


def _soft_cap(score: float, cap: float = 97.0, hard_max: float = 99.0) -> float:
    """pattern_score가 너무 빨리 100에 고정되는 문제를 막기 위한 소프트 캡."""
    s = float(score)
    if s <= cap:
        return s
    boosted = cap + (s - cap) * 0.3
    return min(hard_max, boosted)


def compute_temporal_features(
    pb_stats: Dict[str, Any],
    streak_info: Dict[str, Any],
) -> Dict[str, float]:
    """시계열/변동성 Feature 계산.

    - pattern_drift   : 최근 pattern_score 표준편차
    - run_speed       : 최근 스트릭 평균 길이
    - tie_volatility  : 최근 10판 tie 비율 vs 전체 tie 비율 차이
    """
    n = 5
    if len(pattern.pattern_score_history) >= 2:
        window = pattern.pattern_score_history[-n:]
        drift = float(statistics.pstdev(window)) if len(window) >= 2 else 0.0
    else:
        drift = 0.0

    streaks = streak_info.get("streaks") or []
    last_runs = streaks[-5:]
    if last_runs:
        avg_run_len = sum(s.get("len", 0) for s in last_runs) / len(last_runs)
    else:
        avg_run_len = 0.0

    last_10 = road.big_road[-10:]
    recent_tie_ratio = (last_10.count("T") / len(last_10)) if last_10 else 0.0
    tie_ratio_total = float(pb_stats.get("tie_ratio") or 0.0)
    tie_vol = abs(recent_tie_ratio - tie_ratio_total)

    return {
        "pattern_drift": drift,
        "run_speed": avg_run_len,
        "tie_volatility": tie_vol,
    }


def _simulate_future_path(pb_seq: List[str], path: Tuple[str, ...]) -> Dict[str, float]:
    """지정된 P/B path 를 추가했을 때의 pattern/flow 상태를 간단 평가."""
    seq_next = pb_seq + list(path)

    matrix_next, positions_next = road.build_big_road_structure(seq_next)
    big_eye_next, small_road_next, cockroach_next = road.compute_chinese_roads(
        matrix_next, positions_next, seq_next
    )

    # 내부 코어 사용(원본 로직 유지)
    pattern_next = pattern.compute_pattern_features(seq_next)

    score_next = float(pattern_next["pattern_score"])
    ptype_next = str(pattern_next["pattern_type"])

    streak_next = road.compute_streaks(seq_next)
    flow_next = flow.compute_flow_features(
        big_eye_next, small_road_next, cockroach_next, streak_next
    )

    return {
        "pattern_score": score_next,
        "pattern_type": ptype_next,
        "flow_strength": float(flow_next["flow_strength"]),
        "flow_chaos_risk": float(flow_next["flow_chaos_risk"]),
    }


def compute_future_scenarios(pb_seq: List[str]) -> Dict[str, Dict[str, float]]:
    """다음 1~2판 P/B 시나리오(P,B,PP,PB,BP,BB) 평가."""
    scenarios: Dict[str, Dict[str, float]] = {}
    paths = {
        "P": ("P",),
        "B": ("B",),
        "PP": ("P", "P"),
        "PB": ("P", "B"),
        "BP": ("B", "P"),
        "BB": ("B", "B"),
    }
    for key, path in paths.items():
        scenarios[key] = _simulate_future_path(pb_seq, path)
    return scenarios


def compute_regime_forecast(
    pb_seq: List[str],
    pattern_dict: Dict[str, Any],
    temporal: Dict[str, Any],
    flow_dict: Dict[str, Any],
    adv: Dict[str, Any],
    future_scenarios: Dict[str, Any],
) -> Dict[str, Any]:
    """v9.0 Regime Forecast."""
    pattern_type = str(pattern_dict.get("pattern_type") or "none")
    chaos_index = float(adv.get("chaos_index") or 0.0)
    regime_shift_score = float(adv.get("regime_shift_score") or 0.0)
    flow_chaos = float(flow_dict.get("flow_chaos_risk") or 0.0)
    shoe_regime = str(adv.get("shoe_regime") or "mixed_shoe")

    # 2판 내 줄 시작
    base_line = 0.1
    if pattern_type in ("streak", "blocks"):
        base_line += 0.2

    if isinstance(future_scenarios, dict) and future_scenarios:
        pp = future_scenarios.get("PP", {}) if isinstance(future_scenarios.get("PP", {}), dict) else {}
        bb = future_scenarios.get("BB", {}) if isinstance(future_scenarios.get("BB", {}), dict) else {}

        best_score = max(float(pp.get("pattern_score") or 0.0), float(bb.get("pattern_score") or 0.0))
        best_type = pp.get("pattern_type") or bb.get("pattern_type") or "none"

        if best_type in ("streak", "blocks"):
            base_line += 0.15
        base_line += min(0.25, max(0.0, (best_score - 50.0) / 100.0))

    line2 = max(0.0, min(1.0, base_line))

    # 3판 내 chaos 시작
    base_chaos = 0.05 + max(0.0, chaos_index - 0.4) + max(0.0, flow_chaos - 0.5)
    if shoe_regime in ("rotation_shoe", "mixed_shoe"):
        base_chaos += 0.1
    chaos3 = max(0.0, min(1.0, base_chaos))

    # 5판 내 Regime Shift 가능성
    shift5 = max(0.0, min(1.0, regime_shift_score))

    return {
        "regime_forecast_line2": line2,
        "regime_forecast_chaos3": chaos3,
        "regime_forecast_shift5": shift5,
    }


def compute_beauty_score(
    pb_seq: List[str],
    pattern_dict: Dict[str, Any],
    flow_dict: Dict[str, Any],
    adv: Dict[str, Any],
) -> float:
    """
    사람 눈 기준 '그림 예쁨 점수' (0~100)를 계산한다.
    """
    if not pb_seq:
        return 0.0

    pattern_score = float(pattern_dict.get("pattern_score") or 0.0)
    pattern_sym = float(pattern_dict.get("pattern_symmetry") or 0.0)
    flow_stability = float(flow_dict.get("flow_stability") or 0.0)
    chaos_index = float(adv.get("chaos_index") or 0.0)
    chaos_risk = float(flow_dict.get("flow_chaos_risk") or 0.0)
    china_agree = float(adv.get("china_agree_last12") or 0.0)

    be_flips = float(adv.get("big_eye_flips_last10") or 0.0)
    sm_flips = float(adv.get("small_road_flips_last10") or 0.0)
    ck_flips = float(adv.get("cockroach_flips_last10") or 0.0)
    flip_sum = be_flips + sm_flips + ck_flips

    flip_smooth = 1.0 - min(flip_sum / 30.0, 1.0)
    pattern_norm = _soft_cap(pattern_score) / 100.0

    base = (
        0.35 * pattern_norm
        + 0.20 * pattern_sym
        + 0.20 * flow_stability
        + 0.15 * china_agree
        + 0.10 * (1.0 - chaos_index)
    )
    base = max(0.0, min(base, 1.0))

    beauty = 0.7 * base + 0.3 * flip_smooth

    if chaos_risk >= 0.90:
        beauty *= 0.6
    elif chaos_risk >= 0.80:
        beauty *= 0.75

    return max(0.0, min(100.0, beauty * 100.0))
