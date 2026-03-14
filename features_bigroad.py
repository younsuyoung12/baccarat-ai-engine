# -*- coding: utf-8 -*-
# features_bigroad.py
"""
features_bigroad.py
====================================================
BigRoad/시계열/미래 시나리오/Regime Forecast/Beauty Score 모듈 v12.1
(RULE-ONLY · STRICT · NO-FALLBACK · FAIL-FAST)

역할
- pattern_score soft-cap, temporal features 계산
- 미래 1~2판 시나리오 평가
- 레짐 예측(regime forecast)
- beauty_score(그림 예쁨) 계산

중요 정책
----------------------------------------------------
- STRICT · NO-FALLBACK · FAIL-FAST
- 입력 타입/필수키/값 범위 위반 시 즉시 예외
- get(... ) or ... 형태의 폴백 금지
- warmup에서만 허용되는 값은 명시적으로 결정론적으로 계산
- 출력 스키마는 features.py / predictor_adapter.py / app.py 하위 호환 유지
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Tuple

import flow
import pattern
import road

VALID_PB = ("P", "B")
VALID_WINNER = ("P", "B", "T")
VALID_RB = ("R", "B")
FUTURE_SCENARIO_KEYS = ("P", "B", "PP", "PB", "BP", "BB")


def _require_dict(v: Any, name: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        raise TypeError(f"{name} must be dict, got {type(v).__name__}")
    return v


def _require_list(v: Any, name: str) -> List[Any]:
    if not isinstance(v, list):
        raise TypeError(f"{name} must be list, got {type(v).__name__}")
    return v


def _require_key(d: Dict[str, Any], key: str, *, name: str) -> Any:
    if key not in d:
        raise KeyError(f"{name} missing required key: {key}")
    return d[key]


def _as_float(v: Any, *, name: str) -> float:
    if isinstance(v, bool):
        raise TypeError(f"{name} must be float, got bool")
    if not isinstance(v, (int, float)):
        raise TypeError(f"{name} must be float, got {type(v).__name__}")
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def _as_int(v: Any, *, name: str) -> int:
    if isinstance(v, bool):
        raise TypeError(f"{name} must be int, got bool")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    raise TypeError(f"{name} must be int, got {type(v).__name__}")


def _require_unit_interval(v: Any, *, name: str) -> float:
    x = _as_float(v, name=name)
    if x < 0.0 or x > 1.0:
        raise ValueError(f"{name} must be in [0,1], got {x}")
    return x


def _require_score_0_100(v: Any, *, name: str) -> float:
    x = _as_float(v, name=name)
    if x < 0.0 or x > 100.0:
        raise ValueError(f"{name} must be in [0,100], got {x}")
    return x


def _require_nonnegative_float(v: Any, *, name: str) -> float:
    x = _as_float(v, name=name)
    if x < 0.0:
        raise ValueError(f"{name} must be >= 0, got {x}")
    return x


def _require_nonempty_str(v: Any, *, name: str) -> str:
    if not isinstance(v, str):
        raise TypeError(f"{name} must be str, got {type(v).__name__}")
    s = v.strip()
    if not s:
        raise ValueError(f"{name} must be non-empty str")
    return s


def _validate_pb_seq(pb_seq: Any, *, name: str) -> List[str]:
    raw = _require_list(pb_seq, name)
    out: List[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise TypeError(f"{name}[{i}] must be str, got {type(item).__name__}")
        s = item.strip().upper()
        if s not in VALID_PB:
            raise ValueError(f"{name}[{i}] invalid: {item!r} (allowed: {VALID_PB})")
        out.append(s)
    return out


def _validate_big_road_raw(big_road_raw: Any, *, name: str) -> List[str]:
    raw = _require_list(big_road_raw, name)
    out: List[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise TypeError(f"{name}[{i}] must be str, got {type(item).__name__}")
        s = item.strip().upper()
        if s not in VALID_WINNER:
            raise ValueError(f"{name}[{i}] invalid: {item!r} (allowed: {VALID_WINNER})")
        out.append(s)
    return out


def _validate_rb_seq(seq: Any, *, name: str) -> List[str]:
    raw = _require_list(seq, name)
    out: List[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise TypeError(f"{name}[{i}] must be str, got {type(item).__name__}")
        s = item.strip().upper()
        if s not in VALID_RB:
            raise ValueError(f"{name}[{i}] invalid: {item!r} (allowed: {VALID_RB})")
        out.append(s)
    return out


def _validate_positions(positions: Any, *, expected_len: int, name: str) -> List[Tuple[int, int]]:
    raw = _require_list(positions, name)
    if len(raw) != expected_len:
        raise RuntimeError(f"{name} length mismatch: {len(raw)} != {expected_len}")

    out: List[Tuple[int, int]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{name}[{i}] must be tuple[int,int], got {type(item).__name__}")
        col, row = item
        if not isinstance(col, int) or not isinstance(row, int):
            raise TypeError(f"{name}[{i}] must be tuple[int,int]")
        if col < 0 or row < 0:
            raise ValueError(f"{name}[{i}] must be non-negative, got {(col, row)}")
        out.append((col, row))
    return out


def _validate_pattern_history() -> List[float]:
    history = getattr(pattern, "pattern_score_history", None)
    raw = _require_list(history, "pattern.pattern_score_history")
    out: List[float] = []
    for i, v in enumerate(raw):
        out.append(_require_score_0_100(v, name=f"pattern.pattern_score_history[{i}]"))
    return out


def _mean_len_from_streaks(streaks: List[Any]) -> float:
    if not streaks:
        return 0.0

    lengths: List[int] = []
    for i, s in enumerate(streaks):
        sd = _require_dict(s, f"streaks[{i}]")
        ln = _as_int(_require_key(sd, "len", name=f"streaks[{i}]"), name=f"streaks[{i}].len")
        if ln < 0:
            raise ValueError(f"streaks[{i}].len must be >= 0, got {ln}")
        lengths.append(ln)

    if not lengths:
        return 0.0
    return float(sum(lengths)) / float(len(lengths))


def _soft_cap(score: float, cap: float = 97.0, hard_max: float = 99.0) -> float:
    """pattern_score가 너무 빨리 100에 고정되는 문제를 막기 위한 소프트 캡."""
    s = _require_score_0_100(score, name="score")
    cap_v = _require_score_0_100(cap, name="cap")
    hard_max_v = _require_score_0_100(hard_max, name="hard_max")
    if cap_v > hard_max_v:
        raise ValueError(f"cap must be <= hard_max, got cap={cap_v}, hard_max={hard_max_v}")

    if s <= cap_v:
        return s
    boosted = cap_v + (s - cap_v) * 0.3
    return min(hard_max_v, boosted)


def _validate_future_scenario_payload(payload: Dict[str, Any], *, name: str) -> Dict[str, Any]:
    obj = _require_dict(payload, name)

    pattern_score = _require_score_0_100(
        _require_key(obj, "pattern_score", name=name),
        name=f"{name}.pattern_score",
    )
    pattern_type = _require_nonempty_str(
        _require_key(obj, "pattern_type", name=name),
        name=f"{name}.pattern_type",
    )
    flow_strength = _require_unit_interval(
        _require_key(obj, "flow_strength", name=name),
        name=f"{name}.flow_strength",
    )
    flow_chaos_risk = _require_unit_interval(
        _require_key(obj, "flow_chaos_risk", name=name),
        name=f"{name}.flow_chaos_risk",
    )

    return {
        "pattern_score": float(pattern_score),
        "pattern_type": pattern_type,
        "flow_strength": float(flow_strength),
        "flow_chaos_risk": float(flow_chaos_risk),
    }


def compute_temporal_features(
    pb_stats: Dict[str, Any],
    streak_info: Dict[str, Any],
) -> Dict[str, float]:
    """시계열/변동성 Feature 계산."""
    pb_stats_v = _require_dict(pb_stats, "pb_stats")
    streak_info_v = _require_dict(streak_info, "streak_info")

    tie_ratio_total = _require_unit_interval(
        _require_key(pb_stats_v, "tie_ratio", name="pb_stats"),
        name="pb_stats.tie_ratio",
    )

    streaks = _require_list(_require_key(streak_info_v, "streaks", name="streak_info"), "streak_info.streaks")
    last_runs = streaks[-5:]
    avg_run_len = _mean_len_from_streaks(last_runs)

    hist = _validate_pattern_history()
    if len(hist) >= 2:
        window = hist[-5:]
        drift = float(statistics.pstdev(window)) if len(window) >= 2 else 0.0
    else:
        drift = 0.0

    big_road_raw = _validate_big_road_raw(getattr(road, "big_road", None), name="road.big_road")
    last_10 = big_road_raw[-10:]
    recent_tie_ratio = (last_10.count("T") / len(last_10)) if last_10 else 0.0
    tie_vol = abs(recent_tie_ratio - tie_ratio_total)

    if not math.isfinite(drift):
        raise ValueError("pattern_drift must be finite")
    if drift < 0.0:
        raise ValueError(f"pattern_drift must be >= 0, got {drift}")
    if not math.isfinite(avg_run_len):
        raise ValueError("run_speed must be finite")
    if avg_run_len < 0.0:
        raise ValueError(f"run_speed must be >= 0, got {avg_run_len}")
    if not math.isfinite(tie_vol):
        raise ValueError("tie_volatility must be finite")
    if tie_vol < 0.0:
        raise ValueError(f"tie_volatility must be >= 0, got {tie_vol}")

    return {
        "pattern_drift": float(drift),
        "run_speed": float(avg_run_len),
        "tie_volatility": float(tie_vol),
    }


def _simulate_future_path(pb_seq: List[str], path: Tuple[str, ...]) -> Dict[str, Any]:
    """지정된 P/B path를 추가했을 때의 pattern/flow 상태를 평가한다."""
    pb_seq_v = _validate_pb_seq(pb_seq, name="pb_seq")
    if not pb_seq_v:
        raise ValueError("pb_seq must not be empty")

    if not isinstance(path, tuple):
        raise TypeError(f"path must be tuple[str, ...], got {type(path).__name__}")
    if not path:
        raise ValueError("path must not be empty")

    path_list: List[str] = []
    for i, p in enumerate(path):
        if not isinstance(p, str):
            raise TypeError(f"path[{i}] must be str, got {type(p).__name__}")
        s = p.strip().upper()
        if s not in VALID_PB:
            raise ValueError(f"path[{i}] invalid: {p!r} (allowed: {VALID_PB})")
        path_list.append(s)

    seq_next = pb_seq_v + path_list

    matrix_next, positions_next = road.build_big_road_structure(seq_next)
    _require_list(matrix_next, "road.build_big_road_structure().matrix")
    _validate_positions(
        positions_next,
        expected_len=len(seq_next),
        name="road.build_big_road_structure().positions",
    )

    big_eye_next, small_road_next, cockroach_next = road.compute_chinese_roads(
        matrix_next,
        positions_next,
        seq_next,
    )
    _validate_rb_seq(big_eye_next, name="big_eye_next")
    _validate_rb_seq(small_road_next, name="small_road_next")
    _validate_rb_seq(cockroach_next, name="cockroach_next")

    pattern_next = _require_dict(pattern.compute_pattern_features(seq_next), "pattern_next")
    score_next = _require_score_0_100(
        _require_key(pattern_next, "pattern_score", name="pattern_next"),
        name="pattern_next.pattern_score",
    )
    ptype_next = _require_nonempty_str(
        _require_key(pattern_next, "pattern_type", name="pattern_next"),
        name="pattern_next.pattern_type",
    )

    streak_next = _require_dict(road.compute_streaks(seq_next), "streak_next")
    flow_next = _require_dict(
        flow.compute_flow_features(big_eye_next, small_road_next, cockroach_next, streak_next),
        "flow_next",
    )

    flow_strength = _require_unit_interval(
        _require_key(flow_next, "flow_strength", name="flow_next"),
        name="flow_next.flow_strength",
    )
    flow_chaos_risk = _require_unit_interval(
        _require_key(flow_next, "flow_chaos_risk", name="flow_next"),
        name="flow_next.flow_chaos_risk",
    )

    return {
        "pattern_score": float(score_next),
        "pattern_type": ptype_next,
        "flow_strength": float(flow_strength),
        "flow_chaos_risk": float(flow_chaos_risk),
    }


def compute_future_scenarios(pb_seq: List[str]) -> Dict[str, Dict[str, Any]]:
    """다음 1~2판 P/B 시나리오(P,B,PP,PB,BP,BB) 평가."""
    pb_seq_v = _validate_pb_seq(pb_seq, name="pb_seq")
    if not pb_seq_v:
        raise ValueError("pb_seq must not be empty")

    scenarios: Dict[str, Dict[str, Any]] = {}
    paths: Dict[str, Tuple[str, ...]] = {
        "P": ("P",),
        "B": ("B",),
        "PP": ("P", "P"),
        "PB": ("P", "B"),
        "BP": ("B", "P"),
        "BB": ("B", "B"),
    }

    for key in FUTURE_SCENARIO_KEYS:
        scenarios[key] = _validate_future_scenario_payload(
            _simulate_future_path(pb_seq_v, paths[key]),
            name=f"future_scenarios.{key}",
        )

    return scenarios


def compute_regime_forecast(
    pb_seq: List[str],
    pattern_dict: Dict[str, Any],
    temporal: Dict[str, Any],
    flow_dict: Dict[str, Any],
    adv: Dict[str, Any],
    future_scenarios: Dict[str, Any],
) -> Dict[str, Any]:
    """레짐 예측(regime forecast) 계산."""
    _validate_pb_seq(pb_seq, name="pb_seq")
    pattern_dict_v = _require_dict(pattern_dict, "pattern_dict")
    temporal_v = _require_dict(temporal, "temporal")
    flow_dict_v = _require_dict(flow_dict, "flow_dict")
    adv_v = _require_dict(adv, "adv")
    future_v = _require_dict(future_scenarios, "future_scenarios")

    pattern_type = _require_nonempty_str(
        _require_key(pattern_dict_v, "pattern_type", name="pattern_dict"),
        name="pattern_dict.pattern_type",
    ).lower()

    chaos_index = _require_unit_interval(
        _require_key(adv_v, "chaos_index", name="adv"),
        name="adv.chaos_index",
    )
    regime_shift_score = _require_unit_interval(
        _require_key(adv_v, "regime_shift_score", name="adv"),
        name="adv.regime_shift_score",
    )
    flow_chaos = _require_unit_interval(
        _require_key(flow_dict_v, "flow_chaos_risk", name="flow_dict"),
        name="flow_dict.flow_chaos_risk",
    )
    shoe_regime = _require_nonempty_str(
        _require_key(adv_v, "shoe_regime", name="adv"),
        name="adv.shoe_regime",
    )

    _require_nonnegative_float(
        _require_key(temporal_v, "pattern_drift", name="temporal"),
        name="temporal.pattern_drift",
    )

    pp = _validate_future_scenario_payload(
        _require_dict(_require_key(future_v, "PP", name="future_scenarios"), "future_scenarios.PP"),
        name="future_scenarios.PP",
    )
    bb = _validate_future_scenario_payload(
        _require_dict(_require_key(future_v, "BB", name="future_scenarios"), "future_scenarios.BB"),
        name="future_scenarios.BB",
    )

    base_line = 0.1
    if pattern_type in ("streak", "blocks"):
        base_line += 0.2

    if pp["pattern_score"] >= bb["pattern_score"]:
        best_score = pp["pattern_score"]
        best_type = pp["pattern_type"].lower()
    else:
        best_score = bb["pattern_score"]
        best_type = bb["pattern_type"].lower()

    if best_type in ("streak", "blocks"):
        base_line += 0.15
    base_line += min(0.25, max(0.0, (best_score - 50.0) / 100.0))
    line2 = max(0.0, min(1.0, base_line))

    base_chaos = 0.05 + max(0.0, chaos_index - 0.4) + max(0.0, flow_chaos - 0.5)
    if shoe_regime in ("rotation_shoe", "mixed_shoe"):
        base_chaos += 0.1
    chaos3 = max(0.0, min(1.0, base_chaos))

    shift5 = max(0.0, min(1.0, regime_shift_score))

    return {
        "regime_forecast_line2": float(line2),
        "regime_forecast_chaos3": float(chaos3),
        "regime_forecast_shift5": float(shift5),
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
    pb_seq_v = _validate_pb_seq(pb_seq, name="pb_seq")
    if not pb_seq_v:
        raise ValueError("pb_seq must not be empty")

    pattern_dict_v = _require_dict(pattern_dict, "pattern_dict")
    flow_dict_v = _require_dict(flow_dict, "flow_dict")
    adv_v = _require_dict(adv, "adv")

    pattern_score = _require_score_0_100(
        _require_key(pattern_dict_v, "pattern_score", name="pattern_dict"),
        name="pattern_dict.pattern_score",
    )
    pattern_sym = _require_unit_interval(
        _require_key(pattern_dict_v, "pattern_symmetry", name="pattern_dict"),
        name="pattern_dict.pattern_symmetry",
    )
    flow_stability = _require_unit_interval(
        _require_key(flow_dict_v, "flow_stability", name="flow_dict"),
        name="flow_dict.flow_stability",
    )
    chaos_index = _require_unit_interval(
        _require_key(adv_v, "chaos_index", name="adv"),
        name="adv.chaos_index",
    )
    chaos_risk = _require_unit_interval(
        _require_key(flow_dict_v, "flow_chaos_risk", name="flow_dict"),
        name="flow_dict.flow_chaos_risk",
    )
    china_agree = _require_unit_interval(
        _require_key(adv_v, "china_agree_last12", name="adv"),
        name="adv.china_agree_last12",
    )

    be_flips = _require_nonnegative_float(
        _require_key(adv_v, "big_eye_flips_last10", name="adv"),
        name="adv.big_eye_flips_last10",
    )
    sm_flips = _require_nonnegative_float(
        _require_key(adv_v, "small_road_flips_last10", name="adv"),
        name="adv.small_road_flips_last10",
    )
    ck_flips = _require_nonnegative_float(
        _require_key(adv_v, "cockroach_flips_last10", name="adv"),
        name="adv.cockroach_flips_last10",
    )

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

    final_score = max(0.0, min(100.0, beauty * 100.0))
    if not math.isfinite(final_score):
        raise ValueError("beauty_score must be finite")

    return float(final_score)