# -*- coding: utf-8 -*-
# features_china.py
"""
features_china.py
====================================================
중국점/Chaos/Regime 고급 Feature 모듈 v12.1
(RULE-ONLY · STRICT · NO-FALLBACK · FAIL-FAST)

역할
- 중국점(BE/SM/CK) 기반 고급 feature 계산(플립/합의도/컬럼 높이 등)
- Chaos Index / Regime(슈 성격) / 전환(transition) 신호 산출
- features.py / recommend.py / app.py가 소비하는 v12 rule-only contract 유지

변경 요약 (2026-03-14)
----------------------------------------------------
1) STRICT 계약 강화
   - 입력 타입/필수키/값 범위를 즉시 검증
   - get(... ) or ... 형태 제거
   - road 전역 시퀀스/매트릭스 계약도 검증
2) pb_stats.pb_ratio 스키마 충돌 해결
   - features.py가 만드는 dict(p_ratio/ratio/value/p/b/denom) 계약과
     legacy dict(player/banker) 계약을 모두 엄격 검증 후 수용
   - 값 불일치 시 즉시 예외
3) RULE-ONLY 정렬
   - GPT/LLM 의존 없음
   - 무상태(Stateless) 계산 유지
4) 기존 출력 키 유지
   - adv 딕셔너리의 기존 키를 유지하여 하위 호환 보장

추가 변경 요약 (2026-03-14)
----------------------------------------------------
1) road runtime 무결성 최종 검증 추가
   - road.validate_roadmap_integrity() 결과를 직접 확인
   - 전역 캐시 오염 상태를 조용히 통과시키지 않음
2) matrix 사용 용도별 허용 심볼 분리
   - big_road_matrix 는 P/B만 허용
   - china matrix 는 R/B만 허용
3) 문자열 계약 강화
   - pattern_type 은 non-empty str 강제
   - flow_direction 은 str/None만 허용하고 P/B 외 값은 명시적으로 비방향 취급
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import flow
import pattern
import road

from features_bigroad import _soft_cap

VALID_PB = ("P", "B")
VALID_RB = ("R", "B")
_BIGROAD_ALLOWED = ("P", "B")


def _require_key(d: Dict[str, Any], key: str, *, name: str) -> Any:
    if key not in d:
        raise KeyError(f"{name} missing key: {key}")
    return d[key]


def _require_dict(v: Any, name: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        raise TypeError(f"{name} must be dict, got {type(v).__name__}")
    return v


def _require_list(v: Any, name: str) -> List[Any]:
    if not isinstance(v, list):
        raise TypeError(f"{name} must be list, got {type(v).__name__}")
    return v


def _as_int(v: Any, *, name: str) -> int:
    if isinstance(v, bool):
        raise TypeError(f"{name} must be int, got bool")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    raise TypeError(f"{name} must be int, got {type(v).__name__}")


def _as_float(v: Any, *, name: str) -> float:
    if isinstance(v, bool):
        raise TypeError(f"{name} must be float, got bool")
    if not isinstance(v, (int, float)):
        raise TypeError(f"{name} must be float, got {type(v).__name__}")
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    return x


def _require_unit_interval(v: Any, *, name: str) -> float:
    x = _as_float(v, name=name)
    if x < 0.0 or x > 1.0:
        raise ValueError(f"{name} must be in [0,1], got {x}")
    return x


def _require_minus1_plus1(v: Any, *, name: str) -> float:
    x = _as_float(v, name=name)
    if x < -1.0 or x > 1.0:
        raise ValueError(f"{name} must be in [-1,1], got {x}")
    return x


def _require_nonempty_str(v: Any, *, name: str) -> str:
    if not isinstance(v, str):
        raise TypeError(f"{name} must be str, got {type(v).__name__}")
    s = v.strip()
    if not s:
        raise ValueError(f"{name} must be non-empty str")
    return s


def _validate_pb_symbol(v: Any, *, name: str) -> str:
    if not isinstance(v, str):
        raise TypeError(f"{name} must be str, got {type(v).__name__}")
    s = v.strip().upper()
    if s not in VALID_PB:
        raise ValueError(f"{name} invalid: {v!r} (allowed: {VALID_PB})")
    return s


def _validate_rb_symbol(v: Any, *, name: str) -> str:
    if not isinstance(v, str):
        raise TypeError(f"{name} must be str, got {type(v).__name__}")
    s = v.strip().upper()
    if s not in VALID_RB:
        raise ValueError(f"{name} invalid: {v!r} (allowed: {VALID_RB})")
    return s


def _validate_pb_seq(seq: Any, *, name: str) -> List[str]:
    raw = _require_list(seq, name)
    return [_validate_pb_symbol(v, name=f"{name}[{i}]") for i, v in enumerate(raw)]


def _validate_rb_seq(seq: Any, *, name: str) -> List[str]:
    raw = _require_list(seq, name)
    return [_validate_rb_symbol(v, name=f"{name}[{i}]") for i, v in enumerate(raw)]


def _validate_numeric_list(seq: Any, *, name: str) -> List[float]:
    raw = _require_list(seq, name)
    return [_as_float(v, name=f"{name}[{i}]") for i, v in enumerate(raw)]


def _validate_row_major_matrix(
    matrix: Any,
    *,
    name: str,
    allowed_nonempty: Tuple[str, ...],
) -> List[List[str]]:
    raw = _require_list(matrix, name)
    if not raw:
        return []

    normalized: List[List[str]] = []
    width: Optional[int] = None

    for r, row in enumerate(raw):
        row_raw = _require_list(row, f"{name}[{r}]")
        if width is None:
            width = len(row_raw)
        elif len(row_raw) != width:
            raise RuntimeError(f"{name} row width mismatch at row={r}")

        new_row: List[str] = []
        for c, cell in enumerate(row_raw):
            if not isinstance(cell, str):
                raise TypeError(f"{name}[{r}][{c}] must be str, got {type(cell).__name__}")
            s = cell.strip().upper()
            if s == "":
                new_row.append("")
                continue
            if s not in allowed_nonempty:
                raise ValueError(
                    f"{name}[{r}][{c}] invalid value: {cell!r} "
                    f"(allowed empty or {allowed_nonempty})"
                )
            new_row.append(s)
        normalized.append(new_row)

    return normalized


def _assert_runtime_road_contract(pb_seq: List[str]) -> None:
    if not pb_seq:
        raise ValueError("pb_seq is empty")

    _validate_rb_seq(road.big_eye_seq, name="road.big_eye_seq")
    _validate_rb_seq(road.small_road_seq, name="road.small_road_seq")
    _validate_rb_seq(road.cockroach_seq, name="road.cockroach_seq")

    if not road.big_road_matrix:
        raise RuntimeError("road.big_road_matrix missing/empty while pb_seq exists")

    _validate_row_major_matrix(
        road.big_road_matrix,
        name="road.big_road_matrix",
        allowed_nonempty=_BIGROAD_ALLOWED,
    )
    _validate_row_major_matrix(
        road.big_eye_matrix,
        name="road.big_eye_matrix",
        allowed_nonempty=VALID_RB,
    )
    _validate_row_major_matrix(
        road.small_road_matrix,
        name="road.small_road_matrix",
        allowed_nonempty=VALID_RB,
    )
    _validate_row_major_matrix(
        road.cockroach_matrix,
        name="road.cockroach_matrix",
        allowed_nonempty=VALID_RB,
    )

    if not hasattr(road, "validate_roadmap_integrity"):
        raise RuntimeError("road.validate_roadmap_integrity missing")
    ok, reason = road.validate_roadmap_integrity()
    if not isinstance(ok, bool):
        raise RuntimeError("road.validate_roadmap_integrity must return bool in first slot")
    if not ok:
        raise RuntimeError(f"road runtime integrity failed: {reason}")


def _extract_player_ratio_strict(pb_ratio_obj: Any) -> float:
    """
    허용 계약:
    1) legacy: {"player": x, "banker": y}
    2) v12 features.py: {"p_ratio": x, "ratio": x, "value": x, "p": int, "b": int, "denom": int}
    """
    obj = _require_dict(pb_ratio_obj, "pb_stats.pb_ratio")

    ratio_candidates: Dict[str, float] = {}

    for key in ("player", "p_ratio", "ratio", "value"):
        if key in obj:
            ratio_candidates[key] = _require_unit_interval(obj[key], name=f"pb_stats.pb_ratio.{key}")

    if not ratio_candidates:
        if "p" in obj and "b" in obj:
            p_count = _as_int(obj["p"], name="pb_stats.pb_ratio.p")
            b_count = _as_int(obj["b"], name="pb_stats.pb_ratio.b")
            denom = _as_int(obj.get("denom", p_count + b_count), name="pb_stats.pb_ratio.denom")
            if p_count < 0 or b_count < 0:
                raise ValueError("pb_stats.pb_ratio.p/b must be >= 0")
            if denom <= 0:
                raise ValueError("pb_stats.pb_ratio.denom must be > 0")
            if denom != p_count + b_count:
                raise ValueError(
                    f"pb_stats.pb_ratio.denom mismatch: denom={denom}, p+b={p_count + b_count}"
                )
            return float(p_count) / float(denom)
        raise KeyError(
            "pb_stats.pb_ratio must contain one of "
            "player/p_ratio/ratio/value or p/b(/denom)"
        )

    canonical = None
    for _, v in ratio_candidates.items():
        if canonical is None:
            canonical = v
            continue
        if abs(v - canonical) > 1e-9:
            raise ValueError(f"pb_stats.pb_ratio conflicting values: {ratio_candidates}")

    assert canonical is not None

    if "banker" in obj:
        banker = _require_unit_interval(obj["banker"], name="pb_stats.pb_ratio.banker")
        if abs((canonical + banker) - 1.0) > 1e-6:
            raise ValueError(
                f"pb_stats.pb_ratio player+banker must sum to 1.0, got {canonical + banker}"
            )

    if "p" in obj and "b" in obj and "denom" in obj:
        p_count = _as_int(obj["p"], name="pb_stats.pb_ratio.p")
        b_count = _as_int(obj["b"], name="pb_stats.pb_ratio.b")
        denom = _as_int(obj["denom"], name="pb_stats.pb_ratio.denom")
        if p_count < 0 or b_count < 0:
            raise ValueError("pb_stats.pb_ratio.p/b must be >= 0")
        if denom <= 0:
            raise ValueError("pb_stats.pb_ratio.denom must be > 0")
        if denom != p_count + b_count:
            raise ValueError(
                f"pb_stats.pb_ratio.denom mismatch: denom={denom}, p+b={p_count + b_count}"
            )
        derived = float(p_count) / float(denom)
        if abs(derived - canonical) > 1e-9:
            raise ValueError(
                f"pb_stats.pb_ratio derived ratio mismatch: {derived} != {canonical}"
            )

    return canonical


def _count_color_flips(seq: List[str], window: int, allowed: Tuple[str, ...]) -> int:
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    for i, x in enumerate(seq):
        if x not in allowed:
            raise ValueError(f"seq[{i}] invalid: {x!r} (allowed: {allowed})")
    tail = seq[-window:] if window > 0 else []
    if len(tail) < 2:
        return 0
    flips = 0
    for i in range(1, len(tail)):
        if tail[i] != tail[i - 1]:
            flips += 1
    return flips


def _column_heights(matrix: List[List[str]], *, name: str, allowed_nonempty: Tuple[str, ...]) -> List[int]:
    validated = _validate_row_major_matrix(
        matrix,
        name=name,
        allowed_nonempty=allowed_nonempty,
    )
    if not validated:
        return []

    rows = len(validated)
    cols = len(validated[0]) if rows > 0 else 0
    heights: List[int] = []

    for c in range(cols):
        h = 0
        for r in range(rows):
            if validated[r][c]:
                h += 1
        heights.append(h)
    return heights


def _compute_flip_cycle_pb(pb_seq: List[str]) -> float:
    if len(pb_seq) < 3:
        return 0.0

    last_side = pb_seq[0]
    last_flip_idx: Optional[int] = None
    distances: List[int] = []

    for i, side in enumerate(pb_seq[1:], start=1):
        if side != last_side:
            if last_flip_idx is not None:
                distances.append(i - last_flip_idx)
            last_flip_idx = i
            last_side = side

    if not distances:
        return 0.0
    return float(sum(distances)) / float(len(distances))


def _compute_global_chaos_ratio_from_scratch(pb_seq: List[str]) -> float:
    """
    전체 슈 기준 Chaos 비율 추정.
    warmup 구간(len < 20)은 정의값 0.0을 반환한다.
    """
    if len(pb_seq) < 20:
        return 0.0

    chaos_cnt = 0
    rounds = 0

    for i in range(10, len(pb_seq) + 1):
        prefix = pb_seq[:i]
        matrix_i, positions_i = road.build_big_road_structure(prefix)
        be_i, sm_i, ck_i = road.compute_chinese_roads(matrix_i, positions_i, prefix)
        _validate_rb_seq(be_i, name="prefix.big_eye_seq")
        _validate_rb_seq(sm_i, name="prefix.small_seq")
        _validate_rb_seq(ck_i, name="prefix.cock_seq")

        streak_i = road.compute_streaks(prefix)
        streak_i = _require_dict(streak_i, "road.compute_streaks(prefix)")
        flow_i = flow.compute_flow_features(be_i, sm_i, ck_i, streak_i)
        flow_i = _require_dict(flow_i, "flow.compute_flow_features(prefix)")

        chaos_risk_i = _require_unit_interval(
            _require_key(flow_i, "flow_chaos_risk", name="flow_i"),
            name="flow_i.flow_chaos_risk",
        )
        if chaos_risk_i >= 0.80:
            chaos_cnt += 1
        rounds += 1

    ratio = float(chaos_cnt) / float(max(rounds, 1))
    return max(0.0, min(1.0, ratio))


def _r_streak_len(seq: List[str]) -> int:
    if not seq:
        return 0
    length = 0
    for v in reversed(seq):
        if v == "R":
            length += 1
        else:
            break
    return length


def _bottom_touch_flag_for_matrix(
    matrix: List[List[str]],
    *,
    name: str,
    allowed_nonempty: Tuple[str, ...],
) -> bool:
    validated = _validate_row_major_matrix(
        matrix,
        name=name,
        allowed_nonempty=allowed_nonempty,
    )
    if not validated:
        return False

    rows = len(validated)
    cols = len(validated[0]) if rows > 0 else 0

    for c in range(cols - 1, -1, -1):
        for r in range(rows - 1, -1, -1):
            if validated[r][c]:
                return r == rows - 1
    return False


def _compute_decalcomania_features(pb_seq: List[str], window: int = 6) -> Dict[str, Any]:
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")

    if len(pb_seq) <= window:
        return {
            "decalcomania_found": False,
            "decalcomania_hint": None,
            "decalcomania_support": 0.0,
        }

    tail = pb_seq[-window:]
    total_matches = 0
    next_counts = {"P": 0, "B": 0}

    for start in range(0, len(pb_seq) - window):
        seg = pb_seq[start : start + window]
        if seg == tail and start + window < len(pb_seq):
            nxt = pb_seq[start + window]
            if nxt in VALID_PB:
                next_counts[nxt] += 1
                total_matches += 1

    if total_matches == 0:
        return {
            "decalcomania_found": False,
            "decalcomania_hint": None,
            "decalcomania_support": 0.0,
        }

    if next_counts["P"] > next_counts["B"]:
        hint = "P"
    elif next_counts["B"] > next_counts["P"]:
        hint = "B"
    else:
        return {
            "decalcomania_found": False,
            "decalcomania_hint": None,
            "decalcomania_support": 0.0,
        }

    support = float(next_counts[hint]) / float(total_matches)
    return {
        "decalcomania_found": True,
        "decalcomania_hint": hint,
        "decalcomania_support": support,
    }


def _classify_shoe_regime(frame_mode: str, segment_type: str, global_chaos_ratio: float) -> str:
    if frame_mode == "chaos_shoe" or global_chaos_ratio >= 0.55:
        return "chaos_shuffle_shoe"
    if frame_mode == "streak_shoe" or segment_type == "streak":
        return "trend_shoe"
    if frame_mode in ("pingpong_shoe", "block_shoe") or segment_type in ("pingpong", "blocks"):
        return "rotation_shoe"
    if global_chaos_ratio <= 0.20 and frame_mode == "stable_shoe":
        return "stable_shoe"
    return "mixed_shoe"


def compute_advanced_features(
    pb_seq: List[str],
    pb_stats: Dict[str, Any],
    streak_info: Dict[str, Any],
    pattern_dict: Dict[str, Any],
    temporal: Dict[str, Any],
    flow_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """v12 rule-only 중국점/Chaos/Regime Feature 계산."""
    pb_seq_v = _validate_pb_seq(pb_seq, name="pb_seq")
    pb_stats_v = _require_dict(pb_stats, "pb_stats")
    streak_info_v = _require_dict(streak_info, "streak_info")
    pattern_dict_v = _require_dict(pattern_dict, "pattern_dict")
    temporal_v = _require_dict(temporal, "temporal")
    flow_dict_v = _require_dict(flow_dict, "flow_dict")

    _assert_runtime_road_contract(pb_seq_v)

    rounds_total = _as_int(_require_key(pb_stats_v, "total_rounds", name="pb_stats"), name="pb_stats.total_rounds")
    if rounds_total <= 0:
        raise ValueError(f"pb_stats.total_rounds must be > 0, got {rounds_total}")

    pattern_type_raw = _require_nonempty_str(
        _require_key(pattern_dict_v, "pattern_type", name="pattern_dict"),
        name="pattern_dict.pattern_type",
    )
    pattern_type = pattern_type_raw.lower()

    chaos_risk = _require_unit_interval(
        _require_key(flow_dict_v, "flow_chaos_risk", name="flow_dict"),
        name="flow_dict.flow_chaos_risk",
    )
    reversal_risk = _require_unit_interval(
        _require_key(flow_dict_v, "flow_reversal_risk", name="flow_dict"),
        name="flow_dict.flow_reversal_risk",
    )
    noise = _require_unit_interval(
        _require_key(pattern_dict_v, "pattern_noise_ratio", name="pattern_dict"),
        name="pattern_dict.pattern_noise_ratio",
    )
    pattern_energy = _require_minus1_plus1(
        _require_key(pattern_dict_v, "pattern_energy", name="pattern_dict"),
        name="pattern_dict.pattern_energy",
    )
    pattern_drift = _as_float(
        _require_key(temporal_v, "pattern_drift", name="temporal"),
        name="temporal.pattern_drift",
    )
    if pattern_drift < 0.0:
        raise ValueError(f"temporal.pattern_drift must be >= 0, got {pattern_drift}")

    flow_strength = _require_unit_interval(
        _require_key(flow_dict_v, "flow_strength", name="flow_dict"),
        name="flow_dict.flow_strength",
    )
    flow_stability = _require_unit_interval(
        _require_key(flow_dict_v, "flow_stability", name="flow_dict"),
        name="flow_dict.flow_stability",
    )
    pattern_score = _as_float(
        _require_key(pattern_dict_v, "pattern_score", name="pattern_dict"),
        name="pattern_dict.pattern_score",
    )

    flow_direction_raw = _require_key(flow_dict_v, "flow_direction", name="flow_dict")
    if flow_direction_raw is not None and not isinstance(flow_direction_raw, str):
        raise TypeError("flow_dict.flow_direction must be str or None")
    flow_direction = flow_direction_raw.strip().upper() if isinstance(flow_direction_raw, str) else None

    adv: Dict[str, Any] = {}

    # 1) 로드 동조 스코어
    p_score = 0
    b_score = 0
    last_pb = pb_seq_v[-1]

    if last_pb == "P":
        p_score += 1
    elif last_pb == "B":
        b_score += 1

    def _apply_vote(side: Optional[str]) -> None:
        nonlocal p_score, b_score
        if side == "P":
            p_score += 1
        elif side == "B":
            b_score += 1

    if flow_direction in ("P", "B"):
        _apply_vote(flow_direction)

    opp = "B" if last_pb == "P" else "P"
    for seq in (road.big_eye_seq, road.small_road_seq, road.cockroach_seq):
        if not seq:
            continue
        last = seq[-1]
        if last == "R":
            _apply_vote(last_pb)
        elif last == "B":
            _apply_vote(opp)

    adv["road_sync_p"] = p_score
    adv["road_sync_b"] = b_score
    adv["road_sync_gap"] = abs(p_score - b_score)

    # 2) 구간 유형
    if chaos_risk >= 0.75 or noise >= 0.6:
        segment_type = "chaos"
    elif pattern_type in ("streak", "pingpong", "blocks"):
        segment_type = pattern_type
    else:
        segment_type = "mixed"
    adv["segment_type"] = segment_type

    # 3) 전환 플래그
    be_flips = _count_color_flips(road.big_eye_seq, 10, VALID_RB)
    sm_flips = _count_color_flips(road.small_road_seq, 10, VALID_RB)
    ck_flips = _count_color_flips(road.cockroach_seq, 10, VALID_RB)
    flip_sum = be_flips + sm_flips + ck_flips

    transition_flag = abs(pattern_energy) >= 0.15 or pattern_drift >= 10.0 or flip_sum >= 8
    adv["transition_flag"] = transition_flag

    # 4) Mini Trend
    last6 = pb_seq_v[-6:]
    adv["mini_trend_p"] = last6.count("P")
    adv["mini_trend_b"] = last6.count("B")

    # 5) 중국점 방향 일치율(최근 12판)
    n = min(len(road.big_eye_seq), len(road.small_road_seq), len(road.cockroach_seq), 12)
    agree_cnt = 0
    for i in range(n):
        be = road.big_eye_seq[-n + i]
        sm = road.small_road_seq[-n + i]
        ck = road.cockroach_seq[-n + i]
        if be in VALID_RB and be == sm == ck:
            agree_cnt += 1
    adv["china_agree_last12"] = (float(agree_cnt) / float(n)) if n > 0 else 0.0

    # 6) 중국점 컬럼 높이 변화
    def _height_change(m: List[List[str]], *, name: str) -> int:
        heights = _column_heights(m, name=name, allowed_nonempty=VALID_RB)
        if len(heights) >= 2:
            return heights[-1] - heights[-2]
        return 0

    adv["big_eye_height_change"] = _height_change(road.big_eye_matrix, name="road.big_eye_matrix")
    adv["small_road_height_change"] = _height_change(road.small_road_matrix, name="road.small_road_matrix")
    adv["cockroach_height_change"] = _height_change(road.cockroach_matrix, name="road.cockroach_matrix")

    # 7) Chaos Index
    chaos_index = 0.6 * chaos_risk + 0.2 * noise + 0.2 * reversal_risk
    if rounds_total < 15:
        chaos_index *= 0.6
    chaos_index = max(0.0, min(1.0, chaos_index))
    adv["chaos_index"] = chaos_index

    # 8) pattern_score 전구간/최근구간
    history_raw = getattr(pattern, "pattern_score_history", None)
    if history_raw is None:
        raise RuntimeError("pattern.pattern_score_history missing")
    hist = _validate_numeric_list(history_raw, name="pattern.pattern_score_history")

    if hist:
        weights = [0.9 ** (len(hist) - 1 - i) for i in range(len(hist))]
        denom = sum(weights)
        if denom <= 0:
            raise RuntimeError("pattern_score_history weights invalid")
        global_raw = sum(s * w for s, w in zip(hist, weights)) / denom
        pattern_score_global = _soft_cap(global_raw)

        last10 = hist[-10:]
        last5 = hist[-5:]
        last10_mean = _soft_cap(sum(last10) / len(last10)) if last10 else _soft_cap(pattern_score)
        last5_mean = _soft_cap(sum(last5) / len(last5)) if last5 else _soft_cap(pattern_score)

        adv["pattern_score_global"] = float(pattern_score_global)
        adv["pattern_score_last10"] = float(last10_mean)
        adv["pattern_score_last5"] = float(last5_mean)
    else:
        base = _soft_cap(pattern_score)
        adv["pattern_score_global"] = float(base)
        adv["pattern_score_last10"] = float(base)
        adv["pattern_score_last5"] = float(base)

    adv["pattern_stability"] = 1.0 / (pattern_drift + 1.0)

    adv["big_eye_flips_last10"] = be_flips
    adv["small_road_flips_last10"] = sm_flips
    adv["cockroach_flips_last10"] = ck_flips

    adv["flip_cycle_pb"] = _compute_flip_cycle_pb(pb_seq_v)

    # 9) 전체 슈 Frame Mode & global chaos ratio
    global_chaos_ratio = _compute_global_chaos_ratio_from_scratch(pb_seq_v)
    adv["global_chaos_ratio"] = global_chaos_ratio

    ps_global = float(adv["pattern_score_global"])
    if global_chaos_ratio >= 0.40:
        frame_mode = "chaos_shoe"
    elif pattern_type == "streak":
        frame_mode = "streak_shoe"
    elif pattern_type == "pingpong":
        frame_mode = "pingpong_shoe"
    elif pattern_type == "blocks":
        frame_mode = "block_shoe"
    elif ps_global >= 60.0:
        frame_mode = "stable_shoe"
    else:
        frame_mode = "mixed_shoe"
    adv["frame_mode"] = frame_mode

    shoe_regime = _classify_shoe_regime(frame_mode, segment_type, global_chaos_ratio)
    adv["shoe_regime"] = shoe_regime

    # 10) 후반부 선호 패턴
    if len(hist) >= 20:
        first10 = hist[:10]
        last10_hist = hist[-10:]
        adv["frame_trend_delta"] = float((sum(last10_hist) / len(last10_hist)) - (sum(first10) / len(first10)))
    else:
        adv["frame_trend_delta"] = 0.0

    # 11) 중국점 상호 반응성
    response_delay_score = abs(be_flips - sm_flips) + abs(be_flips - ck_flips)
    adv["response_delay_score"] = float(response_delay_score)

    # 12) odd_run_length / odd_run_spike_flag
    current_streak = _require_key(streak_info_v, "current_streak", name="streak_info")
    current_streak = _require_dict(current_streak, "streak_info.current_streak")
    odd_run_length = _as_int(
        _require_key(current_streak, "len", name="streak_info.current_streak"),
        name="streak_info.current_streak.len",
    )
    if odd_run_length < 0:
        raise ValueError(f"streak_info.current_streak.len must be >= 0, got {odd_run_length}")

    adv["odd_run_length"] = odd_run_length
    adv["odd_run_spike_flag"] = bool(odd_run_length >= 3 and segment_type == "pingpong")

    # 13) three_rule_signal
    def _su_tail(seq: List[str]) -> str:
        tail = seq[-3:]
        if not tail:
            return "---"
        return "".join("S" if v == "R" else "U" if v == "B" else "-" for v in tail)

    be_su = _su_tail(road.big_eye_seq)
    sm_su = _su_tail(road.small_road_seq)
    ck_su = _su_tail(road.cockroach_seq)
    adv["three_rule_signal"] = f"BE:{be_su}|SM:{sm_su}|CK:{ck_su}"

    # 14) v7.6 추가 Feature
    adv["china_r_streak_be"] = _r_streak_len(road.big_eye_seq)
    adv["china_r_streak_sm"] = _r_streak_len(road.small_road_seq)
    adv["china_r_streak_ck"] = _r_streak_len(road.cockroach_seq)

    def _last_depth(m: List[List[str]], *, name: str) -> int:
        heights = _column_heights(m, name=name, allowed_nonempty=VALID_RB)
        return heights[-1] if heights else 0

    adv["china_depth_be"] = _last_depth(road.big_eye_matrix, name="road.big_eye_matrix")
    adv["china_depth_sm"] = _last_depth(road.small_road_matrix, name="road.small_road_matrix")
    adv["china_depth_ck"] = _last_depth(road.cockroach_matrix, name="road.cockroach_matrix")

    adv["bottom_touch_bigroad"] = _bottom_touch_flag_for_matrix(
        road.big_road_matrix,
        name="road.big_road_matrix",
        allowed_nonempty=_BIGROAD_ALLOWED,
    )
    adv["bottom_touch_bigeye"] = _bottom_touch_flag_for_matrix(
        road.big_eye_matrix,
        name="road.big_eye_matrix",
        allowed_nonempty=VALID_RB,
    )
    adv["bottom_touch_small"] = _bottom_touch_flag_for_matrix(
        road.small_road_matrix,
        name="road.small_road_matrix",
        allowed_nonempty=VALID_RB,
    )
    adv["bottom_touch_cockroach"] = _bottom_touch_flag_for_matrix(
        road.cockroach_matrix,
        name="road.cockroach_matrix",
        allowed_nonempty=VALID_RB,
    )

    deca = _compute_decalcomania_features(pb_seq_v, window=6)
    adv["decalcomania_found"] = deca["decalcomania_found"]
    adv["decalcomania_hint"] = deca["decalcomania_hint"]
    adv["decalcomania_support"] = deca["decalcomania_support"]

    pb_ratio_global_obj = _require_key(pb_stats_v, "pb_ratio", name="pb_stats")
    p_global = _extract_player_ratio_strict(pb_ratio_global_obj)

    tail_pb = pb_seq_v[-10:]
    p_local = (tail_pb.count("P") / len(tail_pb)) if tail_pb else 0.0
    pb_diff_score = abs(p_local - p_global)
    adv["pb_diff_score"] = pb_diff_score

    approx_max_rounds = 72.0
    phase_progress = min(1.0, float(rounds_total) / approx_max_rounds)
    if rounds_total <= 20:
        shoe_phase = "early"
    elif rounds_total <= 45:
        shoe_phase = "mid"
    else:
        shoe_phase = "late"

    adv["shoe_phase"] = shoe_phase
    adv["phase_progress"] = phase_progress

    chaos_end_flag = (
        chaos_index < 0.5
        and flow_strength >= 0.4
        and flow_stability >= 0.5
        and segment_type != "chaos"
        and frame_mode != "chaos_shoe"
    )
    adv["chaos_end_flag"] = chaos_end_flag

    # 15) Regime Shift Score
    regime_shift_score = 0.0
    if transition_flag:
        regime_shift_score += 0.4
    regime_shift_score += min(0.3, max(0.0, chaos_index - 0.5))
    regime_shift_score += min(0.3, max(0.0, abs(pb_diff_score - 0.15)))
    if shoe_phase == "late":
        regime_shift_score += 0.1
    regime_shift_score = max(0.0, min(1.0, regime_shift_score))
    adv["regime_shift_score"] = regime_shift_score

    return adv