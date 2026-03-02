# -*- coding: utf-8 -*-
# features_china.py
"""
features_china.py
====================================================
중국점/Chaos/Regime 고급 Feature 모듈 (옵션 A: 완전 무상태)

역할
- 중국점(BE/SM/CK) 기반 고급 feature 계산(플립/합의도/컬럼 높이 등)
- Chaos Index / Regime(슈 성격) / 전환(transition) 신호 산출

변경 요약 (2025-12-23)
----------------------------------------------------
1) features_entry(EntryState/STATE) 의존 완전 제거 (옵션 A: 무상태)
2) compute_advanced_features(..., state=...) 제거
3) entry_momentum / tie_turbulence_rounds 등 상태 누적 기반 Feature 삭제
4) 폴백 금지 강화: 필수 입력 키 누락 시 즉시 예외
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

import flow
import pattern
import road

from features_bigroad import _soft_cap


def _require_key(d: Dict[str, Any], key: str, *, name: str) -> Any:
    if key not in d:
        raise KeyError(f"{name} missing key: {key}")
    return d[key]


def _count_color_flips(seq: List[str], window: int, allowed: Tuple[str, ...]) -> int:
    """최근 window 구간에서 색상(R/B) 전환 횟수 계산."""
    tail = [x for x in seq[-window:] if x in allowed]
    if len(tail) < 2:
        return 0
    flips = 0
    for i in range(1, len(tail)):
        if tail[i] != tail[i - 1]:
            flips += 1
    return flips


def _column_heights(matrix: List[List[str]]) -> List[int]:
    """각 컬럼별 높이(칩 개수) 계산."""
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    heights: List[int] = []
    for c in range(cols):
        h = 0
        for r in range(rows):
            if matrix[r][c]:
                h += 1
        heights.append(h)
    return heights


def _compute_flip_cycle_pb(pb_seq: List[str]) -> float:
    """P/B 사이 전환 주기 평균 (flip_cycle_pb)."""
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
    return sum(distances) / len(distances)


def _compute_global_chaos_ratio_from_scratch(pb_seq: List[str]) -> float:
    """전체 슈 기준 Chaos 비율 추정 (v9.x)."""
    # 초반(데이터 부족)에서는 '정의된 값'으로 0.0 반환한다 (엔진 중단 금지).
    if len(pb_seq) < 20:
        return 0.0

    chaos_cnt = 0
    rounds = 0
    for i in range(10, len(pb_seq) + 1):
        prefix = pb_seq[:i]
        matrix_i, positions_i = road.build_big_road_structure(prefix)
        be_i, sm_i, ck_i = road.compute_chinese_roads(matrix_i, positions_i, prefix)
        streak_i = road.compute_streaks(prefix)
        flow_i = flow.compute_flow_features(be_i, sm_i, ck_i, streak_i)
        if float(flow_i["flow_chaos_risk"]) >= 0.80:
            chaos_cnt += 1
        rounds += 1
    return chaos_cnt / max(rounds, 1)


def _r_streak_len(seq: List[str]) -> int:
    """해당 R/B 시퀀스의 마지막 R 연속 길이."""
    if not seq:
        return 0
    length = 0
    for v in reversed(seq):
        if v == "R":
            length += 1
        else:
            break
    return length


def _bottom_touch_flag_for_matrix(matrix: List[List[str]]) -> bool:
    """6행 보드 기준, 마지막 칩이 바닥에 닿았는지 여부."""
    if not matrix:
        return False
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    for c in range(cols - 1, -1, -1):
        for r in range(rows - 1, -1, -1):
            if matrix[r][c]:
                return r == rows - 1
    return False


def _compute_decalcomania_features(pb_seq: List[str], window: int = 6) -> Dict[str, Any]:
    """데칼코마니 패턴 감지."""
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
            if nxt in ("P", "B"):
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

    support = next_counts[hint] / total_matches
    return {
        "decalcomania_found": True,
        "decalcomania_hint": hint,
        "decalcomania_support": support,
    }


def _classify_shoe_regime(frame_mode: str, segment_type: str, global_chaos_ratio: float) -> str:
    """슈 전체 성격을 간단 레짐으로 분류 (v9.x 우선순위)."""
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
    """v7.5~v9.x 중국점/Chaos/Regime Feature 계산 (옵션 A: 무상태)."""
    if not pb_seq:
        raise ValueError("pb_seq is empty")

    # 필수 키 검증 (폴백 금지)
    rounds_total = int(_require_key(pb_stats, "total_rounds", name="pb_stats"))
    pattern_type = str(_require_key(pattern_dict, "pattern_type", name="pattern_dict"))
    chaos_risk = float(_require_key(flow_dict, "flow_chaos_risk", name="flow_dict"))
    reversal_risk = float(_require_key(flow_dict, "flow_reversal_risk", name="flow_dict"))
    noise = float(_require_key(pattern_dict, "pattern_noise_ratio", name="pattern_dict"))
    pattern_energy = float(_require_key(pattern_dict, "pattern_energy", name="pattern_dict"))
    pattern_drift = float(_require_key(temporal, "pattern_drift", name="temporal"))
    flow_strength = float(_require_key(flow_dict, "flow_strength", name="flow_dict"))
    flow_stability = float(_require_key(flow_dict, "flow_stability", name="flow_dict"))

    adv: Dict[str, Any] = {}

    # 1) 로드 동조 스코어
    p_score = 0
    b_score = 0
    last_pb = pb_seq[-1]

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

    fd = flow_dict.get("flow_direction")
    if fd in ("P", "B"):
        _apply_vote(fd)

    # 중국점 R/B → 현재 Big Road 방향 or 반대 방향으로 투표
    if last_pb in ("P", "B"):
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

    # 2) 구간 유형 (segment_type)
    if chaos_risk >= 0.75 or noise >= 0.6:
        segment_type = "chaos"
    elif pattern_type in ("streak", "pingpong", "blocks"):
        segment_type = pattern_type
    else:
        segment_type = "mixed"
    adv["segment_type"] = segment_type

    # 3) 전환 플래그
    be_flips = _count_color_flips(road.big_eye_seq, 10, ("R", "B"))
    sm_flips = _count_color_flips(road.small_road_seq, 10, ("R", "B"))
    ck_flips = _count_color_flips(road.cockroach_seq, 10, ("R", "B"))
    flip_sum = be_flips + sm_flips + ck_flips

    transition_flag = abs(pattern_energy) >= 15.0 or pattern_drift >= 10.0 or flip_sum >= 8
    adv["transition_flag"] = transition_flag

    # 4) Mini Trend
    last6 = pb_seq[-6:]
    adv["mini_trend_p"] = last6.count("P")
    adv["mini_trend_b"] = last6.count("B")

    # 5) 중국점 방향 일치율(최근 12판)
    n = min(len(road.big_eye_seq), len(road.small_road_seq), len(road.cockroach_seq), 12)
    agree_cnt = 0
    for i in range(n):
        be = road.big_eye_seq[-n + i]
        sm = road.small_road_seq[-n + i]
        ck = road.cockroach_seq[-n + i]
        if be in ("R", "B") and be == sm == ck:
            agree_cnt += 1
    adv["china_agree_last12"] = (agree_cnt / n) if n > 0 else 0.0

    # 6) 중국점 컬럼 높이 변화
    def _height_change(m: List[List[str]]) -> int:
        heights = _column_heights(m)
        if len(heights) >= 2:
            return heights[-1] - heights[-2]
        return 0

    adv["big_eye_height_change"] = _height_change(road.big_eye_matrix)
    adv["small_road_height_change"] = _height_change(road.small_road_matrix)
    adv["cockroach_height_change"] = _height_change(road.cockroach_matrix)

    # 7) Chaos Index
    chaos_index = 0.6 * chaos_risk + 0.2 * noise + 0.2 * reversal_risk
    if rounds_total < 15:
        chaos_index *= 0.6
    chaos_index = max(0.0, min(1.0, chaos_index))
    adv["chaos_index"] = chaos_index

    # 8) pattern_score 전구간/최근구간 (소프트 캡)
    if pattern.pattern_score_history:
        hist = list(pattern.pattern_score_history)
        weights = [0.9 ** (len(hist) - 1 - i) for i in range(len(hist))]
        denom = sum(weights) if weights else 1.0
        global_raw = sum(s * w for s, w in zip(hist, weights)) / denom
        pattern_score_global = _soft_cap(global_raw)

        last10 = hist[-10:]
        last5 = hist[-5:]
        last10_mean = _soft_cap(statistics.mean(last10)) if last10 else _soft_cap(float(pattern_dict["pattern_score"]))
        last5_mean = _soft_cap(statistics.mean(last5)) if last5 else _soft_cap(float(pattern_dict["pattern_score"]))

        adv["pattern_score_global"] = float(pattern_score_global)
        adv["pattern_score_last10"] = float(last10_mean)
        adv["pattern_score_last5"] = float(last5_mean)
    else:
        base = _soft_cap(float(pattern_dict["pattern_score"]))
        adv["pattern_score_global"] = base
        adv["pattern_score_last10"] = base
        adv["pattern_score_last5"] = base

    adv["pattern_stability"] = 1.0 / (pattern_drift + 1.0)

    adv["big_eye_flips_last10"] = be_flips
    adv["small_road_flips_last10"] = sm_flips
    adv["cockroach_flips_last10"] = ck_flips

    adv["flip_cycle_pb"] = _compute_flip_cycle_pb(pb_seq)

    # 9) 전체 슈 Frame Mode & global chaos ratio
    global_chaos_ratio = _compute_global_chaos_ratio_from_scratch(pb_seq)
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
    if len(pattern.pattern_score_history) >= 20:
        first10 = pattern.pattern_score_history[:10]
        last10_hist = pattern.pattern_score_history[-10:]
        adv["frame_trend_delta"] = float(statistics.mean(last10_hist) - statistics.mean(first10))
    else:
        adv["frame_trend_delta"] = 0.0

    # 11) 중국점 상호 반응성
    response_delay_score = abs(be_flips - sm_flips) + abs(be_flips - ck_flips)
    adv["response_delay_score"] = float(response_delay_score)

    # 12) odd_run_length / odd_run_spike_flag
    current_streak = _require_key(streak_info, "current_streak", name="streak_info")
    if not isinstance(current_streak, dict):
        raise TypeError("streak_info.current_streak must be dict")
    odd_run_length = int(current_streak.get("len") or 0)
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

    # v7.6 추가 Feature
    adv["china_r_streak_be"] = _r_streak_len(road.big_eye_seq)
    adv["china_r_streak_sm"] = _r_streak_len(road.small_road_seq)
    adv["china_r_streak_ck"] = _r_streak_len(road.cockroach_seq)

    def _last_depth(m: List[List[str]]) -> int:
        heights = _column_heights(m)
        return heights[-1] if heights else 0

    adv["china_depth_be"] = _last_depth(road.big_eye_matrix)
    adv["china_depth_sm"] = _last_depth(road.small_road_matrix)
    adv["china_depth_ck"] = _last_depth(road.cockroach_matrix)

    adv["bottom_touch_bigroad"] = _bottom_touch_flag_for_matrix(road.big_road_matrix)
    adv["bottom_touch_bigeye"] = _bottom_touch_flag_for_matrix(road.big_eye_matrix)
    adv["bottom_touch_small"] = _bottom_touch_flag_for_matrix(road.small_road_matrix)
    adv["bottom_touch_cockroach"] = _bottom_touch_flag_for_matrix(road.cockroach_matrix)

    deca = _compute_decalcomania_features(pb_seq, window=6)
    adv["decalcomania_found"] = deca["decalcomania_found"]
    adv["decalcomania_hint"] = deca["decalcomania_hint"]
    adv["decalcomania_support"] = deca["decalcomania_support"]

    pb_ratio_global = _require_key(pb_stats, "pb_ratio", name="pb_stats")
    if not isinstance(pb_ratio_global, dict):
        raise TypeError("pb_stats.pb_ratio must be dict")
    p_global = float(_require_key(pb_ratio_global, "player", name="pb_stats.pb_ratio"))

    tail_pb = [x for x in pb_seq[-10:] if x in ("P", "B")]
    if tail_pb:
        p_local = tail_pb.count("P") / len(tail_pb)
        pb_diff_score = abs(p_local - p_global)
    else:
        pb_diff_score = 0.0
    adv["pb_diff_score"] = pb_diff_score

    approx_max_rounds = 72.0
    phase_progress = min(1.0, rounds_total / approx_max_rounds) if approx_max_rounds > 0 else 0.0
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

    # v9.0 Regime Shift Score (무상태)
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
