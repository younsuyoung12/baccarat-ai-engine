# -*- coding: utf-8 -*-
"""
pattern.py
====================================================
Pattern Engine for Baccarat Predictor AI Engine v11.x

역할(재정의)
- pattern.py는 “방향 생성기”가 아니다.
- 오직:
  1) Big Road 구조가 ‘패턴으로 분류 가능한 상태인지’ 판정
  2) 패턴의 종류(pattern_type)와 에너지(pattern_energy)만 제공
- 베팅 방향(P/B)을 직접 또는 간접적으로 생성·암시하지 않는다.

변경 요약 (2026-01-06)
----------------------------------------------------
1) ✅ features.py 호환 계약 보강:
   - compute_pattern_features() 반환 dict에 pattern_stability(0.0~1.0)를 항상 포함
   - pattern_stability는 side-free 지표이며 symmetry/noise/run_count만 사용
   - 0~1 clamp + finite 강제(비정상 시 즉시 예외, 폴백 금지)
----------------------------------------------------

변경 요약 (2026-01-02)
----------------------------------------------------
1) 출력 계약(features.py 호환) 완전성 보장:
   - compute_pattern_features() 반환 dict에 아래 필수 키를 항상 포함
     pattern_noise_ratio (0.0~1.0)
     pattern_reversal_signal (중립값 0.0 고정)
2) 방향 암시 금지 유지:
   - noise_ratio / reversal_signal은 side-free 지표로만 제공
3) PatternNotReadyError 정책 유지:
   - 데이터 부족/구조 불가 시 예외 발생(워밍업 정상 상태, 폴백 금지)
----------------------------------------------------

변경 요약 (2026-01-01)
----------------------------------------------------
1) 입력 신뢰 기준 변경:
   - Big Road의 “논리 run(블록)” 기반 정보만 사용
   - 중국점(R/B), 스트릭 강도 수치 단독 사용 금지
2) 패턴 분류 체계 고정:
   - pattern_type ∈ {streak, pingpong, blocks, mixed, random, not_ready}
3) PatternNotReadyError 정책 강화:
   - run 개수 < 3 또는 구조 판단 불가 시 반드시 예외 발생
   - 데이터 부족은 “정상 워밍업”이며 폴백 금지
4) 출력 계약을 단일 형태로 고정:
   {
     "pattern_type": str,
     "pattern_energy": float,     # -1.0 ~ +1.0
     "pattern_score": float,      # 0 ~ 100 (NORMAL 승격 참고용)
     "pattern_symmetry": float,   # 0 ~ 1
     "pattern_noise_ratio": float,# 0 ~ 1 (구조 불안정성 지표)
     "pattern_stability": float,  # 0 ~ 1 (구조 안정성 지표, side-free)
     "pattern_reversal_signal": float, # 중립값 0.0 (방향 암시 금지)
     "pattern_tags": [str, ...]
   }
   - 방향 힌트/side 추론 정보 절대 포함 금지
----------------------------------------------------
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import road

logger = logging.getLogger(__name__)


class PatternNotReadyError(ValueError):
    """패턴 분석에 필요한 최소 데이터가 부족할 때 발생(워밍업 정상 상태, 폴백 금지)."""


# pattern_score_history는 “평활 점수(0~100)”로 유지
pattern_score_history: List[float] = []


# -----------------------------
# Helpers (side-free)
# -----------------------------
def _build_runs_from_pb(pb_seq: List[str]) -> List[Tuple[str, int]]:
    runs: List[Tuple[str, int]] = []
    if not pb_seq:
        return runs
    cur = pb_seq[0]
    ln = 1
    for v in pb_seq[1:]:
        if v == cur:
            ln += 1
        else:
            runs.append((cur, ln))
            cur = v
            ln = 1
    runs.append((cur, ln))
    return runs


def _last_n_runs(runs: List[Tuple[str, int]], n: int) -> List[Tuple[str, int]]:
    if n <= 0:
        return []
    return runs[-n:] if len(runs) > n else runs[:]


def _is_strict_alternating(runs: List[Tuple[str, int]]) -> bool:
    if len(runs) < 2:
        return False
    for i in range(1, len(runs)):
        if runs[i][0] == runs[i - 1][0]:
            return False
    return True


def _symmetry_from_run_lengths(runs: List[Tuple[str, int]], window: int = 8) -> float:
    tail = _last_n_runs(runs, window)
    if not tail:
        return 0.0
    lens = [int(ln) for _side, ln in tail]
    if len(lens) <= 1:
        return 1.0
    avg = sum(lens) / len(lens)
    var = sum((x - avg) ** 2 for x in lens) / len(lens)
    denom = (avg**2) + 1e-9
    sym = 1.0 - (var / denom)
    if sym < 0.0:
        sym = 0.0
    if sym > 1.0:
        sym = 1.0
    return float(sym)


def _classify_pattern(runs: List[Tuple[str, int]]) -> Tuple[str, List[str]]:
    """
    pattern_type ∈ {streak, pingpong, blocks, mixed, random, not_ready}
    - 방향(P/B) 값은 반환/태그에 절대 포함하지 않는다.
    """
    tags: List[str] = []

    if len(runs) < 3:
        return "not_ready", ["NOT_READY:RUNS_LT_3"]

    last4 = _last_n_runs(runs, 4)
    last6 = _last_n_runs(runs, 6)

    lens4 = [ln for _s, ln in last4]
    lens6 = [ln for _s, ln in last6]

    alt4 = _is_strict_alternating(last4) if len(last4) >= 4 else False
    alt6 = _is_strict_alternating(last6) if len(last6) >= 4 else False

    # 1) pingpong: 최근 최소 4 run, 길이 모두 1, 완전 교대
    if len(last4) >= 4 and alt4 and all(ln == 1 for ln in lens4):
        return "pingpong", ["TYPE:pingpong", "RULE:ALT4_LEN1"]

    # 2) blocks: 최근 2개 이상 run이 서로 다른 side이고 length>=2
    if len(runs) >= 2:
        a_side, a_len = runs[-2]
        b_side, b_len = runs[-1]
        if a_side != b_side and a_len >= 2 and b_len >= 2:
            return "blocks", ["TYPE:blocks", "RULE:LAST2_GE2"]

    # 3) mixed: 길이 1과 >=2 혼재 + 교대가 반복(최근 6 기준)
    has_single = any(ln == 1 for ln in lens6)
    has_block = any(ln >= 2 for ln in lens6)
    if has_single and has_block and alt6:
        return "mixed", ["TYPE:mixed", "RULE:ALT6_MIXEDLEN"]

    # 4) streak: 동일 side run 길이 >=4 (다른 타입과 충돌 없을 때만)
    max_len = max(lens6) if lens6 else 0
    if max_len >= 4:
        return "streak", ["TYPE:streak", "RULE:MAXLEN_GE4"]

    # 5) random
    return "random", ["TYPE:random"]


def _score_from_type_and_symmetry(pattern_type: str, symmetry: float, runs: List[Tuple[str, int]]) -> float:
    """
    pattern_score: 0~100 (NORMAL 승격 참고용)
    - 방향 정보 미사용
    - 과대평가 방지: random / not_ready는 상한 강제
    """
    base_map = {
        "streak": 72.0,
        "pingpong": 70.0,
        "blocks": 66.0,
        "mixed": 56.0,
        "random": 40.0,
        "not_ready": 0.0,
    }
    base = float(base_map.get(pattern_type, 40.0))

    # 구조 신뢰(근거) 스케일: run 개수가 늘수록 점수 반영 폭 확대
    confidence = min(1.0, max(0.0, len(runs) / 12.0))

    # 대칭성 가중(0~1)
    sym_boost = 18.0 * symmetry

    # 최근 run 길이 안정성(방향 미사용)
    tail = _last_n_runs(runs, 6)
    lens = [ln for _s, ln in tail]
    avg = sum(lens) / len(lens) if lens else 0.0
    var = sum((x - avg) ** 2 for x in lens) / len(lens) if lens else 0.0
    denom = (avg**2) + 1e-9
    var_norm = var / denom if denom > 0 else 0.0
    stability = max(0.0, 1.0 - min(1.0, var_norm))  # 0~1
    stab_boost = 10.0 * stability

    raw = base + sym_boost + stab_boost

    # confidence로 초기 과대평가 억제
    score = base + (raw - base) * confidence

    # 상한/하한
    if pattern_type == "random":
        score = min(score, 45.0)
    if pattern_type == "not_ready":
        score = 0.0

    if score < 0.0:
        score = 0.0
    if score > 100.0:
        score = 100.0
    return float(score)


def _energy_from_score_delta(raw_score: float, prev_score: float) -> float:
    """
    -1.0 ~ +1.0
    점수 변화량 기반.
    """
    delta = raw_score - prev_score
    e = delta / 100.0
    if e < -1.0:
        e = -1.0
    if e > 1.0:
        e = 1.0
    return float(e)


# -----------------------------
# Public API
# -----------------------------
def compute_pattern_features(pb_seq: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    출력 계약(고정):
    {
      "pattern_type": str,
      "pattern_energy": float,          # -1.0 ~ +1.0
      "pattern_score": float,           # 0~100
      "pattern_symmetry": float,        # 0~1
      "pattern_noise_ratio": float,     # 0~1 (구조 불안정성)
      "pattern_stability": float,       # 0~1 (구조 안정성, side-free)
      "pattern_reversal_signal": float, # 중립값 0.0 (방향 암시 금지)
      "pattern_tags": [str, ...]
    }
    """
    global pattern_score_history

    # 입력 확보
    if pb_seq is None:
        pb_seq = road.get_pb_sequence()
    if not pb_seq:
        msg = "PatternNotReady: pb_seq empty"
        logger.info(msg)
        raise PatternNotReadyError(msg)

    # run 구성(road 메타 우선, 없으면 자체 구성)
    try:
        runs = road.get_run_sequence(pb_seq)
    except Exception:
        runs = _build_runs_from_pb(pb_seq)

    if not runs or len(runs) < 3:
        msg = f"PatternNotReady: runs insufficient: {0 if not runs else len(runs)} < 3"
        logger.info(msg)
        raise PatternNotReadyError(msg)

    # 분류
    pattern_type, type_tags = _classify_pattern(runs)
    if pattern_type == "not_ready":
        msg = "PatternNotReady: classifier not_ready"
        logger.info(msg)
        raise PatternNotReadyError(msg)

    # 대칭성/스코어
    symmetry = _symmetry_from_run_lengths(runs, window=8)
    raw_score = _score_from_type_and_symmetry(pattern_type, symmetry, runs)

    # 점수 평활(0.3 new + 0.7 prev)
    if pattern_score_history:
        prev_score = float(pattern_score_history[-1])
        score = 0.3 * raw_score + 0.7 * prev_score
    else:
        prev_score = raw_score
        score = raw_score

    # 에너지
    energy = _energy_from_score_delta(raw_score, prev_score)

    # noise ratio (0~1): run 길이 분산 기반 구조 불안정성 (side-free)
    tail = _last_n_runs(runs, 6)
    lens = [ln for _s, ln in tail]
    avg = sum(lens) / len(lens) if lens else 0.0
    var = sum((x - avg) ** 2 for x in lens) / len(lens) if lens else 0.0
    denom = (avg**2) + 1e-9
    var_norm = var / denom if denom > 0 else 0.0
    if var_norm < 0.0:
        var_norm = 0.0
    if var_norm > 1.0:
        var_norm = 1.0
    pattern_noise_ratio = float(var_norm)

    # -----------------------------
    # pattern_stability (0~1, side-free)
    # -----------------------------
    # symmetry/noise/run_count 기반(방향 암시 금지)
    run_confidence = min(1.0, max(0.0, len(runs) / 12.0))
    raw_stability = (symmetry**0.8) * ((1.0 - pattern_noise_ratio) ** 0.9)
    pattern_stability = raw_stability * run_confidence

    if not math.isfinite(pattern_stability):
        raise ValueError("pattern_stability must be finite")
    if pattern_stability < 0.0:
        pattern_stability = 0.0
    if pattern_stability > 1.0:
        pattern_stability = 1.0
    pattern_stability = float(pattern_stability)

    # reversal signal (side-free): 중립값 고정 (방향 암시 금지)
    pattern_reversal_signal = 0.0

    # history 갱신(ready 통과 후만)
    pattern_score_history.append(float(score))
    if len(pattern_score_history) > 200:
        pattern_score_history = pattern_score_history[-200:]

    # tags (방향 암시 금지: side/last winner/run who 포함 금지)
    tags: List[str] = []
    tags.extend(type_tags)
    tags.append(f"RUNS={len(runs)}")
    tags.append(f"SYM={symmetry:.3f}")
    tags.append(f"SCORE_RAW={raw_score:.2f}")
    tags.append(f"SCORE_SMOOTH={score:.2f}")
    tags.append(f"ENERGY={energy:.3f}")
    tags.append(f"NOISE={pattern_noise_ratio:.3f}")
    tags.append(f"STAB={pattern_stability:.3f}")
    tags.append("REV=0.0")

    # road 구조 메타(방향 정보 없이 구조명만)
    try:
        struct = road.get_recent_structure(pb_seq)
        if struct:
            tags.append(f"BR_STRUCT={str(struct).lower()}")
    except Exception:
        pass

    # 반환 계약(고정)
    return {
        "pattern_type": str(pattern_type),
        "pattern_energy": float(energy),
        "pattern_score": float(score),
        "pattern_symmetry": float(symmetry),
        "pattern_noise_ratio": float(pattern_noise_ratio),
        "pattern_stability": float(pattern_stability),
        "pattern_reversal_signal": float(pattern_reversal_signal),
        "pattern_tags": tags,
    }
