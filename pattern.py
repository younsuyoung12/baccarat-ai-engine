# -*- coding: utf-8 -*-
"""
pattern.py
====================================================
Pattern Engine for Baccarat Predictor AI Engine v12.1
(RULE-ONLY · STRICT · NO-FALLBACK · FAIL-FAST)

역할
- pattern.py는 방향 생성기가 아니다.
- 오직:
  1) Big Road 구조가 패턴으로 분류 가능한 상태인지 판정
  2) 패턴 종류(pattern_type)와 에너지(pattern_energy) 계산
  3) 구조 안정성/대칭성/노이즈 같은 side-free 지표 계산
- 베팅 방향(P/B)을 직접 또는 간접적으로 생성·암시하지 않는다.

변경 요약 (2026-03-14)
----------------------------------------------------
1) STRICT 위반 제거
   - road.get_run_sequence() 예외 시 자체 계산으로 넘어가던 fallback 제거
   - road.get_recent_structure() 예외 무시(silent pass) 제거
2) 입력 계약 강화
   - pb_seq는 반드시 list[str] 이고 각 원소는 P/B 여야 한다
   - pattern_score_history는 finite 0~100 float만 허용
3) 출력 계약 유지
   - compute_pattern_features() 반환 스키마 유지
4) side-free 유지
   - 방향 힌트/side 추론 정보 반환 금지
----------------------------------------------------
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import road

logger = logging.getLogger(__name__)

VALID_PB = ("P", "B")


class PatternNotReadyError(ValueError):
    """패턴 분석에 필요한 최소 데이터가 부족할 때 발생(워밍업 정상 상태, 폴백 금지)."""


# pattern_score_history는 “평활 점수(0~100)”로 유지
pattern_score_history: List[float] = []


# -----------------------------
# Strict helpers
# -----------------------------
def _require_list(v: Any, name: str) -> List[Any]:
    if not isinstance(v, list):
        raise TypeError(f"{name} must be list, got {type(v).__name__}")
    return v


def _require_score_0_100(v: Any, *, name: str) -> float:
    if isinstance(v, bool):
        raise TypeError(f"{name} must be float, got bool")
    if not isinstance(v, (int, float)):
        raise TypeError(f"{name} must be float, got {type(v).__name__}")
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    if x < 0.0 or x > 100.0:
        raise ValueError(f"{name} must be in [0,100], got {x}")
    return x


def _clamp_01(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError("value must be finite")
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def _clamp_m11(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError("value must be finite")
    if v < -1.0:
        return -1.0
    if v > 1.0:
        return 1.0
    return float(v)


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


def _validate_pattern_score_history() -> List[float]:
    global pattern_score_history
    raw = _require_list(pattern_score_history, "pattern_score_history")
    validated: List[float] = []
    for i, v in enumerate(raw):
        validated.append(_require_score_0_100(v, name=f"pattern_score_history[{i}]"))
    return validated


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
    return _clamp_01(sym)


def _classify_pattern(runs: List[Tuple[str, int]]) -> Tuple[str, List[str]]:
    """
    pattern_type ∈ {streak, pingpong, blocks, mixed, random, not_ready}
    - 방향(P/B) 값은 반환/태그에 절대 포함하지 않는다.
    """
    if len(runs) < 3:
        return "not_ready", ["NOT_READY:RUNS_LT_3"]

    last4 = _last_n_runs(runs, 4)
    last6 = _last_n_runs(runs, 6)

    lens4 = [ln for _s, ln in last4]
    lens6 = [ln for _s, ln in last6]

    alt4 = _is_strict_alternating(last4) if len(last4) >= 4 else False
    alt6 = _is_strict_alternating(last6) if len(last6) >= 4 else False

    # 1) pingpong
    if len(last4) >= 4 and alt4 and all(ln == 1 for ln in lens4):
        return "pingpong", ["TYPE:pingpong", "RULE:ALT4_LEN1"]

    # 2) blocks
    if len(runs) >= 2:
        a_side, a_len = runs[-2]
        b_side, b_len = runs[-1]
        if a_side != b_side and a_len >= 2 and b_len >= 2:
            return "blocks", ["TYPE:blocks", "RULE:LAST2_GE2"]

    # 3) mixed
    has_single = any(ln == 1 for ln in lens6)
    has_block = any(ln >= 2 for ln in lens6)
    if has_single and has_block and alt6:
        return "mixed", ["TYPE:mixed", "RULE:ALT6_MIXEDLEN"]

    # 4) streak
    max_len = max(lens6) if lens6 else 0
    if max_len >= 4:
        return "streak", ["TYPE:streak", "RULE:MAXLEN_GE4"]

    # 5) random
    return "random", ["TYPE:random"]


def _score_from_type_and_symmetry(pattern_type: str, symmetry: float, runs: List[Tuple[str, int]]) -> float:
    """
    pattern_score: 0~100
    - 방향 정보 미사용
    - random / not_ready는 상한 강제
    """
    base_map = {
        "streak": 72.0,
        "pingpong": 70.0,
        "blocks": 66.0,
        "mixed": 56.0,
        "random": 40.0,
        "not_ready": 0.0,
    }
    if pattern_type not in base_map:
        raise ValueError(f"invalid pattern_type: {pattern_type!r}")

    base = float(base_map[pattern_type])

    confidence = _clamp_01(len(runs) / 12.0)
    sym_boost = 18.0 * _clamp_01(symmetry)

    tail = _last_n_runs(runs, 6)
    lens = [ln for _s, ln in tail]
    avg = sum(lens) / len(lens) if lens else 0.0
    var = sum((x - avg) ** 2 for x in lens) / len(lens) if lens else 0.0
    denom = (avg**2) + 1e-9
    var_norm = var / denom if denom > 0 else 0.0
    stability = max(0.0, 1.0 - min(1.0, var_norm))
    stab_boost = 10.0 * stability

    raw = base + sym_boost + stab_boost
    score = base + (raw - base) * confidence

    if pattern_type == "random":
        score = min(score, 45.0)
    if pattern_type == "not_ready":
        score = 0.0

    return _require_score_0_100(score, name="pattern_score")


def _energy_from_score_delta(raw_score: float, prev_score: float) -> float:
    """
    -1.0 ~ +1.0
    점수 변화량 기반.
    """
    delta = raw_score - prev_score
    e = delta / 100.0
    return _clamp_m11(e)


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
      "pattern_noise_ratio": float,     # 0~1
      "pattern_stability": float,       # 0~1
      "pattern_reversal_signal": float, # 0.0 고정
      "pattern_tags": [str, ...]
    }
    """
    global pattern_score_history

    if pb_seq is None:
        pb_seq = road.get_pb_sequence()

    pb_seq_v = _validate_pb_seq(pb_seq, name="pb_seq")
    if not pb_seq_v:
        msg = "PatternNotReady: pb_seq empty"
        logger.info(msg)
        raise PatternNotReadyError(msg)

    runs = _build_runs_from_pb(pb_seq_v)
    if len(runs) < 3:
        msg = f"PatternNotReady: runs insufficient: {len(runs)} < 3"
        logger.info(msg)
        raise PatternNotReadyError(msg)

    pattern_type, type_tags = _classify_pattern(runs)
    if pattern_type == "not_ready":
        msg = "PatternNotReady: classifier not_ready"
        logger.info(msg)
        raise PatternNotReadyError(msg)

    symmetry = _symmetry_from_run_lengths(runs, window=8)
    raw_score = _score_from_type_and_symmetry(pattern_type, symmetry, runs)

    history = _validate_pattern_score_history()
    if history:
        prev_score = history[-1]
        score = (0.3 * raw_score) + (0.7 * prev_score)
    else:
        prev_score = raw_score
        score = raw_score

    score = _require_score_0_100(score, name="smoothed_pattern_score")
    energy = _energy_from_score_delta(raw_score, prev_score)

    tail = _last_n_runs(runs, 6)
    lens = [ln for _s, ln in tail]
    avg = sum(lens) / len(lens) if lens else 0.0
    var = sum((x - avg) ** 2 for x in lens) / len(lens) if lens else 0.0
    denom = (avg**2) + 1e-9
    var_norm = var / denom if denom > 0 else 0.0
    pattern_noise_ratio = _clamp_01(var_norm)

    run_confidence = _clamp_01(len(runs) / 12.0)
    raw_stability = (symmetry**0.8) * ((1.0 - pattern_noise_ratio) ** 0.9)
    pattern_stability = _clamp_01(raw_stability * run_confidence)

    pattern_reversal_signal = 0.0

    pattern_score_history.append(float(score))
    if len(pattern_score_history) > 200:
        pattern_score_history = pattern_score_history[-200:]

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

    struct = road.get_recent_structure(pb_seq_v)
    if not isinstance(struct, str) or not struct.strip():
        raise RuntimeError("road.get_recent_structure() must return non-empty str")
    tags.append(f"BR_STRUCT={struct.strip().lower()}")

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