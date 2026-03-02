# -*- coding: utf-8 -*-
"""
Meta-Learning & 전략 모드 엔진 v11.0

역할:
- 패턴 + 중국점 조합(micro-learning) 승률 학습
- 슈 레짐/구간/중국점 조합(meta-learning) 승률 학습
- 신뢰도 보정 + 전략 모드 결정

변경 요약 (2025-12-24)
----------------------------------------------------
1) reverse_applied 계약 완전 제거(역배 기능 제거와 계약 동기화)
   - decide_strategy_mode()에서 bet_info.reverse_applied 요구/검증/분기 제거
   - SHADOW-REVERSAL 모드 제거(더 이상 사용하지 않음)
2) core_mode(None) 허용: GPT 분석 실패 시에도 전략 모드가 산출되도록 개선
   - core_mode가 None/빈값이면 features 기반으로 deterministic하게 파생(폴백이 아니라 1차 규칙)
   - 필수 features 키가 없으면 즉시 예외(폴백 금지 유지)
3) 기존 Strict(폴백 금지) 정책 유지
   - 필수 키 누락/타입 위반 시 즉시 예외
----------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# micro-learning: pattern_type + china_sig
micro_learning_stats: Dict[str, Dict[str, int]] = {}

# meta-learning: pattern/frame/segment/shoe_phase/regime + china_sig
meta_learning_stats: Dict[str, Dict[str, Any]] = {}


# ---------------- Strict Helpers (폴백 금지) ----------------
def _require_key(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ValueError(f"[meta_learning] missing required key: {key}")
    return d[key]


def _require_str(d: Dict[str, Any], key: str) -> str:
    v = _require_key(d, key)
    if not isinstance(v, str):
        raise TypeError(f"[meta_learning] '{key}' must be str, got {type(v).__name__}")
    if not v.strip():
        raise ValueError(f"[meta_learning] '{key}' must be non-empty string")
    return v


def _require_float(d: Dict[str, Any], key: str) -> float:
    v = _require_key(d, key)
    if isinstance(v, bool):
        raise TypeError(f"[meta_learning] '{key}' must be number, got bool")
    if not isinstance(v, (int, float)):
        raise TypeError(f"[meta_learning] '{key}' must be number, got {type(v).__name__}")
    return float(v)


def _require_int(d: Dict[str, Any], key: str) -> int:
    v = _require_key(d, key)
    if isinstance(v, bool):
        raise TypeError(f"[meta_learning] '{key}' must be int, got bool")
    if not isinstance(v, (int, float)):
        raise TypeError(f"[meta_learning] '{key}' must be int-like, got {type(v).__name__}")
    if isinstance(v, float) and not v.is_integer():
        raise ValueError(f"[meta_learning] '{key}' must be integer, got {v}")
    return int(v)


def _require_bool(d: Dict[str, Any], key: str) -> bool:
    v = _require_key(d, key)
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    raise TypeError(f"[meta_learning] '{key}' must be bool (or 0/1), got {type(v).__name__}:{v!r}")


def _validate_rb_seq(seq: List[str], name: str) -> None:
    if not isinstance(seq, list):
        raise TypeError(f"[meta_learning] {name} must be list[str]")
    for x in seq:
        if x not in ("R", "B"):
            raise ValueError(f"[meta_learning] {name} contains invalid symbol: {x!r} (only 'R'/'B')")


def _build_china_signature(big_eye: List[str], small_road: List[str], cockroach: List[str], tail: int = 8) -> str:
    """
    중국점 3종을 고정 길이 시그니처로 요약.
    - 폴백 금지: 입력이 잘못되면 즉시 예외
    - 목적: 학습 Key 안정성(짧고 deterministic)
    """
    _validate_rb_seq(big_eye, "big_eye")
    _validate_rb_seq(small_road, "small_road")
    _validate_rb_seq(cockroach, "cockroach")

    def _tail(sig: List[str]) -> str:
        t = sig[-tail:] if sig else []
        return "".join(t) if t else "_"

    be = _tail(big_eye)
    sm = _tail(small_road)
    ck = _tail(cockroach)

    return f"BE{len(big_eye)}:{be}|SM{len(small_road)}:{sm}|CK{len(cockroach)}:{ck}"


# ---------------- 학습 스냅샷 ----------------
def build_learning_snapshot(
    features: Dict[str, Any],
    big_eye: List[str],
    small_road: List[str],
    cockroach: List[str],
) -> Dict[str, Any]:
    """Meta-Learning 용 핵심 상태 스냅샷(폴백 금지: 필수 키 누락/타입 위반 즉시 예외)."""
    if not isinstance(features, dict):
        raise TypeError(f"[meta_learning] features must be dict, got {type(features).__name__}")

    pattern_type = _require_str(features, "pattern_type")
    frame_mode = _require_str(features, "frame_mode")
    segment_type = _require_str(features, "segment_type")
    shoe_phase = _require_str(features, "shoe_phase")
    shoe_regime = _require_str(features, "shoe_regime")
    chaos_index = _require_float(features, "chaos_index")
    pb_diff_score = _require_float(features, "pb_diff_score")
    bottom_touch = _require_bool(features, "bottom_touch_bigroad")

    china_sig = _build_china_signature(big_eye, small_road, cockroach)

    chaos_bucket = "low" if chaos_index < 0.4 else "mid" if chaos_index < 0.7 else "high"
    pb_bucket = "flat" if pb_diff_score < 0.08 else "biased" if pb_diff_score < 0.18 else "extreme"
    bottom_flag = "bottom" if bottom_touch else "midair"

    meta_key = (
        f"{pattern_type}|{segment_type}|{frame_mode}|"
        f"{shoe_phase}|{shoe_regime}|{chaos_bucket}|{pb_bucket}|"
        f"{bottom_flag}|{china_sig}"
    )
    micro_key = f"{pattern_type}|{china_sig}"

    return {
        "key": micro_key,
        "pattern_type": pattern_type,
        "china_sig": china_sig,
        "meta_key": meta_key,
        "frame_mode": frame_mode,
        "segment_type": segment_type,
        "shoe_phase": shoe_phase,
        "shoe_regime": shoe_regime,
    }


def _update_micro_learning(snapshot: Dict[str, Any], correct: bool) -> None:
    """micro-learning 통계 업데이트(폴백 금지: key 누락 시 즉시 예외)."""
    if not isinstance(snapshot, dict):
        raise TypeError(f"[meta_learning] snapshot must be dict, got {type(snapshot).__name__}")
    if not isinstance(correct, bool):
        raise TypeError("[meta_learning] correct must be bool")

    key = snapshot.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("[meta_learning] snapshot.key missing/invalid (폴백 금지)")

    stats = micro_learning_stats.setdefault(key, {"total": 0, "correct": 0})
    if "total" not in stats or "correct" not in stats:
        raise ValueError("[meta_learning] micro_learning_stats corrupted schema (폴백 금지)")

    stats["total"] += 1
    if correct:
        stats["correct"] += 1


def _update_meta_learning(snapshot: Dict[str, Any], correct: bool) -> None:
    """meta-learning 통계 업데이트(폴백 금지: meta_key 누락 시 즉시 예외)."""
    if not isinstance(snapshot, dict):
        raise TypeError(f"[meta_learning] snapshot must be dict, got {type(snapshot).__name__}")
    if not isinstance(correct, bool):
        raise TypeError("[meta_learning] correct must be bool")

    meta_key = snapshot.get("meta_key")
    if not isinstance(meta_key, str) or not meta_key.strip():
        raise ValueError("[meta_learning] snapshot.meta_key missing/invalid (폴백 금지)")

    stats = meta_learning_stats.setdefault(meta_key, {"total": 0, "correct": 0})
    if "total" not in stats or "correct" not in stats:
        raise ValueError("[meta_learning] meta_learning_stats corrupted schema (폴백 금지)")

    stats["total"] += 1
    if correct:
        stats["correct"] += 1


# ---------------- 신뢰도 보정 ----------------
def adjust_confidence_by_chinese_roads(
    confidence: float,
    big_eye: List[str],
    small_road: List[str],
    cockroach: List[str],
) -> Tuple[float, str]:
    """중국점 최근 R/B 조합을 기반으로 confidence 1차 보정(입력 무결성 위반 시 즉시 예외)."""
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise TypeError("[meta_learning] confidence must be number")
    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("[meta_learning] confidence must be within [0,1]")

    _validate_rb_seq(big_eye, "big_eye")
    _validate_rb_seq(small_road, "small_road")
    _validate_rb_seq(cockroach, "cockroach")

    last_be = big_eye[-1] if big_eye else None
    last_sm = small_road[-1] if small_road else None
    last_ck = cockroach[-1] if cockroach else None

    vals = [last_be, last_sm, last_ck]
    r_cnt = sum(1 for v in vals if v == "R")
    b_cnt = sum(1 for v in vals if v == "B")

    factor = 1.0
    note = ""

    if r_cnt >= 2 and b_cnt == 0:
        factor = 1.05
        note = "중국점 안정 패턴(R 우세)"
    elif b_cnt >= 2 and r_cnt == 0:
        factor = 0.95
        note = "중국점 불안정/전환 압력(B 우세)"

    new_conf = max(0.0, min(1.0, confidence * factor))
    return new_conf, note


def apply_micro_learning_boost(
    confidence: float,
    features: Dict[str, Any],
    big_eye: List[str],
    small_road: List[str],
    cockroach: List[str],
) -> Tuple[float, str]:
    """micro-learning 승률 기반 confidence 2차 보정(학습량 부족이면 보정 생략은 정상 로직)."""
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise TypeError("[meta_learning] confidence must be number")
    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("[meta_learning] confidence must be within [0,1]")
    if not isinstance(features, dict):
        raise TypeError(f"[meta_learning] features must be dict, got {type(features).__name__}")

    pattern_type = _require_str(features, "pattern_type")
    china_sig = _build_china_signature(big_eye, small_road, cockroach)
    key = f"{pattern_type}|{china_sig}"

    stats = micro_learning_stats.get(key)
    if not stats:
        return confidence, ""
    if "total" not in stats or "correct" not in stats:
        raise ValueError("[meta_learning] micro_learning_stats corrupted schema (폴백 금지)")
    if stats["total"] < 5:
        return confidence, ""

    acc = stats["correct"] / max(stats["total"], 1)
    factor = 1.0
    if acc >= 0.6:
        factor = 1.05
    elif acc <= 0.4:
        factor = 0.95

    new_conf = max(0.0, min(1.0, confidence * factor))
    if factor > 1.0:
        note = f"동일 패턴/중국점 과거 승률 {acc:.2f} (신뢰도 소폭 상승)"
    elif factor < 1.0:
        note = f"동일 패턴/중국점 과거 승률 {acc:.2f} (신뢰도 소폭 감소)"
    else:
        note = ""
    return new_conf, note


def apply_meta_learning_enrichment(
    confidence: float,
    snapshot: Dict[str, Any],
) -> Tuple[float, str, Optional[Dict[str, Any]]]:
    """meta-learning 승률 기반 confidence 3차 보정(학습량 부족이면 보정 생략은 정상 로직)."""
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise TypeError("[meta_learning] confidence must be number")
    confidence = float(confidence)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("[meta_learning] confidence must be within [0,1]")
    if not isinstance(snapshot, dict):
        raise TypeError(f"[meta_learning] snapshot must be dict, got {type(snapshot).__name__}")

    meta_key = snapshot.get("meta_key")
    if not isinstance(meta_key, str) or not meta_key.strip():
        raise ValueError("[meta_learning] snapshot.meta_key missing/invalid (폴백 금지)")

    stats = meta_learning_stats.get(meta_key)
    if not stats:
        return confidence, "", None
    if "total" not in stats or "correct" not in stats:
        raise ValueError("[meta_learning] meta_learning_stats corrupted schema (폴백 금지)")
    if stats["total"] < 10:
        return confidence, "", None

    acc = stats["correct"] / max(stats["total"], 1)
    factor = 1.0
    if acc >= 0.62:
        factor = 1.07
    elif acc >= 0.55:
        factor = 1.04
    elif acc <= 0.45:
        factor = 0.93

    new_conf = max(0.0, min(1.0, confidence * factor))
    if factor > 1.0:
        note = f"Meta-Learning 레짐 승률 {acc:.2f} (강화)"
    elif factor < 1.0:
        note = f"Meta-Learning 레짐 승률 {acc:.2f} (약화)"
    else:
        note = ""

    meta_info = {
        "meta_key": meta_key,
        "meta_total": stats["total"],
        "meta_acc": acc,
    }
    return new_conf, note, meta_info


# ---------------- 전략 모드 결정 ----------------
def _derive_core_mode_from_features(features: Dict[str, Any]) -> str:
    """
    core_mode가 None/빈값일 때 features 기반으로 deterministic하게 파생한다.
    - 폴백이 아니라, GPT 없이도 일관된 모드 산출을 위한 1차 규칙.
    - 필요한 키는 decide_strategy_mode()에서 이미 strict로 검증된다.
    """
    segment_type = _require_str(features, "segment_type")
    shoe_regime = _require_str(features, "shoe_regime")
    chaos_index = _require_float(features, "chaos_index")
    tie_turbulence = _require_int(features, "tie_turbulence_rounds")

    # 강 카오스 우선
    if chaos_index >= 0.72 or tie_turbulence > 0 or shoe_regime == "chaos_shuffle_shoe":
        return "chaos"

    # 패턴/줄 중심
    if segment_type in ("streak", "blocks"):
        return "pattern"

    # 회전/퐁당 중심
    if segment_type == "pingpong" or shoe_regime == "rotation_shoe":
        return "flow"

    return "mixed"


def decide_strategy_mode(
    core_mode: Optional[str],
    features: Dict[str, Any],
    bet_info: Dict[str, Any],
    meta_info: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    전략 모드 결정:

    - AUTO-PATTERN / AUTO-FLOW / ANTI-FLOW
    - DELAY-IN / DELAY-OUT
    - MICRO-SCALP
    - REGIME-SHIFT MODE

    폴백 금지(Strict):
    - features / bet_info 필수 계약 위반 시 즉시 예외
    - 단, core_mode는 None 허용: GPT 실패 시 features로 파생한다.
    """
    if not isinstance(features, dict):
        raise TypeError(f"[meta_learning] features must be dict, got {type(features).__name__}")
    if not isinstance(bet_info, dict):
        raise TypeError(f"[meta_learning] bet_info must be dict, got {type(bet_info).__name__}")

    # bet_info 계약(최소)
    unit = _require_int(bet_info, "unit")

    # features 계약(필수)
    segment_type = _require_str(features, "segment_type")
    shoe_regime = _require_str(features, "shoe_regime")
    chaos_index = _require_float(features, "chaos_index")
    shoe_phase = _require_str(features, "shoe_phase")
    regime_shift_score = _require_float(features, "regime_shift_score")
    tie_turbulence = _require_int(features, "tie_turbulence_rounds")
    entry_momentum = int(features.get("entry_momentum", 0))
    chaos_end_flag = _require_bool(features, "chaos_end_flag")

    # core_mode: None/빈값이면 features에서 파생
    if core_mode is None or (isinstance(core_mode, str) and not core_mode.strip()):
        core_mode = _derive_core_mode_from_features(features)
    if not isinstance(core_mode, str) or not core_mode.strip():
        raise TypeError("[meta_learning] core_mode must be str or None")

    # 기본 베이스 전략
    if core_mode == "pattern":
        strategy = "AUTO-PATTERN"
    elif core_mode == "flow":
        strategy = "AUTO-FLOW"
    elif core_mode == "chaos":
        strategy = "DELAY-OUT"
    elif core_mode == "mixed":
        strategy = "AUTO-FLOW"
    else:
        raise ValueError(f"[meta_learning] core_mode invalid: {core_mode!r}")

    # 1) 카오스/난기류가 강하면 DELAY-OUT
    if chaos_index >= 0.7 or tie_turbulence > 0 or shoe_regime == "chaos_shuffle_shoe":
        return "DELAY-OUT", "카오스/난기류 구간 – 진입 지연(DELAY-OUT)"

    # 2) Chaos-end + 진입(unit>0) → DELAY-IN
    if chaos_end_flag and unit > 0:
        return "DELAY-IN", "Chaos 종료 신호 – 소액 진입(DELAY-IN)"

    # 3) 강한 줄/블록 + 모멘텀 → AUTO-PATTERN
    if segment_type in ("streak", "blocks") and entry_momentum >= 2 and unit > 0:
        return "AUTO-PATTERN", "줄/블록 + 모멘텀 – 패턴 추종(AUTO-PATTERN)"

    # 4) 회전/퐁당 위주 슈 → AUTO-FLOW / MICRO-SCALP
    if segment_type == "pingpong" or shoe_regime == "rotation_shoe":
        if unit <= 1:
            return "MICRO-SCALP", "회전/퐁당 구간 – 소액 스캘프(MICRO-SCALP)"
        return "AUTO-FLOW", "회전/퐁당 구간 – 흐름 추종(AUTO-FLOW)"

    # 5) Regime Shift 점수가 높으면 REGIME-SHIFT MODE
    if regime_shift_score >= 0.7:
        return "REGIME-SHIFT MODE", "슈 레짐 전환 가능성↑ – 포지션 축소 / 재평가"

    # 6) Meta-Learning 승률이 좋고 단위가 작으면 MICRO-SCALP
    if meta_info is not None:
        if not isinstance(meta_info, dict):
            raise TypeError(f"[meta_learning] meta_info must be dict or None, got {type(meta_info).__name__}")
        if "meta_acc" not in meta_info:
            raise ValueError("[meta_learning] meta_info.meta_acc missing (폴백 금지)")
        acc = meta_info["meta_acc"]
        if not isinstance(acc, (int, float)) or isinstance(acc, bool):
            raise TypeError("[meta_learning] meta_info.meta_acc must be number")
        acc = float(acc)

        if unit > 0 and acc >= 0.6 and chaos_index <= 0.5 and unit <= 2:
            return "MICRO-SCALP", f"Meta 레짐 승률 {acc:.2f} – 반복 구간 스캘프"

    # 7) 후반부(late) + 혼합 구간이면 ANTI-FLOW 탐색
    if shoe_phase == "late" and unit > 0:
        if segment_type == "mixed" and chaos_index <= 0.5:
            return "ANTI-FLOW", "후반 혼합 구간 – 역방향(ANTI-FLOW) 탐색"
        return "AUTO-FLOW", "후반 – 흐름 위주(AUTO-FLOW)"

    # 기본값
    return strategy, f"기본 코어 모드({core_mode}) 기반 전략"
