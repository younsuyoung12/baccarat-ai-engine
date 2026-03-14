# -*- coding: utf-8 -*-
"""
flow.py
====================================================
Flow Lifecycle Manager for Baccarat Predictor AI Engine v12.1
(RULE-ONLY · STRICT · NO-FALLBACK · FAIL-FAST)

역할
- flow.py는 방향 생성기가 아니다.
- 오직 흐름 생명 주기(flow_state: DEAD/TEST/ALIVE)만 관리한다.
- 방향(P/B), PASS/PROBE/NORMAL 결정, unit 계산을 절대 수행하지 않는다.

출력 계약(고정)
{
  "flow_state": "DEAD|TEST|ALIVE",
  "flow_reason": str,
  "flow_confidence": float,     # 0~1
  "flow_strength": float,       # 0~1 (흐름 강도: 상태/신뢰 기반, side-free)
  "flow_stability": float,      # 0~1 (흐름 안정성: 구조/에너지 기반, side-free)
  "flow_chaos_risk": float,     # 0~1 (혼돈/불안정 리스크, side-free)
  "flow_reversal_risk": float,  # 0~1 (반전/흐름 붕괴 리스크, side-free)
  "flow_direction": str,        # "neutral" (방향 암시 금지)
  "flow_tags": [str, ...]
}

변경 요약 (2026-03-14)
----------------------------------------------------
1) v12 호출 구조 정합성 수정
   - features.py가 pattern_type/pattern_energy/outcome을 주입하지 않아도
     road 구조 메타만으로 deterministic 하게 상태 계산 가능하도록 수정
2) STRICT 계약 강화
   - big_eye/small_road/cockroach 입력 시퀀스 검증
   - streak_info 계약 검증
   - road.get_pb_sequence()/get_structure_meta() 반환 계약 검증
3) 방향 암시 금지 유지
   - flow_direction은 "neutral" 고정
   - 모든 수치는 side-free 구조 지표
4) 예외 정책 정리
   - 계약 위반/상태 오염만 예외
   - 데이터 부족/워밍업은 결정론적 DEAD/중립 수치 반환
----------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import road


# -----------------------------
# States / constants
# -----------------------------
FLOW_DEAD = "DEAD"
FLOW_TEST = "TEST"
FLOW_ALIVE = "ALIVE"

ENTRY_PROBE = "PROBE"
ENTRY_NORMAL = "NORMAL"

CHINA_ALIVE = "ALIVE"
CHINA_WEAK = "WEAK"
CHINA_BROKEN = "BROKEN"
CHINA_UNKNOWN = "UNKNOWN"

FLOW_DIRECTION_NEUTRAL = "neutral"

MIN_PB_FOR_ALIVE = 10

VALID_PB = ("P", "B")
VALID_RB = ("R", "B")
VALID_STRUCTURES = ("pingpong", "blocks", "mixed", "streak", "random")
VALID_PATTERN_TYPES = ("streak", "pingpong", "blocks", "mixed", "random")


# -----------------------------
# Context (module-local)
# -----------------------------
@dataclass
class FlowContext:
    state: str = FLOW_DEAD
    probe_hit_count: int = 0

    last_seen_pb_len: int = 0
    last_seen_shoe_id: Optional[str] = None

    last_bigroad_structure: str = "random"
    last_pattern_type: Optional[str] = None
    last_pattern_energy: float = 0.0
    last_china_state: str = CHINA_UNKNOWN

    def reset(self) -> None:
        self.state = FLOW_DEAD
        self.probe_hit_count = 0
        self.last_bigroad_structure = "random"
        self.last_pattern_type = None
        self.last_pattern_energy = 0.0
        self.last_china_state = CHINA_UNKNOWN


_FLOW_CTX = FlowContext()


# -----------------------------
# Strict helpers
# -----------------------------
def _require_dict(v: Any, name: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        raise TypeError(f"{name} must be dict, got {type(v).__name__}")
    return v


def _require_list(v: Any, name: str) -> List[Any]:
    if not isinstance(v, list):
        raise TypeError(f"{name} must be list, got {type(v).__name__}")
    return v


def _require_key(d: Dict[str, Any], key: str, *, name: str) -> Any:
    if not isinstance(d, dict):
        raise TypeError(f"{name} must be dict, got {type(d).__name__}")
    if key not in d:
        raise KeyError(f"{name} missing required key: {key}")
    return d[key]


def _require_int(v: Any, name: str) -> int:
    if isinstance(v, bool):
        raise TypeError(f"{name} must be int, got bool")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    raise TypeError(f"{name} must be int, got {type(v).__name__}")


def _require_float(v: Any, name: str) -> float:
    if isinstance(v, bool):
        raise TypeError(f"{name} must be float, got bool")
    if not isinstance(v, (int, float)):
        raise TypeError(f"{name} must be float, got {type(v).__name__}")
    x = float(v)
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return x


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _clamp_m11(x: float) -> float:
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return float(x)


def _normalize_optional_str(v: Any, name: str) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        raise TypeError(f"{name} must be str or None, got {type(v).__name__}")
    s = v.strip()
    return s if s else None


def _normalize_optional_bool(v: Any, name: str) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if float(v) in (0.0, 1.0):
            return bool(int(v))
        raise ValueError(f"{name} numeric value must be 0/1, got {v}")
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
        raise ValueError(f"{name} invalid bool string: {v!r}")
    raise TypeError(f"{name} must be bool-compatible or None, got {type(v).__name__}")


def _validate_pb_seq(seq: Any, name: str) -> List[str]:
    raw = _require_list(seq, name)
    out: List[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise TypeError(f"{name}[{i}] must be str, got {type(item).__name__}")
        s = item.strip().upper()
        if s not in VALID_PB:
            raise ValueError(f"{name}[{i}] invalid: {item!r} (allowed: {VALID_PB})")
        out.append(s)
    return out


def _validate_rb_seq(seq: Any, name: str) -> List[str]:
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


def _normalize_pattern_type(v: Any, name: str) -> Optional[str]:
    s = _normalize_optional_str(v, name)
    if s is None:
        return None
    s = s.lower()
    if s not in VALID_PATTERN_TYPES:
        raise ValueError(f"{name} invalid: {v!r} (allowed: {VALID_PATTERN_TYPES})")
    return s


def _normalize_pattern_energy(v: Any, name: str) -> Optional[float]:
    if v is None:
        return None
    x = _require_float(v, name)
    return _clamp_m11(x)


def _extract_current_streak_len(streak_info: Dict[str, Any]) -> int:
    current = streak_info.get("current_streak")
    if current is None:
        return 0
    current = _require_dict(current, "streak_info.current_streak")

    if "len" in current:
        ln = _require_int(current["len"], "streak_info.current_streak.len")
    elif "length" in current:
        ln = _require_int(current["length"], "streak_info.current_streak.length")
    else:
        raise KeyError("streak_info.current_streak missing len/length")

    if ln < 0:
        raise ValueError(f"current_streak length must be >= 0, got {ln}")
    return ln


# -----------------------------
# Extraction helpers
# -----------------------------
def _extract_shoe_id(streak_info: Dict[str, Any]) -> Optional[str]:
    if "shoe_id" in streak_info:
        return _normalize_optional_str(streak_info.get("shoe_id"), "streak_info.shoe_id")
    if "shoe" in streak_info:
        return _normalize_optional_str(streak_info.get("shoe"), "streak_info.shoe")
    return None


def _extract_outcome_info(streak_info: Dict[str, Any]) -> Tuple[Optional[str], Optional[bool], List[str]]:
    """
    상위(recommend)가 전달할 수 있는 직전 결과 정보.
    - last_entry_type: PROBE|NORMAL
    - last_hit: bool
    """
    tags: List[str] = []

    entry_raw = None
    if "last_entry_type" in streak_info:
        entry_raw = streak_info.get("last_entry_type")
    elif "prev_entry_type" in streak_info:
        entry_raw = streak_info.get("prev_entry_type")

    entry = _normalize_optional_str(entry_raw, "streak_info.last_entry_type")
    if entry is not None:
        entry = entry.upper()
        if entry not in (ENTRY_PROBE, ENTRY_NORMAL):
            raise ValueError(f"invalid entry type: {entry!r}")

    hit_raw = None
    if "last_hit" in streak_info:
        hit_raw = streak_info.get("last_hit")
    elif "prev_hit" in streak_info:
        hit_raw = streak_info.get("prev_hit")
    hit = _normalize_optional_bool(hit_raw, "streak_info.last_hit")

    if entry is None:
        tags.append("OUTCOME=none")
    else:
        tags.append(f"OUTCOME_ENTRY={entry}")

    if hit is None:
        tags.append("OUTCOME_HIT=unknown")
    else:
        tags.append(f"OUTCOME_HIT={int(hit)}")

    return entry, hit, tags


def _china_state_from_last_marks(
    big_eye: List[str],
    small_road: List[str],
    cockroach: List[str],
) -> Tuple[str, Dict[str, Optional[str]], int]:
    """
    중국점은 색(R/B) 자체를 해석하지 않는다.
    - 마지막 mark를 기반으로 상태 요약만 생성
    """
    last_be = big_eye[-1] if big_eye else None
    last_sr = small_road[-1] if small_road else None
    last_cr = cockroach[-1] if cockroach else None

    marks = {
        "big_eye_last": last_be,
        "small_road_last": last_sr,
        "cockroach_last": last_cr,
    }

    observed = sum(1 for v in marks.values() if v in VALID_RB)
    if observed <= 0:
        return CHINA_UNKNOWN, marks, 0

    bcnt = sum(1 for v in marks.values() if v == "B")
    if bcnt >= 2:
        return CHINA_BROKEN, marks, bcnt
    if bcnt == 1:
        return CHINA_WEAK, marks, bcnt
    return CHINA_ALIVE, marks, bcnt


def _pattern_gate_ok(pattern_type: Optional[str]) -> bool:
    return pattern_type in ("streak", "pingpong", "blocks")


def _picture_present_from_bigroad(structure: str) -> bool:
    return structure in ("pingpong", "blocks", "mixed", "streak")


def _derive_pattern_info_from_structure(
    structure: str,
    current_run_len: int,
    china_state: str,
) -> Tuple[str, float]:
    """
    외부 pattern 주입이 없을 때 road 구조만으로 side-free 패턴 정보를 결정론적으로 생성한다.
    """
    if structure not in VALID_STRUCTURES:
        raise ValueError(f"invalid structure: {structure!r}")

    if structure == "streak":
        energy = 0.25 + (0.10 * min(max(current_run_len - 2, 0), 4))
    elif structure == "blocks":
        energy = 0.22 + (0.06 * min(max(current_run_len - 1, 0), 4))
    elif structure == "pingpong":
        energy = 0.30
    elif structure == "mixed":
        energy = -0.08
    else:  # random
        energy = -0.25

    if china_state == CHINA_ALIVE:
        energy += 0.05
    elif china_state == CHINA_WEAK:
        energy -= 0.05
    elif china_state == CHINA_UNKNOWN:
        energy -= 0.08
    elif china_state == CHINA_BROKEN:
        energy -= 0.15

    return structure, _clamp_m11(energy)


def _extract_pattern_info(
    streak_info: Dict[str, Any],
    bigroad_structure: str,
    china_state: str,
) -> Tuple[str, float, List[str]]:
    """
    우선순위
    1) streak_info.pattern_dict.{pattern_type,pattern_energy}
    2) streak_info.{pattern_type,pattern_energy}
    3) road structure 기반 deterministic derive
    """
    tags: List[str] = []

    current_run_len = _extract_current_streak_len(streak_info)

    pattern_type: Optional[str] = None
    pattern_energy: Optional[float] = None

    if "pattern_dict" in streak_info:
        pd = _require_dict(streak_info.get("pattern_dict"), "streak_info.pattern_dict")
        if "pattern_type" in pd:
            pattern_type = _normalize_pattern_type(pd.get("pattern_type"), "streak_info.pattern_dict.pattern_type")
        if "pattern_energy" in pd:
            pattern_energy = _normalize_pattern_energy(pd.get("pattern_energy"), "streak_info.pattern_dict.pattern_energy")

    if pattern_type is None and "pattern_type" in streak_info:
        pattern_type = _normalize_pattern_type(streak_info.get("pattern_type"), "streak_info.pattern_type")
    if pattern_energy is None and "pattern_energy" in streak_info:
        pattern_energy = _normalize_pattern_energy(streak_info.get("pattern_energy"), "streak_info.pattern_energy")

    if pattern_type is None or pattern_energy is None:
        derived_type, derived_energy = _derive_pattern_info_from_structure(
            structure=bigroad_structure,
            current_run_len=current_run_len,
            china_state=china_state,
        )
        if pattern_type is None:
            pattern_type = derived_type
        if pattern_energy is None:
            pattern_energy = derived_energy
        tags.append("PATTERN_SRC=DERIVED")
    else:
        tags.append("PATTERN_SRC=INJECTED")

    tags.append(f"PTYPE={pattern_type}")
    tags.append(f"PENERGY={pattern_energy:.3f}")
    return pattern_type, float(pattern_energy), tags


# -----------------------------
# Side-free metrics
# -----------------------------
def _compute_confidence(
    state: str,
    picture_present: bool,
    pattern_ok: bool,
    pattern_energy: float,
    china_state: str,
    probe_hit_count: int,
    pb_len: int,
) -> float:
    if state == FLOW_DEAD:
        base = 0.20
    elif state == FLOW_TEST:
        base = 0.58
    elif state == FLOW_ALIVE:
        base = 0.80
    else:
        raise RuntimeError(f"invalid flow state: {state!r}")

    adj = 0.0
    if picture_present:
        adj += 0.05
    if pattern_ok:
        adj += 0.06
    if probe_hit_count >= 1:
        adj += 0.05

    if pattern_energy > 0.0:
        adj += 0.06 * min(1.0, pattern_energy)
    elif pattern_energy < 0.0:
        adj -= 0.08 * min(1.0, abs(pattern_energy))

    if china_state == CHINA_ALIVE:
        adj += 0.04
    elif china_state == CHINA_WEAK:
        adj -= 0.04
    elif china_state == CHINA_UNKNOWN:
        adj -= 0.07
    elif china_state == CHINA_BROKEN:
        adj -= 0.12

    if pb_len < MIN_PB_FOR_ALIVE:
        adj -= 0.05

    return _clamp01(base + adj)


def _compute_strength(state: str, confidence: float) -> float:
    conf = _clamp01(confidence)
    if state == FLOW_DEAD:
        return 0.0
    if state == FLOW_TEST:
        return _clamp01(0.30 + (0.22 * conf))
    if state == FLOW_ALIVE:
        return _clamp01(0.62 + (0.38 * conf))
    raise RuntimeError(f"invalid flow state: {state!r}")


def _compute_stability(
    state: str,
    bigroad_structure: str,
    pattern_type: str,
    pattern_energy: float,
    china_state: str,
    probe_hit_count: int,
    pb_len: int,
) -> float:
    if state == FLOW_DEAD:
        return 0.0

    base_map = {
        "pingpong": 0.72,
        "blocks": 0.68,
        "streak": 0.70,
        "mixed": 0.54,
        "random": 0.38,
    }
    if bigroad_structure not in base_map:
        raise RuntimeError(f"invalid bigroad_structure: {bigroad_structure!r}")

    base = float(base_map[bigroad_structure])

    if _pattern_gate_ok(pattern_type):
        base += 0.05
    elif pattern_type == "mixed":
        base -= 0.03
    elif pattern_type == "random":
        base -= 0.08

    base -= 0.15 * min(1.0, abs(pattern_energy))

    if probe_hit_count >= 1:
        base += 0.05

    if china_state == CHINA_UNKNOWN:
        base -= 0.08
    elif china_state == CHINA_WEAK:
        base -= 0.04
    elif china_state == CHINA_BROKEN:
        base -= 0.16

    if pb_len < MIN_PB_FOR_ALIVE:
        base -= 0.05

    return _clamp01(base)


def _compute_chaos_risk(stability: float, bigroad_structure: str, china_state: str) -> float:
    risk = 1.0 - _clamp01(stability)

    if bigroad_structure == "mixed":
        risk += 0.06
    elif bigroad_structure == "random":
        risk += 0.16

    if china_state == CHINA_UNKNOWN:
        risk += 0.10
    elif china_state == CHINA_WEAK:
        risk += 0.06
    elif china_state == CHINA_BROKEN:
        risk += 0.20

    return _clamp01(risk)


def _compute_reversal_risk(
    state: str,
    pattern_energy: float,
    bigroad_structure: str,
    china_state: str,
    pb_len: int,
) -> float:
    if state == FLOW_DEAD:
        base = 0.52
    elif state == FLOW_TEST:
        base = 0.44
    elif state == FLOW_ALIVE:
        base = 0.34
    else:
        raise RuntimeError(f"invalid flow state: {state!r}")

    e = _clamp_m11(pattern_energy)
    if e < 0.0:
        base += 0.25 * abs(e)
    elif e > 0.0:
        base -= 0.15 * e

    if china_state == CHINA_WEAK:
        base += 0.12
    elif china_state == CHINA_UNKNOWN:
        base += 0.08
    elif china_state == CHINA_BROKEN:
        base += 0.25

    if bigroad_structure == "mixed":
        base += 0.06
    elif bigroad_structure == "random":
        base += 0.10

    if pb_len < MIN_PB_FOR_ALIVE:
        base += 0.05

    return _clamp01(base)


# -----------------------------
# Core transition logic
# -----------------------------
def _detect_shoe_reset(pb_len: int, shoe_id: Optional[str]) -> Optional[str]:
    if shoe_id and _FLOW_CTX.last_seen_shoe_id and shoe_id != _FLOW_CTX.last_seen_shoe_id:
        return "shoe_id_changed"
    if pb_len < _FLOW_CTX.last_seen_pb_len:
        return "pb_len_decreased"
    return None


def _transition(
    pb_len: int,
    shoe_reset: Optional[str],
    picture_present: bool,
    bigroad_structure: str,
    pattern_type: str,
    pattern_energy: float,
    china_state: str,
    last_entry_type: Optional[str],
    last_hit: Optional[bool],
) -> Dict[str, Any]:
    tags: List[str] = []
    reason = ""

    _FLOW_CTX.last_bigroad_structure = bigroad_structure
    _FLOW_CTX.last_pattern_type = pattern_type
    _FLOW_CTX.last_pattern_energy = pattern_energy
    _FLOW_CTX.last_china_state = china_state

    tags.append(f"BR_STRUCT={bigroad_structure}")
    tags.append(f"CHINA={china_state}")
    tags.append(f"PB_LEN={pb_len}")
    tags.append(f"PTYPE={pattern_type}")
    tags.append(f"PENERGY={pattern_energy:.3f}")

    if shoe_reset:
        _FLOW_CTX.reset()
        reason = f"SHOE_RESET({shoe_reset})"
        tags.append("RESET")
        return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

    if china_state == CHINA_BROKEN:
        _FLOW_CTX.reset()
        reason = "CHINA_BROKEN"
        tags.append("HARD_DEAD")
        return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

    if not picture_present:
        _FLOW_CTX.reset()
        reason = "NO_PICTURE"
        tags.append("HARD_DEAD")
        return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

    pattern_ok = _pattern_gate_ok(pattern_type)
    cur = _FLOW_CTX.state

    if cur == FLOW_DEAD:
        if pattern_ok:
            _FLOW_CTX.state = FLOW_TEST
            _FLOW_CTX.probe_hit_count = 0
            reason = "DEAD_TO_TEST(PICTURE+PATTERN_OK)"
            tags.append("PROMOTE_TO_TEST")
        else:
            _FLOW_CTX.state = FLOW_DEAD
            reason = "DEAD_STAY(INSUFF_SIGNALS)"
            tags.append("STAY_DEAD")

    elif cur == FLOW_TEST:
        if last_entry_type == ENTRY_PROBE and last_hit is False:
            _FLOW_CTX.reset()
            reason = "TEST_TO_DEAD(PROBE_MISS)"
            tags.append("DROP_TO_DEAD")
            return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

        if not pattern_ok:
            _FLOW_CTX.reset()
            reason = "TEST_TO_DEAD(PATTERN_LOST)"
            tags.append("DROP_TO_DEAD")
            return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

        if last_entry_type == ENTRY_PROBE and last_hit is True:
            _FLOW_CTX.probe_hit_count += 1
            tags.append(f"PROBE_HIT_COUNT={_FLOW_CTX.probe_hit_count}")
            if pb_len >= MIN_PB_FOR_ALIVE:
                _FLOW_CTX.state = FLOW_ALIVE
                reason = "TEST_TO_ALIVE(PROBE_HIT)"
                tags.append("PROMOTE_TO_ALIVE")
            else:
                _FLOW_CTX.state = FLOW_TEST
                reason = "TEST_STAY(EARLY_SHOE_BLOCK_ALIVE)"
                tags.append("EARLY_SHOE_BLOCK")
        elif pb_len >= MIN_PB_FOR_ALIVE and china_state == CHINA_ALIVE and pattern_energy >= 0.20:
            _FLOW_CTX.state = FLOW_ALIVE
            reason = "TEST_TO_ALIVE(STRUCTURE_MATURED)"
            tags.append("PROMOTE_TO_ALIVE")
        else:
            _FLOW_CTX.state = FLOW_TEST
            reason = "TEST_STAY(VALIDATING)"
            tags.append("STAY_TEST")

    elif cur == FLOW_ALIVE:
        if last_entry_type == ENTRY_NORMAL and last_hit is False:
            _FLOW_CTX.reset()
            reason = "ALIVE_TO_DEAD(NORMAL_MISS)"
            tags.append("DROP_TO_DEAD")
            return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

        if china_state == CHINA_WEAK:
            _FLOW_CTX.state = FLOW_TEST
            reason = "ALIVE_TO_TEST(CHINA_WEAK)"
            tags.append("DOWNGRADE_TO_TEST")
        elif not pattern_ok:
            _FLOW_CTX.state = FLOW_TEST
            reason = "ALIVE_TO_TEST(PATTERN_LOST)"
            tags.append("DOWNGRADE_TO_TEST")
        elif pattern_energy < -0.15:
            _FLOW_CTX.state = FLOW_TEST
            reason = "ALIVE_TO_TEST(PATTERN_ENERGY_DOWN)"
            tags.append("DOWNGRADE_TO_TEST")
        else:
            _FLOW_CTX.state = FLOW_ALIVE
            reason = "ALIVE_STAY"
            tags.append("STAY_ALIVE")

    else:
        raise RuntimeError(f"INVALID_FLOW_STATE:{cur!r}")

    return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)


def _final(
    reason: str,
    tags: List[str],
    pb_len: int,
    bigroad_structure: str,
    picture_present: bool,
    pattern_type: str,
    pattern_energy: float,
    china_state: str,
) -> Dict[str, Any]:
    pattern_ok = _pattern_gate_ok(pattern_type)

    conf = _compute_confidence(
        state=_FLOW_CTX.state,
        picture_present=picture_present,
        pattern_ok=pattern_ok,
        pattern_energy=pattern_energy,
        china_state=china_state,
        probe_hit_count=_FLOW_CTX.probe_hit_count,
        pb_len=pb_len,
    )

    flow_stability = _compute_stability(
        state=_FLOW_CTX.state,
        bigroad_structure=bigroad_structure,
        pattern_type=pattern_type,
        pattern_energy=pattern_energy,
        china_state=china_state,
        probe_hit_count=_FLOW_CTX.probe_hit_count,
        pb_len=pb_len,
    )

    flow_strength = _compute_strength(state=_FLOW_CTX.state, confidence=conf)
    flow_chaos_risk = _compute_chaos_risk(
        stability=flow_stability,
        bigroad_structure=bigroad_structure,
        china_state=china_state,
    )
    flow_reversal_risk = _compute_reversal_risk(
        state=_FLOW_CTX.state,
        pattern_energy=pattern_energy,
        bigroad_structure=bigroad_structure,
        china_state=china_state,
        pb_len=pb_len,
    )

    flow_direction = FLOW_DIRECTION_NEUTRAL

    tags.append(f"FLOW_DIR={flow_direction}")
    tags.append(f"FLOW_STR={flow_strength:.3f}")
    tags.append(f"FLOW_STAB={flow_stability:.3f}")
    tags.append(f"FLOW_CHAOS={flow_chaos_risk:.3f}")
    tags.append(f"FLOW_REV={flow_reversal_risk:.3f}")

    return {
        "flow_state": _FLOW_CTX.state,
        "flow_reason": reason,
        "flow_confidence": float(conf),
        "flow_strength": float(flow_strength),
        "flow_stability": float(flow_stability),
        "flow_chaos_risk": float(flow_chaos_risk),
        "flow_reversal_risk": float(flow_reversal_risk),
        "flow_direction": str(flow_direction),
        "flow_tags": tags,
    }


# -----------------------------
# Public API
# -----------------------------
def compute_flow_features(
    big_eye: List[str],
    small_road: List[str],
    cockroach: List[str],
    streak_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    이름은 유지하되, 반환 의미는 flow lifecycle + side-free metrics 이다.
    - 방향/베팅 결정을 하지 않는다.
    - 외부 결과(last_entry_type/last_hit)가 주입되면 그것도 반영한다.
    - 외부 pattern_type/pattern_energy가 없어도 road 구조 메타만으로 deterministic 계산한다.
    """
    streak_info_v = _require_dict(streak_info, "streak_info")
    big_eye_v = _validate_rb_seq(big_eye, "big_eye")
    small_road_v = _validate_rb_seq(small_road, "small_road")
    cockroach_v = _validate_rb_seq(cockroach, "cockroach")

    runtime_pb = _validate_pb_seq(road.get_pb_sequence(), "road.get_pb_sequence()")
    pb_len = len(runtime_pb)

    if "pb_len" in streak_info_v:
        provided_pb_len = _require_int(streak_info_v["pb_len"], "streak_info.pb_len")
        if provided_pb_len != pb_len:
            raise RuntimeError(f"pb_len mismatch: streak_info.pb_len={provided_pb_len} != runtime_pb_len={pb_len}")

    shoe_id = _extract_shoe_id(streak_info_v)
    shoe_reset = _detect_shoe_reset(pb_len=pb_len, shoe_id=shoe_id)

    _FLOW_CTX.last_seen_pb_len = pb_len
    if shoe_id is not None:
        _FLOW_CTX.last_seen_shoe_id = shoe_id

    if pb_len == 0:
        _FLOW_CTX.reset()
        return {
            "flow_state": FLOW_DEAD,
            "flow_reason": "WARMUP_PB_EMPTY",
            "flow_confidence": 0.0,
            "flow_strength": 0.0,
            "flow_stability": 0.0,
            "flow_chaos_risk": 1.0,
            "flow_reversal_risk": 0.5,
            "flow_direction": FLOW_DIRECTION_NEUTRAL,
            "flow_tags": ["WARMUP", "PB_LEN=0", "FLOW_DIR=neutral"],
        }

    structure_meta = _require_dict(road.get_structure_meta(runtime_pb), "road.get_structure_meta()")
    structure_raw = _require_key(structure_meta, "structure", name="road.get_structure_meta()")
    if not isinstance(structure_raw, str):
        raise TypeError("road.get_structure_meta().structure must be str")
    structure = structure_raw.strip().lower()
    if structure not in VALID_STRUCTURES:
        raise ValueError(f"invalid structure from road.get_structure_meta(): {structure!r}")

    picture_present = _picture_present_from_bigroad(structure)
    china_state, china_marks, china_bcnt = _china_state_from_last_marks(big_eye_v, small_road_v, cockroach_v)
    pattern_type, pattern_energy, pt_tags = _extract_pattern_info(
        streak_info=streak_info_v,
        bigroad_structure=structure,
        china_state=china_state,
    )
    last_entry_type, last_hit, out_tags = _extract_outcome_info(streak_info_v)

    res = _transition(
        pb_len=pb_len,
        shoe_reset=shoe_reset,
        picture_present=picture_present,
        bigroad_structure=structure,
        pattern_type=pattern_type,
        pattern_energy=pattern_energy,
        china_state=china_state,
        last_entry_type=last_entry_type,
        last_hit=last_hit,
    )

    res["flow_tags"].extend(pt_tags)
    res["flow_tags"].extend(out_tags)
    res["flow_tags"].append(f"CHINA_BCNT={china_bcnt}")
    res["flow_tags"].append(f"CHINA_MARKS={china_marks}")

    return res


def reset_flow_state() -> None:
    """외부 reset 훅."""
    _FLOW_CTX.reset()
    _FLOW_CTX.last_seen_pb_len = 0
    _FLOW_CTX.last_seen_shoe_id = None


def get_flow_state_snapshot() -> Dict[str, Any]:
    """디버그/모니터링용 스냅샷."""
    return {
        "state": _FLOW_CTX.state,
        "probe_hit_count": _FLOW_CTX.probe_hit_count,
        "last_seen_pb_len": _FLOW_CTX.last_seen_pb_len,
        "last_seen_shoe_id": _FLOW_CTX.last_seen_shoe_id,
        "last_bigroad_structure": _FLOW_CTX.last_bigroad_structure,
        "last_pattern_type": _FLOW_CTX.last_pattern_type,
        "last_pattern_energy": _FLOW_CTX.last_pattern_energy,
        "last_china_state": _FLOW_CTX.last_china_state,
    }