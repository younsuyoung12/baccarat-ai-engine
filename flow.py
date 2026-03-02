# -*- coding: utf-8 -*-
"""
flow.py
====================================================
Flow Lifecycle Manager for Baccarat Predictor AI Engine v11.x

역할
- flow.py는 “방향 생성기”가 아니다.
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

변경 요약 (2026-01-02)
----------------------------------------------------
1) features.build_feature_payload_v3() 계약 호환:
   - flow_dict에 필수 키를 모든 정상 반환 경로에서 항상 포함
     flow_strength / flow_stability / flow_chaos_risk / flow_reversal_risk / flow_direction
2) 방향 암시 금지 유지:
   - flow_direction은 "neutral" 고정
   - 모든 수치는 side-free(구조/상태/신뢰) 지표로만 계산
3) 예외 정책 유지:
   - 진짜 오류(상태 오염 등)에만 예외
   - 데이터 부족/흐름 미형성은 중립 수치 반환(폴백/무시는 아님: 명시적 계산)
----------------------------------------------------

변경 요약 (2026-01-01)
----------------------------------------------------
1) 기존 “중국점 + 스트릭 기반 방향/수치(flow_direction/chaos/strength)” 계산 제거.
2) 입력 신뢰 기준을 아래로 고정:
   - Big Road 구조 메타(road.py 제공)
   - 중국점 상태 요약(ALIVE/WEAK/BROKEN/UNKNOWN)
   - pattern_type / pattern_energy (pattern.py 산출값이 상위에서 주입)
   - PROBE/NORMAL 결과(적중/실패)는 상위(recommend)에서 주입
3) 상태 전이 규칙을 명시적으로 구현:
   - DEAD↔TEST↔ALIVE 전이 및 새 슈 리셋 처리
4) 중국점은 색(R/B) 해석 금지: 오직 상태 요약만 사용.
5) 예외는 “진짜 오류”에만 사용. 데이터 부족은 정상 상태로 처리.
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

# 새 슈 초반 보호: 어떤 경우에도 ALIVE 진입 금지(pb_len 기준)
MIN_PB_FOR_ALIVE = 10


# -----------------------------
# Context (module-local)
# -----------------------------
@dataclass
class FlowContext:
    state: str = FLOW_DEAD
    probe_hit_count: int = 0

    last_seen_pb_len: int = 0
    last_seen_shoe_id: Optional[str] = None

    # 최근 입력(디버그/태그용)
    last_bigroad_structure: str = "random"
    last_pattern_type: Optional[str] = None
    last_pattern_energy: float = 0.0
    last_china_state: str = CHINA_UNKNOWN

    def reset(self) -> None:
        self.state = FLOW_DEAD
        self.probe_hit_count = 0
        # last_seen_* 는 슈 감지에 필요하므로 유지 가능.
        self.last_bigroad_structure = "random"
        self.last_pattern_type = None
        self.last_pattern_energy = 0.0
        self.last_china_state = CHINA_UNKNOWN


_FLOW_CTX = FlowContext()


# -----------------------------
# Extraction helpers
# -----------------------------
def _as_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "t", "1", "yes", "y"):
            return True
        if s in ("false", "f", "0", "no", "n"):
            return False
    return None


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _extract_pattern_info(streak_info: Dict[str, Any]) -> Tuple[Optional[str], float, List[str]]:
    tags: List[str] = []

    pt: Optional[str] = None
    pe: float = 0.0

    # candidates (상위 파이프라인 주입을 허용)
    # 1) streak_info["pattern_dict"] = {"pattern_type":..., "pattern_energy":...}
    pd = streak_info.get("pattern_dict")
    if isinstance(pd, dict):
        pt = _as_str(pd.get("pattern_type"))
        pe = _as_float(pd.get("pattern_energy"), default=0.0)

    # 2) flat keys
    if pt is None:
        pt = _as_str(streak_info.get("pattern_type"))
    if "pattern_energy" in streak_info:
        pe = _as_float(streak_info.get("pattern_energy"), default=pe)

    if pt is not None:
        pt = pt.lower()
        tags.append(f"PTYPE={pt}")
    else:
        tags.append("PTYPE=missing")

    tags.append(f"PENERGY={pe:.3f}")
    return pt, pe, tags


def _extract_outcome_info(streak_info: Dict[str, Any]) -> Tuple[Optional[str], Optional[bool], List[str]]:
    """
    상위(recommend)가 전달할 수 있는 '직전 베팅 결과' 정보.
    - last_entry_type: PROBE|NORMAL
    - last_hit: bool
    """
    tags: List[str] = []

    entry = _as_str(streak_info.get("last_entry_type")) or _as_str(streak_info.get("prev_entry_type"))
    hit = _as_bool(streak_info.get("last_hit"))
    if hit is None:
        hit = _as_bool(streak_info.get("prev_hit"))

    if entry is not None:
        entry = entry.strip().upper()
        if entry not in (ENTRY_PROBE, ENTRY_NORMAL):
            entry = None

    if entry is None:
        tags.append("OUTCOME=none")
    else:
        tags.append(f"OUTCOME_ENTRY={entry}")

    if hit is None:
        tags.append("OUTCOME_HIT=unknown")
    else:
        tags.append(f"OUTCOME_HIT={int(hit)}")

    return entry, hit, tags


def _extract_shoe_id(streak_info: Dict[str, Any]) -> Optional[str]:
    return _as_str(streak_info.get("shoe_id")) or _as_str(streak_info.get("shoe"))


def _china_state_from_last_marks(
    big_eye: List[str],
    small_road: List[str],
    cockroach: List[str],
) -> Tuple[str, Dict[str, Optional[str]], int]:
    """
    중국점은 R/B 색 자체를 해석하지 않는다.
    - 오직 상태 요약만 만든다: ALIVE/WEAK/BROKEN/UNKNOWN
    - 규칙:
      * 관측 0개 -> UNKNOWN
      * 마지막 3개(각 로드의 last)가 존재하는 값 중 'B'가 2개 이상 -> BROKEN
      * 'B'가 1개 -> WEAK
      * 그 외 -> ALIVE
    """
    last_be = _as_str(big_eye[-1]) if big_eye else None
    last_sr = _as_str(small_road[-1]) if small_road else None
    last_cr = _as_str(cockroach[-1]) if cockroach else None

    marks = {
        "big_eye_last": last_be.upper() if last_be else None,
        "small_road_last": last_sr.upper() if last_sr else None,
        "cockroach_last": last_cr.upper() if last_cr else None,
    }

    observed = sum(1 for v in marks.values() if v in ("R", "B"))
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
    # road.py: pingpong / blocks / mixed / streak / random
    return structure in ("pingpong", "blocks", "mixed", "streak")


# -----------------------------
# Side-free metrics (contract keys)
# -----------------------------
def _compute_strength(state: str, confidence: float) -> float:
    # 상태에 따른 강도 구간을 명시적으로 분리(방향 암시 금지)
    conf = _clamp01(confidence)
    if state == FLOW_DEAD:
        return 0.0
    if state == FLOW_TEST:
        # 0.3 ~ 0.5
        return _clamp01(0.3 + 0.2 * conf)
    # ALIVE: 0.6 ~ 1.0
    return _clamp01(0.6 + 0.4 * conf)


def _compute_stability(
    state: str,
    bigroad_structure: str,
    pattern_type: Optional[str],
    pattern_energy: float,
    china_state: str,
    probe_hit_count: int,
    pb_len: int,
) -> float:
    if state == FLOW_DEAD:
        return 0.0

    # 구조 기반 안정성(방향 정보 미사용)
    base_map = {
        "pingpong": 0.70,
        "blocks": 0.66,
        "streak": 0.68,
        "mixed": 0.55,
        "random": 0.40,
    }
    base = float(base_map.get(str(bigroad_structure).lower(), 0.45))

    # 패턴 게이트 OK는 구조 일관성 보조 신호
    if _pattern_gate_ok(pattern_type):
        base += 0.05
    else:
        base -= 0.05

    # 에너지 변동(절대값)이 클수록 불안정(방향 미사용)
    e = abs(float(pattern_energy))
    if e > 1.0:
        e = 1.0
    base -= 0.15 * e

    # PROBE 적중 누적은 안정성에 소폭 가산(방향 미사용)
    if probe_hit_count >= 1:
        base += 0.05

    # 중국점 상태 요약 반영(색 해석 금지)
    if china_state == CHINA_UNKNOWN:
        base -= 0.08
    elif china_state == CHINA_WEAK:
        base -= 0.03
    elif china_state == CHINA_BROKEN:
        base -= 0.15

    # 새 슈 초반은 구조가 덜 고정되므로 소폭 감산
    if int(pb_len) < int(MIN_PB_FOR_ALIVE):
        base -= 0.05

    return _clamp01(base)


def _compute_chaos_risk(stability: float, bigroad_structure: str, china_state: str) -> float:
    # 혼돈 리스크는 안정성의 보Complement + 상태 보정
    risk = 1.0 - _clamp01(stability)

    if str(bigroad_structure).lower() == "mixed":
        risk += 0.05
    elif str(bigroad_structure).lower() == "random":
        risk += 0.15

    if china_state == CHINA_UNKNOWN:
        risk += 0.10
    elif china_state == CHINA_WEAK:
        risk += 0.05
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
    # 반전/붕괴 리스크(방향 미사용): 에너지 하락/약화 상태에서 증가
    if state == FLOW_DEAD:
        base = 0.50
    elif state == FLOW_TEST:
        base = 0.45
    else:
        base = 0.35

    e = float(pattern_energy)
    if e < -1.0:
        e = -1.0
    if e > 1.0:
        e = 1.0

    if e < 0.0:
        base += 0.25 * min(1.0, abs(e))
    elif e > 0.0:
        base -= 0.15 * min(1.0, e)

    if china_state == CHINA_WEAK:
        base += 0.12
    elif china_state == CHINA_UNKNOWN:
        base += 0.08
    elif china_state == CHINA_BROKEN:
        base += 0.25

    if str(bigroad_structure).lower() == "mixed":
        base += 0.06

    if int(pb_len) < int(MIN_PB_FOR_ALIVE):
        base += 0.05

    return _clamp01(base)


# -----------------------------
# Core transition logic
# -----------------------------
def _detect_shoe_reset(pb_len: int, shoe_id: Optional[str]) -> Optional[str]:
    # shoe_id 변경 우선
    if shoe_id and _FLOW_CTX.last_seen_shoe_id and shoe_id != _FLOW_CTX.last_seen_shoe_id:
        return "shoe_id_changed"
    # pb_len 감소 (리셋)
    if pb_len < _FLOW_CTX.last_seen_pb_len:
        return "pb_len_decreased"
    return None


def _compute_confidence(
    state: str,
    picture_present: bool,
    pattern_ok: bool,
    pattern_energy: float,
    china_state: str,
    probe_hit_count: int,
) -> float:
    if state == FLOW_DEAD:
        base = 0.25
    elif state == FLOW_TEST:
        base = 0.60
    else:
        base = 0.82

    adj = 0.0
    if picture_present:
        adj += 0.05
    if pattern_ok:
        adj += 0.05
    if probe_hit_count >= 1:
        adj += 0.05
    if pattern_energy > 0:
        adj += 0.05
    elif pattern_energy < 0:
        adj -= 0.05

    if china_state == CHINA_UNKNOWN:
        adj -= 0.08
    elif china_state == CHINA_WEAK:
        adj -= 0.03
    elif china_state == CHINA_BROKEN:
        adj -= 0.10

    v = base + adj
    return _clamp01(v)


def _transition(
    pb_len: int,
    shoe_reset: Optional[str],
    picture_present: bool,
    bigroad_structure: str,
    pattern_type: Optional[str],
    pattern_energy: float,
    china_state: str,
    last_entry_type: Optional[str],
    last_hit: Optional[bool],
) -> Dict[str, Any]:
    tags: List[str] = []
    reason = ""

    # update debug fields
    _FLOW_CTX.last_bigroad_structure = bigroad_structure
    _FLOW_CTX.last_pattern_type = pattern_type
    _FLOW_CTX.last_pattern_energy = pattern_energy
    _FLOW_CTX.last_china_state = china_state

    tags.append(f"BR_STRUCT={bigroad_structure}")
    tags.append(f"CHINA={china_state}")
    tags.append(f"PB_LEN={pb_len}")
    if pattern_type is None:
        tags.append("PTYPE=missing")
    else:
        tags.append(f"PTYPE={pattern_type}")
    tags.append(f"PENERGY={pattern_energy:.3f}")

    # 0) shoe reset
    if shoe_reset:
        _FLOW_CTX.reset()
        reason = f"SHOE_RESET({shoe_reset})"
        tags.append("RESET")
        return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

    # 1) hard dead conditions
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

    # 2) state machine
    cur = _FLOW_CTX.state

    if cur == FLOW_DEAD:
        if picture_present and pattern_ok and china_state != CHINA_BROKEN:
            _FLOW_CTX.state = FLOW_TEST
            _FLOW_CTX.probe_hit_count = 0
            reason = "DEAD_TO_TEST(PICTURE+PATTERN_OK)"
            tags.append("PROMOTE_TO_TEST")
        else:
            _FLOW_CTX.state = FLOW_DEAD
            reason = "DEAD_STAY(INSUFF_SIGNALS)"
            tags.append("STAY_DEAD")

    elif cur == FLOW_TEST:
        # TEST → DEAD
        if (last_entry_type == ENTRY_PROBE) and (last_hit is False):
            _FLOW_CTX.reset()
            reason = "TEST_TO_DEAD(PROBE_MISS)"
            tags.append("DROP_TO_DEAD")
            return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

        if china_state == CHINA_BROKEN:
            _FLOW_CTX.reset()
            reason = "TEST_TO_DEAD(CHINA_BROKEN)"
            tags.append("DROP_TO_DEAD")
            return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

        if not picture_present:
            _FLOW_CTX.reset()
            reason = "TEST_TO_DEAD(PICTURE_LOST)"
            tags.append("DROP_TO_DEAD")
            return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

        # TEST → ALIVE (조건 만족 + 새 슈 초반 보호)
        promoted = False
        if (last_entry_type == ENTRY_PROBE) and (last_hit is True):
            _FLOW_CTX.probe_hit_count += 1
            tags.append(f"PROBE_HIT_COUNT={_FLOW_CTX.probe_hit_count}")
            if _FLOW_CTX.probe_hit_count >= 1:
                if pb_len >= MIN_PB_FOR_ALIVE:
                    _FLOW_CTX.state = FLOW_ALIVE
                    reason = "TEST_TO_ALIVE(PROBE_HIT)"
                    tags.append("PROMOTE_TO_ALIVE")
                    promoted = True
                else:
                    reason = "TEST_STAY(EARLY_SHOE_BLOCK_ALIVE)"
                    tags.append("EARLY_SHOE_BLOCK")

        if not promoted:
            if pattern_energy > 0:
                if pb_len >= MIN_PB_FOR_ALIVE:
                    _FLOW_CTX.state = FLOW_ALIVE
                    reason = "TEST_TO_ALIVE(PATTERN_ENERGY_UP)"
                    tags.append("PROMOTE_TO_ALIVE")
                else:
                    reason = "TEST_STAY(EARLY_SHOE_BLOCK_ALIVE)"
                    tags.append("EARLY_SHOE_BLOCK")

        if reason == "":
            _FLOW_CTX.state = FLOW_TEST
            reason = "TEST_STAY(VALIDATING)"
            tags.append("STAY_TEST")

    elif cur == FLOW_ALIVE:
        # ALIVE → DEAD
        if (last_entry_type == ENTRY_NORMAL) and (last_hit is False):
            _FLOW_CTX.reset()
            reason = "ALIVE_TO_DEAD(NORMAL_MISS)"
            tags.append("DROP_TO_DEAD")
            return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

        if china_state == CHINA_BROKEN:
            _FLOW_CTX.reset()
            reason = "ALIVE_TO_DEAD(CHINA_BROKEN)"
            tags.append("DROP_TO_DEAD")
            return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

        if not picture_present:
            _FLOW_CTX.reset()
            reason = "ALIVE_TO_DEAD(PICTURE_LOST)"
            tags.append("DROP_TO_DEAD")
            return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)

        # ALIVE → TEST
        if china_state == CHINA_WEAK:
            _FLOW_CTX.state = FLOW_TEST
            reason = "ALIVE_TO_TEST(CHINA_WEAK)"
            tags.append("DOWNGRADE_TO_TEST")
        elif pattern_energy < 0:
            _FLOW_CTX.state = FLOW_TEST
            reason = "ALIVE_TO_TEST(PATTERN_ENERGY_DOWN)"
            tags.append("DOWNGRADE_TO_TEST")
        else:
            _FLOW_CTX.state = FLOW_ALIVE
            reason = "ALIVE_STAY"
            tags.append("STAY_ALIVE")

    else:
        # 진짜 오류 (상태 값 오염)
        raise RuntimeError(f"INVALID_FLOW_STATE:{cur!r}")

    return _final(reason, tags, pb_len, bigroad_structure, picture_present, pattern_type, pattern_energy, china_state)


def _final(
    reason: str,
    tags: List[str],
    pb_len: int,
    bigroad_structure: str,
    picture_present: bool,
    pattern_type: Optional[str],
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
    flow_chaos_risk = _compute_chaos_risk(stability=flow_stability, bigroad_structure=bigroad_structure, china_state=china_state)
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
# Public API (kept name)
# -----------------------------
def compute_flow_features(
    big_eye: List[str],
    small_road: List[str],
    cockroach: List[str],
    streak_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    ⚠️ 이름은 유지하되, 반환 의미는 “flow lifecycle + side-free metrics”로 유지한다.
    - 방향/베팅 결정을 하지 않는다.
    - 상태 전이에 필요한 정보는 streak_info에 주입될 수 있다.
      * pattern_type / pattern_energy
      * last_entry_type / last_hit
      * shoe_id
      * pb_len (없으면 road.get_pb_sequence() 기반)
    """
    if streak_info is None or not isinstance(streak_info, dict):
        raise TypeError("streak_info must be dict")

    # pb_len / shoe_id
    pb_len = int(streak_info.get("pb_len") or len(road.get_pb_sequence()))
    shoe_id = _extract_shoe_id(streak_info)

    # shoe reset detect
    shoe_reset = _detect_shoe_reset(pb_len=pb_len, shoe_id=shoe_id)

    # update last seen
    _FLOW_CTX.last_seen_pb_len = pb_len
    if shoe_id:
        _FLOW_CTX.last_seen_shoe_id = shoe_id

    # Big Road structure meta (road.py)
    smeta = road.get_structure_meta()
    structure = str((smeta or {}).get("structure") or "random").lower()
    picture_present = _picture_present_from_bigroad(structure)

    # China state summary (R/B -> state only)
    china_state, china_marks, china_bcnt = _china_state_from_last_marks(big_eye, small_road, cockroach)

    # Pattern info (must be injected by upstream)
    pattern_type, pattern_energy, pt_tags = _extract_pattern_info(streak_info)
    if pattern_type is not None:
        pattern_type = pattern_type.lower()

    # Outcome info (must be injected by upstream)
    last_entry_type, last_hit, out_tags = _extract_outcome_info(streak_info)

    # Transition
    res = _transition(
        pb_len=pb_len,
        shoe_reset=shoe_reset,
        picture_present=picture_present,
        bigroad_structure=structure,
        pattern_type=pattern_type,
        pattern_energy=float(pattern_energy),
        china_state=china_state,
        last_entry_type=last_entry_type,
        last_hit=last_hit,
    )

    # Append detail tags
    res["flow_tags"].extend(pt_tags)
    res["flow_tags"].extend(out_tags)
    res["flow_tags"].append(f"CHINA_BCNT={china_bcnt}")
    # marks 자체는 외부 계약을 바꾸므로 tag로만 제공(디버그)
    if china_state != CHINA_UNKNOWN:
        res["flow_tags"].append(f"CHINA_MARKS={china_marks}")

    return res


def reset_flow_state() -> None:
    """외부 reset 훅(테스트/슈 리셋 시 사용 가능)."""
    _FLOW_CTX.reset()
    _FLOW_CTX.last_seen_pb_len = 0
    _FLOW_CTX.last_seen_shoe_id = None


def get_flow_state_snapshot() -> Dict[str, Any]:
    """디버그/모니터링용 스냅샷(계약 외 사용)."""
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
