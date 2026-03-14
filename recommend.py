# -*- coding: utf-8 -*-
"""
recommend.py
====================================================
Baccarat Predictor AI Engine v12.1
Deterministic Rule-based Betting Recommender
(STRICT · NO-FALLBACK · FAIL-FAST)

변경 요약 (2026-03-14)
----------------------------------------------------
1) GPT / LLM 완전 제거
   - gpt_engine import 제거
   - gpt_decide 호출 제거
   - engine+gpt hybrid score 제거
   - recommend_bet 시그니처에서 gpt_analysis/mode/alerts 제거
2) deterministic rule engine 전환
   - Big Road picture 기반 방향 생성
   - Future China Roads 기반 2-of-3 확증
   - chaos / entropy 필터
   - signal_strength 기반 PASS / BET / bet_unit 결정
3) bet_unit 1~3 도입
   - signal_strength 기반 1~3 강도 부여
   - PROBE / WEAK 상태에서는 bet_unit 상한 1로 제한
4) 정책 충돌 제거
   - 단일 streak를 임의로 그림으로 취급하지 않음
   - fallback / default substitution 금지
5) 출력 스키마 유지
   - bet_side / bet_unit / entry_type / reason / tags / metrics / pass_reason
----------------------------------------------------

역할
------
- pb_seq(누적 P/B 시퀀스) + features를 기반으로
  bet_side / bet_unit / entry_type(PASS/PROBE/NORMAL)를 결정한다.
- 이 모듈은 deterministic rule engine이며,
  확률 모델/LLM/자연어 해석에 의존하지 않는다.

운영 원칙
------
STRICT · NO-FALLBACK · FAIL-FAST
- 필수 입력 누락/스키마 위반/불일치 → 즉시 RuntimeError
- 조용한 continue/pass 금지
- 임의 기본값 생성 금지
- 다음 수 확률(%) 반환 금지

출력 스키마
------
{
  "bet_side": "P"|"B"|None,
  "bet_unit": int,
  "entry_type": "PROBE"|"NORMAL"|None,
  "reason": str,
  "tags": [str, ...],
  "metrics": { ... },
  "pass_reason": optional str
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json


# ----------------------------------------------------
# Constants
# ----------------------------------------------------

SIDE_P = "P"
SIDE_B = "B"

ENTRY_PROBE = "PROBE"
ENTRY_NORMAL = "NORMAL"

FLOW_DEAD = "DEAD"
FLOW_TEST = "TEST"
FLOW_ALIVE = "ALIVE"

MIN_PB_FOR_SIGNAL = 5
FLOW_NORMAL_MIN_PBLEN = 10

MIN_UNIT = 1
MAX_UNIT = 3

CHAOS_PASS_THRESHOLD = 0.55
ENTROPY_PASS_THRESHOLD = 0.92
SIGNAL_PASS_THRESHOLD = 0.50
SIGNAL_BET2_THRESHOLD = 0.55
SIGNAL_BET3_THRESHOLD = 0.70

NORMAL_MIN_BEAUTY = 60.0
NORMAL_MIN_STABILITY = 0.55
NORMAL_MAX_CHAOS = 0.45
NORMAL_MIN_SIGNAL_STRENGTH = 0.70


# ----------------------------------------------------
# Local-only FLOW_LIFE context
# ----------------------------------------------------

@dataclass
class FlowLifeContext:
    state: str = FLOW_DEAD
    last_side: Optional[str] = None
    consecutive_probe_hits: int = 0

    pending_bet_side: Optional[str] = None
    pending_entry_type: Optional[str] = None
    pending_at_pb_len: Optional[int] = None

    last_seen_pb_len: int = 0
    last_seen_shoe_sig: Optional[str] = None

    probe_fail_count: int = 0
    no_play_shoe: bool = False
    probe_since_last_normal: bool = False
    force_probe_next: bool = False

    def reset_dead(self) -> None:
        # 실패/슈 리셋 시 런타임 상태만 초기화.
        # 실패 카운트/force_probe_next는 유지한다.
        self.state = FLOW_DEAD
        self.last_side = None
        self.consecutive_probe_hits = 0
        self.pending_bet_side = None
        self.pending_entry_type = None
        self.pending_at_pb_len = None
        self.no_play_shoe = False


def _flow_get_ctx(meta: Dict[str, Any]) -> FlowLifeContext:
    if not isinstance(meta, dict):
        raise TypeError("meta must be dict")
    ctx = meta.get("_flow_life_ctx")
    if ctx is None:
        ctx = FlowLifeContext()
        meta["_flow_life_ctx"] = ctx
    if not isinstance(ctx, FlowLifeContext):
        raise TypeError("meta._flow_life_ctx must be FlowLifeContext")
    return ctx


def _extract_shoe_sig(meta: Dict[str, Any]) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    shoe_id = meta.get("shoe_id")
    if isinstance(shoe_id, str) and shoe_id.strip():
        return shoe_id.strip()
    return None


def _pb_clean(pb_seq: List[str]) -> List[str]:
    if not isinstance(pb_seq, list):
        raise TypeError("pb_seq must be list")
    out: List[str] = []
    for x in pb_seq:
        s = str(x).upper()
        if s in (SIDE_P, SIDE_B):
            out.append(s)
    return out


def _flow_resolve_pending(
    flow_ctx: FlowLifeContext,
    pb_seq: List[str],
    shoe_sig: Optional[str],
) -> Dict[str, Any]:
    dbg: Dict[str, Any] = {"resolved": False}

    pb_only = _pb_clean(pb_seq)
    pb_len = len(pb_only)

    if shoe_sig and flow_ctx.last_seen_shoe_sig and shoe_sig != flow_ctx.last_seen_shoe_sig:
        dbg["shoe_reset"] = "shoe_sig_changed"
        flow_ctx.reset_dead()
    elif pb_len < flow_ctx.last_seen_pb_len:
        dbg["shoe_reset"] = "pb_len_decreased"
        flow_ctx.reset_dead()

    flow_ctx.last_seen_shoe_sig = shoe_sig
    flow_ctx.last_seen_pb_len = pb_len

    if flow_ctx.pending_bet_side in (SIDE_P, SIDE_B) and flow_ctx.pending_entry_type in (ENTRY_PROBE, ENTRY_NORMAL):
        at_pb_len = int(flow_ctx.pending_at_pb_len or 0)
        if pb_len >= at_pb_len + 1:
            last_winner = str(pb_only[-1]).upper() if pb_len > 0 else None
            hit = bool(last_winner == flow_ctx.pending_bet_side)

            dbg.update(
                {
                    "resolved": True,
                    "pending_side": flow_ctx.pending_bet_side,
                    "pending_entry": flow_ctx.pending_entry_type,
                    "at_pb_len": at_pb_len,
                    "last_winner": last_winner,
                    "hit": hit,
                }
            )

            if flow_ctx.pending_entry_type == ENTRY_PROBE:
                if hit:
                    flow_ctx.consecutive_probe_hits += 1
                    if flow_ctx.consecutive_probe_hits >= 2:
                        flow_ctx.state = FLOW_ALIVE
                else:
                    flow_ctx.consecutive_probe_hits = 0
                    flow_ctx.probe_fail_count += 1
                    flow_ctx.force_probe_next = True
                    flow_ctx.reset_dead()
            else:
                if hit:
                    flow_ctx.state = FLOW_ALIVE
                else:
                    flow_ctx.probe_fail_count += 1
                    flow_ctx.force_probe_next = True
                    flow_ctx.reset_dead()

            flow_ctx.pending_bet_side = None
            flow_ctx.pending_entry_type = None
            flow_ctx.pending_at_pb_len = None

    return dbg


# ----------------------------------------------------
# Strict helpers
# ----------------------------------------------------

def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"missing required field: {key}")
    return d[key]


def _require_dict(v: Any, name: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        raise TypeError(f"{name} must be dict, got {type(v).__name__}")
    return v


def _as_float(v: Any, name: str) -> float:
    try:
        x = float(v)
    except Exception as e:
        raise TypeError(f"{name} must be float-compatible") from e
    if x != x:
        raise RuntimeError(f"{name} must be finite (NaN)")
    if x in (float("inf"), float("-inf")):
        raise RuntimeError(f"{name} must be finite (inf)")
    return float(x)


def _as_int(v: Any, name: str) -> int:
    try:
        return int(v)
    except Exception as e:
        raise TypeError(f"{name} must be int-compatible") from e


def _require_unit_interval(v: Any, name: str) -> float:
    x = _as_float(v, name)
    if x < 0.0 or x > 1.0:
        raise RuntimeError(f"{name} must be in [0,1], got {x}")
    return x


def _require_nonempty_str(v: Any, name: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise RuntimeError(f"{name} must be non-empty str")
    return v.strip()


# ----------------------------------------------------
# Big Road picture detection
# ----------------------------------------------------

def _rle_runs(seq: List[str]) -> List[Tuple[str, int]]:
    if not seq:
        return []
    runs: List[Tuple[str, int]] = []
    cur = seq[0]
    n = 1
    for s in seq[1:]:
        if s == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur = s
            n = 1
    runs.append((cur, n))
    return runs


def _is_alternating(seq: List[str]) -> bool:
    if len(seq) < 2:
        return False
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            return False
    return True


def _infer_target_block_len(runs: List[Tuple[str, int]]) -> Optional[int]:
    if not runs:
        return None
    tail = runs[-4:]
    lens = [ln for _, ln in tail if isinstance(ln, int) and ln >= 2]
    if not lens:
        return None
    return int(max(lens))


def _analyze_bigroad_structure(pb_seq: List[str]) -> Dict[str, Any]:
    seq = _pb_clean(pb_seq)
    n = len(seq)
    runs = _rle_runs(seq)

    info: Dict[str, Any] = {
        "picture_present": False,
        "structure_type": "NONE",
        "is_random": False,
        "target_block_len": None,
        "current_run_len": runs[-1][1] if runs else 0,
        "expected_next_side": None,
    }

    if n < 4:
        return info

    if _is_alternating(seq[-4:]):
        info["picture_present"] = True
        info["structure_type"] = "PINGPONG"
        info["target_block_len"] = 1
        info["expected_next_side"] = SIDE_B if seq[-1] == SIDE_P else SIDE_P
        return info

    if len(runs) >= 2:
        (s1, l1), (s2, l2) = runs[-2], runs[-1]
        if s1 != s2 and l1 >= 2 and l2 >= 2:
            target = _infer_target_block_len(runs)
            info["picture_present"] = True
            info["structure_type"] = "BLOCKS"
            info["target_block_len"] = target
            if target and l2 < target:
                info["expected_next_side"] = s2
            else:
                info["expected_next_side"] = s1
            return info

    if len(runs) >= 2 and runs[-1][1] >= 3:
        info["picture_present"] = True
        info["structure_type"] = "STREAK"
        info["target_block_len"] = runs[-1][1]
        info["expected_next_side"] = runs[-1][0]
        return info

    if len(runs) >= 4:
        tail = runs[-4:]
        lens = [ln for _, ln in tail]
        has_single = any(ln == 1 for ln in lens)
        has_block = any(ln >= 2 for ln in lens)
        alternating_runs = all(tail[i][0] != tail[i - 1][0] for i in range(1, len(tail)))
        if has_single and has_block and alternating_runs:
            target = _infer_target_block_len(runs)
            info["picture_present"] = True
            info["structure_type"] = "MIXED_BLOCKS"
            info["target_block_len"] = target
            last_side, last_len = runs[-1]
            prev_side, _ = runs[-2]
            if target and last_len < target:
                info["expected_next_side"] = last_side
            else:
                info["expected_next_side"] = prev_side
            return info

    if len(runs) >= 6 and all(ln == 1 for _, ln in runs[-6:]) and not _is_alternating(seq[-6:]):
        info["is_random"] = True
        info["structure_type"] = "STRICT_RANDOM"
        return info

    return info


def _extract_pattern_type(features: Dict[str, Any]) -> Optional[str]:
    pt = features.get("pattern_type")
    if isinstance(pt, str) and pt.strip():
        return pt.strip().lower()

    pd = features.get("pattern_dict")
    if isinstance(pd, dict):
        pt2 = pd.get("pattern_type")
        if isinstance(pt2, str) and pt2.strip():
            return pt2.strip().lower()

    return None


def _decide_side_from_picture(
    pb_seq: List[str],
    bigroad_info: Dict[str, Any],
    pattern_type: Optional[str],
) -> Tuple[Optional[str], List[str]]:
    tags: List[str] = []
    seq = _pb_clean(pb_seq)
    runs = _rle_runs(seq)

    br_pic = bool(bigroad_info.get("picture_present"))
    br_type = str(bigroad_info.get("structure_type") or "NONE")
    br_next = bigroad_info.get("expected_next_side")

    tags.append(f"BR_PIC={int(br_pic)}")
    tags.append(f"BR_TYPE={br_type}")
    if pattern_type is not None:
        tags.append(f"PTYPE={pattern_type}")

    if br_pic and br_next in (SIDE_P, SIDE_B):
        tags.append("SIDE_SRC=BIGROAD")
        return str(br_next), tags

    # 패턴 타입은 보조용.
    # 단일 streak를 pattern_type만으로 방향 생성하는 것은 금지한다.
    if pattern_type == "pingpong" and seq:
        tags.append("SIDE_SRC=PATTERN_PINGPONG")
        return (SIDE_B if seq[-1] == SIDE_P else SIDE_P), tags

    if pattern_type == "blocks" and len(runs) >= 2 and runs[-1][1] >= 2 and runs[-2][1] >= 2:
        target = _infer_target_block_len(runs) or 2
        last_side, last_len = runs[-1]
        prev_side, _ = runs[-2]
        tags.append("SIDE_SRC=PATTERN_BLOCKS")
        tags.append(f"PT_BLOCKLEN={target}")
        if last_len < target:
            return last_side, tags
        return prev_side, tags

    tags.append("SIDE_DENY_NO_PICTURE")
    return None, tags


# ----------------------------------------------------
# China roads helpers
# ----------------------------------------------------

def _last_rb_from_matrix(matrix_json: Any) -> Optional[str]:
    if matrix_json is None:
        return None

    if isinstance(matrix_json, str):
        try:
            mat = json.loads(matrix_json)
        except Exception as e:
            raise RuntimeError(f"china matrix json parse failed: {type(e).__name__}") from e
    elif isinstance(matrix_json, list):
        mat = matrix_json
    else:
        raise RuntimeError("china matrix must be json string or list")

    if not isinstance(mat, list):
        raise RuntimeError("china matrix root must be list")

    for col in reversed(mat):
        if not isinstance(col, list):
            raise RuntimeError("china matrix column must be list")
        for cell in reversed(col):
            c = str(cell).upper().strip()
            if c in ("R", "B"):
                return c
            if c == "":
                continue
            raise RuntimeError(f"china matrix contains invalid cell: {c!r}")

    return None


def _china_marks_from_features(features: Dict[str, Any]) -> Dict[str, Optional[str]]:
    big_eye_src = features.get("big_eye_matrix_json") if "big_eye_matrix_json" in features else features.get("big_eye_matrix")
    small_src = features.get("small_road_matrix_json") if "small_road_matrix_json" in features else features.get("small_road_matrix")
    cock_src = features.get("cockroach_matrix_json") if "cockroach_matrix_json" in features else features.get("cockroach_matrix")

    if big_eye_src is None or small_src is None or cock_src is None:
        raise RuntimeError("missing china matrices in features (big_eye/small/cockroach)")

    return {
        "big_eye_last": _last_rb_from_matrix(big_eye_src),
        "small_last": _last_rb_from_matrix(small_src),
        "cockroach_last": _last_rb_from_matrix(cock_src),
    }


def _china_b_count(china_marks: Dict[str, Optional[str]]) -> int:
    return sum(1 for v in china_marks.values() if v == "B")


def _china_health_state(china_bcnt: int, china_marks: Dict[str, Optional[str]]) -> str:
    observed = sum(1 for v in china_marks.values() if v in ("R", "B"))
    if observed <= 0:
        return "UNKNOWN"
    if china_bcnt >= 2:
        return "BROKEN"
    if china_bcnt == 1:
        return "WEAK"
    return "ALIVE"


def _extract_future_scenarios_strict(features: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fs = features.get("future_scenarios")
    if fs is None:
        raise RuntimeError("future_scenarios missing (required)")
    fs = _require_dict(fs, "future_scenarios")

    if "P" not in fs or "B" not in fs:
        raise RuntimeError("future_scenarios must contain keys 'P' and 'B'")

    fP = _require_dict(fs["P"], "future_scenarios.P")
    fB = _require_dict(fs["B"], "future_scenarios.B")

    def pick_field(d: Dict[str, Any], keys: List[str], name: str) -> Optional[str]:
        present = [k for k in keys if k in d]
        if len(present) != 1:
            raise RuntimeError(f"future scenario field '{name}' must exist in exactly one of {keys} (present={present})")
        v = d[present[0]]
        if v is None:
            return None
        if not isinstance(v, str):
            raise RuntimeError(f"future scenario '{name}' must be string or null")
        s = v.strip().upper()
        if s not in ("R", "B"):
            raise RuntimeError(f"future scenario '{name}' must be 'R' or 'B' or null")
        return s

    future_if_P = {
        "big_eye": pick_field(fP, ["big_eye", "bigEye"], "big_eye"),
        "small": pick_field(fP, ["small_road", "smallRoad", "small"], "small"),
        "cockroach": pick_field(fP, ["cockroach", "cock"], "cockroach"),
    }
    future_if_B = {
        "big_eye": pick_field(fB, ["big_eye", "bigEye"], "big_eye"),
        "small": pick_field(fB, ["small_road", "smallRoad", "small"], "small"),
        "cockroach": pick_field(fB, ["cockroach", "cock"], "cockroach"),
    }
    return future_if_P, future_if_B


def _china_confirm_from_future(
    future_if_P: Dict[str, Any],
    future_if_B: Dict[str, Any],
) -> Tuple[Optional[str], Dict[str, Optional[str]], str]:
    """
    각 중국점 road에 대해:
    - if P -> R and if B -> B : P vote
    - if P -> B and if B -> R : B vote
    - 그 외 : no vote
    """
    votes: Dict[str, Optional[str]] = {"big_eye": None, "small": None, "cockroach": None}

    for road_name in ("big_eye", "small", "cockroach"):
        p_mark = future_if_P.get(road_name)
        b_mark = future_if_B.get(road_name)

        if p_mark == "R" and b_mark == "B":
            votes[road_name] = SIDE_P
        elif p_mark == "B" and b_mark == "R":
            votes[road_name] = SIDE_B
        else:
            votes[road_name] = None

    p_count = sum(1 for v in votes.values() if v == SIDE_P)
    b_count = sum(1 for v in votes.values() if v == SIDE_B)

    if p_count >= 2 and p_count > b_count:
        strength = "STRONG" if p_count == 3 else "CONFIRM"
        return SIDE_P, votes, strength

    if b_count >= 2 and b_count > p_count:
        strength = "STRONG" if b_count == 3 else "CONFIRM"
        return SIDE_B, votes, strength

    return None, votes, "NONE"


# ----------------------------------------------------
# Leader payload / signal strength
# ----------------------------------------------------

def _extract_leader_payload(leader_state: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(leader_state, dict):
        raise TypeError("leader_state must be dict")

    required = [
        "leader_confidence",
        "leader_trust_state",
        "leader_signal",
    ]
    for k in required:
        if k not in leader_state:
            raise RuntimeError(f"leader_state missing required key: {k}")

    return leader_state


def _compute_signal_strength(
    *,
    beauty_score: float,
    stability: float,
    pattern_symmetry: float,
    chaos: float,
    leader_confidence: float,
) -> float:
    beauty_norm = beauty_score / 100.0

    if beauty_norm < 0.0 or beauty_norm > 1.0:
        raise RuntimeError(f"beauty_score_norm out of range: {beauty_norm}")
    if stability < 0.0 or stability > 1.0:
        raise RuntimeError(f"stability out of range: {stability}")
    if pattern_symmetry < 0.0 or pattern_symmetry > 1.0:
        raise RuntimeError(f"pattern_symmetry out of range: {pattern_symmetry}")
    if chaos < 0.0 or chaos > 1.0:
        raise RuntimeError(f"chaos out of range: {chaos}")
    if leader_confidence < 0.0 or leader_confidence > 1.0:
        raise RuntimeError(f"leader_confidence out of range: {leader_confidence}")

    strength = (
        0.30 * beauty_norm +
        0.25 * stability +
        0.20 * pattern_symmetry +
        0.15 * leader_confidence +
        0.10 * (1.0 - chaos)
    )

    if strength < 0.0 or strength > 1.0:
        raise RuntimeError(f"signal_strength out of range: {strength}")
    return float(strength)


def _compute_base_bet_unit(signal_strength: float) -> int:
    if signal_strength < SIGNAL_PASS_THRESHOLD:
        return 0
    if signal_strength < SIGNAL_BET2_THRESHOLD:
        return 1
    if signal_strength < SIGNAL_BET3_THRESHOLD:
        return 2
    return 3


# ----------------------------------------------------
# Main API
# ----------------------------------------------------

def recommend_bet(
    pb_seq: List[str],
    features: Dict[str, Any],
    leader_state: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(features, dict):
        raise TypeError("features must be dict")
    if not isinstance(meta, dict):
        raise TypeError("meta must be dict")

    # ----------------------------------------------------
    # FLOW_LIFE
    # ----------------------------------------------------
    flow_ctx = _flow_get_ctx(meta)
    shoe_sig = _extract_shoe_sig(meta)
    flow_resolve_dbg = _flow_resolve_pending(flow_ctx, pb_seq, shoe_sig)

    # ----------------------------------------------------
    # Required fields
    # ----------------------------------------------------
    rounds_total = _as_int(_require(features, "rounds_total"), "rounds_total")
    pb_ratio = _as_float(_require(features, "pb_ratio"), "pb_ratio")
    entropy = _require_unit_interval(_require(features, "entropy"), "entropy")
    beauty_score = _as_float(_require(features, "beauty_score"), "beauty_score")
    chaos = _require_unit_interval(_require(features, "chaos"), "chaos")
    stability = _require_unit_interval(_require(features, "stability"), "stability")
    pattern_score = _as_float(_require(features, "pattern_score"), "pattern_score")
    pattern_symmetry = _require_unit_interval(_require(features, "pattern_symmetry"), "pattern_symmetry")
    pattern_energy = _as_float(_require(features, "pattern_energy"), "pattern_energy")

    pb_only = _pb_clean(pb_seq)
    pb_len = len(pb_only)

    # ----------------------------------------------------
    # Warm-up guard
    # ----------------------------------------------------
    if pb_len < MIN_PB_FOR_SIGNAL:
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len
        tags = ["HOLD", "WARMUP_PBLEN_LT_5"]
        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD_WARMUP_PBLEN_LT_5",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "pb_ratio": pb_ratio,
                "entropy": entropy,
                "chaos": chaos,
                "stability": stability,
                "beauty_score": beauty_score,
                "pattern_score": pattern_score,
                "pattern_symmetry": pattern_symmetry,
                "pattern_energy": pattern_energy,
                "flow_life_state": flow_ctx.state,
                "flow_life_resolve_dbg": flow_resolve_dbg,
            },
            "pass_reason": "HOLD_WARMUP_PBLEN_LT_5",
        }

    # ----------------------------------------------------
    # Chaos / entropy filters
    # ----------------------------------------------------
    if chaos > CHAOS_PASS_THRESHOLD:
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len
        tags = ["HOLD", "CHAOS_HIGH", f"CHAOS={chaos:.4f}"]
        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD_CHAOS_HIGH",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "entropy": entropy,
                "chaos": chaos,
                "stability": stability,
                "beauty_score": beauty_score,
                "flow_life_state": flow_ctx.state,
                "flow_life_resolve_dbg": flow_resolve_dbg,
            },
            "pass_reason": "HOLD_CHAOS_HIGH",
        }

    if entropy > ENTROPY_PASS_THRESHOLD:
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len
        tags = ["HOLD", "ENTROPY_HIGH", f"ENTROPY={entropy:.4f}"]
        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD_ENTROPY_HIGH",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "entropy": entropy,
                "chaos": chaos,
                "stability": stability,
                "beauty_score": beauty_score,
                "flow_life_state": flow_ctx.state,
                "flow_life_resolve_dbg": flow_resolve_dbg,
            },
            "pass_reason": "HOLD_ENTROPY_HIGH",
        }

    # ----------------------------------------------------
    # Big Road picture → engine side
    # ----------------------------------------------------
    bigroad_info = _analyze_bigroad_structure(pb_seq)
    br_type = str(bigroad_info.get("structure_type") or "NONE")
    br_is_random = bool(bigroad_info.get("is_random"))

    pattern_type = _extract_pattern_type(features)
    engine_side, side_tags = _decide_side_from_picture(pb_seq, bigroad_info, pattern_type)

    if br_is_random:
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len
        tags = ["HOLD", "STRICT_RANDOM", f"BR_TYPE={br_type}"] + side_tags
        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD_STRICT_RANDOM",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "bigroad_structure": br_type,
                "pattern_type": pattern_type,
                "entropy": entropy,
                "chaos": chaos,
                "flow_life_state": flow_ctx.state,
                "flow_life_resolve_dbg": flow_resolve_dbg,
            },
            "pass_reason": "HOLD_STRICT_RANDOM",
        }

    if engine_side not in (SIDE_P, SIDE_B):
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len
        tags = ["HOLD", "NO_PICTURE_OR_DIRECTION", f"BR_TYPE={br_type}"] + side_tags
        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD_NO_PICTURE_OR_DIRECTION",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "bigroad_structure": br_type,
                "pattern_type": pattern_type,
                "entropy": entropy,
                "chaos": chaos,
                "flow_life_state": flow_ctx.state,
                "flow_life_resolve_dbg": flow_resolve_dbg,
            },
            "pass_reason": "HOLD_NO_PICTURE_OR_DIRECTION",
        }

    # ----------------------------------------------------
    # China health + future confirm
    # ----------------------------------------------------
    china_marks = _china_marks_from_features(features)
    china_bcnt = _china_b_count(china_marks)
    china_state = _china_health_state(china_bcnt, china_marks)

    if china_state == "BROKEN":
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len
        tags = ["HOLD", "CHINA_BROKEN", f"BR_TYPE={br_type}"] + side_tags
        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD_CHINA_BROKEN",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "china_state": china_state,
                "china_bcnt": china_bcnt,
                "china_marks": china_marks,
                "engine_side": engine_side,
                "entropy": entropy,
                "chaos": chaos,
                "flow_life_state": flow_ctx.state,
                "flow_life_resolve_dbg": flow_resolve_dbg,
            },
            "pass_reason": "HOLD_CHINA_BROKEN",
        }

    future_if_P, future_if_B = _extract_future_scenarios_strict(features)
    china_confirm_side, china_votes, china_confirm_strength = _china_confirm_from_future(future_if_P, future_if_B)

    if china_confirm_side is None:
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len
        tags = ["HOLD", "CHINA_NO_CONFIRM", f"BR_TYPE={br_type}"] + side_tags
        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD_CHINA_NO_CONFIRM",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "china_state": china_state,
                "china_votes": china_votes,
                "future_if_P": future_if_P,
                "future_if_B": future_if_B,
                "engine_side": engine_side,
                "entropy": entropy,
                "chaos": chaos,
                "flow_life_state": flow_ctx.state,
                "flow_life_resolve_dbg": flow_resolve_dbg,
            },
            "pass_reason": "HOLD_CHINA_NO_CONFIRM",
        }

    if china_confirm_side != engine_side:
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len
        tags = [
            "HOLD",
            "CHINA_DISAGREE",
            f"ENGINE_SIDE={engine_side}",
            f"CHINA_CONFIRM_SIDE={china_confirm_side}",
            f"CHINA_CONFIRM={china_confirm_strength}",
            f"BR_TYPE={br_type}",
        ] + side_tags
        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD_CHINA_DISAGREE",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "china_state": china_state,
                "china_votes": china_votes,
                "future_if_P": future_if_P,
                "future_if_B": future_if_B,
                "engine_side": engine_side,
                "china_confirm_side": china_confirm_side,
                "entropy": entropy,
                "chaos": chaos,
                "flow_life_state": flow_ctx.state,
                "flow_life_resolve_dbg": flow_resolve_dbg,
            },
            "pass_reason": "HOLD_CHINA_DISAGREE",
        }

    # ----------------------------------------------------
    # Leader payload / confidence
    # ----------------------------------------------------
    leader_payload = _extract_leader_payload(leader_state)
    leader_confidence = _require_unit_interval(leader_payload.get("leader_confidence"), "leader_state.leader_confidence")
    leader_trust_state = _require_nonempty_str(leader_payload.get("leader_trust_state"), "leader_state.leader_trust_state").upper()
    leader_signal = leader_payload.get("leader_signal")

    if leader_signal is not None and leader_signal not in (SIDE_P, SIDE_B):
        raise RuntimeError(f"leader_state.leader_signal invalid: {leader_signal!r}")
    if leader_trust_state not in ("NONE", "WEAK", "MID", "STRONG"):
        raise RuntimeError(f"leader_state.leader_trust_state invalid: {leader_trust_state!r}")

    if leader_signal in (SIDE_P, SIDE_B) and leader_trust_state in ("MID", "STRONG") and leader_signal != engine_side:
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len
        tags = [
            "HOLD",
            "LEADER_DISAGREE",
            f"ENGINE_SIDE={engine_side}",
            f"LEADER_SIGNAL={leader_signal}",
            f"LEADER_TRUST={leader_trust_state}",
            f"BR_TYPE={br_type}",
        ] + side_tags
        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD_LEADER_DISAGREE",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "leader_confidence": leader_confidence,
                "leader_trust_state": leader_trust_state,
                "leader_signal": leader_signal,
                "engine_side": engine_side,
                "china_confirm_side": china_confirm_side,
                "entropy": entropy,
                "chaos": chaos,
                "flow_life_state": flow_ctx.state,
                "flow_life_resolve_dbg": flow_resolve_dbg,
            },
            "pass_reason": "HOLD_LEADER_DISAGREE",
        }

    # ----------------------------------------------------
    # Signal strength
    # ----------------------------------------------------
    signal_strength = _compute_signal_strength(
        beauty_score=beauty_score,
        stability=stability,
        pattern_symmetry=pattern_symmetry,
        chaos=chaos,
        leader_confidence=leader_confidence,
    )

    if signal_strength < SIGNAL_PASS_THRESHOLD:
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len
        tags = [
            "HOLD",
            "SIGNAL_TOO_WEAK",
            f"SIGNAL_STRENGTH={signal_strength:.4f}",
            f"BR_TYPE={br_type}",
            f"LEADER_TRUST={leader_trust_state}",
            f"CHINA_CONFIRM={china_confirm_strength}",
        ] + side_tags
        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD_SIGNAL_TOO_WEAK",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "signal_strength": signal_strength,
                "leader_confidence": leader_confidence,
                "leader_trust_state": leader_trust_state,
                "china_state": china_state,
                "china_confirm_side": china_confirm_side,
                "entropy": entropy,
                "chaos": chaos,
                "beauty_score": beauty_score,
                "stability": stability,
                "pattern_symmetry": pattern_symmetry,
                "flow_life_state": flow_ctx.state,
                "flow_life_resolve_dbg": flow_resolve_dbg,
            },
            "pass_reason": "HOLD_SIGNAL_TOO_WEAK",
        }

    # ----------------------------------------------------
    # Entry type
    # ----------------------------------------------------
    if flow_ctx.force_probe_next:
        internal_entry_type = ENTRY_PROBE
        flow_ctx.force_probe_next = False
    elif (
        pb_len >= FLOW_NORMAL_MIN_PBLEN
        and signal_strength >= NORMAL_MIN_SIGNAL_STRENGTH
        and beauty_score >= NORMAL_MIN_BEAUTY
        and stability >= NORMAL_MIN_STABILITY
        and chaos <= NORMAL_MAX_CHAOS
        and china_state == "ALIVE"
        and leader_trust_state in ("MID", "STRONG")
    ):
        internal_entry_type = ENTRY_NORMAL
    else:
        internal_entry_type = ENTRY_PROBE

    # ----------------------------------------------------
    # Final side / state update
    # ----------------------------------------------------
    final_side = engine_side

    if internal_entry_type == ENTRY_NORMAL:
        flow_ctx.state = FLOW_ALIVE
    else:
        flow_ctx.state = FLOW_TEST

    flow_ctx.last_side = final_side
    flow_ctx.pending_bet_side = final_side
    flow_ctx.pending_entry_type = internal_entry_type
    flow_ctx.pending_at_pb_len = pb_len
    flow_ctx.last_seen_pb_len = pb_len
    flow_ctx.probe_since_last_normal = (internal_entry_type == ENTRY_PROBE)

    # ----------------------------------------------------
    # Bet unit
    # ----------------------------------------------------
    base_bet_unit = _compute_base_bet_unit(signal_strength)
    if base_bet_unit == 0:
        raise RuntimeError("base_bet_unit cannot be 0 after PASS threshold passed")

    if internal_entry_type == ENTRY_PROBE or china_state == "WEAK":
        bet_unit = 1
    else:
        bet_unit = base_bet_unit

    if bet_unit < MIN_UNIT or bet_unit > MAX_UNIT:
        raise RuntimeError(f"bet_unit out of range: {bet_unit}")

    tags: List[str] = [
        f"BR_TYPE={br_type}",
        f"ENGINE_SIDE={engine_side}",
        f"CHINA_CONFIRM_SIDE={china_confirm_side}",
        f"CHINA_CONFIRM={china_confirm_strength}",
        f"CHINA_STATE={china_state}",
        f"LEADER_TRUST={leader_trust_state}",
        f"SIGNAL_STRENGTH={signal_strength:.4f}",
        f"ENTRY={internal_entry_type}",
        f"BET_UNIT={bet_unit}",
    ] + side_tags

    metrics: Dict[str, Any] = {
        "rounds_total": rounds_total,
        "pb_len": pb_len,
        "pb_ratio": pb_ratio,
        "entropy": entropy,
        "chaos": chaos,
        "stability": stability,
        "beauty_score": beauty_score,
        "pattern_score": pattern_score,
        "pattern_symmetry": pattern_symmetry,
        "pattern_energy": pattern_energy,
        "bigroad_structure": br_type,
        "pattern_type": pattern_type,
        "engine_side": engine_side,
        "final_side": final_side,
        "signal_strength": signal_strength,
        "china_state": china_state,
        "china_bcnt": china_bcnt,
        "china_marks": china_marks,
        "china_votes": china_votes,
        "china_confirm_side": china_confirm_side,
        "china_confirm_strength": china_confirm_strength,
        "future_if_P": future_if_P,
        "future_if_B": future_if_B,
        "leader_confidence": leader_confidence,
        "leader_trust_state": leader_trust_state,
        "leader_signal": leader_signal,
        "flow_life_state": flow_ctx.state,
        "flow_life_last_side": flow_ctx.last_side,
        "flow_life_probe_hit_streak": flow_ctx.consecutive_probe_hits,
        "flow_life_resolve_dbg": flow_resolve_dbg,
        "probe_fail_count": int(flow_ctx.probe_fail_count),
        "probe_since_last_normal": bool(flow_ctx.probe_since_last_normal),
        "force_probe_next": bool(flow_ctx.force_probe_next),
        "entry_type": internal_entry_type,
        "bet_unit": bet_unit,
    }

    return {
        "bet_side": final_side,
        "bet_unit": int(bet_unit),
        "entry_type": internal_entry_type,
        "reason": f"{internal_entry_type}_RULE_SIGNAL",
        "tags": tags,
        "metrics": metrics,
    }