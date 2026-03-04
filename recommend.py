# -*- coding: utf-8 -*-
"""
recommend.py
====================================================
Baccarat Predictor AI Engine v11.x
Rule-based Betting Recommender (실전 플레이어 기준) + GPT 패턴 해석 (Hybrid)

역할
------
- pb_seq(누적 P/B 시퀀스) + features를 기반으로 bet_side / bet_unit / entry_type(PASS/PROBE/NORMAL)를 결정한다.
- 이 모듈은 룰 기반 엔진이며, 폴백(근거 없는 기본값 방향 생성)을 금지한다.

✅ 운영 원칙(추가)
------
STRICT · NO-FALLBACK · FAIL-FAST
- 필수 입력 누락/스키마 위반/불일치 → 즉시 RuntimeError
- 조용한 continue/pass 금지
- 임의 기본값 생성 금지

출력 스키마(유지)
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

변경 요약 (2026-03-04)
----------------------------------------------------
1) ✅ Hybrid Engine + GPT 패턴 해석 결합:
   - 최근 5판(P/B only) + runs + pattern_type + flow_state + china_state + future china roads 를 GPT에 전달
   - engine_weight(0.6) + gpt_confidence 를 점수 결합하여 최종 P/B 산출
   - |scoreP - scoreB| < 0.1 → HOLD (bet_side=None, bet_unit=0)
2) ✅ 5판 안전장치:
   - "최근 P/B 5개"가 확보되지 않으면 절대 방향을 내지 않는다(HOLD).
3) ✅ 학습 기반 line_transition_table(learned) 사용 중지:
   - 사용자의 "학습 사용 안함" 정책에 맞춰 line_transition_table 적용을 비활성화한다.
   - 태그에는 LINE_*를 DISABLED로 명시한다.
4) ✅ UI 단순화를 위한 출력 정책:
   - UI에는 bet_side(P/B)만 보이도록, 출력 entry_type은 None으로 반환(표시 단순화 목적).
   - 내부 FlowLifeContext 상태 관리를 위해 internal_entry_type은 유지한다(외부 표시와 분리).

절대 금지(유지)
----------------------------------------------------
- 확률(%) 예측 또는 반환
- 다음 수 예측 표현(맞춘다/우세 등)
- recommend.py 외 파일 수정(이 파일 단독 변경 요청에 한함)
- engine_state / road / feature 계산 로직 침범
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import os

from gpt_engine import gpt_decide, GptDecision


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

# NORMAL은 슈 초반에는 금지
FLOW_NORMAL_MIN_PBLEN = 10

# unit range
MIN_UNIT = 1
MAX_UNIT = 3

# NORMAL 승격 최소 조건(실전 플레이어)
NORMAL_MIN_BEAUTY = 60.0
NORMAL_MAX_CHAOS = 0.60
NORMAL_MIN_STABILITY = 0.40

# Line transition table (history action) — 학습 기반 기능. (현재 비활성)
LINE_TABLE_PATH = os.path.join("models", "line_transition_table.json")
ENABLE_LEARNED_LINE_TABLE = False  # ✅ 학습 사용 안함 정책: 항상 False

# Hybrid weights
ENGINE_WEIGHT = 0.60
EDGE_HOLD_THRESHOLD = 0.10  # |scoreP - scoreB| < 0.1 => HOLD

# 5판 안전장치 (P/B only 기준)
MIN_PB_FOR_GPT = 5


# ----------------------------------------------------
# Local-only FLOW_LIFE context (recommend.py 내부 상태)
# ----------------------------------------------------

@dataclass
class FlowLifeContext:
    # current state
    state: str = FLOW_DEAD
    last_side: Optional[str] = None
    consecutive_probe_hits: int = 0

    # pending bet from previous call (to be resolved on next call)
    pending_bet_side: Optional[str] = None
    pending_entry_type: Optional[str] = None
    pending_at_pb_len: Optional[int] = None  # pb_len at the time of decision (PB only 기준)

    # guard for shoe reset / call skipping
    last_seen_pb_len: int = 0  # PB only len 기준으로 운용

    # last seen shoe signature (meta 기반 새 슈 감지용)
    last_seen_shoe_sig: Optional[str] = None

    # legacy fields kept for output schema compatibility
    probe_fail_count: int = 0
    no_play_shoe: bool = False

    # "최근 PROBE 1회 이상 진입 이력" 추적 (NORMAL 1회 허용 토큰)
    probe_since_last_normal: bool = False

    # 직전 실패 후 다음 1회는 PROBE 1 강제
    force_probe_next: bool = False

    def reset_dead(self) -> None:
        # last_seen_pb_len 은 호출 정렬/슈 리셋 감지에 사용되므로 유지한다.
        self.state = FLOW_DEAD
        self.last_side = None
        self.consecutive_probe_hits = 0
        self.pending_bet_side = None
        self.pending_entry_type = None
        self.pending_at_pb_len = None
        self.probe_fail_count = 0
        self.no_play_shoe = False
        # probe_since_last_normal / force_probe_next 는 “다음 진입 시점” 정책에 필요하므로 유지


def _flow_get_ctx(meta: Dict[str, Any]) -> FlowLifeContext:
    if meta is None or not isinstance(meta, dict):
        raise TypeError("meta must be dict")
    ctx = meta.get("_flow_life_ctx")
    if ctx is None:
        ctx = FlowLifeContext()
        meta["_flow_life_ctx"] = ctx
    if not isinstance(ctx, FlowLifeContext):
        # 외부가 덮어썼을 가능성(계약 위반). 진짜 오류로 처리.
        raise TypeError("meta._flow_life_ctx must be FlowLifeContext")
    return ctx


def _extract_shoe_sig(meta: Dict[str, Any]) -> Optional[str]:
    """
    새 슈를 감지하기 위한 시그니처. 없으면 None.
    - app/predictor_adapter가 shoe_id 를 meta에 넣는 경우를 우선 사용.
    - 없으면 pb_len 변화로만 간접 감지한다.
    """
    if not isinstance(meta, dict):
        return None
    shoe_id = meta.get("shoe_id")
    if isinstance(shoe_id, str) and shoe_id.strip():
        return shoe_id.strip()
    return None


def _flow_resolve_pending(flow_ctx: FlowLifeContext, pb_seq: List[str], shoe_sig: Optional[str]) -> Dict[str, Any]:
    """
    이전 호출에서 PROBE/NORMAL로 베팅을 '시도'했던 것을,
    이번 호출에서 결과(직전 winner)로 적중/실패를 판정한다.
    - 통계/상태 전이: consecutive_probe_hits, probe_fail_count, force_probe_next
    """
    dbg: Dict[str, Any] = {"resolved": False}

    pb_only = _pb_clean(pb_seq)
    pb_len = len(pb_only)

    # shoe change: shoe_sig 우선, 없으면 pb_len 감소로 감지
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
        # pending은 “결정 당시 pb_len(PB only)” 기준으로 “그 다음 1개 PB 결과”에서만 해석한다.
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
                else:
                    flow_ctx.consecutive_probe_hits = 0
                    flow_ctx.probe_fail_count += 1
                    flow_ctx.force_probe_next = True  # 직전 실패 후 다음 1회 PROBE 강제
                    # 실패하면 흐름을 "깨짐"으로 보고 DEAD로 리셋
                    flow_ctx.reset_dead()
            else:
                # NORMAL 결과는 흐름 유지/붕괴 판단(보수적으로)
                if hit:
                    flow_ctx.state = FLOW_ALIVE
                else:
                    flow_ctx.reset_dead()
                    flow_ctx.force_probe_next = True

            # pending clear
            flow_ctx.pending_bet_side = None
            flow_ctx.pending_entry_type = None
            flow_ctx.pending_at_pb_len = None

    return dbg


# ----------------------------------------------------
# Helpers: strict contract (no fallback)
# ----------------------------------------------------

def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"missing required field: {key}")
    return d[key]


def _as_float(v: Any, name: str) -> float:
    try:
        x = float(v)
    except Exception as e:
        raise TypeError(f"{name} must be float-compatible") from e
    if not (x == x):  # NaN
        raise RuntimeError(f"{name} must be finite (NaN)")
    if x in (float("inf"), float("-inf")):
        raise RuntimeError(f"{name} must be finite (inf)")
    return float(x)


def _as_int(v: Any, name: str) -> int:
    try:
        return int(v)
    except Exception as e:
        raise TypeError(f"{name} must be int-compatible") from e


# ----------------------------------------------------
# Helpers: PB only extraction
# ----------------------------------------------------

def _pb_clean(pb_seq: List[str]) -> List[str]:
    out: List[str] = []
    if not isinstance(pb_seq, list):
        raise TypeError("pb_seq must be list")
    for x in pb_seq:
        s = str(x).upper()
        if s in (SIDE_P, SIDE_B):
            out.append(s)
    return out


def _last5_and_runs_strict(pb_seq: List[str]) -> Tuple[str, List[int]]:
    """
    STRICT:
    - P/B only 기준으로 last5(길이=5)와 runs를 계산한다.
    - P/B가 5개 미만이면 RuntimeError가 아니라 '상위 정책'에서 HOLD 처리한다(5판 안전장치).
    """
    pb_only = _pb_clean(pb_seq)
    if len(pb_only) < MIN_PB_FOR_GPT:
        raise RuntimeError("PB_ONLY_LEN_LT_5")
    last5 = "".join(pb_only[-5:])  # 길이=5 보장
    # runs: 전체 PB only로 계산 후 최근 3개만
    runs: List[int] = []
    last: Optional[str] = None
    cnt = 0
    for s in pb_only:
        if s != last:
            if cnt > 0:
                runs.append(cnt)
            cnt = 1
            last = s
        else:
            cnt += 1
    if cnt > 0:
        runs.append(cnt)
    if not runs:
        raise RuntimeError("runs empty (unexpected)")
    return last5, runs[-3:]


def _derive_pattern_type_for_gpt(pattern_type: Optional[str], br_type: str) -> str:
    """
    STRICT:
    - pattern_type이 있으면 그대로 사용(정규화).
    - 없으면 BigRoad 구조(br_type)에서 deterministic하게 파생.
    - 둘 다 불가능하면 예외.
    """
    if isinstance(pattern_type, str) and pattern_type.strip():
        pt = pattern_type.strip().lower()
        return pt

    bt = str(br_type or "").strip().upper()
    if bt == "PINGPONG":
        return "pingpong"
    if bt == "BLOCKS":
        return "blocks"
    if bt == "MIXED_BLOCKS":
        return "mixed"
    # picture가 있는데 pattern_type까지 없으면 파이프라인 입력 계약이 약한 상태
    raise RuntimeError("pattern_type missing and cannot derive from BigRoad structure")


# ----------------------------------------------------
# Helpers: Big Road picture detection (structure-based)
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
    """
    최근 run들(최대 4개)에서 '블록 길이'를 추정한다.
    - len>=2 인 run만 사용
    - 관측치가 없으면 None
    """
    if not runs:
        return None
    tail = runs[-4:]
    lens = [ln for _, ln in tail if isinstance(ln, int) and ln >= 2]
    if not lens:
        return None
    # 보수적으로 최대값을 채택(블록이 커지는 구간에서 과소추정 방지)
    target = max(lens)
    if target < 2:
        return None
    return int(target)


def _analyze_bigroad_structure(pb_seq: List[str]) -> Dict[str, Any]:
    """
    Big Road(pb_seq)에서 '그림'이 있는지 판단하고, 가능하면 다음 1판 예상(side 후보)를 생성한다.
    - 단일 스트릭만으로는 그림으로 취급하지 않는다(초반 추종 금지).
    """
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

    # 1) pingpong (1212 / 2121 ...) : 최근 4개가 완전 교대
    if _is_alternating(seq[-4:]):
        info["picture_present"] = True
        info["structure_type"] = "PINGPONG"
        info["target_block_len"] = 1
        info["expected_next_side"] = SIDE_B if seq[-1] == SIDE_P else SIDE_P
        return info

    # 2) blocks
    if len(runs) >= 2:
        (s1, l1), (s2, l2) = runs[-2], runs[-1]
        if s1 in (SIDE_P, SIDE_B) and s2 in (SIDE_P, SIDE_B) and s1 != s2 and l1 >= 2 and l2 >= 2:
            target = _infer_target_block_len(runs)
            info["picture_present"] = True
            info["structure_type"] = "BLOCKS"
            info["target_block_len"] = target
            if target and l2 < target:
                info["expected_next_side"] = s2
            else:
                info["expected_next_side"] = s1
            return info

    # 3) mixed blocks
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

    # strict random
    if len(runs) >= 6 and all(ln == 1 for _, ln in runs[-6:]) and not _is_alternating(seq[-6:]):
        info["is_random"] = True
        info["structure_type"] = "STRICT_RANDOM"
        return info

    return info


# ----------------------------------------------------
# Helpers: pattern gate (optional; already computed upstream)
# ----------------------------------------------------

def _extract_pattern_type(features: Dict[str, Any]) -> Optional[str]:
    """
    pattern.py 결과는 파이프라인에서 계산되어 features에 포함될 수 있다.
    - 우선순위: features.pattern_dict.pattern_type -> features.pattern_type
    """
    pd = features.get("pattern_dict")
    if isinstance(pd, dict):
        pt = pd.get("pattern_type")
        if isinstance(pt, str) and pt.strip():
            return pt.strip().lower()
    pt2 = features.get("pattern_type")
    if isinstance(pt2, str) and pt2.strip():
        return pt2.strip().lower()
    return None


def _pattern_gate_ok(pt: Optional[str]) -> bool:
    return pt in ("streak", "pingpong", "blocks")


def _decide_side_from_picture(pb_seq: List[str], bigroad_info: Dict[str, Any], pattern_type: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """
    방향 생성은 오직 '그림'이 있을 때만.
    - Big Road 구조 기반 예상이 있으면 우선 사용
    - 그 외 pattern_type 기반 최소 규칙만 사용(폴백 금지)
    """
    tags: List[str] = []
    seq = _pb_clean(pb_seq)
    last = seq[-1] if seq else None

    br_pic = bool(bigroad_info.get("picture_present"))
    br_type = str(bigroad_info.get("structure_type") or "NONE")
    br_next = bigroad_info.get("expected_next_side")

    pt = (pattern_type or "").lower() if isinstance(pattern_type, str) else None
    pt_ok = _pattern_gate_ok(pt)

    tags.append(f"BR_PIC={int(br_pic)}")
    tags.append(f"BR_TYPE={br_type}")
    if pt is not None:
        tags.append(f"PTYPE={pt}")

    # Big Road 구조 우선
    if br_pic and br_next in (SIDE_P, SIDE_B):
        tags.append("SIDE_SRC=BIGROAD")
        return str(br_next), tags

    # pattern gate 보조
    if pt_ok and last in (SIDE_P, SIDE_B):
        if pt == "pingpong":
            tags.append("SIDE_SRC=PATTERN_PINGPONG")
            return (SIDE_B if last == SIDE_P else SIDE_P), tags
        if pt == "streak":
            tags.append("SIDE_SRC=PATTERN_STREAK")
            return last, tags
        if pt == "blocks":
            runs = _rle_runs(seq)
            if len(runs) >= 2 and runs[-1][1] >= 2 and runs[-2][1] >= 2:
                target = _infer_target_block_len(runs) or 2
                last_side, last_len = runs[-1]
                prev_side, _ = runs[-2]
                tags.append(f"PT_BLOCKLEN={target}")
                if last_len < target:
                    return last_side, tags
                return prev_side, tags
            tags.append("SIDE_DENY_PATTERN_BLOCKS_INSUFF")
            return None, tags

    tags.append("SIDE_DENY_NO_PICTURE")
    return None, tags


# ----------------------------------------------------
# Helpers: China roads (validation only; never direction)
# ----------------------------------------------------

def _last_rb_from_matrix(matrix_json: Any) -> Optional[str]:
    """
    matrix_json: list[list[str]] (JSON string or python list)
    Returns last non-empty mark among ("R","B").
    STRICT:
    - 형식이 파싱 불가하면 RuntimeError (조용히 None 반환 금지)
    """
    if matrix_json is None:
        return None

    mat: Any
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

    # scan from end
    for col in reversed(mat):
        if not isinstance(col, list):
            raise RuntimeError("china matrix column must be list")
        for cell in reversed(col):
            c = str(cell).upper().strip()
            if c in ("R", "B"):
                return c
            if c == "":
                continue
            # 허용: 빈값 외의 값이 들어오면 스키마 위반
            raise RuntimeError(f"china matrix contains invalid cell: {c!r}")

    return None


def _china_b_count(features: Dict[str, Any]) -> Tuple[int, Dict[str, Optional[str]]]:
    """
    중국점은 색(R/B)만 사용하며, Bank/Player 방향과 무관하다.
    - 붕괴 판정용으로만 B 카운트를 사용(간단/보수적).
    STRICT:
    - 관련 키가 누락되면 예외(계약 위반).
    """
    # 키는 반드시 존재해야 한다(없으면 파이프라인 불일치)
    # json string 또는 list 모두 허용
    big_eye_src = features.get("big_eye_matrix_json") if "big_eye_matrix_json" in features else features.get("big_eye_matrix")
    small_src = features.get("small_road_matrix_json") if "small_road_matrix_json" in features else features.get("small_road_matrix")
    cock_src = features.get("cockroach_matrix_json") if "cockroach_matrix_json" in features else features.get("cockroach_matrix")

    if big_eye_src is None or small_src is None or cock_src is None:
        raise RuntimeError("missing china matrices in features (big_eye/small/cockroach)")

    big_eye_last = _last_rb_from_matrix(big_eye_src)
    small_last = _last_rb_from_matrix(small_src)
    cock_last = _last_rb_from_matrix(cock_src)

    marks = {
        "big_eye_last": big_eye_last,
        "small_last": small_last,
        "cockroach_last": cock_last,
    }
    bcnt = sum(1 for v in marks.values() if v == "B")
    return int(bcnt), marks


def _china_health_state(china_bcnt: int, china_marks: Dict[str, Optional[str]]) -> str:
    observed = sum(1 for v in china_marks.values() if v in ("R", "B"))
    if observed <= 0:
        return "UNKNOWN"
    if china_bcnt >= 2:
        return "BROKEN"
    if china_bcnt == 1:
        return "WEAK"
    return "ALIVE"


# ----------------------------------------------------
# Future China Roads (STRICT)
# ----------------------------------------------------

def _extract_future_scenarios_strict(features: Dict[str, Any], gpt_analysis: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    STRICT:
    - future_scenarios는 features 또는 gpt_analysis 중 '정확히 하나'에 존재해야 한다.
      (둘 다 있으면 내용이 동일해야 한다)
    - future_scenarios는 {"P": {...}, "B": {...}} 형태여야 한다.
    - 각 {...}는 big_eye/small_road/cockroach 값을 제공해야 하며 R/B/None만 허용.
    """
    fs_from_features = features.get("future_scenarios") if isinstance(features, dict) else None
    fs_from_analysis = gpt_analysis.get("future_scenarios") if isinstance(gpt_analysis, dict) else None

    if fs_from_features is None and fs_from_analysis is None:
        raise RuntimeError("future_scenarios missing (required)")

    if fs_from_features is not None and fs_from_analysis is not None:
        if fs_from_features != fs_from_analysis:
            raise RuntimeError("future_scenarios mismatch between features and gpt_analysis")
        fs = fs_from_features
    else:
        fs = fs_from_features if fs_from_features is not None else fs_from_analysis

    if not isinstance(fs, dict):
        raise RuntimeError("future_scenarios must be dict")

    if "P" not in fs or "B" not in fs:
        raise RuntimeError("future_scenarios must contain keys 'P' and 'B'")

    fP = fs["P"]
    fB = fs["B"]
    if not isinstance(fP, dict) or not isinstance(fB, dict):
        raise RuntimeError("future_scenarios.P/B must be dict")

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


# ----------------------------------------------------
# Leader (trust only; do not use leader_signal/source)
# ----------------------------------------------------

def _extract_leader_trust_state(leader_state: Dict[str, Any]) -> str:
    """
    leader_state.leader_trust_state: NONE|WEAK|MID|STRONG
    - 누락/이상 값은 "NONE"으로 정규화(엔진 크래시 방지)
    """
    if not isinstance(leader_state, dict):
        return "NONE"
    v = leader_state.get("leader_trust_state")
    if not isinstance(v, str):
        return "NONE"
    s = v.strip().upper()
    if s in ("NONE", "WEAK", "MID", "STRONG"):
        return s
    return "NONE"


# ----------------------------------------------------
# NORMAL gate logic
# ----------------------------------------------------

def _should_allow_normal(
    pb_seq: List[str],
    flow_ctx: FlowLifeContext,
    beauty_score: float,
    chaos: float,
    stability: float,
    structure_flags: Dict[str, Any],
    confirm_flags: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    NORMAL 허용 판단(기존 정책 유지).

    - 최소 조건 경로(실전 플레이어):
      * 마지막 NORMAL 이후 PROBE 이력 1회 이상(probe_since_last_normal)
      * beauty_score >= 60
      * chaos < 0.6
      * stability > 0.4
      * pb_len >= 10
      -> leader/GPT/alerts/structure/confirm 필수조건 제거, NORMAL 1회 허용
    - 그 외: 기존 경로(구조/확증 플래그) 유지
    """
    tags: List[str] = []
    pb_len = len(_pb_clean(pb_seq))

    if pb_len < FLOW_NORMAL_MIN_PBLEN:
        tags.append("NORMAL_DENY_EARLY_SHOE")
        return False, tags

    minimal_ok = (
        bool(flow_ctx.probe_since_last_normal)
        and beauty_score >= NORMAL_MIN_BEAUTY
        and chaos < NORMAL_MAX_CHAOS
        and stability > NORMAL_MIN_STABILITY
    )

    if minimal_ok:
        tags.append("NORMAL_ALLOW_MINIMAL")
        return True, tags

    # 기존 경로: FLOW_ALIVE + 구조/확증
    if flow_ctx.state != FLOW_ALIVE:
        tags.append("NORMAL_DENY_FLOW_NOT_ALIVE")
        return False, tags

    structure_ok = bool(structure_flags.get("structure_ok", False))
    confirm_ok = bool(confirm_flags.get("confirm_ok", False))

    if not structure_ok:
        tags.append("NORMAL_DENY_STRUCTURE")
        return False, tags
    if not confirm_ok:
        tags.append("NORMAL_DENY_CONFIRM")
        return False, tags

    tags.append("NORMAL_ALLOW_GATED")
    return True, tags


# ----------------------------------------------------
# Hybrid Decision (Engine + GPT)
# ----------------------------------------------------

def _hybrid_decide(engine_side: str, gpt: GptDecision) -> Optional[str]:
    """
    STRICT:
    - engine_side: P/B only
    - gpt.side: P/B/HOLD
    - 반환: P/B or None(HOLD)
    """
    if engine_side not in (SIDE_P, SIDE_B):
        raise RuntimeError(f"invalid engine_side: {engine_side!r}")

    score_p = 0.0
    score_b = 0.0

    if engine_side == SIDE_P:
        score_p += ENGINE_WEIGHT
    else:
        score_b += ENGINE_WEIGHT

    if gpt.side == SIDE_P:
        score_p += float(gpt.confidence)
    elif gpt.side == SIDE_B:
        score_b += float(gpt.confidence)
    elif gpt.side == "HOLD":
        pass
    else:
        raise RuntimeError(f"invalid gpt.side: {gpt.side!r}")

    if abs(score_p - score_b) < EDGE_HOLD_THRESHOLD:
        return None
    return SIDE_P if score_p > score_b else SIDE_B


# ----------------------------------------------------
# Main API
# ----------------------------------------------------

def recommend_bet(
    pb_seq: List[str],
    features: Dict[str, Any],
    leader_state: Dict[str, Any],
    gpt_analysis: Dict[str, Any],
    mode: str,
    alerts: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    # ----------------------------------------------------
    # FLOW_LIFE: resolve pending from previous call (if any)
    # ----------------------------------------------------
    flow_ctx = _flow_get_ctx(meta)
    shoe_sig = _extract_shoe_sig(meta)
    flow_resolve_dbg = _flow_resolve_pending(flow_ctx, pb_seq, shoe_sig)

    # ----------------------------------------------------
    # Required fields (no fallback)
    # ----------------------------------------------------
    rounds_total = _as_int(_require(features, "rounds_total"), "rounds_total")
    pb_ratio = _as_float(_require(features, "pb_ratio"), "pb_ratio")                     # -1~+1
    beauty_score = _as_float(_require(features, "beauty_score"), "beauty_score")         # 0~100
    chaos = _as_float(_require(features, "chaos"), "chaos")                               # 0~1
    stability = _as_float(_require(features, "stability"), "stability")                   # 0~1
    pattern_score = _as_float(_require(features, "pattern_score"), "pattern_score")       # -1~+1
    pattern_symmetry = _as_float(_require(features, "pattern_symmetry"), "pattern_symmetry")
    pattern_energy = _as_float(_require(features, "pattern_energy"), "pattern_energy")

    pb_only = _pb_clean(pb_seq)
    pb_len = len(pb_only)

    # ----------------------------------------------------
    # 5판 안전장치: 최근 P/B 5개 미만이면 절대 방향을 내지 않는다.
    # ----------------------------------------------------
    if pb_len < MIN_PB_FOR_GPT:
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len

        tags: List[str] = ["HOLD", "WARMUP_PBLEN_LT_5"]
        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD (WARMUP_PBLEN_LT_5)",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "pb_ratio": pb_ratio,
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
    # Picture gate: Big Road structure OR pattern_type(streak/pingpong/blocks)
    # ----------------------------------------------------
    bigroad_info = _analyze_bigroad_structure(pb_seq)
    br_pic = bool(bigroad_info.get("picture_present"))
    br_type = str(bigroad_info.get("structure_type") or "NONE")
    br_is_random = bool(bigroad_info.get("is_random"))

    pattern_type = _extract_pattern_type(features)
    pt_ok = _pattern_gate_ok(pattern_type)

    picture_present = bool(br_pic or pt_ok)

    side: Optional[str] = None
    side_tags: List[str] = []
    if picture_present:
        side, side_tags = _decide_side_from_picture(pb_seq, bigroad_info, pattern_type)

    # ----------------------------------------------------
    # HOLD: 그림 미형성 / 방향 미결정
    # ----------------------------------------------------
    if not picture_present or side not in (SIDE_P, SIDE_B):
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len

        tags = []
        if br_is_random:
            tags.extend(["HOLD", "STRICT_RANDOM"])
        else:
            tags.extend(["HOLD", "NO_PICTURE" if not picture_present else "NO_DIRECTION"])
        tags.append(f"BR_TYPE={br_type}")
        if pattern_type is not None:
            tags.append(f"PTYPE={pattern_type}")
        tags.extend(side_tags)

        # 학습 기반 라인 테이블 비활성 태그
        tags.append("LINE_KEY=DISABLED")
        tags.append("LINE_ACTION=DISABLED")
        tags.append("LINE_SRC=DISABLED")

        leader_trust_state = _extract_leader_trust_state(leader_state)
        tags.append(f"LEADER_TRUST={leader_trust_state}")

        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD (NO_PICTURE)" if not picture_present else "HOLD (NO_DIRECTION)",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "chaos": chaos,
                "stability": stability,
                "beauty_score": beauty_score,
                "pattern_score": pattern_score,
                "pattern_symmetry": pattern_symmetry,
                "pattern_energy": pattern_energy,
                "picture_present": bool(picture_present),
                "bigroad_structure": br_type,
                "pattern_type": pattern_type,
                "pass_reason": "HOLD_NO_PICTURE" if not picture_present else "HOLD_NO_DIRECTION",
                "flow_life_state": flow_ctx.state,
                "flow_life_last_side": flow_ctx.last_side,
                "flow_life_probe_hit_streak": flow_ctx.consecutive_probe_hits,
                "flow_life_resolve_dbg": flow_resolve_dbg,
                "no_play_shoe": bool(getattr(flow_ctx, "no_play_shoe", False)),
                "probe_fail_count": int(getattr(flow_ctx, "probe_fail_count", 0)),
                "probe_since_last_normal": bool(flow_ctx.probe_since_last_normal),
                "force_probe_next": bool(flow_ctx.force_probe_next),
                "leader_trust_state": leader_trust_state,
            },
            "pass_reason": "HOLD_NO_PICTURE" if not picture_present else "HOLD_NO_DIRECTION",
        }

    tags: List[str] = []
    tags.append(f"BR_TYPE={br_type}")
    tags.extend(side_tags)

    # ----------------------------------------------------
    # Leader trust (로그/게이트용)
    # ----------------------------------------------------
    leader_trust_state = _extract_leader_trust_state(leader_state)
    tags.append(f"LEADER_TRUST={leader_trust_state}")

    # ----------------------------------------------------
    # China roads (STRICT): validate only
    # ----------------------------------------------------
    china_bcnt, china_marks = _china_b_count(features)
    china_state = _china_health_state(china_bcnt, china_marks)
    tags.append(f"CHINA_STATE={china_state}")
    tags.append(f"CHINA_BCNT={china_bcnt}")  # B=Black(중국점 색), not Banker

    # ----------------------------------------------------
    # Future China Roads (STRICT): required
    # ----------------------------------------------------
    future_if_P, future_if_B = _extract_future_scenarios_strict(features, gpt_analysis)

    # ----------------------------------------------------
    # Prepare GPT input (STRICT)
    # ----------------------------------------------------
    last5, runs = _last5_and_runs_strict(pb_seq)

    pt_for_gpt = _derive_pattern_type_for_gpt(pattern_type, br_type)
    # flow_state는 recommend 내부 lifecycle state 사용(외부 flow.py 상태와 혼동 금지)
    flow_state_for_gpt = str(flow_ctx.state).upper()
    if flow_state_for_gpt not in (FLOW_DEAD, FLOW_TEST, FLOW_ALIVE):
        raise RuntimeError(f"invalid flow_state_for_gpt: {flow_state_for_gpt!r}")

    gpt_input: Dict[str, Any] = {
        "last5": last5,
        "runs": runs,
        "pattern_type": pt_for_gpt,
        "flow_state": flow_state_for_gpt,
        "china_state": str(china_state).upper(),
        "future_if_P": future_if_P,
        "future_if_B": future_if_B,
    }

    # ----------------------------------------------------
    # GPT decision (STRICT): no fallback
    # ----------------------------------------------------
    gpt_dec: GptDecision = gpt_decide(gpt_input)

    # ----------------------------------------------------
    # Hybrid combine: engine(side) + GPT
    # ----------------------------------------------------
    engine_side = str(side)
    final_side = _hybrid_decide(engine_side=engine_side, gpt=gpt_dec)

    # 학습 기반 line table 비활성 태그
    tags.append("LINE_KEY=DISABLED")
    tags.append("LINE_ACTION=DISABLED")
    tags.append("LINE_SRC=DISABLED")

    # ----------------------------------------------------
    # HOLD if weak edge
    # ----------------------------------------------------
    if final_side is None:
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len
        tags.append("HOLD_EDGE_TOO_SMALL")
        return {
            "bet_side": None,
            "bet_unit": 0,
            "entry_type": None,
            "reason": "HOLD (LOW_EDGE)",
            "tags": tags,
            "metrics": {
                "rounds_total": rounds_total,
                "pb_len": pb_len,
                "chaos": chaos,
                "stability": stability,
                "beauty_score": beauty_score,
                "pattern_score": pattern_score,
                "pattern_symmetry": pattern_symmetry,
                "pattern_energy": pattern_energy,
                "picture_present": True,
                "bigroad_structure": br_type,
                "pattern_type": pattern_type,
                "china_state": china_state,
                "china_marks": china_marks,
                "future_if_P": future_if_P,
                "future_if_B": future_if_B,
                "gpt_side": gpt_dec.side,
                "gpt_confidence": gpt_dec.confidence,
                "engine_side": engine_side,
            },
            "pass_reason": "HOLD_LOW_EDGE",
        }

    # ----------------------------------------------------
    # NORMAL eligibility (기존 게이트 유지하되, UI 단순화를 위해 외부 entry_type은 None)
    # ----------------------------------------------------
    normal_structure_flags = features.get("normal_structure_flags") or {}
    normal_confirmation_flags = features.get("normal_confirmation_flags") or {}

    if not isinstance(normal_structure_flags, dict):
        normal_structure_flags = {}
    if not isinstance(normal_confirmation_flags, dict):
        normal_confirmation_flags = {}

    allow_normal, normal_tags = _should_allow_normal(
        pb_seq=pb_seq,
        flow_ctx=flow_ctx,
        beauty_score=beauty_score,
        chaos=chaos,
        stability=stability,
        structure_flags=normal_structure_flags,
        confirm_flags=normal_confirmation_flags,
    )
    tags.extend(normal_tags)

    # 내부 상태 머신/멱등성 유지용 internal_entry_type (외부 표시와 분리)
    internal_entry_type: str
    if flow_ctx.force_probe_next:
        internal_entry_type = ENTRY_PROBE
        tags.append("FORCE_PROBE_NEXT")
        flow_ctx.force_probe_next = False
    else:
        internal_entry_type = ENTRY_NORMAL if allow_normal else ENTRY_PROBE

    # China validation: 붕괴/약화 시 내부적으로는 TEST로 강등
    if china_state == "BROKEN":
        flow_ctx.state = FLOW_TEST
        internal_entry_type = ENTRY_PROBE
        tags.append("CHINA_BROKEN_FORCE_PROBE")
    elif china_state == "WEAK":
        flow_ctx.state = FLOW_TEST
        if internal_entry_type == ENTRY_NORMAL:
            internal_entry_type = ENTRY_PROBE
            tags.append("CHINA_WEAK_DOWNGRADE_PROBE")

    # flow state update
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

    # UI 단순화를 위해 bet_unit은 1 고정, entry_type은 None 반환
    # (UI에서 PROBE 표기/강도 표기 혼란 최소화 목적)
    bet_unit = 1

    metrics: Dict[str, Any] = {
        "rounds_total": rounds_total,
        "pb_len": pb_len,
        "chaos": chaos,
        "stability": stability,
        "beauty_score": beauty_score,
        "pattern_score": pattern_score,
        "pattern_symmetry": pattern_symmetry,
        "pattern_energy": pattern_energy,
        "picture_present": True,
        "bigroad_structure": br_type,
        "pattern_type": pattern_type,
        "china_state": china_state,
        "china_bcnt": china_bcnt,
        "china_marks": china_marks,
        "future_if_P": future_if_P,
        "future_if_B": future_if_B,
        "engine_side": engine_side,
        "gpt_side": gpt_dec.side,
        "gpt_confidence": gpt_dec.confidence,
        "final_side": final_side,
        "flow_life_state": flow_ctx.state,
        "flow_life_last_side": flow_ctx.last_side,
        "flow_life_probe_hit_streak": flow_ctx.consecutive_probe_hits,
        "flow_life_resolve_dbg": flow_resolve_dbg,
        "no_play_shoe": bool(getattr(flow_ctx, "no_play_shoe", False)),
        "probe_fail_count": int(getattr(flow_ctx, "probe_fail_count", 0)),
        "probe_since_last_normal": bool(flow_ctx.probe_since_last_normal),
        "force_probe_next": bool(flow_ctx.force_probe_next),
        "pb_ratio": pb_ratio,
        "leader_trust_state": leader_trust_state,
        "last5": last5,
        "runs": runs,
        "flow_state_for_gpt": flow_state_for_gpt,
        "pattern_type_for_gpt": pt_for_gpt,
        "learned_line_table_enabled": bool(ENABLE_LEARNED_LINE_TABLE),
    }

    return {
        "bet_side": final_side,
        "bet_unit": int(bet_unit),
        "entry_type": None,          # ✅ UI 혼란 제거: 표시용 entry_type 제거
        "reason": "",                # ✅ UI 혼란 제거: 설명 미표시(필요 시 서버 로그에서만 확인)
        "tags": tags,
        "metrics": metrics,
    }