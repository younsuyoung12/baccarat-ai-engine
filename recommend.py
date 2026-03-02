
# -*- coding: utf-8 -*-
"""
recommend.py
====================================================
Baccarat Predictor AI Engine v11.x
Rule-based Betting Recommender (실전 플레이어 기준)

역할
------
- pb_seq(누적 P/B 시퀀스) + features를 기반으로 bet_side / bet_unit / entry_type(PASS/PROBE/NORMAL)를 결정한다.
- 이 모듈은 룰 기반 엔진이며, 폴백(근거 없는 기본값 방향 생성)을 금지한다.

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

변경 요약 (2026-01-01)
----------------------------------------------------
1) 방향 생성은 “그림(picture)”이 확인된 경우에만 허용한다.
   - 그림 정의: (A) Big Road(pb_seq)에서 블록/교대 구조가 확인되거나, (B) pattern.py 결과의 pattern_type이 streak/pingpong/blocks 중 하나.
   - 그림이 없으면 bet_side=None, entry_type=None (판단 보류; PASS 강제 금지).
2) 새 슈/초반 보호: pb_len(길이)만으로 방향을 만들지 않으며, 스트릭/트렌드 수치 단독 기반 방향 생성 로직을 제거한다.
3) Big Road 인식은 “줄/핑퐁 2분법”이 아니라, 단일 스트릭 / 블록 교대 / 핑퐁 교대 / 혼합 블록을 구분한다.
   - 마지막 1~2개 P/B 값만 추종하는 방향 결정 로직을 제거한다.
4) 중국점(BigEye/Small/Cockroach)은 방향을 만들지 않는다.
   - 역할: 현재 Big Road 기반 방향이 “살아있는지/깨졌는지”만 판정.
   - 붕괴 시: NORMAL→PROBE 강등, PROBE→판단 보류.
5) pattern.py는 “방향 생성 입구 필터”로 취급한다.
   - PatternNotReadyError는 파이프라인에서 WARMUP으로 처리되며, 본 모듈은 그 상태에서 방향을 생성하지 않는다.
   - pattern_score는 NORMAL 승격 판단에만 간접 사용(기존 구조/확증 플래그 유지).

변경 요약 (2026-01-23)
----------------------------------------------------
1) models/line_transition_table.json 기반 “FOLLOW/REVERSE 행동 결정” 추가
   - 기존 그림 인식 / 방향 생성 로직은 절대 변경하지 않는다.
   - 방향(side) 결정 직후, 줄 상태(Line State)를 산출하고
     preferred_action(FOLLOW/REVERSE)에 따라 side를 최종 확정한다.
   - tags에 LINE_KEY / LINE_ACTION / LINE_SRC 를 반드시 포함한다.

2) HOLD 빈도 감소(중국점 붕괴 시 PROBE→HOLD 완화)
   - CHINA_BROKEN 상황에서 기존: PROBE면 HOLD로 판단 보류
   - 변경: PROBE 유지(유닛 1 고정), 이유/태그에 명시
   - 주의: 방향 생성/그림 로직은 건드리지 않는다.

3) leader_trust_state 기반 “NORMAL 승격 보조 게이트” 추가
   - 기존 NORMAL 후보(allow_normal=True)인 경우에만 leader_trust_state를 참고한다.
   - leader_trust_state == STRONG 일 때만 NORMAL 유지
   - 그 외(NONE/WEAK/MID/누락)는 NORMAL → PROBE로 강등
   - 방향(side) 변경 금지, leader_signal/leader_source/can_override_side 사용 금지
----------------------------------------------------

절대 금지(유지)
----------------------------------------------------
- 확률(%) 예측 또는 반환
- 다음 수 예측 표현
- 추천/우세/맞출 수 있다는 표현
- recommend.py 외 파일 수정
- engine_state / road / feature 계산 로직 침범
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import os


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

# Line transition table (history action)
LINE_TABLE_PATH = os.path.join("models", "line_transition_table.json")


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
    pending_at_pb_len: Optional[int] = None  # pb_len at the time of decision

    # guard for shoe reset / call skipping
    last_seen_pb_len: int = 0

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

    pb_len = len(pb_seq)
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
        # pending은 “결정 당시 pb_len” 기준으로 “그 다음 1개 결과”에서만 해석한다.
        if pb_len >= at_pb_len + 1:
            last_winner = str(pb_seq[-1]).upper() if pb_len > 0 else None
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
        return float(v)
    except Exception as e:
        raise TypeError(f"{name} must be float-compatible") from e


def _as_int(v: Any, name: str) -> int:
    try:
        return int(v)
    except Exception as e:
        raise TypeError(f"{name} must be int-compatible") from e


# ----------------------------------------------------
# Helpers: Big Road picture detection (structure-based)
# ----------------------------------------------------

def _pb_clean(pb_seq: List[str]) -> List[str]:
    out: List[str] = []
    for x in pb_seq:
        s = str(x).upper()
        if s in (SIDE_P, SIDE_B):
            out.append(s)
    return out


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

    # 2) blocks (11->22, 111->222, 112211 ...)
    #    - 최근 2개 run이 모두 2 이상이면 블록 교대 구조로 인정
    if len(runs) >= 2:
        (s1, l1), (s2, l2) = runs[-2], runs[-1]
        if s1 in (SIDE_P, SIDE_B) and s2 in (SIDE_P, SIDE_B) and s1 != s2 and l1 >= 2 and l2 >= 2:
            target = _infer_target_block_len(runs)
            info["picture_present"] = True
            info["structure_type"] = "BLOCKS"
            info["target_block_len"] = target
            # 블록 길이가 아직 다 안 찼으면 유지, 다 찼으면 전환
            if target and l2 < target:
                info["expected_next_side"] = s2
            else:
                info["expected_next_side"] = s1
            return info

    # 3) mixed blocks (12122 등): 교대 중에 2+블록이 끼어드는 형태
    #    - 최근 run들에서 (len==1과 len>=2)가 공존하고, run의 방향 전환이 반복될 때만 인정
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

    # strict random: 최근 구간에 연속이 없고(전부 1-run), pingpong도 아님
    # (판단 보류로만 사용)
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
            # blocks는 run-length 기반 추정이 불가능하면 방향을 만들지 않는다.
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
# NEW: Line state + History action (FOLLOW/REVERSE)
# ----------------------------------------------------

_LINE_TABLE_CACHE: Optional[Dict[str, Any]] = None
_LINE_TABLE_ERR: Optional[str] = None


def _load_line_table() -> Dict[str, Any]:
    """
    models/line_transition_table.json 로드 (캐시)
    - 파일이 없거나 파싱 실패 시: 빈 dict 반환 + 내부 에러 문자열 저장
    - 엔진 크래시 방지 목적(명시 태그로 노출)
    """
    global _LINE_TABLE_CACHE, _LINE_TABLE_ERR
    if _LINE_TABLE_CACHE is not None:
        return _LINE_TABLE_CACHE

    try:
        if not os.path.exists(LINE_TABLE_PATH):
            _LINE_TABLE_ERR = "FILE_NOT_FOUND"
            _LINE_TABLE_CACHE = {}
            return _LINE_TABLE_CACHE

        with open(LINE_TABLE_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            _LINE_TABLE_ERR = "INVALID_ROOT_TYPE"
            _LINE_TABLE_CACHE = {}
            return _LINE_TABLE_CACHE

        _LINE_TABLE_CACHE = obj
        _LINE_TABLE_ERR = None
        return _LINE_TABLE_CACHE

    except Exception as e:
        _LINE_TABLE_ERR = f"LOAD_ERROR:{type(e).__name__}"
        _LINE_TABLE_CACHE = {}
        return _LINE_TABLE_CACHE


def _flip_side(side: str) -> str:
    return SIDE_B if side == SIDE_P else SIDE_P


def _phase_from_runlen(run_len: int) -> str:
    if run_len <= 2:
        return "START"
    if run_len <= 4:
        return "GROW"
    return "MATURE"


def _infer_line_type_phase(pb_seq: List[str]) -> Tuple[str, str, Dict[str, Any]]:
    """
    LINE_TYPE  : STREAK / BLOCK / ALT_CYCLE / MIXED
    LINE_PHASE : START / GROW / MATURE
    """
    seq = _pb_clean(pb_seq)
    runs = _rle_runs(seq)
    dbg: Dict[str, Any] = {"runs": runs[-6:] if runs else []}

    if not runs:
        return "MIXED", "START", dbg

    _, cur_len = runs[-1]
    phase = _phase_from_runlen(int(cur_len))

    # ALT_CYCLE: 최근 4개가 완전 교대
    if len(seq) >= 4 and _is_alternating(seq[-4:]):
        return "ALT_CYCLE", "START", dbg  # 교대는 run_len=1 기반으로 START 고정

    # BLOCK: 최근 2개 run이 2 이상이며 서로 교대
    if len(runs) >= 2:
        (s1, l1), (s2, l2) = runs[-2], runs[-1]
        if s1 != s2 and int(l1) >= 2 and int(l2) >= 2:
            return "BLOCK", phase, dbg

    # STREAK: 현재 run이 3 이상
    if int(cur_len) >= 3:
        return "STREAK", phase, dbg

    return "MIXED", phase, dbg


def _choose_line_action(line_type: str, line_phase: str) -> Tuple[str, str, str]:
    """
    returns: (line_key_used, action, src)
    action: FOLLOW / REVERSE
    src   : HIST_TABLE / HIST_TABLE_ANY / DEFAULT_FOLLOW / TABLE_MISSING
    """
    table = _load_line_table()

    key = f"{line_type}:{line_phase}"
    key_any = f"{line_type}:ANY"

    # table missing / invalid
    if not table:
        if _LINE_TABLE_ERR is not None:
            return key, "FOLLOW", f"TABLE_MISSING({_LINE_TABLE_ERR})"
        return key, "FOLLOW", "DEFAULT_FOLLOW"

    # exact key
    if key in table and isinstance(table.get(key), dict):
        pref = str(table[key].get("preferred_action") or "").upper().strip()
        if pref in ("FOLLOW", "REVERSE"):
            return key, pref, "HIST_TABLE"

    # ANY fallback
    if key_any in table and isinstance(table.get(key_any), dict):
        pref = str(table[key_any].get("preferred_action") or "").upper().strip()
        if pref in ("FOLLOW", "REVERSE"):
            return key_any, pref, "HIST_TABLE_ANY"

    # if table exists but no key matched -> explicit default
    return key, "FOLLOW", "DEFAULT_FOLLOW"


def _apply_follow_reverse_action(
    pb_seq: List[str],
    side: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    side(그림 기반 방향) 결정 이후,
    line_transition_table 기반 preferred_action(FOLLOW/REVERSE)로 side를 최종 확정한다.
    """
    line_type, line_phase, dbg = _infer_line_type_phase(pb_seq)
    used_key, action, src = _choose_line_action(line_type, line_phase)

    final_side = side
    if action == "REVERSE":
        final_side = _flip_side(side)

    out_dbg: Dict[str, Any] = {
        "line_type": line_type,
        "line_phase": line_phase,
        "line_key": used_key,
        "line_action": action,
        "line_src": src,
        "line_dbg": dbg,
        "side_before": side,
        "side_after": final_side,
    }
    return final_side, out_dbg


# ----------------------------------------------------
# Helpers: China roads (validation only; never direction)
# ----------------------------------------------------

def _last_rb_from_matrix(matrix_json: Any) -> Optional[str]:
    """
    matrix_json: list[list[str]] (JSON string or python list)
    Returns last non-empty mark among ("R","B").
    """
    mat = None
    if matrix_json is None:
        return None
    if isinstance(matrix_json, str):
        try:
            mat = json.loads(matrix_json)
        except Exception:
            return None
    elif isinstance(matrix_json, list):
        mat = matrix_json
    else:
        return None

    try:
        # scan from end
        for col in reversed(mat):
            if not isinstance(col, list):
                continue
            for cell in reversed(col):
                c = str(cell).upper().strip()
                if c in ("R", "B"):
                    return c
    except Exception:
        return None
    return None


def _china_b_count(features: Dict[str, Any]) -> Tuple[int, Dict[str, Optional[str]]]:
    """
    중국점은 색(R/B)만 사용하며, Bank/Player 방향과 무관하다.
    - 붕괴 판정용으로만 B 카운트를 사용(간단/보수적).
    """
    big_eye_last = _last_rb_from_matrix(features.get("big_eye_matrix_json") or features.get("big_eye_matrix"))
    small_last = _last_rb_from_matrix(features.get("small_road_matrix_json") or features.get("small_road_matrix"))
    cock_last = _last_rb_from_matrix(features.get("cockroach_matrix_json") or features.get("cockroach_matrix"))

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
    pb_len = len(pb_seq)

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

    pb_len = len(pb_seq)

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

    # side decision is allowed only when picture_present is True
    side: Optional[str] = None
    side_tags: List[str] = []
    if picture_present:
        side, side_tags = _decide_side_from_picture(pb_seq, bigroad_info, pattern_type)

    # ----------------------------------------------------
    # HOLD 1) 그림 미형성 / 방향 미결정 (정상 상태)
    # ----------------------------------------------------
    if not picture_present or side not in (SIDE_P, SIDE_B):
        flow_ctx.reset_dead()
        flow_ctx.last_seen_pb_len = pb_len

        tags: List[str] = []
        if br_is_random:
            tags.extend(["HOLD", "STRICT_RANDOM"])
        else:
            tags.extend(["HOLD", "NO_PICTURE" if not picture_present else "NO_DIRECTION"])
        tags.append(f"BR_TYPE={br_type}")
        if pattern_type is not None:
            tags.append(f"PTYPE={pattern_type}")
        tags.extend(side_tags)

        # 줄 상태 태그는 참고용(행동 결정을 못 하는 경우)
        line_type, line_phase, _ = _infer_line_type_phase(pb_seq)
        tags.append(f"LINE_KEY={line_type}:{line_phase}")
        tags.append("LINE_ACTION=NA")
        tags.append("LINE_SRC=NA")

        # leader trust는 로그용으로만 남김(결정에는 사용 불가)
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
    if pattern_type is not None:
        tags.append(f"PTYPE={pattern_type}")
    tags.extend(side_tags)

    # ----------------------------------------------------
    # NEW: Apply FOLLOW/REVERSE action from history table
    # ----------------------------------------------------
    side_before = str(side)
    side, line_dbg = _apply_follow_reverse_action(pb_seq=pb_seq, side=str(side))

    tags.append(f"LINE_KEY={line_dbg['line_key']}")
    tags.append(f"LINE_ACTION={line_dbg['line_action']}")
    tags.append(f"LINE_SRC={line_dbg['line_src']}")

    # ----------------------------------------------------
    # Leader trust (로그/게이트용)
    # ----------------------------------------------------
    leader_trust_state = _extract_leader_trust_state(leader_state)
    tags.append(f"LEADER_TRUST={leader_trust_state}")

    # ----------------------------------------------------
    # China roads: validate only (never direction)
    # ----------------------------------------------------
    china_bcnt, china_marks = _china_b_count(features)
    china_state = _china_health_state(china_bcnt, china_marks)
    tags.append(f"CHINA_STATE={china_state}")
    tags.append(f"CHINA_BCNT={china_bcnt}")  # B=Black(중국점 색), not Banker

    # ----------------------------------------------------
    # NORMAL eligibility (gated) + base entry selection
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

    # base unit (beauty 기반)
    if beauty_score >= 85:
        normal_unit = 3
    elif beauty_score >= 70:
        normal_unit = 2
    else:
        normal_unit = 1

    probe_unit = 1

    # ----------------------------------------------------
    # Forced PROBE after miss (PASS 고착 제거)
    # ----------------------------------------------------
    if flow_ctx.force_probe_next:
        entry_type = ENTRY_PROBE
        bet_unit = probe_unit
        reason = "PROBE (FORCED_AFTER_MISS)"
        tags.append("FORCE_PROBE_NEXT")
        # 강제 PROBE는 1회 소비
        flow_ctx.force_probe_next = False
    else:
        # 기본: allow_normal이면 NORMAL, 아니면 PROBE
        if allow_normal:
            entry_type = ENTRY_NORMAL
            bet_unit = normal_unit
            reason = "NORMAL (ALLOW_GATED)"
        else:
            entry_type = ENTRY_PROBE
            bet_unit = probe_unit
            reason = "PROBE (DEFAULT)"

    # ----------------------------------------------------
    # leader_trust_state NORMAL gate
    # - 기존 NORMAL 후보일 때만 적용 (단독 조건 금지)
    # ----------------------------------------------------
    if entry_type == ENTRY_NORMAL:
        if leader_trust_state == "STRONG":
            tags.append("LEADER_TRUST_GATE_ALLOW")
        else:
            entry_type = ENTRY_PROBE
            bet_unit = probe_unit
            reason = "PROBE (DOWNGRADE_BY_LEADER_TRUST)"
            tags.append("LEADER_TRUST_GATE_DENY")
            tags.append("NORMAL_TO_PROBE_BY_LEADER_TRUST")

    # ----------------------------------------------------
    # China validation: 붕괴 시 강등/판단보류 (HOLD 빈도 감소)
    # ----------------------------------------------------
    if china_state == "BROKEN":
        if entry_type == ENTRY_NORMAL:
            entry_type = ENTRY_PROBE
            bet_unit = 1
            reason = "PROBE (DOWNGRADE_BY_CHINA_BROKEN)"
            tags.append("CHINA_DOWNGRADE_NORMAL_TO_PROBE")
            # 흐름도 TEST로 되돌림(살아있지 않음)
            flow_ctx.state = FLOW_TEST
        else:
            # 변경: HOLD 대신 PROBE 유지 (유닛 1 고정)
            entry_type = ENTRY_PROBE
            bet_unit = 1
            reason = "PROBE (CHINA_BROKEN_KEEP)"
            tags.append("CHINA_BROKEN_KEEP_PROBE")
            flow_ctx.state = FLOW_TEST

    elif china_state == "WEAK":
        if entry_type == ENTRY_NORMAL:
            entry_type = ENTRY_PROBE
            bet_unit = 1
            reason = "PROBE (DOWNGRADE_BY_CHINA_WEAK)"
            tags.append("CHINA_DOWNGRADE_NORMAL_TO_PROBE")
            flow_ctx.state = FLOW_TEST

    # ----------------------------------------------------
    # FLOW_LIFE state transition + pending set
    # ----------------------------------------------------
    # sanity clamp
    if bet_unit < 0:
        bet_unit = 0
    if bet_unit > MAX_UNIT:
        bet_unit = MAX_UNIT

    if entry_type == ENTRY_NORMAL:
        flow_ctx.state = FLOW_ALIVE
        flow_ctx.last_side = side
        flow_ctx.probe_since_last_normal = False
    elif entry_type == ENTRY_PROBE:
        flow_ctx.state = FLOW_TEST
        flow_ctx.last_side = side
        flow_ctx.probe_since_last_normal = True
    else:
        # should not happen; contract error
        raise RuntimeError("invalid entry_type")

    # set pending for next resolution
    flow_ctx.pending_bet_side = side
    flow_ctx.pending_entry_type = entry_type
    flow_ctx.pending_at_pb_len = pb_len

    flow_ctx.last_seen_pb_len = pb_len

    # ----------------------------------------------------
    # Output
    # ----------------------------------------------------
    metrics: Dict[str, Any] = {
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
        "china_state": china_state,
        "china_bcnt": china_bcnt,
        "china_marks": china_marks,
        "flow_life_state": flow_ctx.state,
        "flow_life_last_side": flow_ctx.last_side,
        "flow_life_probe_hit_streak": flow_ctx.consecutive_probe_hits,
        "flow_life_resolve_dbg": flow_resolve_dbg,
        "no_play_shoe": bool(getattr(flow_ctx, "no_play_shoe", False)),
        "probe_fail_count": int(getattr(flow_ctx, "probe_fail_count", 0)),
        "probe_since_last_normal": bool(flow_ctx.probe_since_last_normal),
        "force_probe_next": bool(flow_ctx.force_probe_next),
        "pb_ratio": pb_ratio,
        # line action debug
        "line_action_dbg": line_dbg,
        "side_before_line_action": side_before,
        "side_after_line_action": side,
        # leader trust
        "leader_trust_state": leader_trust_state,
    }

    return {
        "bet_side": side,
        "bet_unit": int(bet_unit),
        "entry_type": entry_type,
        "reason": reason,
        "tags": tags,
        "metrics": metrics,
    }