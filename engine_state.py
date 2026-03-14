# -*- coding: utf-8 -*-
# engine_state.py
"""
engine_state.py
====================================================
Baccarat Predictor AI Engine – 엔진 상태 저장/복원 모듈 (RULE-ONLY)

역할
- Big Road(원본), 패턴 히스토리, Road Leader 상태를 JSON 파일로 저장/복원한다.
- 로드 파생 데이터(Big Eye/Small/Cockroach/매트릭스 등)는 복원 후
  road.recompute_all_roads()로 무결성 있게 재계산한다.

중요 정책
- STRICT · NO-FALLBACK · FAIL-FAST
- 스키마/타입/필수키/값 무결성 위반 시 즉시 예외
- readiness 미달은 load 실패 사유가 아니다(UI/연속 운용을 위해).
- trade readiness 와 ui readiness 는 분리한다.

변경 요약 (2026-03-14) — v12.1
----------------------------------------------------
1) 상태 파일 무결성 검증 강화
   - big_road 원소는 P/B/T만 허용
   - pattern_history 원소는 finite number만 허용
   - leader_state 는 dict 타입을 강제하고 road_leader.set_state()로 최종 검증
2) 복원 후 파생 로드 무결성 검증 추가
   - road.recompute_all_roads() 이후 road.validate_roadmap_integrity()를 호출
   - 복원된 상태가 손상되었으면 즉시 예외
3) runtime readiness 계약 강화
   - get_ui_readiness / get_trade_readiness 에서도 big_road 계약을 검증
   - 메모리 오염 상태를 조용히 통과시키지 않는다
4) schema version 업
   - SCHEMA_VERSION = 4
   - load는 schema_version 1/2/3/4 지원
   - v1/v2/v3 잔재 키(micro_learning/meta_learning)는 있으면 dict 타입만 검증 후 무시
----------------------------------------------------

(기존 변경 요약 2026-03-04)
----------------------------------------------------
1) meta_learning 완전 제거
   - import meta_learning 삭제
   - save/load에서 micro_learning/meta_learning 저장/복원 제거
2) load는 schema_version 1/2/3/4를 명시적으로 지원
----------------------------------------------------

(기존 변경 요약 2025-12-24)
----------------------------------------------------
- Deadlock 제거: Leader 통계를 "차단 조건"에서 "워밍업 조건"으로 변경
  - MIN_BIG_ROAD_LEN(기본 5)만 충족하면 get_trade_readiness()는 True를 반환한다.
  - Leader stats 부족 / last_signals 없음은 NOT READY로 막지 않고,
    reason에 "READY (leader warmup: ...)" 형태로만 기록한다.
----------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pattern
import road
import road_leader

logger = logging.getLogger(__name__)

STATE_FILE = "engine_state.json"

# v4: rule-only strict validation 강화
SCHEMA_VERSION = 4
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3, 4)

# -----------------------------
# UI 표시(패널 렌더) 최소 조건
# -----------------------------
MIN_UI_BIG_ROAD_LEN = 1  # BigRoad가 비어있지 않으면 UI는 표시 가능

# -----------------------------
# 트레이딩(진입 판단) 가능 최소 조건
# -----------------------------
MIN_BIG_ROAD_LEN = 5       # 최소 판수(원본 Big Road)

# Leader는 "차단 조건"이 아니라 "워밍업 완료 기준"으로만 사용한다.
MIN_LEADER_TOTAL = 5       # 리더 통계(최근 적중 추적) 워밍업 완료 기준
REQUIRE_ANY_SIGNAL = True  # last_signals 중 하나라도 존재해야 함(워밍업 완료 기준)


class EngineStateError(RuntimeError):
    """엔진 상태 저장/복원/검증 오류(폴백 금지)."""


class EngineStateNotReadyError(EngineStateError):
    """
    베팅 준비 미달(=베팅 판단/진입 금지).
    - 이 예외는 '베팅 직전'에만 사용해야 한다.
    - load 단계에서 이 예외로 앱을 리셋시키면 BigRoad가 비어 UI 패널이 사라지는 문제가 발생한다.
    """


def _as_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{key} must be int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"{key} must be int, got {type(value).__name__}")


def _as_finite_float(value: Any, *, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{key} must be float, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be float, got {type(value).__name__}")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{key} must be finite")
    return x


def _require_dict(state: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in state:
        raise ValueError(f"state missing required key: {key}")
    val = state[key]
    if not isinstance(val, dict):
        raise TypeError(f"state[{key}] must be dict, got {type(val).__name__}")
    return val


def _require_list(state: Dict[str, Any], key: str) -> list:
    if key not in state:
        raise ValueError(f"state missing required key: {key}")
    val = state[key]
    if not isinstance(val, list):
        raise TypeError(f"state[{key}] must be list, got {type(val).__name__}")
    return val


def _optional_dict_strict(state: Dict[str, Any], key: str) -> None:
    """
    옵션 키는 '있으면 타입만 엄격 검증'하고 사용하지 않는다.
    - 호환성 목적(구버전 state 파일에 남아있을 수 있음)
    """
    if key not in state:
        return
    val = state[key]
    if not isinstance(val, dict):
        raise TypeError(f"state[{key}] must be dict if present, got {type(val).__name__}")


def _validate_big_road_list(value: Any, *, key: str) -> List[str]:
    if not isinstance(value, list):
        raise TypeError(f"{key} must be list, got {type(value).__name__}")

    out: List[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(f"{key}[{idx}] must be str, got {type(item).__name__}")
        s = item.strip().upper()
        if s not in ("P", "B", "T"):
            raise ValueError(f"{key}[{idx}] invalid value: {item!r} (allowed: 'P','B','T')")
        out.append(s)
    return out


def _validate_pattern_history_list(value: Any, *, key: str) -> List[float]:
    if not isinstance(value, list):
        raise TypeError(f"{key} must be list, got {type(value).__name__}")

    out: List[float] = []
    for idx, item in enumerate(value):
        out.append(_as_finite_float(item, key=f"{key}[{idx}]"))
    return out


def _serialize_leader_state(leader_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    leader_state 내부 window(deque)를 JSON 저장 가능한 list로 변환.
    road_leader.get_state()는 deepcopy를 반환한다고 가정하며, 원본 상태는 오염시키지 않는다.
    """
    if not leader_state:
        return leader_state

    stats = leader_state.get("stats")
    if isinstance(stats, dict):
        for _, stat in stats.items():
            if isinstance(stat, dict) and "window" in stat:
                w = stat.get("window")
                stat["window"] = [] if w is None else list(w)
    return leader_state


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix="engine_state_", suffix=".json", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.debug("[ENGINE-STATE] tmp cleanup failed: %s", e)


def _extract_leader_max_total(leader_state: Any) -> int:
    if not leader_state:
        return 0
    if not isinstance(leader_state, dict):
        raise TypeError(f"leader_state must be dict or None, got {type(leader_state).__name__}")

    stats = leader_state.get("stats")
    if stats is None:
        return 0
    if not isinstance(stats, dict):
        raise TypeError(f"leader_state.stats must be dict, got {type(stats).__name__}")

    max_total = 0
    for _, st in stats.items():
        if not isinstance(st, dict):
            raise TypeError("leader_state.stats[*] must be dict")
        total = st.get("total", 0)
        total_i = _as_int(total, key="leader_state.stats.total")
        if total_i < 0:
            raise ValueError("leader_state.stats.total must be >= 0")
        if total_i > max_total:
            max_total = total_i
    return max_total


def _extract_any_signal_exists(leader_state: Any) -> bool:
    if not leader_state:
        return False
    if not isinstance(leader_state, dict):
        raise TypeError(f"leader_state must be dict or None, got {type(leader_state).__name__}")

    last_signals = leader_state.get("last_signals")
    if last_signals is None:
        return False
    if not isinstance(last_signals, dict):
        raise TypeError(f"leader_state.last_signals must be dict, got {type(last_signals).__name__}")

    for _, sig in last_signals.items():
        if sig is not None:
            return True
    return False


def _get_runtime_big_road_strict() -> List[str]:
    big_road = getattr(road, "big_road", None)
    return _validate_big_road_list(big_road, key="road.big_road")


def _validate_restored_runtime_integrity() -> None:
    if not hasattr(road, "recompute_all_roads"):
        raise EngineStateError("road.recompute_all_roads missing (contract violation)")

    road.recompute_all_roads()

    if not hasattr(road, "validate_roadmap_integrity"):
        raise EngineStateError("road.validate_roadmap_integrity missing (contract violation)")

    ok, reason = road.validate_roadmap_integrity()
    if not isinstance(ok, bool):
        raise EngineStateError("road.validate_roadmap_integrity must return (bool, str)")
    if not ok:
        raise EngineStateError(f"restored roadmap integrity failed: {reason}")


# -----------------------------
# UI 표시 준비도 (패널 렌더)
# -----------------------------
def get_ui_readiness() -> Tuple[bool, str]:
    """
    UI 표시를 위한 최소 준비도.
    - 베팅 준비도와 분리한다.
    """
    big_road = _get_runtime_big_road_strict()
    big_road_len = len(big_road)
    if big_road_len < MIN_UI_BIG_ROAD_LEN:
        return False, f"BigRoad empty: {big_road_len} < {MIN_UI_BIG_ROAD_LEN}"
    return True, "READY"


# -----------------------------
# 베팅(진입 판단) 준비도
# -----------------------------
def get_trade_readiness() -> Tuple[bool, str]:
    """
    현재 메모리 상태가 '분석 기반 진입 판단(=베팅)'을 수행할 준비가 되었는지 점검한다.

    정책:
    - MIN_BIG_ROAD_LEN(기본 5)만 충족하면 트레이딩은 READY로 본다.
    - Leader 통계/시그널은 "워밍업 완료 기준"으로만 사용하며,
      부족하더라도 트레이딩을 차단하지 않는다(Deadlock 제거).
    """
    big_road = _get_runtime_big_road_strict()
    big_road_len = len(big_road)

    if big_road_len < MIN_BIG_ROAD_LEN:
        return False, f"BigRoad insufficient: {big_road_len} < {MIN_BIG_ROAD_LEN}"

    # Leader 상태는 타입/스키마만 엄격 검증(폴백 금지).
    leader_state = road_leader.get_state()
    max_total = _extract_leader_max_total(leader_state)
    any_sig = _extract_any_signal_exists(leader_state)

    leader_total_ok = (max_total >= MIN_LEADER_TOTAL)
    leader_sig_ok = (not REQUIRE_ANY_SIGNAL) or any_sig

    if leader_total_ok and leader_sig_ok:
        return True, "READY"

    parts = []
    if not leader_total_ok:
        parts.append(f"stats max_total {max_total} < {MIN_LEADER_TOTAL}")
    if REQUIRE_ANY_SIGNAL and not leader_sig_ok:
        parts.append("last_signals all None")

    reason = "READY (leader warmup: " + ", ".join(parts) + ")"
    return True, reason


# (호환) 기존 함수명 유지
def get_readiness() -> Tuple[bool, str]:
    return get_trade_readiness()


def assert_ready_or_raise() -> None:
    """
    폴백 금지.
    베팅 준비가 안 됐으면 즉시 예외로 차단한다.

    주의:
    - 이 함수는 '베팅 직전'에만 호출해야 한다.
    - load 단계에서 호출하면 UI/상태가 리셋되는 루프가 발생할 수 있다.
    """
    ok, reason = get_trade_readiness()
    if ok:
        return

    logger.error("[ENGINE-STATE] NOT READY (NO BET): %s", reason)
    raise EngineStateNotReadyError(f"ENGINE NOT READY (NO BET): {reason}")


def save_engine_state(*, last_decision: Dict[str, Any] | None = None) -> None:
    big_road = _get_runtime_big_road_strict()
    pattern_history = _validate_pattern_history_list(
        getattr(pattern, "pattern_score_history", None),
        key="pattern.pattern_score_history",
    )

    leader_state = road_leader.get_state()
    if not isinstance(leader_state, dict):
        raise EngineStateError(f"road_leader.get_state() must return dict, got {type(leader_state).__name__}")
    leader_state = _serialize_leader_state(leader_state)

    if last_decision is None:
        last_decision_obj: Dict[str, Any] = {}
    elif isinstance(last_decision, dict):
        last_decision_obj = last_decision
    else:
        raise TypeError(f"last_decision must be dict or None, got {type(last_decision).__name__}")

    ui_ok, ui_reason = get_ui_readiness()
    trade_ok, trade_reason = get_trade_readiness()

    state: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "saved_at_utc": datetime_utc_iso(),

        # 원본 Big Road(타이 포함)만이 "정답 데이터"
        "big_road": list(big_road),

        # 파생 로드는 저장하지 않는다(복원 후 recompute_all_roads()로 무결성 재계산).
        "pattern_history": list(pattern_history),

        # meta_learning 제거
        "leader_state": leader_state,

        # 관측용(분리)
        "ui_ready": bool(ui_ok),
        "ui_ready_reason": str(ui_reason),

        # 기존 키는 '베팅 준비도'로 유지(leader warmup이어도 True 가능)
        "engine_ready": bool(trade_ok),
        "engine_ready_reason": str(trade_reason),

        "last_decision": last_decision_obj,
    }

    _atomic_write_json(STATE_FILE, state)
    logger.info(
        "[ENGINE-STATE] saved(v%d): ui_ready=%s(%s) trade_ready=%s(%s)",
        SCHEMA_VERSION, ui_ok, ui_reason, trade_ok, trade_reason
    )


def load_engine_state(*, strict_ready: bool = False) -> None:
    """
    strict_ready=False (기본):
      - 스키마/타입/필수키 무결성만 엄격 검증하고 정상 로드한다.
      - 베팅 준비도 미달은 예외로 중단하지 않는다(표시/연속 운용을 위해).

    strict_ready=True:
      - 로드 후 베팅 준비도까지 검사한다.
      - 단, load 자체를 readiness로 실패시키지 않는다(패널/상태 리셋 루프 방지).
      - 대신 로그로만 경고를 남긴다.
    """
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except FileNotFoundError as e:
        logger.exception("[ENGINE-STATE] state file missing: %s", STATE_FILE)
        raise EngineStateNotReadyError(f"state file missing: {STATE_FILE}") from e
    except json.JSONDecodeError as e:
        logger.exception("[ENGINE-STATE] JSON decode failed: %s", STATE_FILE)
        raise EngineStateError(f"engine_state.json JSON 파싱 실패: {e}") from e

    if not isinstance(state, dict):
        raise TypeError("engine_state.json 최상위는 dict여야 합니다.")

    ver = state.get("schema_version")
    if ver is None:
        raise EngineStateError("engine_state.json missing schema_version (폴백 금지)")
    ver_i = _as_int(ver, key="schema_version")
    if ver_i not in SUPPORTED_SCHEMA_VERSIONS:
        raise EngineStateError(
            f"engine_state schema_version unsupported: {ver_i} (supported={SUPPORTED_SCHEMA_VERSIONS})"
        )

    # v1 레거시 필드는 '검증만' 하고 사용하지 않는다.
    if ver_i == 1:
        if "turbulence_counter" in state:
            _as_int(state["turbulence_counter"], key="turbulence_counter")
        if "entry_momentum" in state:
            _as_int(state["entry_momentum"], key="entry_momentum")

    # 구버전 잔재 키는 있으면 dict 타입만 검증 후 무시
    _optional_dict_strict(state, "micro_learning")
    _optional_dict_strict(state, "meta_learning")

    # 필수 키 검증
    big_road_list_raw = _require_list(state, "big_road")
    pattern_hist_raw = _require_list(state, "pattern_history")
    leader_state = _require_dict(state, "leader_state")

    big_road_list = _validate_big_road_list(big_road_list_raw, key="state.big_road")
    pattern_hist = _validate_pattern_history_list(pattern_hist_raw, key="state.pattern_history")

    # ---- 원본 상태 복원 ----
    road.big_road = list(big_road_list)
    pattern.pattern_score_history = list(pattern_hist)
    road_leader.set_state(leader_state)

    # ---- 파생 로드맵 재계산 + 무결성 검증 ----
    _validate_restored_runtime_integrity()

    # ---- readiness는 load 실패 사유가 아니다 ----
    ui_ok, ui_reason = get_ui_readiness()
    trade_ok, trade_reason = get_trade_readiness()

    if strict_ready and not trade_ok:
        logger.warning("[ENGINE-STATE] loaded but TRADE NOT READY: %s", trade_reason)

    logger.info(
        "[ENGINE-STATE] loaded(v%d): ui_ready=%s(%s) trade_ready=%s(%s) big_road_len=%d",
        ver_i, ui_ok, ui_reason, trade_ok, trade_reason, len(road.big_road)
    )


def datetime_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"