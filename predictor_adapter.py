# -*- coding: utf-8 -*-
"""
predictor_adapter.py
====================================================
Baccarat Predictor AI Engine v12.1
STRICT Rule Adapter (Deterministic · No GPT)

변경 요약 (2026-03-14)
----------------------------------------------------
1) recommend.py 시그니처 정합성 수정
   - recommend.recommend_bet(...) 호출에서
     gpt_analysis / mode / alerts 인자 제거
   - 새 시그니처:
     recommend_bet(pb_seq, features, leader_state, meta)
2) Source of truth 정리
   - leader_state는 features_dict["leader_state"]를 그대로 사용
   - app.py / recommend.py / features.py 계약 정합성 유지
3) GPT 잔재 제거
   - 실제 GPT 호출/추론/설명 생성 없음
   - app.py 호환용 필드도 제거하고 최소 응답만 반환
4) STRICT 유지
   - 누락/타입 위반/셀 값 위반 시 즉시 예외
   - 조용한 continue/pass/fallback 금지
----------------------------------------------------

정책
------
STRICT · NO-FALLBACK · FAIL-FAST
- 누락/불일치/스키마 위반 → 즉시 예외(RuntimeError/TypeError/ValueError)
- 조용한 continue/pass 금지
- 임의 기본값 생성 금지

중요
------
- /predict에서 TIE(T)는 app.py에서 선차단되어야 하며,
  이 함수로 들어오면 계약 위반으로 예외 처리한다.
- 이 모듈은 설명 생성기가 아니다.
- rule engine이 최종 bet_side / bet_unit / entry_type을 결정한다.

반환
------
{
  "ai_ok": bool,
  "features": dict,
  "bet": dict,
  "rl_reward": None
}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import road
import features
import future_simulator
import recommend
from engine_state import save_engine_state

IS_RESETTING = False


# -----------------------------
# STRICT helpers
# -----------------------------
def _require_dict(v: Any, name: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        raise TypeError(f"{name} must be dict, got {type(v).__name__}")
    return v


def _require_list(v: Any, name: str) -> List[Any]:
    if not isinstance(v, list):
        raise TypeError(f"{name} must be list, got {type(v).__name__}")
    return v


def _normalize_winner(prev_round_winner: Optional[str]) -> str:
    if prev_round_winner is None:
        raise ValueError("prev_round_winner is required (P/B)")
    if not isinstance(prev_round_winner, str):
        raise TypeError("prev_round_winner must be str")
    s = prev_round_winner.strip().upper()
    if s not in ("P", "B", "T"):
        raise ValueError(f"invalid prev_round_winner: {prev_round_winner!r} (expected 'P'/'B'/'T')")
    return s


def _safe_save_state() -> None:
    save_engine_state()


def _assert_future_scenarios_strict(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    fs = features_dict.get("future_scenarios")
    if fs is None:
        raise RuntimeError("features.future_scenarios missing (required)")
    fs = _require_dict(fs, "features.future_scenarios")

    if "P" not in fs or "B" not in fs:
        raise RuntimeError("features.future_scenarios must contain keys 'P' and 'B'")

    _require_dict(fs.get("P"), "features.future_scenarios.P")
    _require_dict(fs.get("B"), "features.future_scenarios.B")
    return fs


def _assert_bet_contract_strict(bet: Dict[str, Any]) -> None:
    bet = _require_dict(bet, "bet")

    required_keys = {"bet_side", "bet_unit", "entry_type", "reason", "tags", "metrics"}
    missing = required_keys - set(bet.keys())
    if missing:
        raise RuntimeError(f"bet missing required keys: {sorted(missing)}")

    bet_side = bet.get("bet_side")
    if bet_side is not None and bet_side not in ("P", "B"):
        raise RuntimeError(f"bet.bet_side invalid: {bet_side!r} (expected 'P'/'B'/None)")

    bet_unit = bet.get("bet_unit")
    if not isinstance(bet_unit, int):
        raise RuntimeError(f"bet.bet_unit must be int, got {type(bet_unit).__name__}")
    if bet_unit < 0:
        raise RuntimeError(f"bet.bet_unit must be >= 0, got {bet_unit}")

    entry_type = bet.get("entry_type")
    if entry_type is not None and entry_type not in ("PROBE", "NORMAL"):
        raise RuntimeError(f"bet.entry_type invalid: {entry_type!r} (expected 'PROBE'/'NORMAL'/None)")

    if not isinstance(bet.get("reason"), str):
        raise RuntimeError("bet.reason must be str")

    tags = bet.get("tags")
    if not isinstance(tags, list) or any(not isinstance(x, str) for x in tags):
        raise RuntimeError("bet.tags must be list[str]")

    metrics = bet.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("bet.metrics must be dict")


def _normalize_china_matrix_strict(matrix: Any, name: str) -> List[List[str]]:
    if matrix is None:
        raise RuntimeError(f"{name} missing (required)")

    if not isinstance(matrix, list):
        raise TypeError(f"{name} must be list, got {type(matrix).__name__}")

    out: List[List[str]] = []
    for ci, col in enumerate(matrix):
        if not isinstance(col, list):
            raise TypeError(f"{name}[{ci}] must be list, got {type(col).__name__}")
        new_col: List[str] = []
        for ri, cell in enumerate(col):
            if cell is None:
                new_col.append("")
                continue
            if not isinstance(cell, str):
                raise TypeError(f"{name}[{ci}][{ri}] must be str or None, got {type(cell).__name__}")
            s = cell.strip().upper()
            if s not in ("", "R", "B"):
                raise RuntimeError(f"{name}[{ci}][{ri}] invalid cell: {cell!r} (allowed: '', 'R', 'B', None)")
            new_col.append(s)
        out.append(new_col)
    return out


def _inject_china_matrices_strict(features_dict: Dict[str, Any]) -> None:
    if not isinstance(features_dict, dict):
        raise TypeError("features_dict must be dict")

    be = getattr(road, "big_eye_matrix", None)
    sm = getattr(road, "small_road_matrix", None)
    ck = getattr(road, "cockroach_matrix", None)

    features_dict["big_eye_matrix"] = _normalize_china_matrix_strict(be, "road.big_eye_matrix")
    features_dict["small_road_matrix"] = _normalize_china_matrix_strict(sm, "road.small_road_matrix")
    features_dict["cockroach_matrix"] = _normalize_china_matrix_strict(ck, "road.cockroach_matrix")


def _build_leader_state_strict(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Source of truth:
    - features_dict["leader_state"] 가 반드시 있어야 한다.
    - recommend.py는 이 값을 직접 사용한다.
    """
    if not isinstance(features_dict, dict):
        raise TypeError("features_dict must be dict")

    leader_state = features_dict.get("leader_state")
    leader_state = _require_dict(leader_state, "features_dict.leader_state")

    required = {"leader_confidence", "leader_trust_state", "leader_signal"}
    missing = required - set(leader_state.keys())
    if missing:
        raise RuntimeError(f"leader_state missing required keys: {sorted(missing)}")

    return leader_state


# -----------------------------
# Main pipeline
# -----------------------------
def run_ai_pipeline(
    prev_round_winner: Optional[str] = None,
    ai_recent_results: Optional[List[int]] = None,
    ai_streak_lose: Optional[int] = None,
) -> Dict[str, Any]:
    """
    STRICT pipeline:
    - winner는 반드시 P/B 이어야 한다(T는 app.py에서 선차단)
    - future_scenarios는 반드시 존재해야 한다
    - recommend.py는 deterministic rule engine이라고 가정한다.
    """
    if IS_RESETTING:
        raise RuntimeError("RESETTING: run_ai_pipeline called during reset")

    if ai_recent_results is not None:
        _require_list(ai_recent_results, "ai_recent_results")
        if any(not isinstance(x, int) for x in ai_recent_results):
            raise TypeError("ai_recent_results must be list[int]")

    if not isinstance(ai_streak_lose, int):
        raise TypeError(f"ai_streak_lose must be int, got {type(ai_streak_lose).__name__}")

    winner = _normalize_winner(prev_round_winner)
    if winner == "T":
        raise RuntimeError("CONTRACT_VIOLATION: run_ai_pipeline must not be called with winner='T'")

    # 1) Feature 생성
    features_dict = _require_dict(features.build_feature_payload_v3(winner), "features.build_feature_payload_v3()")

    # 2) PB sequence 주입
    pb_seq = road.get_pb_sequence()
    pb_seq = _require_list(pb_seq, "road.get_pb_sequence()")
    features_dict["pb_seq"] = pb_seq

    # 3) China matrices 주입
    _inject_china_matrices_strict(features_dict)

    # 4) chaos/stability alias 매핑 (동등 값 복제)
    if "flow_chaos_risk" not in features_dict:
        raise KeyError("required key missing: flow_chaos_risk")
    if "flow_stability" not in features_dict:
        raise KeyError("required key missing: flow_stability")

    features_dict["chaos"] = float(features_dict["flow_chaos_risk"])
    features_dict["stability"] = float(features_dict["flow_stability"])

    # 5) FUTURE CHINA ROADS 검증 + merge
    fs = _assert_future_scenarios_strict(features_dict)

    merged = future_simulator.merge_future_china_roads(
        fs,
        include_two_step=True,
        max_rows=6,
    )
    merged = _require_dict(merged, "future_simulator.merge_future_china_roads()")
    features_dict["future_scenarios"] = merged
    _assert_future_scenarios_strict(features_dict)

    # 6) leader_state 구성
    leader_state = _build_leader_state_strict(features_dict)

    # 7) meta
    meta: Dict[str, Any] = {}
    if "meta" in features_dict:
        meta_obj = features_dict["meta"]
        meta = _require_dict(meta_obj, "features_dict.meta")

    # 8) recommend 호출 (v12.1 시그니처)
    bet = recommend.recommend_bet(
        pb_seq=pb_seq,
        features=features_dict,
        leader_state=leader_state,
        meta=meta,
    )
    bet = _require_dict(bet, "recommend.recommend_bet()")
    _assert_bet_contract_strict(bet)

    # 9) 최소 응답 구성
    resp: Dict[str, Any] = {
        "ai_ok": True,
        "features": features_dict,
        "bet": bet,
        "rl_reward": None,
    }

    # 10) 상태 저장
    _safe_save_state()

    return resp