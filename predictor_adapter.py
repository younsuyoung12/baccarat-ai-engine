# -*- coding: utf-8 -*-
"""
predictor_adapter.py
====================================================
Baccarat Predictor AI Engine v11.x (STRICT Hybrid: Rule Engine + GPT Pattern Interpreter)

역할
------
- features.build_feature_payload_v3() 로 Feature JSON 생성
- future_simulator 를 이용해 FUTURE CHINA ROADS(매우 중요) 정보를 STRICT로 생성/검증
- recommend.recommend_bet() 으로 최종 bet_side(P/B) 결정 (recommend 내부에서 엔진 판단 + GPT 판단 결합)
- engine_state.save_engine_state() 로 매 라운드 상태 영구 저장

정책
------
STRICT · NO-FALLBACK · FAIL-FAST
- 누락/불일치/스키마 위반 → 즉시 예외(RuntimeError)
- 조용한 continue/pass 금지
- 임의 기본값 생성 금지

중요
------
- 이 모듈은 "GPT 분석/해설"을 만들지 않는다.
- UI에는 P/B만 필요하므로, 분석(comment) 생성은 하지 않는다.
- /predict에서 TIE(T)는 app.py에서 선차단되어야 하며, 이 함수로 들어오면 계약 위반으로 예외 처리한다.

반환
------
{
  "ai_ok": bool,
  "features": dict,
  "gpt_raw": dict|None,          # 사용 안 함(호환용)
  "ml_raw": None,                # 사용 안 함(호환용)
  "ml_reference": None,          # 사용 안 함(호환용)
  "ai_decision": dict,           # 최소 스키마(호환용)
  "alert_message": None,
  "enforced_mode": None,
  "bet": dict,                   # recommend.py 출력
  "strategy_mode": None,
  "strategy_note": str,
  "rl_reward": None
}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import road
import features
import future_simulator
import recommend
from engine_state import save_engine_state  # 매 라운드 상태 영구 저장용

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


def _require_nonempty_str(v: Any, name: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{name} must be non-empty string")
    return v.strip()


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
    # STRICT: 실패는 로깅/관측이 필요하나, 이 함수 자체는 예외를 삼키지 않는다.
    # save_engine_state 자체가 실패하면 호출부에서 예외로 처리한다.
    save_engine_state()


def _assert_future_scenarios_strict(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    STRICT:
    - features.future_scenarios는 반드시 dict여야 한다.
    - keys: "P", "B" 반드시 존재
    - 각 값은 dict
    """
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
    """
    recommend.recommend_bet() 결과 스키마 강제.
    보정/폴백 금지.
    """
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


def _normalize_bet_aliases_strict(bet: Dict[str, Any]) -> Dict[str, Any]:
    """
    호환을 위해 side/unit 별칭을 추가(동등 값 복제).
    보정/변형 아님.
    """
    bet = _require_dict(bet, "bet")
    if "bet_side" in bet and "side" not in bet:
        bet["side"] = bet["bet_side"]
    if "bet_unit" in bet and "unit" not in bet:
        bet["unit"] = bet["bet_unit"]
    if "chaos_limit" not in bet:
        bet["chaos_limit"] = None
    return bet


def _build_leader_state_strict(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    STRICT:
    - leader_state는 dict여야 한다.
    - 계산/추론 없이, features에 이미 존재하는 leader 키만 전달.
    """
    leader_state: Dict[str, Any] = {}
    if not isinstance(features_dict, dict):
        raise TypeError("features_dict must be dict")

    for k in ("leader_trust_state", "leader_ready", "leader_confidence", "leader_source", "leader_signal"):
        if k in features_dict:
            leader_state[k] = features_dict[k]

    # 호환: leader_signal(P/B)이 있으면 leader_side로 복제(추정 아님)
    ls = features_dict.get("leader_signal")
    if ls in ("P", "B") and "leader_side" not in leader_state:
        leader_state["leader_side"] = ls

    return leader_state


def _empty_ai_decision_ok_strict(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    app.py(READY)에서 comment를 요구할 수 있어 최소/결정론적 comment 제공.
    - 절대 예측/확률/퍼센트 금지
    """
    # STRICT: pattern_type이 없으면 예외 (features 계약 위반)
    pt = features_dict.get("pattern_type")
    if not isinstance(pt, str) or not pt.strip():
        raise RuntimeError("features.pattern_type missing/invalid (required for ai_decision.comment)")
    comment = f"PATTERN:{pt.strip().lower()}"  # 결정론적/파생 문자열

    return {
        "ai_ok": True,
        "comment": comment,
        "error": "",
        "risk_tags": [],
        "key_features": [],
        # 호환 키들
        "mode_raw": None,
        "meta_info": None,
        "snapshot": None,
        "engine": "hybrid_rule+gpt",
    }


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
    - future_scenarios는 반드시 존재해야 한다(중요 기능)
    - recommend 내부에서 GPT 호출/결합 수행
    """
    if IS_RESETTING:
        raise RuntimeError("RESETTING: run_ai_pipeline called during reset")

    if not isinstance(ai_streak_lose, int):
        raise TypeError(f"ai_streak_lose must be int, got {type(ai_streak_lose).__name__}")

    winner = _normalize_winner(prev_round_winner)
    if winner == "T":
        # STRICT: app.py에서 선차단이 계약
        raise RuntimeError("CONTRACT_VIOLATION: run_ai_pipeline must not be called with winner='T'")

    # 1) Feature 생성 (STRICT)
    features_dict = _require_dict(features.build_feature_payload_v3(winner), "features.build_feature_payload_v3()")

    # 2) PB 시퀀스 주입 (STRICT)
    pb_seq = road.get_pb_sequence()
    pb_seq = _require_list(pb_seq, "road.get_pb_sequence()")
    features_dict["pb_seq"] = pb_seq

    # 3) chaos/stability alias 매핑 (STRICT: 동등 값 복제)
    if "flow_chaos_risk" not in features_dict:
        raise KeyError("required key missing: flow_chaos_risk")
    if "flow_stability" not in features_dict:
        raise KeyError("required key missing: flow_stability")

    features_dict["chaos"] = float(features_dict["flow_chaos_risk"])
    features_dict["stability"] = float(features_dict["flow_stability"])

    # 4) FUTURE CHINA ROADS (필수)
    fs = _assert_future_scenarios_strict(features_dict)

    # 4-1) future_simulator merge는 "중요 기능"이므로 실패 시 예외
    merged = future_simulator.merge_future_china_roads(
        fs,
        include_two_step=True,
        max_rows=6,
    )
    merged = _require_dict(merged, "future_simulator.merge_future_china_roads()")
    features_dict["future_scenarios"] = merged
    _assert_future_scenarios_strict(features_dict)  # 재검증

    # 5) recommend 호출 (recommend 내부에서 GPT 하이브리드 수행)
    leader_state = _build_leader_state_strict(features_dict)

    # recommend.py는 gpt_analysis를 STRICT로 사용하지 않도록 설계(빈 dict 전달)
    gpt_analysis: Dict[str, Any] = {}
    mode = ""  # 표시/로깅용. 결정 로직 관여 금지.
    alerts: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}

    if isinstance(features_dict.get("meta"), dict):
        meta = features_dict["meta"]

    bet = recommend.recommend_bet(
        pb_seq,
        features_dict,
        leader_state,
        gpt_analysis,
        mode,
        alerts,
        meta,
    )
    bet = _require_dict(bet, "recommend.recommend_bet()")
    _assert_bet_contract_strict(bet)
    bet = _normalize_bet_aliases_strict(bet)

    # 6) ai_decision (호환용, 예측/확률/방향 없음)
    ai_decision = _empty_ai_decision_ok_strict(features_dict)

    resp: Dict[str, Any] = {
        "ai_ok": True,
        "features": features_dict,
        "gpt_raw": {},          # 사용 안함(호환)
        "ml_raw": None,
        "ml_reference": None,
        "ai_decision": ai_decision,
        "alert_message": None,
        "enforced_mode": None,
        "bet": bet,
        "strategy_mode": None,
        "strategy_note": "",
        "rl_reward": None,
    }

    # 7) 상태 저장 (STRICT: 실패하면 예외)
    _safe_save_state()

    return resp