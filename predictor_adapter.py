# -*- coding: utf-8 -*-
"""
predictor_adapter.py
====================================================
Baccarat Predictor AI Engine v11.x (룰 기반 메인 엔진 + GPT 분석 보조)

역할
------
- features.build_feature_payload_v3() 로 Feature JSON 생성
- future_simulator 를 이용해 미래 중국점(FUTURE CHINA ROADS) 정보 merge
- gpt_client.call_gpt_engine() 으로 GPT 분석 호출 (확률/방향 X, 해설/모드만)
- gpt_decider.build_ai_decision() 로 GPT 분석 메타 정리
- recommend.recommend_bet() 으로 최종 베팅(side/unit) 결정 (순수 룰 기반)
- meta_learning.decide_strategy_mode() 로 전략 모드 결정
- engine_state.save_engine_state() 로 매 라운드 상태 영구 저장

정책
------
- GPT는 "분석/해설"만 제공하고, 방향/확률을 제공하지 않는다.
- 베팅 방향/단위는 전적으로 recommend.py(룰 기반 규칙)에서만 결정한다.
- GPT 분석 실패 시에도 룰 엔진은 독립적으로 동작 가능하다.
  (단, ai_decision.ai_ok=False 로 표시되고 error에 사유가 기록된다.)

변경 요약 (2025-12-30)
----------------------------------------------------
1) recommend.recommend_bet() 호출 계약을 recommend.py 시그니처에 정확히 일치시킴
   - (pb_seq, features, leader_state, gpt_analysis, mode, alerts, meta) 순서로 전달
   - BigEye/Small/Cockroach 시퀀스를 인자로 전달하던 잘못된 호출 제거
2) bet 스키마를 recommend.py 출력 스키마에 맞게 검증하도록 수정
   - bet_side/bet_unit/entry_type/reason/tags/metrics 필수 검증
   - 호환을 위해 side/unit/chaos_limit 별칭 키도 함께 제공(보정 아님: 동등 값 복제)
3) Tie(T) 입력 처리 계약 준수(입력 단계 선차단)
   - winner == "T" 인 경우: pb_seq/gridBead에 기록 금지, features/pattern/recommend 호출 금지
   - 마지막 정상 응답을 그대로 반환하고, features.meta.tie_count만 +1 기록
4) run_ai_pipeline() 계약 강화
   - ai_streak_lose는 필수 인자(폴백 금지): None/비정수는 즉시 TypeError
5) UI 참고용 ML 보조 정보(ml_reference) 응답 필드 추가
   - (참고용) 응답에만 포함, recommend/베팅 판단 로직에는 절대 관여하지 않음
----------------------------------------------------

변경 요약 (2026-01-03)
----------------------------------------------------
- v11.x에서 구버전 pkl 기반 ML 모델 로딩(baccarat_ml_model.pkl/joblib) 로직 완전 제거
- ml_reference는 ml_model.analyze_risk_and_stability(features) 기반으로만 생성
- ml_reference 생성 실패/None은 non-fatal:
  [WARN] ML reference unavailable (non-fatal) 출력 후 None 반환
----------------------------------------------------
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import road
import features
import future_simulator
import gpt_client
import gpt_decider
import meta_learning
import ml_model
import recommend
from engine_state import save_engine_state  # 매 라운드 상태 영구 저장용

IS_RESETTING = False

# Tie(T) 입력 시, UI/응답 스키마를 깨지 않기 위해 마지막 정상 응답을 캐시한다.
_LAST_GOOD_RESPONSE: Optional[Dict[str, Any]] = None


# GPT가 절대 제공하면 안 되는 키(방향/확률 계열)
_DISALLOWED_GPT_KEYS = {
    "side",
    "winner_guess",
    "next_winner_guess",
    "player_prob",
    "banker_prob",
    "tie_prob",
    "next_player_prob",
    "next_banker_prob",
    "next_tie_prob",
    "confidence",
    "next_confidence",
}


def _safe_save_state() -> None:
    try:
        save_engine_state()
    except Exception as e:  # pragma: no cover
        print("[AI] save_engine_state 실패:", repr(e), flush=True)


def _empty_ai_decision(error: str = "") -> Dict[str, Any]:
    """AI 판단이 불가능한 경우에 사용하는 기본 구조."""
    return {
        "ai_ok": False,
        "side": None,
        "player_prob": None,
        "banker_prob": None,
        "tie_prob": None,
        "confidence": None,
        "confidence_raw": None,
        "confidence_notes": [],
        "mode_raw": None,
        "comment": "",
        "key_features": [],
        "snapshot": None,
        "meta_info": None,
        "error": error or "",
        "engine": None,
    }


def _empty_bet(reason: str, pass_reason: Optional[str] = None) -> Dict[str, Any]:
    """PASS 기본 구조(recommend.py 출력 스키마 준수)."""
    bet: Dict[str, Any] = {
        "bet_side": None,
        "bet_unit": 0,
        "entry_type": None,
        "reason": reason,
        "tags": ["PASS"],
        "metrics": {},
    }
    if pass_reason:
        bet["pass_reason"] = pass_reason

    # 호환 별칭(동등 값 복제)
    bet["side"] = bet["bet_side"]
    bet["unit"] = bet["bet_unit"]
    bet["chaos_limit"] = None
    return bet


def _sanitize_gpt_raw(gpt_raw: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    GPT raw에서 정책 위반(방향/확률 키)을 제거한다.
    제거가 발생하면 violations에 키 목록을 기록한다.
    """
    if gpt_raw is None:
        return None, []

    if not isinstance(gpt_raw, dict):
        # dict가 아니면 gpt_decider가 처리하기 어려우므로 정책 위반으로 간주
        return None, ["gpt_raw_not_dict"]

    cleaned = dict(gpt_raw)
    violations: List[str] = []

    for k in list(cleaned.keys()):
        if k in _DISALLOWED_GPT_KEYS:
            violations.append(k)
            cleaned.pop(k, None)

    return cleaned, violations


def _assert_bet_contract(bet: Dict[str, Any]) -> None:
    """
    recommend.recommend_bet() 결과 스키마를 강제한다.
    조용히 보정하지 않는다(오염 방지).
    """
    if not isinstance(bet, dict):
        raise TypeError("bet must be dict")

    required_keys = {
        "bet_side",
        "bet_unit",
        "entry_type",
        "reason",
        "tags",
        "metrics",
    }
    missing = required_keys - set(bet.keys())
    if missing:
        raise ValueError(f"bet missing required keys: {sorted(missing)}")

    bet_side = bet.get("bet_side")
    bet_unit = bet.get("bet_unit")
    entry_type = bet.get("entry_type")

    if bet_side is not None and bet_side not in ("P", "B"):
        raise ValueError(f"bet.bet_side invalid: {bet_side!r} (expected 'P'/'B'/None)")

    if not isinstance(bet_unit, int):
        raise TypeError(f"bet.bet_unit must be int, got {type(bet_unit).__name__}")
    if bet_unit < 0:
        raise ValueError(f"bet.bet_unit must be >= 0, got {bet_unit}")

    if entry_type is not None and entry_type not in ("PROBE", "NORMAL"):
        raise ValueError(f"bet.entry_type invalid: {entry_type!r} (expected 'PROBE'/'NORMAL'/None)")

    if not isinstance(bet.get("reason"), str):
        raise TypeError("bet.reason must be str")

    tags = bet.get("tags")
    if not isinstance(tags, list) or any(not isinstance(x, str) for x in tags):
        raise TypeError("bet.tags must be list[str]")

    metrics = bet.get("metrics")
    if not isinstance(metrics, dict):
        raise TypeError("bet.metrics must be dict")

    if "pass_reason" in bet and bet["pass_reason"] is not None and not isinstance(bet["pass_reason"], str):
        raise TypeError("bet.pass_reason must be str or None")


def _build_leader_state(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    recommend.py가 받는 leader_state는 dict여야 한다.
    - 계산/추론 없이, features에 이미 존재하는 leader 관련 키만 선별해 전달한다.
    """
    leader_state: Dict[str, Any] = {}

    # recommend._extract_leader_strength()가 읽는 후보 키들
    for k in ("leader_trust_state", "state", "strength", "bias_strength", "leader_side"):
        if k in features_dict:
            leader_state[k] = features_dict[k]

    # 최소 전달: leader_signal(P/B)이 있으면 leader_side로 전달(동일 값 복제)
    ls = features_dict.get("leader_signal")
    if ls in ("P", "B") and "leader_side" not in leader_state:
        leader_state["leader_side"] = ls

    # 참고용(있으면 전달)
    for k in ("leader_ready", "leader_confidence", "leader_source"):
        if k in features_dict:
            leader_state[k] = features_dict[k]

    return leader_state


def _normalize_bet_aliases(bet: Dict[str, Any]) -> Dict[str, Any]:
    """
    호환을 위해 bet_side/bet_unit → side/unit 별칭을 추가한다(동등 값 복제).
    - 보정/변형 아님.
    """
    if not isinstance(bet, dict):
        raise TypeError("bet must be dict")

    if "bet_side" in bet and "side" not in bet:
        bet["side"] = bet["bet_side"]
    if "bet_unit" in bet and "unit" not in bet:
        bet["unit"] = bet["bet_unit"]
    if "chaos_limit" not in bet:
        bet["chaos_limit"] = None
    return bet


def _normalize_winner(prev_round_winner: Optional[str]) -> Optional[str]:
    """
    입력 winner 정규화.
    - None/"" -> None
    - "p"/"b"/"t" 등 -> "P"/"B"/"T"
    - 그 외는 즉시 예외
    """
    if prev_round_winner is None:
        return None
    if not isinstance(prev_round_winner, str):
        raise TypeError("prev_round_winner must be str or None")

    s = prev_round_winner.strip().upper()
    if s == "":
        return None
    if s not in ("P", "B", "T"):
        raise ValueError(f"invalid prev_round_winner: {prev_round_winner!r} (expected 'P'/'B'/'T'/None)")
    return s


def _load_ml_model() -> None:
    """
    v11.x: 구버전 pkl 기반 ML 모델 로딩 로직 제거.
    - 이 함수는 호환 목적의 더미이며 항상 None을 반환한다.
    """
    return None


def _build_ml_reference(features_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    UI 참고용 ML 보조 정보 생성(결정 로직 관여 금지).

    v11.x 정책
    - pkl 모델(파일/경로/로드) 사용 금지
    - ml_model.analyze_risk_and_stability(features)만 호출
    - 반환값이 None이거나 예외면 non-fatal로 None 반환 + [WARN] 로그 1회 출력
    """
    try:
        ref = ml_model.analyze_risk_and_stability(features_dict)
    except Exception:
        print("[WARN] ML reference unavailable (non-fatal)", flush=True)
        return None

    if ref is None or not isinstance(ref, dict):
        print("[WARN] ML reference unavailable (non-fatal)", flush=True)
        return None

    return ref


def run_ai_pipeline(
    prev_round_winner: Optional[str] = None,
    ai_recent_results: Optional[List[int]] = None,
    ai_streak_lose: Optional[int] = None,  # ✅ 폴백 금지: 누락/비정수 즉시 예외
) -> Dict[str, Any]:
    """
    전체 AI 파이프라인 실행 (v11 룰 기반 모드).

    반환
    -----
    {
        "ai_ok": bool,             # 룰 파이프라인이 정상적으로 bet을 산출했는지
        "features": dict,
        "gpt_raw": dict or None,   # GPT 분석 원본(정책 위반 키 제거된 sanitized)
        "ml_raw": None,            # v11에서는 항상 None
        "ml_reference": dict|None, # UI 참고용 ML 보조 정보(결정 로직 관여 금지)
        "ai_decision": dict,       # GPT 분석 메타 (확률/방향 없음)
        "alert_message": str or None,
        "enforced_mode": str or None,
        "bet": dict,               # recommend.recommend_bet() 결과(룰 기반)
        "strategy_mode": str or None,
        "strategy_note": str,
    }
    """
    global _LAST_GOOD_RESPONSE

    if not isinstance(ai_streak_lose, int):
        raise TypeError(f"ai_streak_lose must be int, got {type(ai_streak_lose).__name__}")

    ai_recent_results = list(ai_recent_results or [])

    try:
        winner = _normalize_winner(prev_round_winner)

        # RESET 직후(또는 슈 리셋)로 pb_seq가 비어 있으면 캐시를 초기화한다.
        pb_now = road.get_pb_sequence()
        if not pb_now:
            _LAST_GOOD_RESPONSE = None

        # --------------------------------------------------
        # [TIE CONTRACT] winner == "T" 선차단
        # - pb_seq/gridBead 기록 금지
        # - features/pattern/recommend 호출 금지
        # - tie_count는 features.meta에만 기록
        # --------------------------------------------------
        if winner == "T":
            if _LAST_GOOD_RESPONSE is not None:
                resp = copy.deepcopy(_LAST_GOOD_RESPONSE)

                feats = resp.get("features")
                if not isinstance(feats, dict):
                    raise TypeError("cached response.features must be dict")

                meta = feats.get("meta")
                if meta is None:
                    meta = {}
                    feats["meta"] = meta
                if not isinstance(meta, dict):
                    raise TypeError("features.meta must be dict")

                tie_count = meta.get("tie_count", 0)
                if tie_count is None:
                    tie_count = 0
                if not isinstance(tie_count, int):
                    raise TypeError("meta.tie_count must be int")
                meta["tie_count"] = tie_count + 1

                resp["features"] = feats

                # 응답 구조 유지: ml_reference 키가 없으면 None으로 유지
                if "ml_reference" not in resp:
                    resp["ml_reference"] = None

                # 캐시도 최신화(연속 Tie에서도 누적되도록)
                _LAST_GOOD_RESPONSE = copy.deepcopy(resp)
                return resp

            # 캐시가 없으면 최소 스키마로 반환(분석/추천 수행 금지)
            reason = "TIE 입력: 승부 결과가 아니므로 기록/분석/추천을 수행하지 않음"
            minimal_features: Dict[str, Any] = {
                "pb_seq": pb_now,
                "rounds_total": len(pb_now),
                "meta": {"tie_count": 1},
            }
            resp = {
                "ai_ok": False,
                "features": minimal_features,
                "gpt_raw": None,
                "bead_seq": road.get_pb_sequence(),  # ✅ 정답
                "ml_raw": None,
                "ml_reference": None,
                "ai_decision": _empty_ai_decision(error=reason),
                "alert_message": None,
                "enforced_mode": None,
                "bet": _empty_bet(reason, pass_reason="PASS_TIE_IGNORED"),
                "strategy_mode": None,
                "strategy_note": reason,
            }
            _LAST_GOOD_RESPONSE = copy.deepcopy(resp)
            return resp

        # 1) Feature 생성
        features_dict = features.build_feature_payload_v3(winner)

        rounds_total = int(features_dict.get("rounds_total") or 0)
        pb_seq = road.get_pb_sequence()

        # BigRoad(P/B) 시퀀스를 features에 주입
        features_dict["pb_seq"] = pb_seq

        # --------------------------------------------------
        # [CONTRACT ADAPTER] chaos / stability 키 매핑
        # recommend.py 계약 충족용 (폴백 아님, alias 매핑)
        # --------------------------------------------------
        if "flow_chaos_risk" not in features_dict:
            raise KeyError("required key missing: flow_chaos_risk")

        if "flow_stability" not in features_dict:
            raise KeyError("required key missing: flow_stability")

        features_dict["chaos"] = float(features_dict["flow_chaos_risk"])
        features_dict["stability"] = float(features_dict["flow_stability"])

        # --- tie turbulence (meta_learning 필수 입력) ---
        tie_state = features_dict.get("tie_state")
        if not isinstance(tie_state, dict):
            raise ValueError("[predictor_adapter] tie_state missing or invalid")

        tie_turbulence_rounds = tie_state.get("turbulence_rounds")
        if not isinstance(tie_turbulence_rounds, int):
            raise ValueError("[predictor_adapter] tie_turbulence_rounds missing or invalid")

        features_dict["tie_turbulence_rounds"] = tie_turbulence_rounds

        if rounds_total <= 0 or not pb_seq:
            reason = "라운드/패턴 데이터 부족 – AI 파이프라인 스킵"
            resp = {
                "ai_ok": False,
                "features": features_dict,
                "gpt_raw": None,
                "ml_raw": None,
                "ml_reference": None,
                "ai_decision": _empty_ai_decision(error=reason),
                "alert_message": None,
                "enforced_mode": None,
                "bet": _empty_bet(reason, pass_reason="PASS_INSUFFICIENT_DATA"),
                "strategy_mode": None,
                "strategy_note": reason,
            }
            _LAST_GOOD_RESPONSE = copy.deepcopy(resp)
            return resp

        # 2) FUTURE CHINA ROADS 시뮬레이션 merge (추가 정보, 실패해도 진행)
        try:
            base_future = features_dict.get("future_scenarios")
            if isinstance(base_future, dict):
                features_dict["future_scenarios"] = future_simulator.merge_future_china_roads(
                    base_future,
                    include_two_step=True,
                    max_rows=6,
                )
        except Exception as e:
            print("[AI] future_simulator merge 오류:", repr(e), flush=True)

        # 3) GPT 분석 호출 (확률/방향 X, 해설/모드만)
        gpt_raw: Optional[Dict[str, Any]] = None
        gpt_error: Optional[str] = None
        try:
            gpt_raw, gpt_error = gpt_client.call_gpt_engine(features_dict)
        except Exception as e:  # pragma: no cover
            gpt_error = repr(e)
            gpt_raw = None
            print("[AI] GPT 호출 예외 –", gpt_error, flush=True)

        # 3-1) GPT raw 정책 위반 키 제거
        gpt_sanitized, violations = _sanitize_gpt_raw(gpt_raw)
        if violations:
            vmsg = f"GPT 정책 위반 키 제거: {violations}"
            gpt_error = f"{gpt_error} | {vmsg}" if gpt_error else vmsg
            print("[AI] " + vmsg, flush=True)

        # 4) GPT 분석 메타 정리
        try:
            ai_dec = gpt_decider.build_ai_decision(
                features_dict,
                gpt_raw=gpt_sanitized,
            )
        except Exception as e:
            err = f"gpt_decider 실패: {repr(e)}"
            print("[AI] " + err, flush=True)
            ai_dec = _empty_ai_decision(error=err)

        # gpt_error가 있으면 ai_dec.error에 합쳐 기록(덮어쓰기 금지)
        if gpt_error:
            prev_err = (ai_dec.get("error") or "").strip()
            ai_dec["error"] = (prev_err + " | " + gpt_error).strip(" |")

        # GPT 코멘트를 alert_message 로 전달 (없으면 빈 문자열)
        alert_message = ai_dec.get("comment") or ""

        # 5) recommend.py 계약에 맞는 인자 구성
        leader_state = _build_leader_state(features_dict)
        gpt_analysis = ai_dec if isinstance(ai_dec, dict) else {}
        mode = ai_dec.get("mode_raw") if isinstance(ai_dec, dict) else None
        if not isinstance(mode, str) or not mode.strip():
            # gpt_raw에 mode가 있으면 사용(추론/보정 아님: 존재 값만 사용)
            if isinstance(gpt_sanitized, dict) and isinstance(gpt_sanitized.get("mode"), str):
                mode = gpt_sanitized["mode"]
            else:
                mode = ""

        alerts: Dict[str, Any] = {"alert_message": alert_message}
        meta: Dict[str, Any] = {}
        if isinstance(features_dict.get("meta"), dict):
            meta = features_dict["meta"]

        # 6) 최종 베팅 추천(recommend_bet) – 순수 룰 기반
        # ✅ recommend.py 시그니처: (pb_seq, features, leader_state, gpt_analysis, mode, alerts, meta)
        bet = recommend.recommend_bet(
            pb_seq,
            features_dict,
            leader_state,
            gpt_analysis,
            mode,
            alerts,
            meta,
        )
        _assert_bet_contract(bet)
        bet = _normalize_bet_aliases(bet)

        # 6-1) UI 참고용 ML 보조 정보(결정 로직 관여 금지) 생성 (non-fatal)
        ml_reference = _build_ml_reference(features_dict)

        # 7) 전략 모드 결정 (GPT 실패면 meta_info는 None일 수 있음)
        meta_info = ai_dec.get("meta_info") if isinstance(ai_dec, dict) and ai_dec.get("ai_ok") else None
        strategy_mode, strategy_note = meta_learning.decide_strategy_mode(
            core_mode=ai_dec.get("mode_raw") if isinstance(ai_dec, dict) else None,
            features=features_dict,
            bet_info=bet,
            meta_info=meta_info,
        )

        resp = {
            "ai_ok": True,
            "features": features_dict,
            "bead_seq": road.get_pb_sequence(),
            "gpt_raw": gpt_sanitized,
            "ml_raw": None,
            "ml_reference": ml_reference,
            "ai_decision": ai_dec,
            "alert_message": alert_message,
            "enforced_mode": None,
            "bet": bet,
            "strategy_mode": strategy_mode,
            "strategy_note": strategy_note,
        }
        _LAST_GOOD_RESPONSE = copy.deepcopy(resp)
        return resp

    finally:
        # ✅ RESET 중에는 상태 저장 금지
        if not IS_RESETTING:
            _safe_save_state()
