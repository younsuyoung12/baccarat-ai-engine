# -*- coding: utf-8 -*-
"""
ml_model.py

Baccarat Predictor AI Engine v11.x – ML 분석 보조 모듈 (위험도·신뢰도 전용)

역할
------
- 이 모듈은 베팅 예측기(방향 P/B/T, 확률 %)가 아니다.
- 오직 현재 슈 상태의 "위험도(risk_level) / 안정성(stability)"을 정량화하여
  GPT 해설의 근거로만 제공한다.
- recommend.py(룰 기반 베팅 엔진)의 판단/방향/유닛/진입(PASS/PROBE/NORMAL)에는
  절대 관여하지 않는다.

변경 (2026-01-03)
-----------------
- 기존 v11 스텁(predict_from_features() -> None) 구조를 유지하되,
  실사용 함수로 analyze_risk_and_stability(features)를 추가하여
  위험도/안정성 등급만 반환하도록 전환.
- 방향 예측/확률 반환/추천/우세 표현을 전면 금지.
- feature가 부족하면 None을 반환(폴백 금지).

반환 스펙 (고정)
----------------
{
  "risk_level": "HIGH" | "MID" | "LOW",
  "stability":  "HIGH" | "MID" | "LOW",
  "confidence_note": "사람이 이해할 수 있는 짧은 근거 문장"
}
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _to_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def analyze_risk_and_stability(features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    위험도/안정성만 평가한다.

    절대 금지:
    - 방향(P/B/T) 예측/추천
    - 확률(%) 계산/반환
    - 우세, 추천, 맞출 수 있다는 표현

    폴백 금지:
    - 위험도 또는 안정성을 산출할 핵심 feature가 부족하면 None 반환
    """
    if not isinstance(features, dict):
        return None

    # ----------------------------
    # 위험도(risk) 입력 후보
    # ----------------------------
    chaos = _to_float(features.get("chaos_index"))
    if chaos is None:
        chaos = _to_float(features.get("flow_chaos_risk"))
    if chaos is None:
        chaos = _to_float(features.get("global_chaos_ratio"))

    reversal = _to_float(features.get("flow_reversal_risk"))
    tie_vol = _to_float(features.get("tie_volatility"))
    regime_shift = _to_float(features.get("regime_shift_score"))

    # risk 산출에 쓸 값이 하나도 없으면 None (폴백 금지)
    risk_inputs = [v for v in (chaos, reversal, tie_vol, regime_shift) if v is not None]
    if not risk_inputs:
        return None

    # ----------------------------
    # 안정성(stability) 입력 후보
    # ----------------------------
    pattern_stab = _to_float(features.get("pattern_stability"))
    flow_stab = _to_float(features.get("flow_stability"))
    beauty = _to_float(features.get("beauty_score"))  # 0~100 가정

    stab_components = []
    if pattern_stab is not None:
        stab_components.append(pattern_stab)
    if flow_stab is not None:
        stab_components.append(flow_stab)
    if beauty is not None:
        # 0~100 -> 0~1 정규화
        beauty01 = max(0.0, min(1.0, beauty / 100.0))
        stab_components.append(beauty01)

    # stability 산출에 쓸 값이 하나도 없으면 None (폴백 금지)
    if not stab_components:
        return None

    # ----------------------------
    # 위험도 점수(0~1) 구성
    # ----------------------------
    # chaos/reversal/regime_shift는 '높을수록 위험' 가정
    # tie_volatility는 환경에 따라 스케일이 다를 수 있어 0~1 클램프만 수행
    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    risk_score_parts = []
    if chaos is not None:
        risk_score_parts.append(_clamp01(chaos))
    if reversal is not None:
        risk_score_parts.append(_clamp01(reversal))
    if regime_shift is not None:
        risk_score_parts.append(_clamp01(regime_shift))
    if tie_vol is not None:
        risk_score_parts.append(_clamp01(tie_vol))

    risk_score = sum(risk_score_parts) / float(len(risk_score_parts))

    if risk_score >= 0.67:
        risk_level = "HIGH"
    elif risk_score <= 0.33:
        risk_level = "LOW"
    else:
        risk_level = "MID"

    # ----------------------------
    # 안정성 점수(0~1) 구성
    # ----------------------------
    stability_score = sum(_clamp01(v) for v in stab_components) / float(len(stab_components))

    if stability_score >= 0.67:
        stability = "HIGH"
    elif stability_score <= 0.33:
        stability = "LOW"
    else:
        stability = "MID"

    # ----------------------------
    # 근거 문장(짧고 단정, 방향/추천 금지)
    # ----------------------------
    if risk_level == "HIGH" and stability == "LOW":
        note = "혼돈·변동 신호가 강하고 안정성 지표가 낮다."
    elif risk_level == "HIGH" and stability != "LOW":
        note = "위험 신호가 강하게 관측되며 변동 가능성이 높다."
    elif risk_level == "LOW" and stability == "HIGH":
        note = "위험 신호가 낮고 안정성 지표가 높게 유지된다."
    elif risk_level == "LOW" and stability != "HIGH":
        note = "위험 신호는 낮지만 안정성은 완전히 고정되지 않았다."
    elif risk_level == "MID" and stability == "LOW":
        note = "위험 신호가 중간 수준이지만 안정성 지표가 약하다."
    elif risk_level == "MID" and stability == "HIGH":
        note = "위험 신호가 중간 수준이며 안정성 지표는 양호하다."
    else:
        note = "위험·안정 신호가 혼재된 상태다."

    return {
        "risk_level": risk_level,
        "stability": stability,
        "confidence_note": note,
    }


def predict_from_features(features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    (호환용) v11 룰 기반 모드에서는 ML 예측(P/B/확률)을 사용하지 않는다.

    - 방향/확률/추천 반환 금지 정책 유지
    - 기존 상위 레이어 호환을 위해 유지하며, 항상 None 반환
    """
    return None
