# -*- coding: utf-8 -*-
"""
gpt_decider.py

Baccarat Predictor AI Engine v11.x – GPT 분석 래퍼 (룰 기반 모드)

역할
------
- GPT 응답(JSON)을 검증하고, 엔진에서 보기 편한 공통 구조로 정리한다.
- 더 이상 P/B/T 확률이나 방향을 계산하지 않는다.
- GPT는 오직 "현재 슈 상태에 대한 해설/모드/리스크 태그"만 제공한다.
- 실제 베팅 방향/단위는 전부 recommend.py(룰 기반 엔진)에서만 결정한다.

추가 (2025-12-30)
-----------------
- UI 전용 필드 `gpt_direction_hint` 추가
  - GPT comment 텍스트에서 방향 힌트만 단순 추출해 제공한다.
  - 베팅 판단(recommend.py), ML, 엔진 상태 저장(engine_state)에는 절대 사용/저장하지 않는다.
  - 값: "gpt_banker" | "gpt_player" | "gpt_neutral"

변경 (2025-12-31)
-----------------
- PASS 여부와 무관하게, GPT comment가 존재하면 gpt_direction_hint를 항상 생성한다.
- payload 검증 실패 시에도 comment가 있으면 힌트를 추출해 포함한다(베팅/로직 영향 없음).
- payload가 None 인 경우에도 gpt_direction_hint 필드는 항상 포함한다(근거 없으면 gpt_neutral).

변경 (2026-01-02)
-----------------
- GPT comment/risk_tags/mode를 바탕으로, UI에 바로 표시 가능한 “AI 판단 결과” 문구를
  이 모듈에서 **결정적으로 재구성**한다.
  - 1행: 아래 4개 중 하나로 고정(애매한 표현 금지)
    * 관망이 유리한 구간
    * 탐색(PROBE)만 허용되는 구간
    * 정상 진입이 가능한 구간
    * 전환을 주의 깊게 관찰해야 하는 구간
  - 2행: 패턴 안정성 / 흐름 지속성 / 혼돈 위험도 관점의 짧은 근거 1문장
  - 베팅 방향/확률/예측 문구는 UI 문구에 절대 포함하지 않는다.
  - 원본 GPT comment는 comment_raw로 보존한다(디버그/로그용).

주의
------
- GPT 응답이 비정상인 경우에도 룰 엔진은 독립적으로 동작할 수 있으나,
  이 모듈은 GPT 분석의 성공 여부만 ai_ok로 알려준다.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List


def _first_non_empty_line(text: str) -> str:
    """첫 문장 판별용: 첫 번째 비어있지 않은 라인을 반환."""
    if not isinstance(text, str):
        return ""
    for ln in text.splitlines():
        s = ln.strip()
        if s:
            return s
    return ""


def _classify_zone(first_line: str, mode_raw: Optional[str], risk_tags: List[str]) -> str:
    """
    UI 1행 결론(4지선다)을 결정한다.

    우선순위:
    1) GPT comment의 첫 문장(명시적 키워드)
    2) risk_tags
    3) mode
    4) 기본값(탐색)
    """

    line = (first_line or "").strip()
    tags = {str(t).strip().lower() for t in (risk_tags or []) if str(t).strip()}
    mode = (mode_raw or "").strip().lower()

    # 1) comment 첫 문장에 결론이 이미 들어온 경우(가장 강함)
    if any(k in line for k in ("관망", "대기", "쉬어", "쉬는", "스킵")):
        return "관망이 유리한 구간"
    if any(k in line for k in ("탐색", "프로브", "probe")):
        return "탐색(PROBE)만 허용되는 구간"
    if any(k in line for k in ("정상 진입", "정상", "진입 가능", "들어갈", "진입")):
        return "정상 진입이 가능한 구간"
    if any(k in line for k in ("전환", "변곡", "뒤집", "반전", "리셋", "모드 변경")):
        return "전환을 주의 깊게 관찰해야 하는 구간"

    # 2) risk_tags 기반 결론
    chaos_hit = ("high_chaos" in tags) or ("danger_zone" in tags) or ("chaos" in tags)
    transition_hit = ("transition" in tags) or ("reversal" in tags) or ("mode_change" in tags) or ("flip" in tags)
    unstable_hit = ("unstable" in tags) or ("low_pattern" in tags) or ("weak_flow" in tags) or ("irregular" in tags)
    stable_hit = ("stable" in tags) or ("trend" in tags) or ("clean" in tags)

    if chaos_hit:
        return "전환을 주의 깊게 관찰해야 하는 구간" if transition_hit else "관망이 유리한 구간"
    if unstable_hit:
        return "탐색(PROBE)만 허용되는 구간"
    if stable_hit:
        return "정상 진입이 가능한 구간"

    # 3) mode 기반 결론
    if mode == "chaos":
        return "관망이 유리한 구간"
    if mode in ("pattern", "flow"):
        return "정상 진입이 가능한 구간"

    # 4) 기본값
    return "탐색(PROBE)만 허용되는 구간"


def _build_reason(conclusion: str, mode_raw: Optional[str], risk_tags: List[str]) -> str:
    """UI 2행 근거 1문장 생성(애매한 표현 금지)."""

    tags = {str(t).strip().lower() for t in (risk_tags or []) if str(t).strip()}
    mode = (mode_raw or "mixed").strip().lower()

    chaos_high = ("high_chaos" in tags) or ("danger_zone" in tags) or (mode == "chaos")
    pattern_low = ("low_pattern" in tags) or ("pattern_break" in tags) or ("unstable" in tags)
    flow_low = ("weak_flow" in tags) or ("irregular" in tags) or ("choppy" in tags)

    if conclusion == "관망이 유리한 구간":
        return "혼돈 위험이 높고, 패턴 안정성과 흐름 지속성이 동시에 약하다."

    if conclusion == "탐색(PROBE)만 허용되는 구간":
        if chaos_high:
            return "혼돈 위험이 남아 있고, 패턴 안정성 또는 흐름 지속성이 아직 확정되지 않았다."
        if pattern_low and flow_low:
            return "패턴 안정성과 흐름 지속성이 모두 약해 탐색 외에는 근거가 부족하다."
        if pattern_low:
            return "패턴 안정성이 약해 확정 진입 근거가 부족하다."
        if flow_low:
            return "흐름 지속성이 약해 확정 진입 근거가 부족하다."
        return "안정 신호가 충분히 쌓이지 않아 탐색만 허용한다."

    if conclusion == "정상 진입이 가능한 구간":
        if chaos_high:
            # 결론은 진입인데 chaos가 높으면 모순이므로, 근거는 '낮다'로 고정하지 않고 최소 문장으로 처리
            return "혼돈 위험이 치솟지 않았고, 패턴 또는 흐름 안정성이 확보됐다."
        return "혼돈 위험이 낮고, 패턴 안정성과 흐름 지속성이 확보됐다."

    # 전환 감시
    return "패턴/흐름이 바뀌는 신호가 있어 혼돈 위험이 올라가며, 안정성이 흔들린다."


def _build_ui_comment(comment_raw: str, mode_raw: Optional[str], risk_tags: List[str]) -> Dict[str, str]:
    """comment_raw를 그대로 노출하지 않고, UI용 문구(2행)를 생성한다."""
    first_line = _first_non_empty_line(comment_raw)
    conclusion = _classify_zone(first_line, mode_raw, risk_tags)
    reason = _build_reason(conclusion, mode_raw, risk_tags)
    ui_comment = f"{conclusion}\n{reason}".strip()
    return {"ui_comment": ui_comment, "conclusion": conclusion, "reason": reason}


def _extract_gpt_direction_hint(comment: str) -> str:
    """
    UI 참고용 방향 힌트 추출기.

    규칙(단순/명시적):
    - banker 키워드만 감지되면  -> "gpt_banker"
    - player 키워드만 감지되면  -> "gpt_player"
    - 둘 다 감지되거나 둘 다 미감지 -> "gpt_neutral"

    주의:
    - 이 값은 UI 표시 전용이다(베팅/ML/상태 저장에 사용 금지).
    - comment 근거가 없으면 반드시 gpt_neutral 을 반환한다(추정/폴백 금지).
    """
    if not isinstance(comment, str):
        return "gpt_neutral"

    s = comment.strip()
    if not s:
        return "gpt_neutral"

    s_low = s.lower()

    banker_hit = (
        ("뱅커" in s)
        or ("banker" in s_low)
        or ("banker favored" in s_low)
        or ("뱅커 선호" in s)
    )

    player_hit = (
        ("플레이어" in s)
        or ("player" in s_low)
        or ("player favored" in s_low)
        or ("플레이어 선호" in s)
    )

    if banker_hit and player_hit:
        return "gpt_neutral"
    if banker_hit:
        return "gpt_banker"
    if player_hit:
        return "gpt_player"
    return "gpt_neutral"


def _empty_decision(error: str) -> Dict[str, Any]:
    """GPT 분석이 불가능할 때 사용하는 기본 구조."""
    return {
        "ai_ok": False,
        "side": None,               # 룰 엔진이 결정하므로 항상 None
        "player_prob": None,        # v11 룰 모드에서는 사용하지 않음
        "banker_prob": None,
        "tie_prob": None,
        "confidence": None,
        "confidence_raw": None,
        "confidence_notes": [],
        "mode_raw": None,
        # UI 표시용(2행 고정). payload 자체가 없으므로 중립적 기본값을 사용한다.
        "comment": "탐색(PROBE)만 허용되는 구간\n안정 신호가 충분히 쌓이지 않아 탐색만 허용한다.",
        "comment_raw": "",
        "ui_conclusion": "탐색(PROBE)만 허용되는 구간",
        "ui_reason": "안정 신호가 충분히 쌓이지 않아 탐색만 허용한다.",
        "key_features": [],
        "gpt_direction_hint": "gpt_neutral",  # UI 전용 (근거 없으면 neutral)
        "snapshot": None,
        "meta_info": None,
        "engine": None,
        "error": error,
    }


def _validate_analysis_payload(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    GPT 분석용 payload 검증.

    요구 필드:
      - mode: str
      - comment: str
      - key_features: list[str] (옵션, 없으면 빈 리스트)
      - risk_tags: list[str] (옵션)
    """
    if payload is None:
        return "payload is None"

    mode = payload.get("mode")
    comment = payload.get("comment")

    if not isinstance(mode, str):
        return "mode 필드가 없거나 문자열이 아님"
    if not isinstance(comment, str):
        return "comment 필드가 없거나 문자열이 아님"

    kf = payload.get("key_features")
    if kf is not None and not isinstance(kf, list):
        return "key_features 필드는 리스트여야 함"

    rt = payload.get("risk_tags")
    if rt is not None and not isinstance(rt, list):
        return "risk_tags 필드는 리스트여야 함"

    return None


def build_ai_decision(
    features: Dict[str, Any],
    gpt_raw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    GPT 응답(JSON) + Feature 를 기반으로,
    엔진에서 사용할 "분석 메타 정보"를 구성한다.

    이 함수는 더 이상:
      - 방향(side)
      - 확률(player_prob/banker_prob/tie_prob)
      - confidence
    를 계산하지 않는다.

    정책:
    - PASS 여부와 무관하게, comment가 있으면 gpt_direction_hint를 항상 생성한다.
    - payload 검증 실패 시에도 가능한 범위에서 comment 기반 힌트를 추출해 포함한다.
    - 단, 어디까지나 UI 표시 전용이며 베팅 판단에는 절대 사용하지 않는다.
    """

    # gpt_raw가 None인 경우에도 출력 계약을 만족해야 한다.
    if gpt_raw is None:
        return _empty_decision(error="payload is None")

    # dict가 아닌 경우는 비정상 payload로 처리
    if not isinstance(gpt_raw, dict):
        return _empty_decision(error="payload is not a dict")

    # validate는 'ai_ok' 판단에만 사용한다.
    err = _validate_analysis_payload(gpt_raw)

    # ---- 부분 파싱(표시 계약 유지) ----
    mode_val = gpt_raw.get("mode")
    mode_raw = mode_val if isinstance(mode_val, str) else None

    comment_val = gpt_raw.get("comment")
    comment_raw = comment_val if isinstance(comment_val, str) else ""

    key_features_raw = gpt_raw.get("key_features") or []
    if isinstance(key_features_raw, list):
        key_features = [str(x) for x in key_features_raw][:10]
    else:
        key_features = []

    risk_tags_raw = gpt_raw.get("risk_tags") or []
    if isinstance(risk_tags_raw, list):
        risk_tags = [str(x) for x in risk_tags_raw][:10]
    else:
        risk_tags = []

    confidence_notes: List[str] = []
    if risk_tags:
        confidence_notes.append("risk_tags=" + ",".join(risk_tags))

    # PASS 포함, 원본 comment 기준으로 항상 힌트 생성(근거 없으면 neutral)
    # (UI 표시 문구는 방향/예측을 포함하지 않도록 재구성하므로, 힌트는 원본에서만 추출한다.)
    gpt_direction_hint = _extract_gpt_direction_hint(comment_raw or "")

    # UI 표시용 comment(2행) 재구성
    ui_pack = _build_ui_comment(comment_raw or "", mode_raw, risk_tags)
    ui_comment = ui_pack["ui_comment"]

    # ---- validate 실패: 분석 성공으로 가장하지 않되, 표시용 힌트는 유지 ----
    if err is not None:
        return {
            "ai_ok": False,
            "side": None,
            "player_prob": None,
            "banker_prob": None,
            "tie_prob": None,
            "confidence": None,
            "confidence_raw": None,
            "confidence_notes": confidence_notes,
            "mode_raw": mode_raw,
            "comment": ui_comment,
            "comment_raw": comment_raw,
            "ui_conclusion": ui_pack["conclusion"],
            "ui_reason": ui_pack["reason"],
            "gpt_direction_hint": gpt_direction_hint,  # UI 전용
            "key_features": key_features,
            "snapshot": None,
            "meta_info": None,
            "engine": "gpt_analysis",
            "error": err,
        }

    # validate 성공: 정상 분석 구조
    return {
        "ai_ok": True,
        "side": None,               # 룰 엔진에서만 방향을 결정
        "player_prob": None,
        "banker_prob": None,
        "tie_prob": None,
        "confidence": None,
        "confidence_raw": None,
        "confidence_notes": confidence_notes,
        "mode_raw": str(mode_raw or "mixed"),
        "comment": ui_comment,
        "comment_raw": comment_raw,
        "ui_conclusion": ui_pack["conclusion"],
        "ui_reason": ui_pack["reason"],
        "gpt_direction_hint": gpt_direction_hint,  # UI 전용
        "key_features": key_features,
        "snapshot": None,
        "meta_info": None,
        "engine": "gpt_analysis",
        "error": "",
    }
