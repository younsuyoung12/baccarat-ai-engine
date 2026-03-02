# -*- coding: utf-8 -*-
# gpt_client.py
"""
OpenAI GPT 클라이언트 모듈 (v11 – 분석 전용 모드)

역할:
- Feature JSON → GPT 프롬프트 구성
- OpenAI Responses API 호출
- GPT 응답 JSON 파싱

v11 변경 사항:
- GPT는 더 이상 P/B/T 확률이나 confidence, 베팅 방향을 반환하지 않는다.
- 오직 다음 정보만 JSON으로 반환한다.
  {
    "mode": "pattern" | "flow" | "mixed" | "chaos",
    "comment": "한국어 해설",
    "key_features": ["..."],
    "risk_tags": ["high_chaos", "stable", ...]  # 옵션
  }
- 확률/방향 관련 필드는 절대 포함하지 않도록 system prompt 에서 강하게 제한한다.

변경 (2026-01-02)
-----------------
- GPT 해설을 “행동형/직관형”으로 강화:
  - comment 첫 문장은 반드시 아래 4개 중 하나의 결론으로 시작하도록 유도
    1) 관망이 유리한 구간
    2) 탐색(PROBE)만 허용되는 구간
    3) 정상 진입이 가능한 구간
    4) 전환을 주의 깊게 관찰해야 하는 구간
- 단, 베팅 방향(P/B)·확률(%)·추천 문구는 계속 절대 금지 (분석 전용 모드 유지)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Tuple, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None

load_dotenv()

OPENAI_MODEL = "gpt-4o-mini"

client: Any = None  # type: ignore[assignment]
if OpenAI is not None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        print(f"[AI] OpenAI 클라이언트 초기화 완료 (모델: {OPENAI_MODEL})", flush=True)
    else:
        print("[AI] 경고: OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다. AI 분석 비활성.", flush=True)
else:
    print("[AI] 경고: 'openai' 패키지를 찾을 수 없습니다. 'pip install openai' 필요.", flush=True)


def _extract_json_from_text(raw_text: str) -> Dict[str, Any]:
    """GPT 응답 텍스트에서 JSON 블록만 파싱."""
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("JSON 블록을 찾을 수 없습니다.")
    json_str = cleaned[start: end + 1]

    try:
        return json.loads(json_str)
    except Exception:
        def _sanitize_json_loose(s: str) -> str:
            s2 = s.strip()
            s2 = s2.replace("None", "null")
            s2 = s2.replace("True", "true")
            s2 = s2.replace("False", "false")
            s2 = re.sub(r",\s*([}\]])", r"\1", s2)
            return s2
        sanitized = _sanitize_json_loose(json_str)
        return json.loads(sanitized)


def _build_gpt_features_from_full_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """GPT에 전달하는 압축 Feature JSON 구성."""

    leader_conf = features.get("leader_confidence")
    if leader_conf is None:
        leader_conf = features.get("leader_confidence_road")

    return {
        "engine_version": features.get("engine_version"),
        "rounds_total": features.get("rounds_total"),

        "pb_stats": {
            "p_count": features.get("p_count"),
            "b_count": features.get("b_count"),
            "t_count": features.get("t_count"),
            "pb_ratio": features.get("pb_ratio"),
            "tie_ratio": features.get("tie_ratio"),
            "entropy": features.get("entropy"),
        },

        "current_streak": features.get("current_streak"),
        "trend_strength": features.get("trend_strength"),
        "momentum": features.get("momentum"),
        "last_20": features.get("last_20"),

        "pattern": {
            "score": features.get("pattern_score"),
            "type": features.get("pattern_type"),
            "energy": features.get("pattern_energy"),
            "symmetry": features.get("pattern_symmetry"),
            "noise_ratio": features.get("pattern_noise_ratio"),
            "reversal_signal": features.get("pattern_reversal_signal"),
            "drift": features.get("pattern_drift"),
        },

        "flow": {
            "strength": features.get("flow_strength"),
            "stability": features.get("flow_stability"),
            "chaos_risk": features.get("flow_chaos_risk"),
            "reversal_risk": features.get("flow_reversal_risk"),
            "direction": features.get("flow_direction"),
        },

        "temporal": {
            "run_speed": features.get("run_speed"),
            "tie_volatility": features.get("tie_volatility"),
        },

        "chinese_roads": {
            "big_eye_recent": features.get("big_eye_recent"),
            "small_road_recent": features.get("small_road_recent"),
            "cockroach_recent": features.get("cockroach_recent"),
        },

        "future_scenarios": features.get("future_scenarios"),

        "advanced": {
            "road_sync_p": features.get("road_sync_p"),
            "road_sync_b": features.get("road_sync_b"),
            "road_sync_gap": features.get("road_sync_gap"),
            "segment_type": features.get("segment_type"),
            "transition_flag": features.get("transition_flag"),
            "mini_trend_p": features.get("mini_trend_p"),
            "mini_trend_b": features.get("mini_trend_b"),
            "china_agree_last12": features.get("china_agree_last12"),
            "big_eye_height_change": features.get("big_eye_height_change"),
            "small_road_height_change": features.get("small_road_height_change"),
            "cockroach_height_change": features.get("cockroach_height_change"),
            "chaos_index": features.get("chaos_index"),
            "pattern_score_global": features.get("pattern_score_global"),
            "pattern_score_last10": features.get("pattern_score_last10"),
            "pattern_score_last5": features.get("pattern_score_last5"),
            "pattern_stability": features.get("pattern_stability"),
            "big_eye_flips_last10": features.get("big_eye_flips_last10"),
            "small_road_flips_last10": features.get("small_road_flips_last10"),
            "cockroach_flips_last10": features.get("cockroach_flips_last10"),
            "flip_cycle_pb": features.get("flip_cycle_pb"),
            "frame_mode": features.get("frame_mode"),
            "response_delay_score": features.get("response_delay_score"),
            "odd_run_length": features.get("odd_run_length"),
            "odd_run_spike_flag": features.get("odd_run_spike_flag"),
            "global_chaos_ratio": features.get("global_chaos_ratio"),
            "frame_trend_delta": features.get("frame_trend_delta"),
            "three_rule_signal": features.get("three_rule_signal"),

            "china_r_streak_be": features.get("china_r_streak_be"),
            "china_r_streak_sm": features.get("china_r_streak_sm"),
            "china_r_streak_ck": features.get("china_r_streak_ck"),
            "china_depth_be": features.get("china_depth_be"),
            "china_depth_sm": features.get("china_depth_sm"),
            "china_depth_ck": features.get("china_depth_ck"),
            "bottom_touch_bigroad": features.get("bottom_touch_bigroad"),
            "bottom_touch_bigeye": features.get("bottom_touch_bigeye"),
            "bottom_touch_small": features.get("bottom_touch_small"),
            "bottom_touch_cockroach": features.get("bottom_touch_cockroach"),

            "decalcomania_found": features.get("decalcomania_found"),
            "decalcomania_hint": features.get("decalcomania_hint"),
            "decalcomania_support": features.get("decalcomania_support"),
            "pb_diff_score": features.get("pb_diff_score"),
            "shoe_phase": features.get("shoe_phase"),
            "phase_progress": features.get("phase_progress"),
            "chaos_end_flag": features.get("chaos_end_flag"),
            "tie_turbulence_rounds": features.get("tie_turbulence_rounds"),
            "entry_momentum": features.get("entry_momentum"),

            "shoe_regime": features.get("shoe_regime"),
            "regime_shift_score": features.get("regime_shift_score"),
            "regime_forecast_line2": features.get("regime_forecast_line2"),
            "regime_forecast_chaos3": features.get("regime_forecast_chaos3"),
            "regime_forecast_shift5": features.get("regime_forecast_shift5"),

            "beauty_score": features.get("beauty_score"),
        },

        "road_leader": {
            "leader_road": features.get("leader_road"),
            "leader_signal": features.get("leader_signal"),
            "leader_confidence": leader_conf,
            "road_hit_rates": features.get("road_hit_rates"),
            "road_prediction_totals": features.get("road_prediction_totals"),
        },
    }


def call_gpt_engine(features: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    GPT 모델 호출 및 JSON 응답 파싱 (분석 전용).

    반환:
        (data, error)
        data 가 None 이 아니면 아래 스펙을 따른다.
        {
          "mode": "pattern" | "flow" | "mixed" | "chaos",
          "comment": str,
          "key_features": [str, ...],
          "risk_tags": [str, ...]   # 옵션
        }
    """
    if client is None:
        return None, "OpenAI 클라이언트가 초기화되지 않았습니다. (API 키 / 라이브러리 확인 필요)"

    rounds = int(features.get("rounds_total") or 0)
    if rounds <= 0:
        reason = "아직 입력된 라운드가 없습니다."
        print("[AI] 최소 판수 미달, GPT 호출 생략:", reason, flush=True)
        return None, reason

    try:
        gpt_features = _build_gpt_features_from_full_features(features)
        payload_str = json.dumps(gpt_features, ensure_ascii=False)
        print(f"[AI] {OPENAI_MODEL} 요청 시작 – 총 라운드 수: {rounds}", flush=True)

        # ---------------------------
        # SYSTEM PROMPT (행동형/직관형 해설 강화, 방향/확률/추천 금지 유지)
        # ---------------------------
        system_prompt = (
            "너의 역할은 바카라 베팅을 대신 판단하거나 다음 결과(P/B)를 예측하는 것이 아니다.\n"
            "너는 현재 슈(shoe)의 상태를 분석하여, 플레이어가 지금 이 구간을 어떻게 다루는 것이 유리한지\n"
            "명확하고 단정적으로 설명하는 “AI 해설자”이다.\n\n"
            "입력 JSON에는 Big Road 기반 본매 패턴(score/type/energy/drift), 중국점 흐름(strength/stability/chaos_risk),\n"
            "P/B/T 통계, 스트릭, Temporal Feature, future_scenarios, 고급 advanced Feature, 슈 레짐(shoe_regime)과\n"
            "Regime Forecast 정보, Road Leader 요약 정보가 포함된다.\n"
            "또한 advanced.beauty_score 는 0~100 범위의 값으로, 사람 눈 기준으로 Big Road/중국점 그림이\n"
            "얼마나 안정적이고 예쁜지에 대한 점수이다.\n\n"
            "절대 금지 사항:\n"
            "- 다음 판 결과(P/B) 예측\n"
            "- 베팅 방향 추천\n"
            "- 확률(%) 제시\n"
            "- 맞출 수 있다는 표현\n"
            "- 아래 필드 출력 금지: player_prob / banker_prob / tie_prob / confidence\n"
            "- P/B/T 쪽을 추천하는 문장(예: '뱅커가 좋아 보입니다', '플레이어 추천')\n\n"
            "반드시 지켜야 할 출력 원칙:\n"
            "1) 분석 결과의 결론을 먼저 제시한다.\n"
            "2) 결론은 반드시 아래 4가지 중 하나로 명확히 선택한다.\n"
            "   - 관망이 유리한 구간\n"
            "   - 탐색(PROBE)만 허용되는 구간\n"
            "   - 정상 진입이 가능한 구간\n"
            "   - 전환을 주의 깊게 관찰해야 하는 구간\n"
            "3) 결론은 애매한 표현 없이 단정적으로 말한다.\n"
            "4) 그 다음 줄에, 왜 그런 판단이 나왔는지 패턴 안정성/흐름 지속성/혼돈 위험도 관점에서 짧게 설명한다.\n"
            "5) 문장은 베팅하는 사람이 즉시 행동을 떠올릴 수 있도록 작성한다.\n\n"
            "톤과 스타일:\n"
            "- 분석가가 아닌 실전 플레이어에게 설명하듯 작성한다.\n"
            "- 불필요한 완곡어법을 사용하지 않는다.\n"
            "- “~으로 보입니다”, “~일 수 있습니다” 같은 표현은 피한다.\n"
            "- 간결하고 명확한 문장을 사용한다.\n\n"
            "출력 언어:\n"
            "- 한국어\n\n"
            "오직 아래 JSON 형식으로만 응답해라(추가 텍스트 금지).\n"
            "{\n"
            "  \"mode\": \"pattern\" | \"flow\" | \"mixed\" | \"chaos\",\n"
            "  \"comment\": \"한국어 해설(첫 문장은 반드시 결론 4개 중 하나로 시작)\",\n"
            "  \"key_features\": [\"핵심 특징 1\", \"핵심 특징 2\", ...],\n"
            "  \"risk_tags\": [\"high_chaos\", \"stable\", \"danger_zone\", ...]\n"
            "}\n"
        )

        user_prompt = (
            "아래는 현재 바카라 슈의 상태에서 추출한 패턴/흐름/시계열/고급/레짐/로드 리더 Feature JSON이다.\n"
            "이 데이터만 사용해서 현재 슈의 상태를 요약하고, 위험도와 특징을 분석해라.\n"
            "단, 다음 한 수가 플레이어/뱅커/타이일 확률이나 베팅 방향은 절대 언급하지 마라.\n"
            "comment 첫 문장은 반드시 아래 결론 4개 중 하나로 시작해라:\n"
            "- 관망이 유리한 구간\n"
            "- 탐색(PROBE)만 허용되는 구간\n"
            "- 정상 진입이 가능한 구간\n"
            "- 전환을 주의 깊게 관찰해야 하는 구간\n\n"
            + payload_str
        )

        response = client.responses.create(  # type: ignore[call-arg]
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_text = getattr(response, "output_text", "") or ""
        print(f"[AI] {OPENAI_MODEL} 원본 응답:", raw_text, flush=True)

        data = _extract_json_from_text(raw_text)

        mode = data.get("mode")
        comment = data.get("comment")
        if not isinstance(mode, str):
            raise ValueError("mode 필드가 없거나 문자열이 아님")
        if not isinstance(comment, str):
            raise ValueError("comment 필드가 없거나 문자열이 아님")

        kf = data.get("key_features")
        if kf is None:
            data["key_features"] = []
        elif not isinstance(kf, list):
            data["key_features"] = [str(kf)]
        else:
            data["key_features"] = [str(x) for x in kf][:10]

        rt = data.get("risk_tags")
        if rt is None:
            data["risk_tags"] = []
        elif not isinstance(rt, list):
            data["risk_tags"] = [str(rt)]
        else:
            data["risk_tags"] = [str(x) for x in rt][:10]

        print(
            f"[AI] 파싱된 분석 – mode:{data['mode']}, "
            f"risk_tags:{','.join(data['risk_tags'])}",
            flush=True,
        )
        return data, ""
    except Exception as e:  # pragma: no cover
        print(f"[AI] {OPENAI_MODEL} 호출/파싱 중 오류:", repr(e), flush=True)
        return None, f"GPT 호출 또는 파싱 중 오류: {repr(e)}"
