# -*- coding: utf-8 -*-
"""
gpt_engine.py
====================================================
Baccarat Predictor AI Engine — GPT Pattern Interpreter (STRICT / NO-FALLBACK)
====================================================

역할
- 중국점/본점 패턴 입력(last5 + runs + 상태값 + future china roads)을 받아
  GPT에게 "패턴 해석 + 방향(side) 제안"을 요청한다.
- 반환은 오직 JSON 스키마로만 허용한다.
- 이 모듈은 UI 표시를 하지 않는다(서버 내부에서만 사용).

운영 원칙
- STRICT · NO-FALLBACK · FAIL-FAST
- 누락/스키마 위반/JSON 파싱 실패 → 즉시 RuntimeError
- 임의 기본값 생성 금지
- 조용한 예외 삼키기 금지
- 민감정보(API Key) 로그 금지

GPT 입력 스키마 (필수)
{
  "last5": "PPBBB",                    # 길이=5, P/B only
  "runs": [3,2,1],                     # 최근 run 길이(양의 정수들)
  "pattern_type": "streak|pingpong|blocks|mixed|random|chaos",
  "flow_state": "DEAD|TEST|ALIVE",
  "china_state": "ALIVE|WEAK|BROKEN|UNKNOWN",
  "future_if_P": {"big_eye": "R|B|null", "small": "R|B|null", "cockroach": "R|B|null"},
  "future_if_B": {"big_eye": "R|B|null", "small": "R|B|null", "cockroach": "R|B|null"}
}

GPT 출력 스키마 (필수)
{
  "side": "P|B|HOLD",
  "confidence": 0.0~1.0
}
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, TypedDict, cast


# -----------------------------
# Types
# -----------------------------
Side = Literal["P", "B", "HOLD"]
FlowState = Literal["DEAD", "TEST", "ALIVE"]
ChinaState = Literal["ALIVE", "WEAK", "BROKEN", "UNKNOWN"]
PatternType = Literal["streak", "pingpong", "blocks", "mixed", "random", "chaos"]
RB = Optional[Literal["R", "B"]]


class FutureChina(TypedDict):
    big_eye: RB
    small: RB
    cockroach: RB


class GptInput(TypedDict):
    last5: str
    runs: list[int]
    pattern_type: str
    flow_state: str
    china_state: str
    future_if_P: FutureChina
    future_if_B: FutureChina


class GptOutput(TypedDict):
    side: str
    confidence: Any


@dataclass(frozen=True, slots=True)
class GptDecision:
    side: Side
    confidence: float


class GptEngineError(RuntimeError):
    pass


# -----------------------------
# Prompt
# -----------------------------
_SYSTEM_PROMPT: str = (
    "You are a baccarat Big Road pattern interpreter.\n"
    "You MUST return ONLY valid JSON (no markdown, no extra text).\n"
    "Input fields describe the last 5 results (P/B only), run lengths, pattern state, "
    "flow state, china state, and future china roads if next is P vs B.\n\n"
    "Return exactly this JSON schema:\n"
    '{ "side": "P|B|HOLD", "confidence": 0.0 }\n\n'
    "Rules:\n"
    "- side must be one of: P, B, HOLD\n"
    "- confidence must be a finite number between 0.0 and 1.0\n"
    "- If uncertain or pattern is chaotic/unsafe, choose HOLD\n"
    "- Do NOT include any additional keys.\n"
)


# -----------------------------
# Strict validators
# -----------------------------
def _die(msg: str) -> None:
    raise GptEngineError(msg)


def _require_str(d: Dict[str, Any], key: str) -> str:
    if key not in d:
        _die(f"missing required field: {key}")
    v = d[key]
    if not isinstance(v, str) or not v.strip():
        _die(f"invalid string field: {key}")
    return v.strip()


def _require_list_int(d: Dict[str, Any], key: str) -> list[int]:
    if key not in d:
        _die(f"missing required field: {key}")
    v = d[key]
    if not isinstance(v, list) or not v:
        _die(f"invalid list field: {key}")
    out: list[int] = []
    for i, x in enumerate(v):
        if isinstance(x, bool):
            _die(f"invalid runs[{i}]: bool not allowed")
        try:
            n = int(x)
        except Exception:
            _die(f"invalid runs[{i}]: int required")
        if n <= 0:
            _die(f"invalid runs[{i}]: must be > 0")
        out.append(n)
    return out


def _require_last5(last5: str) -> str:
    s = last5.strip().upper()
    if len(s) != 5:
        _die("last5 must be length=5 (P/B only)")
    for ch in s:
        if ch not in ("P", "B"):
            _die("last5 must contain only 'P' or 'B'")
    return s


def _require_enum(name: str, value: str, allowed: set[str]) -> str:
    v = value.strip()
    if v not in allowed:
        _die(f"{name} invalid: {v!r} (allowed={sorted(list(allowed))})")
    return v


def _require_future_china(d: Dict[str, Any], key: str) -> FutureChina:
    if key not in d:
        _die(f"missing required field: {key}")
    v = d[key]
    if not isinstance(v, dict):
        _die(f"{key} must be dict")
    for k in ("big_eye", "small", "cockroach"):
        if k not in v:
            _die(f"{key}.{k} missing")
        vv = v[k]
        if vv is None:
            continue
        if not isinstance(vv, str):
            _die(f"{key}.{k} must be 'R'|'B'|null")
        s = vv.strip().upper()
        if s not in ("R", "B"):
            _die(f"{key}.{k} must be 'R'|'B'|null")
        v[k] = s
    return cast(FutureChina, {"big_eye": v["big_eye"], "small": v["small"], "cockroach": v["cockroach"]})


def validate_gpt_input(data: Dict[str, Any]) -> GptInput:
    if not isinstance(data, dict) or not data:
        _die("gpt input must be a non-empty dict")

    last5_raw = _require_str(data, "last5")
    last5 = _require_last5(last5_raw)

    runs = _require_list_int(data, "runs")

    pattern_type = _require_str(data, "pattern_type").lower()
    pattern_type = _require_enum(
        "pattern_type",
        pattern_type,
        {"streak", "pingpong", "blocks", "mixed", "random", "chaos"},
    )

    flow_state = _require_str(data, "flow_state").upper()
    flow_state = _require_enum("flow_state", flow_state, {"DEAD", "TEST", "ALIVE"})

    china_state = _require_str(data, "china_state").upper()
    china_state = _require_enum("china_state", china_state, {"ALIVE", "WEAK", "BROKEN", "UNKNOWN"})

    future_if_p = _require_future_china(data, "future_if_P")
    future_if_b = _require_future_china(data, "future_if_B")

    return cast(
        GptInput,
        {
            "last5": last5,
            "runs": runs,
            "pattern_type": pattern_type,
            "flow_state": flow_state,
            "china_state": china_state,
            "future_if_P": future_if_p,
            "future_if_B": future_if_b,
        },
    )


def _parse_gpt_output_strict(raw: str) -> GptDecision:
    if not isinstance(raw, str) or not raw.strip():
        _die("GPT returned empty content")

    try:
        obj = json.loads(raw)
    except Exception:
        _die("GPT returned non-JSON content")

    if not isinstance(obj, dict):
        _die("GPT output must be a JSON object")

    if "side" not in obj or "confidence" not in obj:
        _die("GPT output missing required keys: side/confidence")

    side_raw = obj["side"]
    if not isinstance(side_raw, str):
        _die("GPT output side must be string")
    side = side_raw.strip().upper()
    if side not in ("P", "B", "HOLD"):
        _die(f"GPT output invalid side: {side!r}")

    conf_raw = obj["confidence"]
    if isinstance(conf_raw, bool):
        _die("GPT output confidence must be numeric (bool not allowed)")
    try:
        conf = float(conf_raw)
    except Exception:
        _die("GPT output confidence must be numeric")

    if not math.isfinite(conf):
        _die("GPT output confidence must be finite")
    if conf < 0.0 or conf > 1.0:
        _die("GPT output confidence must be within [0,1]")

    return GptDecision(side=cast(Side, side), confidence=float(conf))


# -----------------------------
# OpenAI client (STRICT)
# -----------------------------
def _require_openai_api_key(api_key: Optional[str]) -> str:
    if api_key is not None:
        if not isinstance(api_key, str) or not api_key.strip():
            _die("api_key provided but empty/invalid")
        return api_key.strip()

    env_key = os.getenv("OPENAI_API_KEY")
    if not env_key or not str(env_key).strip():
        _die("missing OPENAI_API_KEY (env) and api_key not provided")
    return str(env_key).strip()


def _make_client(api_key: str):
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise GptEngineError(f"openai client import failed: {type(e).__name__}") from None
    return OpenAI(api_key=api_key)


# -----------------------------
# Public API
# -----------------------------
def gpt_decide(
    data: Dict[str, Any],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    timeout_sec: Optional[float] = None,
    api_key: Optional[str] = None,
) -> GptDecision:
    """
    STRICT:
    - 입력 검증 실패 → RuntimeError
    - GPT 응답 JSON 스키마 위반 → RuntimeError
    - 네트워크/API 오류 → 예외 전파 (조용히 무시 금지)

    Returns:
      GptDecision(side, confidence)
    """
    # Validate input strictly
    payload = validate_gpt_input(data)

    # Validate call params (STRICT)
    if not isinstance(model, str) or not model.strip():
        _die("model must be non-empty string")
    if isinstance(temperature, bool):
        _die("temperature must be numeric (bool not allowed)")
    try:
        temp = float(temperature)
    except Exception:
        _die("temperature must be numeric")
    if not math.isfinite(temp) or temp < 0.0 or temp > 2.0:
        _die("temperature must be within [0,2]")

    # OpenAI client
    key = _require_openai_api_key(api_key)
    client = _make_client(key)

    # Prepare messages
    user_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    # Call
    # NOTE: timeout handling depends on the installed openai client. We pass it only if provided.
    kwargs: Dict[str, Any] = {}
    if timeout_sec is not None:
        if isinstance(timeout_sec, bool):
            _die("timeout_sec must be numeric (bool not allowed)")
        try:
            tmo = float(timeout_sec)
        except Exception:
            _die("timeout_sec must be numeric")
        if not math.isfinite(tmo) or tmo <= 0.0:
            _die("timeout_sec must be > 0")
        # openai-python supports `timeout` in many transports; if unsupported, it will raise (STRICT).
        kwargs["timeout"] = tmo

    resp = client.chat.completions.create(
        model=model.strip(),
        temperature=temp,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        **kwargs,
    )

    try:
        raw = resp.choices[0].message.content
    except Exception:
        _die("GPT response missing message content")

    return _parse_gpt_output_strict(raw)


__all__ = [
    "GptEngineError",
    "GptDecision",
    "validate_gpt_input",
    "gpt_decide",
]