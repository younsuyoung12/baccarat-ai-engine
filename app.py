# -*- coding: utf-8 -*-
"""
app.py
====================================================
Baccarat Predictor AI Engine (Flask API)
----------------------------------------------------
정책:
- READY 상태에서는 절대 폴백 금지:
  - ai_ok=False 이면 즉시 예외(500) + 롤백
  - future_scenarios 타입/필수키(P/B) 불일치 시 즉시 예외(500)
- WARMUP 상태(데이터/패턴 준비 미달)에서는 예외로 500을 내지 않는다:
  - bet_side=PASS, ai_ok=False 로 200 OK 반환
  - 상태 누적은 유지(save_engine_state), 엑셀 기록은 하지 않는다
- 실패 시 상태 오염 금지:
  - READY 처리 중 예외 발생 시 직전 스냅샷으로 롤백
  - (이미 엑셀 기록했으면) 마지막 1행 제거
- 입력 계약 위반은 즉시 차단(폴백 금지):
  - /predict는 winner(P/B/T) 필수
  - JSON 타입 오류/누락은 400으로 즉시 반환(서버 500 금지)
- 중복 입력(근본 원인) 원천 차단:
  - (권장) request_id(=클라이언트 이벤트 ID)가 동일하면 서버는 상태 변경 없이 직전 응답을 반환(200)
  - request_id가 없더라도, 매우 짧은 시간 내 동일 winner 연속 요청은 더블클릭으로 간주하고 직전 응답 반환(200)
  - 서버는 라운드 카운터의 단일 진실: len(big_road) 기반 expected_round_id

변경 요약 (2026-03-04, STRICT / NO-LEARNING)
----------------------------------------------------
1) meta_learning 의존 제거(학습 사용 안함 정책)
   - meta_learning import 제거
   - snapshot/restore/reset에서 meta_learning 관련 상태 제거
2) rl_learning_enabled 기본 False (학습 비활성)
   - payload 키는 호환을 위해 유지하되 항상 False로 운용
----------------------------------------------------

변경 요약 (2025-12-30, TIE CONTRACT HOTFIX)
----------------------------------------------------
1) /predict에서 Tie(T) 입력은 파이프라인(예: predictor_adapter/features/pattern/recommend)과 엑셀 로깅을 절대 호출하지 않고,
   road 무결성 검사 이후 즉시 WARMUP(payload)로 종료한다. (Tie는 메타/숫자로만 누적)
2) READY/WARMUP/UNDO 응답의 UI용 P/B 전용 배열(big_road, big)에는 Tie(T)가 값으로 내려가지 않도록
   _pb_seq_no_ties(road.big_road)로 필터링된 배열만 내려보낸다.
   (ties 카운트는 pb_stats["T"]로 유지)
----------------------------------------------------

변경 요약 (2025-12-23, HOTFIX)
----------------------------------------------------
1) /predict에서 round_id 의존 제거(근본 해결)
   - 요청 body의 round_id는 더 이상 요구하지 않으며(있어도 무시), 서버가 내부 상태(len(big_road))로 라운드를 진행
   - 409(ROUND_ID_GAP) 계열 충돌을 구조적으로 제거
2) 중복 POST 근본 방지 강화
   - request_id(client_request_id) 지원: 동일 request_id 재수신 시 상태 변경 없이 직전 응답 반환(200)
   - request_id 미제공 시에도 더블클릭(초단기 동일 winner 연속 요청) 가드로 상태 오염 방지
3) expected_round_id 계산을 단일화
   - last_processed_round_id는 “캐시”로만 유지하고, 모든 진실은 len(road.big_road)에서 계산
4) 응답 키 정리/추가
   - expected_round_id 항상 포함
   - (호환) next_round_id alias 추가
   - received_round_id는 디버깅 용도로만 유지(요청에 있을 때만 echo)
----------------------------------------------------
"""

from __future__ import annotations

import copy
import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request

import features
import pattern
import road
import road_leader
import engine_state

load_engine_state = engine_state.load_engine_state
save_engine_state = engine_state.save_engine_state

from excel_logger import append_round_log_to_excel, new_shoe_id, remove_last_round_log_for_shoe
from predictor_adapter import run_ai_pipeline, IS_RESETTING

# ------------------------------------------------------------
# 파일 경로(단일 프로세스 기준)
# ------------------------------------------------------------
STATE_FILE_PATH = "engine_state.json"
SHOE_ID_FILE_PATH = "current_shoe_id.txt"

app = Flask(__name__)


# ------------------------------------------------------------
# 전역 상태 (AI 통계 / 예측 / UNDO / 입력 정합성)
# ------------------------------------------------------------
current_shoe_id: Optional[str] = None

ai_stats: Dict[str, int] = {"total": 0, "correct": 0}
ai_streak_win: int = 0
ai_streak_lose: int = 0
ai_recent_results: List[int] = []  # 1=win, 0=lose (최근 10판)

last_prediction: Optional[Dict[str, Any]] = None
last_round_snapshot: Optional[Dict[str, Any]] = None
last_ui_state_before_predict: Optional[Dict[str, Any]] = None
last_response_payload: Optional[Dict[str, Any]] = None

# 캐시(진실은 len(big_road))
last_processed_round_id: int = 0

# ✅ 학습 비활성 (호환 키 유지)
rl_learning_enabled: bool = False

# ------------------------------------------------------------
# 중복 요청 방지(서버 단)
# ------------------------------------------------------------
_recent_request_ids: deque[str] = deque(maxlen=200)
_recent_request_id_set: set[str] = set()

_DOUBLE_CLICK_WINDOW_SEC = 0.35
_last_accept_mono: float = 0.0
_last_accept_winner: Optional[str] = None
_last_accept_payload_ref: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------
# 내부 유틸
# ------------------------------------------------------------
def _read_text_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = (f.read() or "").strip()
            return s or None
    except FileNotFoundError:
        return None


def _write_text_file(path: str, value: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)


def _ensure_shoe_id_loaded() -> None:
    global current_shoe_id
    if current_shoe_id:
        return

    sid = _read_text_file(SHOE_ID_FILE_PATH)
    if sid:
        current_shoe_id = sid
        return

    current_shoe_id = new_shoe_id()
    _write_text_file(SHOE_ID_FILE_PATH, current_shoe_id)


def _set_new_shoe_id() -> None:
    global current_shoe_id
    current_shoe_id = new_shoe_id()
    _write_text_file(SHOE_ID_FILE_PATH, current_shoe_id)


def _require_dict(v: Any, name: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        raise TypeError(f"{name} must be dict, got {type(v).__name__}")
    return v


def _require_nonempty_str(v: Any, name: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{name} must be non-empty string")
    return v.strip()
def _require(d: Dict[str, Any], key: str) -> Any:
    """
    STRICT:
    - 키 누락 시 즉시 예외
    - 폴백 금지
    """
    if not isinstance(d, dict):
        raise TypeError(f"_require: d must be dict, got {type(d).__name__}")
    if key not in d:
        raise KeyError(f"missing required field: {key}")
    return d[key]

def _pb_seq_no_ties(big_road: List[str]) -> List[str]:
    return [x for x in big_road if x in ("P", "B")]


def _compute_pb_stats(big_road: List[str]) -> Dict[str, int]:
    out = {"P": 0, "B": 0, "T": 0}
    for x in big_road:
        if x in out:
            out[x] += 1
    return out


def _compute_streak_info(big_road: List[str]) -> Dict[str, Any]:
    seq = big_road
    last = None
    cnt = 0
    for x in reversed(seq):
        if x == "T":
            continue
        if x not in ("P", "B"):
            continue
        if last is None:
            last = x
            cnt = 1
        else:
            if x == last:
                cnt += 1
            else:
                break
    return {"type": last, "count": cnt}


def _ui_ready_flag() -> bool:
    return len(getattr(road, "big_road", []) or []) >= 1


def _ui_ready_reason() -> str:
    br_len = len(getattr(road, "big_road", []) or [])
    if br_len >= 1:
        return "READY"
    return "BigRoad empty"


def _safe_len_big_road() -> int:
    br = getattr(road, "big_road", None)
    if br is None or not isinstance(br, list):
        raise RuntimeError("ROAD_CONTRACT_VIOLATION: road.big_road is missing or not list")
    return len(br)


def _expected_round_id() -> int:
    return _safe_len_big_road() + 1


def _sync_round_id_from_big_road() -> None:
    global last_processed_round_id
    last_processed_round_id = _safe_len_big_road()


def _validate_big_road_matrix_strict() -> None:
    m = getattr(road, "big_road_matrix", None)
    if m is None:
        raise RuntimeError("ROAD_CONTRACT_VIOLATION: road.big_road_matrix is missing")

    if not isinstance(m, list):
        raise TypeError("ROAD_CONTRACT_VIOLATION: road.big_road_matrix must be list")

    for r, row in enumerate(m):
        if not isinstance(row, list):
            raise TypeError(f"ROAD_CONTRACT_VIOLATION: big_road_matrix[{r}] must be list")
        for c, cell in enumerate(row):
            if cell is None:
                continue
            if isinstance(cell, str):
                s = cell.strip().upper()
                if len(s) >= 2 and ("P" in s) and ("B" in s):
                    raise RuntimeError(
                        f"ROADMAP_BROKEN: big_road_matrix cell has both P and B at ({r},{c}): {cell!r}"
                    )
            else:
                raise TypeError(
                    f"ROAD_CONTRACT_VIOLATION: big_road_matrix cell must be str/None, got {type(cell).__name__} at ({r},{c})"
                )


def _json_error(
    status_code: int,
    error_code: str,
    detail: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    payload: Dict[str, Any] = {"status": "error", "error": error_code}
    if detail:
        payload["detail"] = detail
    if extra:
        payload.update(extra)
    return jsonify(payload), status_code


def _extract_request_id(data: Dict[str, Any]) -> Optional[str]:
    v = data.get("request_id") or data.get("client_request_id") or data.get("event_id")
    if v is None:
        return None
    if not isinstance(v, str):
        return None
    s = v.strip()
    return s or None


def _mark_request_id_processed(req_id: str) -> None:
    if req_id in _recent_request_id_set:
        return
    if len(_recent_request_ids) == _recent_request_ids.maxlen:
        old = _recent_request_ids.popleft()
        _recent_request_id_set.discard(old)
    _recent_request_ids.append(req_id)
    _recent_request_id_set.add(req_id)


def _is_request_id_processed(req_id: str) -> bool:
    return req_id in _recent_request_id_set


def _build_not_ready_payload(reason: str, winner_raw: str) -> Dict[str, Any]:
    pb_stats = _compute_pb_stats(road.big_road)
    streak = _compute_streak_info(road.big_road)
    big_tie_matrix = road.build_big_road_tie_matrix()
    big_pb = _pb_seq_no_ties(road.big_road)

    if winner_raw == "T":
        warmup_analysis = f"TIE: {reason}"
        bet_reason = "TIE"
    else:
        warmup_analysis = f"WARMUP: {reason}"
        bet_reason = "WARMUP"

    ai_total = int(ai_stats.get("total") or 0)
    ai_correct = int(ai_stats.get("correct") or 0)

    return {
        "status": "ok",
        "winner": winner_raw,
        "bet_side": "PASS",
        "bet_unit": 0,
        "bet_reason": bet_reason,
        "bet_side_display": "PASS",
        "analysis": warmup_analysis,
        "ai_ok": False,
        "ai_error": reason,
        "ai_engine": "warmup",
        "enforced_mode": None,
        "strategy_mode": None,
        "strategy_comment": None,
        "risk_tags": [],
        "key_features": [],
        "rounds_total": len(road.big_road),
        "shoe_id": current_shoe_id,
        "expected_round_id": _expected_round_id(),
        "next_round_id": _expected_round_id(),
        "ui_ready": bool(_ui_ready_flag()),
        "ui_ready_reason": _ui_ready_reason(),
        "bead_seq": big_pb,
        "big_road": big_pb,
        "big_road_matrix": road.big_road_matrix,
        "big_eye": road.big_eye_seq,
        "small_road": road.small_road_seq,
        "cockroach": road.cockroach_seq,
        "big_eye_matrix": road.big_eye_matrix,
        "small_road_matrix": road.small_road_matrix,
        "cockroach_matrix": road.cockroach_matrix,
        "big_tie_matrix": big_tie_matrix,
        "big": big_pb,
        "china": {
            "big_eye": road.big_eye_seq,
            "small_road": road.small_road_seq,
            "cockroach": road.cockroach_seq,
        },
        "player_wins": pb_stats["P"],
        "banker_wins": pb_stats["B"],
        "ties": pb_stats["T"],
        "current_streak_type": streak["type"],
        "current_streak_count": streak["count"],
        "ai_total": ai_total,
        "ai_correct": ai_correct,
        "ai_win_rate": (ai_correct / ai_total) if ai_total > 0 else 0.0,
        "ai_win_rate_pct": int(round(((ai_correct / ai_total) if ai_total > 0 else 0.0) * 100)),
        "ai_streak_win": ai_streak_win,
        "ai_streak_lose": ai_streak_lose,
        "ai_recent_results": ai_recent_results,
        "rl_learning_enabled": rl_learning_enabled,
        "engine_version": features.ENGINE_VERSION,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "not_ready": True,
        "not_ready_reason": reason,
    }


def _snapshot_all_state() -> Dict[str, Any]:
    return {
        "current_shoe_id": current_shoe_id,
        "ai_stats": copy.deepcopy(ai_stats),
        "ai_streak_win": int(ai_streak_win),
        "ai_streak_lose": int(ai_streak_lose),
        "ai_recent_results": copy.deepcopy(ai_recent_results),
        "last_prediction": copy.deepcopy(last_prediction),
        "last_response_payload": copy.deepcopy(last_response_payload),
        "last_processed_round_id": int(last_processed_round_id),
        "leader_state": road_leader.get_state(),
        "pattern_score_history": copy.deepcopy(pattern.pattern_score_history),
        "big_road": copy.deepcopy(road.big_road),
        "big_road_matrix": copy.deepcopy(road.big_road_matrix),
        "big_road_positions": copy.deepcopy(road.big_road_positions),
        "big_eye_seq": copy.deepcopy(road.big_eye_seq),
        "small_road_seq": copy.deepcopy(road.small_road_seq),
        "cockroach_seq": copy.deepcopy(road.cockroach_seq),
        "big_eye_matrix": copy.deepcopy(road.big_eye_matrix),
        "small_road_matrix": copy.deepcopy(road.small_road_matrix),
        "cockroach_matrix": copy.deepcopy(road.cockroach_matrix),
    }


def _restore_all_state(snap: Dict[str, Any]) -> None:
    global current_shoe_id
    global ai_stats, ai_streak_win, ai_streak_lose, ai_recent_results
    global last_prediction, last_response_payload, last_processed_round_id

    if not isinstance(snap, dict):
        raise TypeError("snapshot must be dict")

    current_shoe_id = snap.get("current_shoe_id")
    if current_shoe_id:
        _write_text_file(SHOE_ID_FILE_PATH, str(current_shoe_id))

    ai_stats = copy.deepcopy(_require_dict(snap.get("ai_stats"), "snap.ai_stats"))
    ai_streak_win = int(snap.get("ai_streak_win", 0))
    ai_streak_lose = int(snap.get("ai_streak_lose", 0))
    ai_recent_results = copy.deepcopy(snap.get("ai_recent_results", []))

    last_prediction = copy.deepcopy(snap.get("last_prediction"))
    last_response_payload = copy.deepcopy(snap.get("last_response_payload"))
    last_processed_round_id = int(snap.get("last_processed_round_id", 0))

    road_leader.set_state(snap.get("leader_state"))
    pattern.pattern_score_history = copy.deepcopy(snap.get("pattern_score_history", []))

    road.big_road = copy.deepcopy(snap.get("big_road", []))
    road.big_road_matrix = copy.deepcopy(snap.get("big_road_matrix", []))
    road.big_road_positions = copy.deepcopy(snap.get("big_road_positions", []))
    road.big_eye_seq = copy.deepcopy(snap.get("big_eye_seq", []))
    road.small_road_seq = copy.deepcopy(snap.get("small_road_seq", []))
    road.cockroach_seq = copy.deepcopy(snap.get("cockroach_seq", []))
    road.big_eye_matrix = copy.deepcopy(snap.get("big_eye_matrix", []))
    road.small_road_matrix = copy.deepcopy(snap.get("small_road_matrix", []))
    road.cockroach_matrix = copy.deepcopy(snap.get("cockroach_matrix", []))


# ------------------------------------------------------------
# 상태 초기화 / 부팅
# ------------------------------------------------------------
def reset_engine_state() -> None:
    global ai_stats, ai_streak_win, ai_streak_lose, ai_recent_results
    global last_prediction, last_round_snapshot, last_ui_state_before_predict
    global last_response_payload, rl_learning_enabled, last_processed_round_id
    global _last_accept_mono, _last_accept_winner, _last_accept_payload_ref
    global _recent_request_ids, _recent_request_id_set

    _set_new_shoe_id()

    road.big_road = []
    road.big_road_matrix = []
    road.big_road_positions = []
    road.big_eye_seq = []
    road.small_road_seq = []
    road.cockroach_seq = []
    road.big_eye_matrix = []
    road.small_road_matrix = []
    road.cockroach_matrix = []

    pattern.pattern_score_history = []

    ai_stats = {"total": 0, "correct": 0}
    ai_streak_win = 0
    ai_streak_lose = 0
    ai_recent_results = []

    last_prediction = None
    last_round_snapshot = None
    last_ui_state_before_predict = None
    last_response_payload = None

    last_processed_round_id = 0
    rl_learning_enabled = False  # ✅ 학습 비활성 고정

    _recent_request_ids = deque(maxlen=200)
    _recent_request_id_set = set()
    _last_accept_mono = 0.0
    _last_accept_winner = None
    _last_accept_payload_ref = None

    road_leader.reset_leader_state()

    app.logger.info("[엔진] RESET – 새 shoe_id=%s", current_shoe_id)


def _bootstrap_on_startup() -> None:
    _ensure_shoe_id_loaded()

    if os.path.exists(STATE_FILE_PATH):
        try:
            load_engine_state(strict_ready=False)
            _sync_round_id_from_big_road()
            app.logger.info("[엔진] load_engine_state() OK (state restored)")
        except Exception:
            app.logger.exception("[엔진] load_engine_state() FAILED → reset_engine_state()")
            reset_engine_state()
            _sync_round_id_from_big_road()
    else:
        app.logger.info("[엔진] engine_state.json 없음 → reset_engine_state()")
        reset_engine_state()
        _sync_round_id_from_big_road()


# ------------------------------------------------------------
# 모드 디버깅용 유틸 (STRICT)
# ------------------------------------------------------------
def build_mode_debug_reason(
    prev_mode: Optional[str],
    next_mode: Optional[str],
    feat: Dict[str, Any],
    future: Dict[str, Any],
) -> Optional[str]:
    if not next_mode:
        return None

    feat = _require_dict(feat, "feat")
    future = _require_dict(future, "future_scenarios")

    # STRICT: 필수 키는 반드시 존재해야 한다(누락 시 예외)
    pattern_score = float(_require(feat, "pattern_score"))
    flow_strength = float(_require(feat, "flow_strength"))
    chaos_risk = float(_require(feat, "flow_chaos_risk"))
    reversal_signal = float(_require(feat, "pattern_reversal_signal"))

    fut_p = _require_dict(_require(future, "P"), "future_scenarios.P")
    fut_b = _require_dict(_require(future, "B"), "future_scenarios.B")

    fp_ps = float(_require(fut_p, "pattern_score"))
    fp_fs = float(_require(fut_p, "flow_strength"))
    fb_ps = float(_require(fut_b, "pattern_score"))
    fb_fs = float(_require(fut_b, "flow_strength"))

    parts: List[str] = []

    if prev_mode is None:
        parts.append(f"초기 모드={next_mode}")
    else:
        parts.append(f"{prev_mode} → {next_mode}")

    if pattern_score >= 70:
        parts.append(f"pattern_score 높음({pattern_score:.1f})")
    elif pattern_score <= 40:
        parts.append(f"pattern_score 낮음({pattern_score:.1f})")
    else:
        parts.append(f"pattern_score 중간({pattern_score:.1f})")

    if flow_strength >= 0.6:
        parts.append(f"flow_strength 강함({flow_strength:.2f})")
    elif flow_strength <= 0.3:
        parts.append(f"flow_strength 약함({flow_strength:.2f})")
    else:
        parts.append(f"flow_strength 보통({flow_strength:.2f})")

    if chaos_risk >= 0.6:
        parts.append(f"chaos_risk 높음({chaos_risk:.2f})")
    elif chaos_risk <= 0.2:
        parts.append(f"chaos_risk 낮음({chaos_risk:.2f})")
    else:
        parts.append(f"chaos_risk 보통({chaos_risk:.2f})")

    if reversal_signal >= 0.7:
        parts.append(f"reversal_signal 강함({reversal_signal:.2f})")

    parts.append(f"future(P) ps={fp_ps:.1f}, fs={fp_fs:.2f}")
    parts.append(f"future(B) ps={fb_ps:.1f}, fs={fb_fs:.2f}")

    return " | ".join(parts)


# ------------------------------------------------------------
# API
# ------------------------------------------------------------
@app.route("/reset", methods=["POST"])
def reset_route():
    global IS_RESETTING
    try:
        IS_RESETTING = True
        reset_engine_state()
        _sync_round_id_from_big_road()

        payload = {
            "status": "ok",
            "shoe_id": current_shoe_id,
            "expected_round_id": _expected_round_id(),
            "next_round_id": _expected_round_id(),
        }
        return jsonify(payload)
    finally:
        IS_RESETTING = False


@app.route("/predict", methods=["POST"])
def predict_route():
    global ai_stats, ai_streak_win, ai_streak_lose, ai_recent_results
    global last_prediction, last_round_snapshot, last_ui_state_before_predict, last_response_payload
    global last_processed_round_id
    global _last_accept_mono, _last_accept_winner, _last_accept_payload_ref

    _ensure_shoe_id_loaded()

    if IS_RESETTING:
        return _json_error(409, "RESETTING", "engine is resetting now")

    data = request.get_json(silent=True)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return _json_error(400, "INVALID_JSON", "JSON body must be an object")

    winner_raw = (data.get("winner") or "").strip().upper()
    if winner_raw not in ("P", "B", "T"):
        return _json_error(400, "INVALID_WINNER", "winner must be one of P/B/T")

    received_round_id = data.get("round_id", None)

    request_id = _extract_request_id(data)
    if request_id:
        if last_response_payload and isinstance(last_response_payload, dict) and last_response_payload.get("request_id") == request_id:
            payload = copy.deepcopy(last_response_payload)
            payload["status"] = "ok"
            payload["note"] = "DUPLICATE_REQUEST_ID_IGNORED"
            payload["expected_round_id"] = _expected_round_id()
            payload["next_round_id"] = _expected_round_id()
            return jsonify(payload), 200

        if _is_request_id_processed(request_id):
            return _json_error(409, "DUPLICATE_REQUEST_ID", "request_id already processed")

    now_mono = time.monotonic()
    if (
        not request_id
        and _last_accept_payload_ref is not None
        and _last_accept_winner == winner_raw
        and (now_mono - _last_accept_mono) <= _DOUBLE_CLICK_WINDOW_SEC
    ):
        payload = copy.deepcopy(_last_accept_payload_ref)
        payload["status"] = "ok"
        payload["note"] = "DOUBLE_CLICK_IGNORED"
        payload["expected_round_id"] = _expected_round_id()
        payload["next_round_id"] = _expected_round_id()
        return jsonify(payload), 200

    last_round_snapshot = _snapshot_all_state()
    last_ui_state_before_predict = copy.deepcopy(last_response_payload)

    excel_written = False
    try:
        before_len = len(road.big_road)
        road.update_road(winner_raw)
        after_len = len(road.big_road)

        if after_len != before_len + 1:
            raise RuntimeError(f"ROADMAP_BROKEN: big_road length jump: {before_len} -> {after_len}")

        last_processed_round_id = after_len

        road_ok, road_error = road.validate_roadmap_integrity()
        if not road_ok:
            app.logger.error("[ROADMAP] integrity failed: %s", road_error)
            _restore_all_state(last_round_snapshot)
            _sync_round_id_from_big_road()
            last_round_snapshot = None
            last_ui_state_before_predict = None
            return _json_error(500, "ROADMAP_BROKEN", f"{road_error}")

        _validate_big_road_matrix_strict()

        if winner_raw == "T":
            payload = _build_not_ready_payload("TIE_INPUT (NO BET)", winner_raw)
            payload["received_round_id"] = received_round_id
            payload["request_id"] = request_id

            save_engine_state()

            last_response_payload = payload
            _last_accept_payload_ref = payload
            _last_accept_mono = now_mono
            _last_accept_winner = winner_raw
            if request_id:
                _mark_request_id_processed(request_id)

            return jsonify(payload), 200

        if last_prediction and winner_raw in ("P", "B"):
            prev_side = str(last_prediction.get("bet_side") or "").upper()
            if prev_side in ("P", "B"):
                is_correct = 1 if prev_side == winner_raw else 0
                ai_stats["total"] += 1
                ai_stats["correct"] += is_correct

                if is_correct == 1:
                    ai_streak_win += 1
                    ai_streak_lose = 0
                else:
                    ai_streak_lose += 1
                    ai_streak_win = 0

                ai_recent_results.append(is_correct)
                if len(ai_recent_results) > 10:
                    ai_recent_results.pop(0)

        ok, reason = engine_state.get_trade_readiness()
        if not ok:
            payload = _build_not_ready_payload(
                f"ENGINE NOT READY (NO BET): {reason}",
                winner_raw,
            )
            payload["received_round_id"] = received_round_id
            payload["request_id"] = request_id

            save_engine_state()

            last_response_payload = payload
            _last_accept_payload_ref = payload
            _last_accept_mono = now_mono
            _last_accept_winner = winner_raw
            if request_id:
                _mark_request_id_processed(request_id)

            return jsonify(payload), 200

        try:
            pipe = _require_dict(
                run_ai_pipeline(
                    prev_round_winner=winner_raw,
                    ai_recent_results=ai_recent_results,
                    ai_streak_lose=ai_streak_lose,
                ),
                "pipe",
            )
        except pattern.PatternNotReadyError as e:
            payload = _build_not_ready_payload(f"ENGINE NOT READY (NO BET): {e}", winner_raw)
            payload["received_round_id"] = received_round_id
            payload["request_id"] = request_id

            save_engine_state()

            last_response_payload = payload
            _last_accept_payload_ref = payload
            _last_accept_mono = now_mono
            _last_accept_winner = winner_raw
            if request_id:
                _mark_request_id_processed(request_id)

            return jsonify(payload), 200

        feat = _require_dict(pipe.get("features"), "pipe.features")
        bet = _require_dict(pipe.get("bet"), "pipe.bet")
        ai_dec = _require_dict(pipe.get("ai_decision"), "pipe.ai_decision")

        ai_ok = bool(ai_dec.get("ai_ok"))
        ai_error = ai_dec.get("error")

        if not ai_ok:
            raise RuntimeError(f"AI_ANALYSIS_FAILED: {ai_error}")

        comment = _require_nonempty_str(ai_dec.get("comment"), "ai_decision.comment")

        future = _require_dict(_require(feat, "future_scenarios"), "features.future_scenarios")
        _require_dict(_require(future, "P"), "features.future_scenarios.P")
        _require_dict(_require(future, "B"), "features.future_scenarios.B")

        bet_side = str(bet.get("bet_side") or "PASS").upper()
        bet_unit = int(bet.get("bet_unit") or 0)
        bet_reason = str(bet.get("bet_reason") or "")

        if bet_side not in ("P", "B", "PASS"):
            raise ValueError(f"BET_CONTRACT_VIOLATION: bet_side={bet_side}")

        bet_side_display = {"P": "PLAYER", "B": "BANKER", "PASS": "PASS"}[bet_side]

        gpt_raw = pipe.get("gpt_raw") or {}
        if gpt_raw is not None and not isinstance(gpt_raw, dict):
            raise TypeError("pipe.gpt_raw must be dict")

        risk_tags: List[str] = []
        key_features: List[str] = []

        if isinstance(gpt_raw, dict):
            rt = gpt_raw.get("risk_tags")
            kf = gpt_raw.get("key_features")

            if rt is not None:
                if not isinstance(rt, list) or not all(isinstance(x, str) for x in rt):
                    raise TypeError("gpt_raw.risk_tags must be list[str]")
                risk_tags = [x.strip() for x in rt if isinstance(x, str) and x.strip()]

            if kf is not None:
                if not isinstance(kf, list) or not all(isinstance(x, str) for x in kf):
                    raise TypeError("gpt_raw.key_features must be list[str]")
                key_features = [x.strip() for x in kf if isinstance(x, str) and x.strip()]

        analysis = comment
        if risk_tags:
            analysis += "\n\n[Risk Tags]\n" + "\n".join(f"- {t}" for t in risk_tags)

        prev_mode = (last_response_payload or {}).get("enforced_mode") if isinstance(last_response_payload, dict) else None
        next_mode = feat.get("enforced_mode")
        mode_changed = bool(prev_mode != next_mode) if next_mode else False
        mode_change_reason = None
        if mode_changed:
            mode_change_reason = build_mode_debug_reason(prev_mode, next_mode, feat, future)

        rounds_total = int(_require(feat, "rounds_total"))
        pb_stats = _compute_pb_stats(road.big_road)
        streak_info = _compute_streak_info(road.big_road)
        big_tie_matrix = road.build_big_road_tie_matrix()
        big_pb = _pb_seq_no_ties(road.big_road)

        ai_total = int(ai_stats.get("total") or 0)
        ai_correct = int(ai_stats.get("correct") or 0)
        ai_win_rate_val = (ai_correct / ai_total) if ai_total > 0 else 0.0
        ai_win_rate_pct = int(round(ai_win_rate_val * 100))

        last_prediction = {
            "bet_side": bet_side,
            "bet_unit": bet_unit,
            "bet_reason": bet_reason,
            "analysis": analysis,
            "enforced_mode": next_mode,
        }

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "shoe_id": current_shoe_id,
            "round_number": rounds_total,
            "winner": winner_raw,
            "bet_side": bet_side,
            "bet_unit": bet_unit,
            "bet_reason": bet_reason,
            "analysis": analysis,
            "ai_ok": ai_ok,
            "ai_error": ai_error,
            "ai_engine": "rule+gpt_analysis",
            "enforced_mode": next_mode,
            "strategy_mode": feat.get("strategy_mode"),
            "strategy_comment": feat.get("strategy_comment"),
            "risk_tags": ",".join(risk_tags),
            "key_features": "|".join(key_features),
            "pattern_score": feat.get("pattern_score"),
            "pattern_reversal_signal": feat.get("pattern_reversal_signal"),
            "flow_strength": feat.get("flow_strength"),
            "flow_chaos_risk": feat.get("flow_chaos_risk"),
            "flow_direction": feat.get("flow_direction"),
            "leader_road": feat.get("leader_road"),
            "leader_signal": feat.get("leader_signal"),
            "leader_confidence": feat.get("leader_confidence"),
            "road_hit_rates": feat.get("road_hit_rates"),
            "ai_total": ai_total,
            "ai_correct": ai_correct,
            "ai_win_rate": ai_win_rate_val,
            "ai_win_rate_pct": ai_win_rate_pct,
            "ai_streak_win": ai_streak_win,
            "ai_streak_lose": ai_streak_lose,
            "adaptive_chaos_limit": feat.get("adaptive_chaos_limit"),
            "reverse_bet_applied": feat.get("reverse_bet_applied"),
            "reverse_bet_original_side": feat.get("reverse_bet_original_side"),
            "future_P_pattern_score": future["P"].get("pattern_score"),
            "future_B_pattern_score": future["B"].get("pattern_score"),
            "future_P_flow_strength": future["P"].get("flow_strength"),
            "future_B_flow_strength": future["B"].get("flow_strength"),
        }

        append_round_log_to_excel(row, current_shoe_id)
        excel_written = True

        response_payload: Dict[str, Any] = {
            "status": "ok",
            "winner": winner_raw,
            "bet_side": bet_side,
            "bet_unit": bet_unit,
            "bet_reason": bet_reason,
            "bet_side_display": bet_side_display,
            "analysis": analysis,
            "ai_ok": ai_ok,
            "ai_error": ai_error,
            "ai_engine": "rule+gpt_analysis",
            "enforced_mode": next_mode,
            "strategy_mode": feat.get("strategy_mode"),
            "strategy_comment": feat.get("strategy_comment"),
            "risk_tags": risk_tags,
            "key_features": key_features,
            "rounds_total": rounds_total,
            "shoe_id": current_shoe_id,
            "expected_round_id": _expected_round_id(),
            "next_round_id": _expected_round_id(),
            "received_round_id": received_round_id,
            "request_id": request_id,
            "ui_ready": bool(_ui_ready_flag()),
            "ui_ready_reason": _ui_ready_reason(),
            "bead_seq": big_pb,
            "big_road": big_pb,
            "big_road_matrix": road.big_road_matrix,
            "big_eye": road.big_eye_seq,
            "small_road": road.small_road_seq,
            "cockroach": road.cockroach_seq,
            "big_eye_matrix": road.big_eye_matrix,
            "small_road_matrix": road.small_road_matrix,
            "cockroach_matrix": road.cockroach_matrix,
            "big_tie_matrix": big_tie_matrix,
            "big": big_pb,
            "china": {
                "big_eye": road.big_eye_seq,
                "small_road": road.small_road_seq,
                "cockroach": road.cockroach_seq,
            },
            "player_wins": pb_stats["P"],
            "banker_wins": pb_stats["B"],
            "ties": pb_stats["T"],
            "current_streak_type": streak_info["type"],
            "current_streak_count": streak_info["count"],
            "ai_total": ai_total,
            "ai_correct": ai_correct,
            "ai_win_rate": ai_win_rate_val,
            "ai_win_rate_pct": ai_win_rate_pct,
            "ai_streak_win": ai_streak_win,
            "ai_streak_lose": ai_streak_lose,
            "ai_recent_results": ai_recent_results,
            "rl_learning_enabled": rl_learning_enabled,
            "rl_reward": pipe.get("rl_reward"),
            "mode_changed": mode_changed,
            "mode_change_reason": mode_change_reason,
            "future_scenarios": future,
            "engine_version": features.ENGINE_VERSION,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "features_raw": feat,
            "not_ready": False,
            "not_ready_reason": None,
        }

        for k, v in feat.items():
            if k not in response_payload:
                response_payload[k] = v

        save_engine_state()

        last_response_payload = response_payload
        _last_accept_payload_ref = response_payload
        _last_accept_mono = now_mono
        _last_accept_winner = winner_raw
        if request_id:
            _mark_request_id_processed(request_id)

        return jsonify(response_payload)

    except Exception as e:
        app.logger.exception("[PREDICT] failed: %s", e)

        if last_round_snapshot:
            try:
                if excel_written and last_round_snapshot.get("current_shoe_id"):
                    remove_last_round_log_for_shoe(str(last_round_snapshot["current_shoe_id"]))
            except Exception:
                app.logger.exception("[PREDICT] failed to rollback excel last row")

            try:
                _restore_all_state(last_round_snapshot)
                _sync_round_id_from_big_road()
            except Exception:
                app.logger.exception("[PREDICT] failed to rollback state snapshot (FATAL)")

        last_round_snapshot = None
        last_ui_state_before_predict = None

        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/undo", methods=["POST"])
def undo_last_round():
    global last_round_snapshot, last_ui_state_before_predict, last_response_payload
    global last_processed_round_id
    global _last_accept_mono, _last_accept_winner, _last_accept_payload_ref

    if not last_round_snapshot:
        return _json_error(409, "NO_SNAPSHOT_TO_UNDO")

    try:
        snap_shoe = last_round_snapshot.get("current_shoe_id") or current_shoe_id
        if snap_shoe:
            remove_last_round_log_for_shoe(str(snap_shoe))

        _restore_all_state(last_round_snapshot)
        _sync_round_id_from_big_road()

        if last_ui_state_before_predict and isinstance(last_ui_state_before_predict, dict):
            last_response_payload = last_ui_state_before_predict
        else:
            big_pb = _pb_seq_no_ties(road.big_road)

            last_response_payload = {
                "status": "ok",
                "note": "UNDO_TO_EMPTY_STATE",
                "shoe_id": current_shoe_id,
                "rounds_total": len(road.big_road),
                "bead_seq": big_pb,
                "big_road": big_pb,
                "big": big_pb,
                "big_road_matrix": road.big_road_matrix,
                "big_eye": road.big_eye_seq,
                "small_road": road.small_road_seq,
                "cockroach": road.cockroach_seq,
                "big_eye_matrix": road.big_eye_matrix,
                "small_road_matrix": road.small_road_matrix,
                "cockroach_matrix": road.cockroach_matrix,
                "big_tie_matrix": road.build_big_road_tie_matrix(),
                "analysis": "UNDO_TO_EMPTY_STATE",
                "ai_ok": False,
                "ai_error": "UNDO_TO_EMPTY_STATE",
                "ui_ready": bool(_ui_ready_flag()),
                "ui_ready_reason": _ui_ready_reason(),
                "engine_version": features.ENGINE_VERSION,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "expected_round_id": _expected_round_id(),
                "next_round_id": _expected_round_id(),
            }

        save_engine_state()

        _last_accept_payload_ref = last_response_payload
        _last_accept_mono = time.monotonic()
        _last_accept_winner = None

        last_round_snapshot = None
        last_ui_state_before_predict = None

        return jsonify(last_response_payload)

    except Exception:
        app.logger.exception("[UNDO] failed")
        return jsonify({"status": "error", "error": "UNDO_FAILED"}), 500


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return render_template("index.html")


_bootstrap_on_startup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)