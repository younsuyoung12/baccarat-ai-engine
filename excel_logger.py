# -*- coding: utf-8 -*-
# excel_logger.py
"""
Excel Logger for Baccarat Predictor AI Engine v10.1 (Full Logging)

역할:
- 날짜별 Excel 파일에 모든 슈/라운드 로그를 누적 저장
  (예: logs/excel/baccarat_2025-11-30.xlsx 에 해당 날짜의 모든 슈 기록)
- 헤더(EXCEL_COLUMNS) 관리
- app.py에서 전달한 row(dict)를 EXCEL_COLUMNS 순서에 맞춰 1행으로 기록

v10.0에서 중요한 점:
- AI 승률(ai_win_rate)·총 예측(ai_total_predictions)·정답 수(ai_total_correct)는
  app.py에서 "GPT가 매 판 예측한 P/B 방향" 기준으로만 계산된 값을 그대로 저장한다.
- 이 모듈은 EV/수익률 계산을 전혀 하지 않는다. 순수 로깅 전용이다.

[2025-11-30] v10.1 UNDO-safe logging
- /undo 가 호출되면, app.py에서 remove_last_round_log_for_shoe(shoe_id)를 호출하여
  같은 날짜 엑셀/CSV 파일에서 해당 shoe_id 의 "마지막 1행"을 삭제한다.
- 실패 시에도 엔진은 계속 돌아가야 하므로, 모든 예외는 내부에서 삼키고 로그만 출력한다.

변경 요약 (2025-12-22)
----------------------------------------------------
1) 엑셀 헤더 보장 로직 개선:
   - 신규 Workbook(기본 A1=None)에서 header가 2행에 붙는 문제를 방지
   - 기존 파일에서 header가 없거나 1행이 데이터인 경우에도 header를 1행에 삽입
2) append 시 필수 메타 자동 보강:
   - timestamp 값이 없으면 ISO timestamp를 자동 주입
3) JSON 컬럼 직렬화 안정화:
   - dict/list 같은 구조만 JSON으로 덤프(이미 문자열이면 그대로 저장)
   - 덤프 실패 시 안전하게 str()로 저장
4) remove_last_round_log_for_shoe 에서도 header 보장 호출(파일 깨짐 방지)

변경 요약 (2026-01-03)
----------------------------------------------------
- 엑셀 로그에 is_correct(int) 컬럼을 추가하고, 모든 행에 0/1을 반드시 기록한다.
  - 1: 해당 판에서 AI가 사용한 방향이 결과(P/B)와 일치
  - 0: 불일치/손실/계산 불가(PASS, TIE, bet_side 없음 포함)
- 기존 prev_ai_correct 등이 존재하면 이를 우선 사용하여 is_correct로 정규화한다.
- 기존 로그 파일의 헤더가 구버전(컬럼 일부만 존재)인 경우, 1행 헤더를 “확장 갱신”한다.
  (데이터 행을 새로 밀어내는 방식으로 중복 헤더 삽입하지 않음)

주의:
- 이 모듈은 단순히 row(dict)를 받아 엑셀에 쓰기만 한다.
- 실제로 위 컬럼에 어떤 값을 넣을지는 app.py에서 row를 구성할 때 채워줘야 한다.
  (단, is_correct는 이 모듈에서 강제 계산/정규화하여 빈칸을 허용하지 않는다.)
"""

import os
import json
import csv
import time
from typing import Any, Dict, Optional, List
from datetime import datetime

try:
    from openpyxl import Workbook, load_workbook  # type: ignore
except ImportError:  # pragma: no cover
    Workbook = None
    load_workbook = None

LOG_BASE_DIR = "logs"
EXCEL_LOG_DIR = os.path.join(LOG_BASE_DIR, "excel")

os.makedirs(LOG_BASE_DIR, exist_ok=True)
os.makedirs(EXCEL_LOG_DIR, exist_ok=True)

# v9.0 엑셀 컬럼 정의
# NOTE: is_correct는 기존 컬럼을 삭제하지 않고 "추가"한다.
EXCEL_COLUMNS: List[str] = [
    # 기본 메타 정보
    "timestamp",
    "date",
    "shoe_id",
    "round_number",
    "winner",

    # 직전 AI 예측 정보
    "prev_ai_used",
    "prev_ai_winner_guess",
    "prev_ai_player_prob",
    "prev_ai_banker_prob",
    "prev_ai_tie_prob",
    "prev_ai_confidence",
    "prev_ai_correct",

    # 누적 AI 성능
    "ai_total_predictions",
    "ai_total_correct",
    "ai_win_rate",

    # 패턴 Feature
    "pattern_score",
    "pattern_type",
    "pattern_energy",
    "pattern_symmetry",
    "pattern_noise_ratio",
    "pattern_reversal_signal",
    "pattern_drift",
    "run_speed",
    "tie_volatility",
    "momentum",

    # Flow Feature
    "flow_strength",
    "flow_stability",
    "flow_chaos_risk",
    "flow_reversal_risk",
    "flow_direction",

    # 모드 정보
    "prev_mode",
    "mode",
    "mode_changed",
    "mode_change_reason",

    # 미래 시나리오 (요약)
    "future_P_pattern_score",
    "future_P_flow_strength",
    "future_B_pattern_score",
    "future_B_flow_strength",

    # 다음 판 GPT 예측
    "next_player_prob",
    "next_banker_prob",
    "next_tie_prob",
    "next_winner_guess",
    "next_confidence",
    "next_comment",

    # 중국점 마지막 값 및 ALERT
    "big_eye_last",
    "small_road_last",
    "cockroach_last",
    "road_ok",
    "road_error",
    "alert_message",

    # XAI / 베팅 결과 / RL Reward
    "ai_key_features",
    "bet_side",
    "bet_unit",
    "bet_reason",
    "rl_reward",

    # 고급 Feature (v7.5~v8.0)
    "road_sync_p",
    "road_sync_b",
    "road_sync_gap",
    "segment_type",
    "transition_flag",
    "mini_trend_p",
    "mini_trend_b",
    "china_agree_last12",
    "big_eye_height_change",
    "small_road_height_change",
    "cockroach_height_change",
    "chaos_index",
    "pattern_score_global",
    "pattern_score_last10",
    "pattern_score_last5",
    "pattern_stability",
    "big_eye_flips_last10",
    "small_road_flips_last10",
    "cockroach_flips_last10",
    "flip_cycle_pb",
    "frame_mode",
    "response_delay_score",
    "odd_run_length",
    "odd_run_spike_flag",
    "global_chaos_ratio",
    "frame_trend_delta",
    "three_rule_signal",

    # v7.6 + v8.0 추가 Feature
    "china_r_streak_be",
    "china_r_streak_sm",
    "china_r_streak_ck",
    "china_depth_be",
    "china_depth_sm",
    "china_depth_ck",
    "bottom_touch_bigroad",
    "bottom_touch_bigeye",
    "bottom_touch_small",
    "bottom_touch_cockroach",
    "decalcomania_found",
    "decalcomania_hint",
    "decalcomania_support",
    "pb_diff_score",
    "shoe_phase",
    "phase_progress",
    "chaos_end_flag",
    "tie_turbulence_rounds",
    "entry_momentum",
    "adaptive_chaos_limit",
    "reverse_bet_applied",
    "reverse_bet_original_side",

    # v9.0 Regime / Meta / 전략 모드
    "shoe_regime",
    "regime_shift_score",
    "regime_forecast_line2",
    "regime_forecast_chaos3",
    "regime_forecast_shift5",
    "strategy_mode",
    "strategy_comment",
    "meta_key",
    "meta_win_rate",
    "leader_road",
    "leader_signal",
    "leader_confidence",
    "leader_hit_rates_json",
    "leader_prediction_totals_json",

    # v9.0 풀 로드맵/Feature/미래 시나리오 JSON
    "big_road_seq",
    "big_eye_seq",
    "small_road_seq",
    "cockroach_seq",
    "big_road_matrix_json",
    "big_eye_matrix_json",
    "small_road_matrix_json",
    "cockroach_matrix_json",
    "tie_matrix_json",

    "future_P_json",
    "future_B_json",
    "future_PP_json",
    "future_PB_json",
    "future_BP_json",
    "future_BB_json",

    "features_json",

    # [2026-01-03] 학습/보정용 “성공/실패” 정규화 컬럼 (반드시 0/1)
    "is_correct",
]


def new_shoe_id() -> str:
    """새 슈 ID 생성 (로깅용 메타 정보)."""
    return datetime.now().strftime("shoe_%Y%m%d_%H%M%S")


def get_excel_path_for_date() -> str:
    """날짜별 Excel 로그 파일 경로 반환.

    예: logs/excel/baccarat_2025-11-30.xlsx
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    fname = f"baccarat_{date_str}.xlsx"
    return os.path.join(EXCEL_LOG_DIR, fname)


def _get_csv_path_for_date(excel_path: str) -> str:
    return excel_path.replace(".xlsx", ".csv")


def _is_empty_row(values: List[Any]) -> bool:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return False
    return True


def _normalize_pb(v: Any) -> str:
    """P/B만 인정. 그 외(PASS/T/빈칸)는 ''."""
    if v is None:
        return ""
    s = str(v).strip().upper()
    if s in ("P", "PLAYER"):
        return "P"
    if s in ("B", "BANKER"):
        return "B"
    return ""


def _normalize_01(v: Any) -> Optional[int]:
    """0/1 정규화. 불가하면 None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        # 숫자/문자 모두 처리
        s = str(v).strip().lower()
        if s in ("1", "true", "t", "yes", "y", "win", "success"):
            return 1
        if s in ("0", "false", "f", "no", "n", "lose", "fail", "loss"):
            return 0
        f = float(s)
        return 1 if f >= 0.5 else 0
    except Exception:
        return None


def _compute_is_correct(row: Dict[str, Any]) -> int:
    """
    is_correct 규칙(고정):
    - 이전 판에 AI가 실제로 베팅한 경우:
        bet_side == winner → 1
        bet_side != winner → 0
    - PASS 또는 bet_side가 없는 경우:
        is_correct = 0
    - TIE 결과인 경우:
        is_correct = 0

    구현 세부:
    - 이미 prev_ai_correct / ai_correct / result_correct / is_correct 등이 있으면 이를 우선 사용
    - 없으면 (prev_ai_used, prev_ai_winner_guess, winner)로 계산
    - 어떤 경우에도 None/빈칸 금지 → 반드시 0 또는 1 반환
    """
    # 1) 이미 존재하는 "정답 여부" 컬럼을 우선 사용
    for k in ("is_correct", "prev_ai_correct", "ai_correct", "result_correct", "is_win", "win", "correct"):
        if k in row:
            n01 = _normalize_01(row.get(k))
            if n01 is not None:
                return int(n01)

    # 2) 로깅 스키마 기준 계산
    #    - "해당 판" 결과(winner)와 "직전 AI의 베팅/예측(prev_ai_winner_guess)" 비교
    prev_used = row.get("prev_ai_used")
    prev_used01 = _normalize_01(prev_used)
    if prev_used01 != 1:
        return 0

    winner = _normalize_pb(row.get("winner"))
    if winner not in ("P", "B"):
        return 0

    guess = _normalize_pb(row.get("prev_ai_winner_guess"))
    if guess not in ("P", "B"):
        return 0

    return 1 if guess == winner else 0


def _ensure_excel_header(ws) -> None:
    """엑셀 시트에 헤더가 없으면 1행에 생성/삽입/확장.

    - 신규 Workbook은 보통 (A1=None) 상태로 max_row=1 이라 header가 2행에 붙기 쉬움.
      -> 1행이 비어있으면 1행을 header로 '덮어쓰기' 한다.
    - 기존 파일에서 1행이 데이터(헤더 없음)인 경우
      -> 1행 위에 header를 삽입한다.
    - 기존 파일의 헤더가 구버전(앞부분만 일치하고 뒷컬럼이 None)인 경우
      -> 1행 헤더를 EXCEL_COLUMNS로 "확장 갱신"한다(중복 헤더 삽입 금지).
    """
    n = len(EXCEL_COLUMNS)

    # 1행의 현재 값 읽기 (새 컬럼까지 포함)
    first_row = [ws.cell(row=1, column=i + 1).value for i in range(n)]

    # 1행이 완전히 빈 경우: 1행에 header를 직접 세팅
    if _is_empty_row(first_row):
        for i, col_name in enumerate(EXCEL_COLUMNS, start=1):
            ws.cell(row=1, column=i, value=col_name)
        return

    # 1행이 이미 최신 header인 경우: OK
    if first_row == EXCEL_COLUMNS:
        return

    # 구버전 헤더(앞부분은 일치, 뒷부분은 None/빈칸) → 확장 갱신
    # (is_correct 등 신규 컬럼 추가 시 중복 헤더 삽입 방지)
    # prefix_len은 "연속해서 일치하는 구간"으로 판단
    prefix_len = 0
    for i, col_name in enumerate(EXCEL_COLUMNS):
        v = first_row[i]
        if v == col_name:
            prefix_len += 1
            continue
        break

    if prefix_len >= 5:  # 최소한 기본 메타/초기 컬럼이 맞는 경우만 헤더로 인정
        # 나머지 칸이 비어있으면 "헤더 확장"으로 처리
        trailing = first_row[prefix_len:]
        trailing_empty = True
        for v in trailing:
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            trailing_empty = False
            break

        if trailing_empty:
            for i, col_name in enumerate(EXCEL_COLUMNS, start=1):
                ws.cell(row=1, column=i, value=col_name)
            return

    # 1행이 header가 아닌데 데이터가 있는 경우: header 삽입
    ws.insert_rows(1)
    for i, col_name in enumerate(EXCEL_COLUMNS, start=1):
        ws.cell(row=1, column=i, value=col_name)


def _safe_json_dump(v: Any) -> Any:
    """JSON 컬럼 안정 직렬화.
    - dict/list/tuple/set 등 구조만 JSON으로 덤프
    - 이미 문자열이면 그대로 둔다(중복 덤프 방지)
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list, tuple, set)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return v


def _row_to_list(row: Dict[str, Any]) -> List[Any]:
    """Dict → EXCEL_COLUMNS 순서의 리스트 변환."""
    json_like_cols = {
        "ai_key_features",
        "big_road_seq",
        "big_eye_seq",
        "small_road_seq",
        "cockroach_seq",
        "big_road_matrix_json",
        "big_eye_matrix_json",
        "small_road_matrix_json",
        "cockroach_matrix_json",
        "tie_matrix_json",
        "future_P_json",
        "future_B_json",
        "future_PP_json",
        "future_PB_json",
        "future_BP_json",
        "future_BB_json",
        "features_json",
        "leader_hit_rates_json",
        "leader_prediction_totals_json",
    }

    out: List[Any] = []
    for col in EXCEL_COLUMNS:
        v = row.get(col)
        if col in json_like_cols:
            v = _safe_json_dump(v)
        out.append(v)
    return out


def _append_excel_row_with_retry(
    excel_path: str,
    row_values: List[Any],
    retries: int = 3,
    delay: float = 0.3,
) -> bool:
    """엑셀 파일에 1행을 추가하되, PermissionError 발생 시 재시도."""
    for attempt in range(retries):
        try:
            if os.path.exists(excel_path):
                wb = load_workbook(excel_path)
            else:
                wb = Workbook()

            ws = wb.active
            _ensure_excel_header(ws)
            ws.append(row_values)
            wb.save(excel_path)
            return True

        except PermissionError as e:
            print(
                f"[ExcelLogger] PermissionError while saving '{excel_path}' "
                f"(attempt {attempt + 1}/{retries}): {e}",
                flush=True,
            )
            if attempt < retries - 1:
                time.sleep(delay)

        except Exception as e:
            print(
                f"[ExcelLogger] Unexpected error while saving '{excel_path}': {e}",
                flush=True,
            )
            break

    return False


def append_round_log_to_excel(row: Dict[str, Any], shoe_id: Optional[str]) -> None:
    """한 라운드 로그를 날짜별 Excel(또는 CSV)에 추가."""
    excel_path = get_excel_path_for_date()
    date_str = datetime.now().strftime("%Y-%m-%d")

    row = dict(row)

    # 로깅용 필수 필드 보강
    row.setdefault("date", date_str)
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    if not row.get("shoe_id"):
        row["shoe_id"] = shoe_id or "unknown_shoe"

    # [2026-01-03] is_correct 강제 기록(0/1, 빈칸/None 금지)
    # - 기존 prev_ai_correct 등이 있으면 그것을 우선 정규화
    # - 없으면 (prev_ai_used, prev_ai_winner_guess, winner)로 계산
    is_correct = _compute_is_correct(row)
    row["is_correct"] = int(is_correct)

    # 구버전 row 구성에서 prev_ai_correct가 비어있으면 동기화(삭제/변경 아님: 빈칸 보강)
    if _normalize_01(row.get("prev_ai_correct")) is None:
        row["prev_ai_correct"] = int(is_correct)

    row_list = _row_to_list(row)

    # 1) Excel 우선
    if Workbook is not None and load_workbook is not None:
        try:
            success = _append_excel_row_with_retry(excel_path, row_list)
        except Exception as e:
            print(
                f"[ExcelLogger] Fatal error in Excel logging for '{excel_path}': {e}",
                flush=True,
            )
            success = False

        if success:
            return

        print(
            f"[ExcelLogger] Excel logging failed for '{excel_path}'. Falling back to CSV logging.",
            flush=True,
        )

    # 2) CSV 폴백
    csv_path = _get_csv_path_for_date(excel_path)
    file_exists = os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(EXCEL_COLUMNS)
            writer.writerow(row_list)
    except Exception as e:
        print(f"[ExcelLogger] Failed to write CSV log '{csv_path}': {e}", flush=True)


def remove_last_round_log_for_shoe(shoe_id: str) -> None:
    """현재 날짜 엑셀/CSV 로그에서 주어진 shoe_id 의 마지막 1행을 삭제한다.

    - /undo 호출 시 사용된다.
    - 파일이 없거나 해당 shoe_id 행이 없으면 조용히 무시.
    - openpyxl 이 있으면 .xlsx 를, 없으면 .csv 를 대상으로 한다.
    """
    excel_path = get_excel_path_for_date()

    # 1) Excel(.xlsx) 우선 처리
    if Workbook is not None and load_workbook is not None and os.path.exists(excel_path):
        try:
            wb = load_workbook(excel_path)
            ws = wb.active

            # 파일이 깨졌거나 header가 없는 경우에도 header를 보장(확장 포함)
            _ensure_excel_header(ws)

            try:
                shoe_col_idx = EXCEL_COLUMNS.index("shoe_id") + 1  # 1-based
            except ValueError:
                shoe_col_idx = None

            target_row = None
            if shoe_col_idx is not None:
                for row_idx in range(ws.max_row, 1, -1):  # 헤더(1행) 제외
                    cell_val = ws.cell(row=row_idx, column=shoe_col_idx).value
                    if cell_val == shoe_id:
                        target_row = row_idx
                        break

            if target_row is not None:
                ws.delete_rows(target_row, 1)
                wb.save(excel_path)
                print(
                    f"[ExcelLogger] Removed last Excel row for shoe_id={shoe_id} (row {target_row})",
                    flush=True,
                )
            return

        except Exception as e:
            print(
                f"[ExcelLogger] Failed to remove last Excel row for shoe_id={shoe_id}: {e}",
                flush=True,
            )
            # 실패 시 CSV 폴백 시도

    # 2) CSV 폴백
    csv_path = _get_csv_path_for_date(excel_path)
    if not os.path.exists(csv_path):
        return

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) <= 1:
            return

        header = rows[0]
        try:
            shoe_idx = header.index("shoe_id")
        except ValueError:
            return

        target_idx = None
        for idx in range(len(rows) - 1, 0, -1):
            r = rows[idx]
            if len(r) > shoe_idx and r[shoe_idx] == shoe_id:
                target_idx = idx
                break

        if target_idx is not None:
            del rows[target_idx]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            print(
                f"[ExcelLogger] Removed last CSV row for shoe_id={shoe_id} (line {target_idx})",
                flush=True,
            )

    except Exception as e:
        print(
            f"[ExcelLogger] Failed to remove CSV row for shoe_id={shoe_id}: {e}",
            flush=True,
        )
