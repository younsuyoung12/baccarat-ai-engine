# -*- coding: utf-8 -*-
"""
train_ml_model.py

Baccarat Predictor AI Engine v11.x – 위험도·안정성 “기준값(cutline) 생성기” (Excel 기반)

역할
------
- 이 스크립트는 더 이상 LightGBM/분류 학습을 하지 않는다.
- logs/excel/*.xlsx 로그를 집계하여, 실전에서 사용할:
  1) chaos_index 기반 임계값(chaos_cut_high / chaos_cut_low)
  2) pattern_stability / flow_stability 최소 유지 임계값
  3) 구간별 실패율 테이블(failure_rate_by_zone)
  을 산출한다.
- 산출 결과는 JSON 1개로 저장되며, ml_model.py가 읽어 위험도/안정성 판단에 사용한다.

[추가 기능] (2026-01-23)
------------------------
- 엑셀 로그를 추가 분석하여 “줄(Line) 상태 전이 기반 행동 통계”를 생성한다.
- 결과는 models/line_transition_table.json 로 저장된다.
- 이 통계는 "FOLLOW(줄대로)" vs "REVERSE(줄 반대로)"가
  어떤 줄 상태에서 더 덜 깨지는지(실패율 낮은지)만 계산한다.
- 방향 예측 / 확률 예측 / 추천 문구 생성은 절대 하지 않는다.

변경 요약 (2026-01-23)
------------------------
1) line_transition_table 키를 2축("{LINE_TYPE}:{LINE_PHASE}")에서
   5축("{LINE_TYPE}:{LINE_PHASE}:{DIR_STATE}:{RUN_PARITY}:{FLIP_BUCKET}")으로 확장.
2) DIR_STATE / RUN_PARITY / FLIP_BUCKET을 추가 계산하여 키 다양성을 확대(30개+ 생성 목표).
3) risk_calibration 산출 로직은 절대 변경하지 않음(기존 결과 보존).

절대 금지
---------
- P/B/T 방향 예측 모델 학습
- 확률(%) 예측 또는 반환
- 추천/우세/맞출 수 있다는 표현
- recommend.py 또는 룰 엔진 침범

입력 데이터(가정)
----------------
- logs/excel/baccarat_*.xlsx

risk_calibration 산출에 필요한 failure 계산 우선순위(기존 유지):
  1) is_correct / correct 계열 컬럼(1/0, True/False)
  2) pnl / profit / ev 계열 컬럼(음수면 실패)
  3) bet_side + winner (P/B 비교, bet_side가 P/B인 경우만)

line_transition_table 산출은 "실제 결과 기반"으로 별도 계산:
  - bet_side(P/B) 와 winner(P/B)만으로 성공/실패를 결정한다.
  - is_correct 는 신뢰하지 않는다. (행 이동/펜딩 해석 문제 가능)

출력 JSON 형식(고정)
--------------------
1) models/risk_calibration.json (기존 유지)

2) models/line_transition_table.json (확장)
{
  "STREAK:GROW:CONTINUE:ODD:LOW": {
    "follow_failure_rate": 0.28,
    "reverse_failure_rate": 0.47,
    "samples": 312,
    "preferred_action": "FOLLOW"
  },
  "ALT_CYCLE:ANY:BREAK:ODD:HIGH": {
    "follow_failure_rate": 0.61,
    "reverse_failure_rate": 0.34,
    "samples": 189,
    "preferred_action": "REVERSE"
  }
}
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# 설정
# ----------------------------
EXCEL_GLOB = "logs/excel/baccarat_*.xlsx"

OUT_DIR = "models"
OUT_JSON = os.path.join(OUT_DIR, "risk_calibration.json")

# [추가] 줄 전이 통계 출력
OUT_LINE_JSON = os.path.join(OUT_DIR, "line_transition_table.json")

# 통계 신뢰 하한
MIN_SAMPLES_REQUIRED = 80

# 실패율 “급증/개선”으로 판정할 최소 격차(전체 실패율 대비)
MIN_LIFT_FOR_HIGH = 0.12   # chaos_cut_high
MIN_LIFT_FOR_LOW = 0.08    # chaos_cut_low / stability_min

# [추가] 줄 전이 통계 최소 표본(키별)
MIN_LINE_SAMPLES_REQUIRED = 20

# [추가] 줄 상태 판단에 필요한 최소 과거 길이
MIN_SEQ_LEN_FOR_LINE_STATE = 4


# ----------------------------
# 유틸
# ----------------------------
def _to_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _norm_pb(x: Any) -> str:
    s = str(x).strip().upper()
    if s in ("P", "PLAYER"):
        return "P"
    if s in ("B", "BANKER"):
        return "B"
    return ""


def _opp_pb(s: str) -> str:
    if s == "P":
        return "B"
    if s == "B":
        return "P"
    return ""


def _find_first_present(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _compute_failure_from_row(row: pd.Series, cols: List[str]) -> Optional[int]:
    """
    [risk_calibration 전용] failure 라벨 계산 (1=실패, 0=성공).
    우선순위:
      1) is_correct/correct 계열
      2) pnl/profit/ev 계열
      3) bet_side + winner 비교

    ⚠ 이 로직은 기존 risk_calibration 생성과의 호환을 위해 "그대로 유지"한다.
    """
    # 1) correct 계열
    correct_col = _find_first_present(
        cols,
        ["is_correct", "correct", "ai_correct", "result_correct", "win", "is_win"],
    )
    if correct_col is not None:
        v = row.get(correct_col)
        if isinstance(v, (bool, np.bool_)):
            return 0 if bool(v) else 1
        vf = _to_float(v)
        if vf is not None:
            # 1이면 성공, 0이면 실패로 해석
            return 0 if vf >= 0.5 else 1
        vs = str(v).strip().lower()
        if vs in ("true", "t", "yes", "y", "1", "win", "success"):
            return 0
        if vs in ("false", "f", "no", "n", "0", "lose", "fail", "loss"):
            return 1

    # 2) pnl/profit 계열
    pnl_col = _find_first_present(cols, ["pnl", "profit", "ev", "net", "delta"])
    if pnl_col is not None:
        vf = _to_float(row.get(pnl_col))
        if vf is not None:
            return 1 if vf < 0 else 0

    # 3) bet_side + winner
    winner_col = _find_first_present(cols, ["winner", "result", "outcome"])
    bet_col = _find_first_present(cols, ["bet_side", "side", "pred_side", "ai_bet_side"])
    if winner_col is not None and bet_col is not None:
        w = _norm_pb(row.get(winner_col))
        b = _norm_pb(row.get(bet_col))
        if w and b:
            return 0 if (w == b) else 1

    return None


# ----------------------------
# cutline 계산 (기존 유지)
# ----------------------------
def _pick_cut_high_safe(
    x: np.ndarray,
    failure: np.ndarray,
    min_samples: int,
    min_lift: float,
) -> float:
    base = float(np.mean(failure))

    qs = np.linspace(0.55, 0.95, 9)
    candidates = np.unique(np.quantile(x, qs))

    for t in sorted(candidates):
        mask = x >= t
        n = int(np.sum(mask))
        if n < min_samples:
            continue
        rate = float(np.mean(failure[mask]))
        if rate >= base + min_lift:
            print(f"[INFO] chaos_cut_high(lift) = {t:.4f}")
            return float(t)

    p80 = float(np.quantile(x, 0.80))
    print("[WARN] chaos_cut_high lift 실패 → 퍼센타일(80%)로 전환:", f"{p80:.4f}")
    return p80


def _pick_cut_low_safe(
    x: np.ndarray,
    failure: np.ndarray,
    min_samples: int,
    min_lift: float,
) -> float:
    base = float(np.mean(failure))

    qs = np.linspace(0.05, 0.45, 9)
    candidates = np.unique(np.quantile(x, qs))

    best_t: Optional[float] = None
    for t in sorted(candidates):
        mask = x <= t
        n = int(np.sum(mask))
        if n < min_samples:
            continue
        rate = float(np.mean(failure[mask]))
        if rate <= base - min_lift:
            best_t = float(t)

    if best_t is not None:
        print(f"[INFO] chaos_cut_low(lift) = {best_t:.4f}")
        return best_t

    p30 = float(np.quantile(x, 0.30))
    print("[WARN] chaos_cut_low lift 실패 → 퍼센타일(30%)로 전환:", f"{p30:.4f}")
    return p30


def _pick_stability_min_safe(
    stab: np.ndarray,
    failure: np.ndarray,
    min_samples: int,
    min_lift: float,
    name: str,
    fallback_percentile: float = 0.30,
) -> float:
    if stab.size == 0:
        raise RuntimeError(f"{name} 산출 실패: stability 데이터가 없습니다(0 samples)")

    base = float(np.mean(failure))
    qs = np.linspace(0.20, 0.90, 15)
    candidates = np.unique(np.quantile(stab, qs))

    for t in sorted(candidates):
        mask = stab >= t
        n = int(np.sum(mask))
        if n < min_samples:
            continue
        rate = float(np.mean(failure[mask]))
        if rate <= base - min_lift:
            print(f"[INFO] {name}(lift) = {t:.4f}")
            return float(t)

    p = float(np.quantile(stab, fallback_percentile))
    print(f"[WARN] {name} lift 실패 → 퍼센타일({int(fallback_percentile * 100)}%)로 전환:", f"{p:.4f}")
    return p


# ----------------------------
# 데이터 로딩 (기존 risk_calibration용)
# ----------------------------
def load_excel_records(pattern: str = EXCEL_GLOB) -> Tuple[pd.DataFrame, int]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"학습할 엑셀 파일을 찾을 수 없습니다: {pattern}")

    rows: List[Dict[str, Any]] = []
    file_count = 0

    for fp in files:
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            print(f"⚠ 파일 읽기 실패: {fp} | {e}")
            continue

        if df is None or df.empty:
            continue

        file_count += 1
        cols = list(df.columns)

        chaos_col = _find_first_present(cols, ["chaos_index", "flow_chaos_risk", "global_chaos_ratio"])
        pstab_col = _find_first_present(cols, ["pattern_stability"])
        fstab_col = _find_first_present(cols, ["flow_stability"])

        if chaos_col is None:
            continue
        if pstab_col is None and fstab_col is None:
            continue

        for _, r in df.iterrows():
            chaos = _to_float(r.get(chaos_col))
            if chaos is None:
                continue

            pstab = _to_float(r.get(pstab_col)) if pstab_col is not None else None
            fstab = _to_float(r.get(fstab_col)) if fstab_col is not None else None
            if pstab is None and fstab is None:
                continue

            failure = _compute_failure_from_row(r, cols)
            if failure is None:
                continue

            rows.append(
                {
                    "chaos_index": float(chaos),
                    "pattern_stability": pstab,
                    "flow_stability": fstab,
                    "failure": int(failure),
                }
            )

    if not rows:
        raise RuntimeError("학습 가능한 레코드가 없습니다(필수 feature 또는 failure 계산 불가).")

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["chaos_index", "failure"])
    return out, file_count


# ----------------------------
# 기준값 생성 (기존 유지)
# ----------------------------
def build_calibration(df: pd.DataFrame, files: int) -> Dict[str, Any]:
    for c in ("chaos_index", "failure"):
        if c not in df.columns:
            raise RuntimeError(f"필수 컬럼 누락: {c}")

    x = df["chaos_index"].astype(float).to_numpy()
    y = df["failure"].astype(int).to_numpy()

    if len(x) < MIN_SAMPLES_REQUIRED:
        raise RuntimeError(f"표본 수 부족: total_rows={len(x)} < {MIN_SAMPLES_REQUIRED}")

    chaos_cut_high = _pick_cut_high_safe(x, y, MIN_SAMPLES_REQUIRED, MIN_LIFT_FOR_HIGH)
    chaos_cut_low = _pick_cut_low_safe(x, y, MIN_SAMPLES_REQUIRED, MIN_LIFT_FOR_LOW)

    if chaos_cut_low >= chaos_cut_high:
        raise RuntimeError("임계값 모순: chaos_cut_low >= chaos_cut_high")

    p_count = int(df["pattern_stability"].notna().sum())
    if p_count <= 0:
        raise RuntimeError("pattern_stability 데이터가 없습니다(산출 불가).")
    if p_count < MIN_SAMPLES_REQUIRED:
        print(
            "[WARN] pattern_stability 표본 부족:",
            f"{p_count} < {MIN_SAMPLES_REQUIRED}",
            "→ lift 기반 산출이 실패할 수 있으며, 실패 시 퍼센타일(30%)로 전환",
        )
    df_p = df.dropna(subset=["pattern_stability"])
    pstab = df_p["pattern_stability"].astype(float).to_numpy()
    y_p = df_p["failure"].astype(int).to_numpy()
    pattern_stability_min = _pick_stability_min_safe(
        pstab, y_p, MIN_SAMPLES_REQUIRED, MIN_LIFT_FOR_LOW, "pattern_stability_min", fallback_percentile=0.30
    )

    f_count = int(df["flow_stability"].notna().sum())
    if f_count <= 0:
        raise RuntimeError("flow_stability 데이터가 없습니다(산출 불가).")
    if f_count < MIN_SAMPLES_REQUIRED:
        print(
            "[WARN] flow_stability 표본 부족:",
            f"{f_count} < {MIN_SAMPLES_REQUIRED}",
            "→ lift 기반 산출이 실패할 수 있으며, 실패 시 퍼센타일(30%)로 전환",
        )
    df_f = df.dropna(subset=["flow_stability"])
    fstab = df_f["flow_stability"].astype(float).to_numpy()
    y_f = df_f["failure"].astype(int).to_numpy()
    flow_stability_min = _pick_stability_min_safe(
        fstab, y_f, MIN_SAMPLES_REQUIRED, MIN_LIFT_FOR_LOW, "flow_stability_min", fallback_percentile=0.30
    )

    failure_rate_by_zone: Dict[str, float] = {}
    chaos_series = df["chaos_index"].astype(float)
    zones = {
        "stable": chaos_series <= chaos_cut_low,
        "mixed": (chaos_series > chaos_cut_low) & (chaos_series < chaos_cut_high),
        "high_chaos": chaos_series >= chaos_cut_high,
    }
    for name, mask in zones.items():
        n = int(mask.sum())
        if n < MIN_SAMPLES_REQUIRED:
            continue
        rate = float(df.loc[mask, "failure"].astype(int).mean())
        failure_rate_by_zone[name] = rate

    result: Dict[str, Any] = {
        "chaos_cut_high": float(chaos_cut_high),
        "chaos_cut_low": float(chaos_cut_low),
        "pattern_stability_min": float(pattern_stability_min),
        "flow_stability_min": float(flow_stability_min),
        "failure_rate_by_zone": failure_rate_by_zone,
        "meta": {
            "total_rows": int(len(df)),
            "files": int(files),
            "source": "excel_logs",
            "min_samples_required": int(MIN_SAMPLES_REQUIRED),
        },
    }
    return result


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =============================================================================
# [추가] 줄(Line) 전이 기반 행동 통계 생성기
# =============================================================================
def _pb_clean(seq: List[str]) -> List[str]:
    out: List[str] = []
    for x in seq:
        s = _norm_pb(x)
        if s in ("P", "B"):
            out.append(s)
    return out


def _rle_runs(seq: List[str]) -> List[Tuple[str, int]]:
    if not seq:
        return []
    runs: List[Tuple[str, int]] = []
    cur = seq[0]
    n = 1
    for s in seq[1:]:
        if s == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur = s
            n = 1
    runs.append((cur, n))
    return runs


def _is_alternating(seq: List[str]) -> bool:
    if len(seq) < 2:
        return False
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            return False
    return True


def _infer_target_block_len(runs: List[Tuple[str, int]]) -> Optional[int]:
    if not runs:
        return None
    tail = runs[-4:]
    lens = [ln for _, ln in tail if isinstance(ln, int) and ln >= 2]
    if not lens:
        return None
    return int(max(lens))


def _count_flips_last6(seq: List[str]) -> int:
    """
    최근 6판 기준 P↔B 전환 횟수.
    """
    s = _pb_clean(seq)
    tail = s[-6:] if len(s) >= 6 else s[:]
    if len(tail) < 2:
        return 0
    flips = 0
    for i in range(1, len(tail)):
        if tail[i] != tail[i - 1]:
            flips += 1
    return int(flips)


def _flip_bucket(seq: List[str]) -> str:
    flips = _count_flips_last6(seq)
    if flips <= 1:
        return "LOW"
    if flips <= 3:
        return "MID"
    return "HIGH"


def _run_parity(last_len: int) -> str:
    return "EVEN" if (int(last_len) % 2 == 0) else "ODD"


def _classify_line_state(
    seq_prior: List[str],
    recent_fail_streak: int,
    prev_expected_next_side: Optional[str],
    after_break_remain: int,
    prev_line_broke: bool,
) -> Optional[Tuple[str, str, str, str, str, str]]:
    """
    seq_prior: 현재 판 결과(winner)가 나오기 전까지의 누적 P/B 시퀀스
    recent_fail_streak: 최근 2판 연속 실패(베팅 실패) 합계 (>=2면 DECAY)
    prev_expected_next_side: 직전 판에서의 expected_next_side (CONTINUE 판정용)
    after_break_remain: BREAK 이후 AFTER_BREAK 카운트다운(2→1→0)
    prev_line_broke: 직전 판에서 (expected_next_side != winner)였는지 여부

    return:
      (line_type, line_phase, dir_state, run_parity, flip_bucket, expected_next_side)
    """
    seq = _pb_clean(seq_prior)
    if len(seq) < MIN_SEQ_LEN_FOR_LINE_STATE:
        return None

    runs = _rle_runs(seq)
    last_side, last_len = runs[-1]
    decay = recent_fail_streak >= 2

    # 공통 축
    parity = _run_parity(last_len)
    fbucket = _flip_bucket(seq)

    # 1) ALT_CYCLE: 최근 4개 완전 교대
    if _is_alternating(seq[-4:]):
        line_type = "ALT_CYCLE"
        phase = "DECAY" if decay else "ANY"
        expected = _opp_pb(seq[-1])  # 교대면 다음은 반대

        # DIR_STATE
        if after_break_remain > 0:
            dir_state = "AFTER_BREAK"
        elif prev_line_broke or recent_fail_streak >= 1:
            dir_state = "BREAK"
        elif last_len in (2, 4):
            dir_state = "EDGE"
        elif prev_expected_next_side in ("P", "B") and expected == prev_expected_next_side:
            dir_state = "CONTINUE"
        else:
            dir_state = "CONTINUE"

        return line_type, phase, dir_state, parity, fbucket, expected

    # 2) BLOCK: 마지막 2 run이 둘 다 2 이상 + 교대
    if len(runs) >= 2:
        (s1, l1), (s2, l2) = runs[-2], runs[-1]
        if s1 in ("P", "B") and s2 in ("P", "B") and s1 != s2 and l1 >= 2 and l2 >= 2:
            line_type = "BLOCK"
            target = _infer_target_block_len(runs) or 2

            if l2 < target:
                expected = s2
                phase = "GROW"
            elif l2 == target:
                expected = s1
                phase = "MATURE"
            else:
                expected = s1
                phase = "DECAY"

            if decay:
                phase = "DECAY"

            # DIR_STATE
            if after_break_remain > 0:
                dir_state = "AFTER_BREAK"
            elif prev_line_broke or recent_fail_streak >= 1:
                dir_state = "BREAK"
            elif last_len in (2, 4):
                dir_state = "EDGE"
            elif prev_expected_next_side in ("P", "B") and expected == prev_expected_next_side:
                dir_state = "CONTINUE"
            else:
                dir_state = "CONTINUE"

            return line_type, phase, dir_state, parity, fbucket, expected

    # 3) STREAK: 마지막 run이 2 이상
    if last_len >= 2:
        line_type = "STREAK"
        if last_len <= 2:
            phase = "START"
        elif last_len <= 4:
            phase = "GROW"
        else:
            phase = "MATURE"

        if decay:
            phase = "DECAY"

        expected = last_side  # 스트릭이면 다음도 같은 쪽(줄 가정)

        # DIR_STATE
        if after_break_remain > 0:
            dir_state = "AFTER_BREAK"
        elif prev_line_broke or recent_fail_streak >= 1:
            dir_state = "BREAK"
        elif last_len in (2, 4):
            dir_state = "EDGE"
        elif prev_expected_next_side in ("P", "B") and expected == prev_expected_next_side:
            dir_state = "CONTINUE"
        else:
            dir_state = "CONTINUE"

        return line_type, phase, dir_state, parity, fbucket, expected

    # 4) MIXED: 그 외
    line_type = "MIXED"
    phase = "DECAY" if decay else "ANY"
    expected = last_side

    # DIR_STATE
    if after_break_remain > 0:
        dir_state = "AFTER_BREAK"
    elif prev_line_broke or recent_fail_streak >= 1:
        dir_state = "BREAK"
    elif last_len in (2, 4):
        dir_state = "EDGE"
    elif prev_expected_next_side in ("P", "B") and expected == prev_expected_next_side:
        dir_state = "CONTINUE"
    else:
        dir_state = "CONTINUE"

    return line_type, phase, dir_state, parity, fbucket, expected


def build_line_transition_table(pattern: str = EXCEL_GLOB) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    엑셀 로그에서 줄 상태별 FOLLOW/REVERSE 실패율을 집계한다.

    FOLLOW / REVERSE 정의:
    - expected_next_side(줄이 "다음에 이렇게 간다"로 가정한 방향)에 대해
      bet_side가 동일하면 FOLLOW, 반대면 REVERSE

    실패(failure) 정의(여기서는 단순/정확):
    - bet_side != winner 이면 실패(1), 같으면 성공(0)

    키 구조(확장):
    "{LINE_TYPE}:{LINE_PHASE}:{DIR_STATE}:{RUN_PARITY}:{FLIP_BUCKET}"

    return:
      - table(dict): 저장할 JSON 본문
      - stats(dict): 출력용 요약 통계
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"엑셀 파일 없음: {pattern}")

    agg = defaultdict(lambda: {"follow_total": 0, "follow_fail": 0, "reverse_total": 0, "reverse_fail": 0})

    used_rows = 0
    used_files = 0

    for fp in files:
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            print(f"⚠ line table 파일 읽기 실패: {fp} | {e}")
            continue

        if df is None or df.empty:
            continue

        cols = list(df.columns)

        ts_col = _find_first_present(cols, ["timestamp", "time", "created_at"])
        shoe_col = _find_first_present(cols, ["shoe_id", "shoe", "shoeid"])
        winner_col = _find_first_present(cols, ["winner", "result", "outcome"])
        bet_col = _find_first_present(cols, ["bet_side", "side", "pred_side", "ai_bet_side"])
        unit_col = _find_first_present(cols, ["bet_unit", "unit", "bet_amount"])

        if ts_col is None or shoe_col is None or winner_col is None or bet_col is None:
            continue

        used_files += 1

        df2 = df[[ts_col, shoe_col, winner_col, bet_col] + ([unit_col] if unit_col else [])].copy()
        df2 = df2.sort_values(ts_col)

        for _, g in df2.groupby(shoe_col):
            g = g.sort_values(ts_col)

            pb_seq: List[str] = []
            recent_fail: List[int] = []          # bet_failure stack (최근 3개)
            line_break_hist: List[int] = []      # line break stack (expected != winner) (최근 필요시만)
            prev_expected_next_side: Optional[str] = None
            after_break_remain = 0               # BREAK 이후 2판 AFTER_BREAK 카운트다운

            for _, row in g.iterrows():
                winner = _norm_pb(row.get(winner_col))
                bet = _norm_pb(row.get(bet_col))

                # unit
                unit = 1
                if unit_col is not None:
                    try:
                        unit = int(row.get(unit_col) or 0)
                    except Exception:
                        unit = 0

                # winner 무효면 아무 것도 못함
                if winner not in ("P", "B"):
                    continue

                # 최근 2판 연속 실패(베팅 실패) 기반 DECAY 판정
                recent_fail_streak = int(sum(recent_fail[-2:])) if recent_fail else 0
                prev_line_broke = bool(line_break_hist[-1] == 1) if line_break_hist else False

                st = _classify_line_state(
                    seq_prior=pb_seq,
                    recent_fail_streak=recent_fail_streak,
                    prev_expected_next_side=prev_expected_next_side,
                    after_break_remain=after_break_remain,
                    prev_line_broke=prev_line_broke,
                )

                # 이번 판 결과를 줄에 반영(항상)
                pb_seq.append(winner)

                # st가 없으면 (초반) AFTER_BREAK 카운트만 소모
                if st is None:
                    if after_break_remain > 0:
                        after_break_remain = max(after_break_remain - 1, 0)
                    continue

                line_type, phase, dir_state, run_parity, flip_bucket, expected_next_side = st

                # line break 계산(줄 가정 expected vs 실제 winner)
                line_broke_now = 1 if expected_next_side != winner else 0
                line_break_hist.append(line_broke_now)
                prev_expected_next_side = expected_next_side

                # AFTER_BREAK 카운트다운 업데이트
                if line_broke_now == 1:
                    after_break_remain = 2
                else:
                    if after_break_remain > 0:
                        after_break_remain = max(after_break_remain - 1, 0)

                # PASS/비베팅/무효 bet 은 샘플에서 제외(줄은 이미 winner로 진행됨)
                if unit <= 0 or bet not in ("P", "B"):
                    continue

                # 실패 계산(베팅 기준)
                failure = 1 if bet != winner else 0
                recent_fail.append(failure)
                if len(recent_fail) > 3:
                    recent_fail = recent_fail[-3:]

                # key + action 집계
                key = f"{line_type}:{phase}:{dir_state}:{run_parity}:{flip_bucket}"
                action = "FOLLOW" if bet == expected_next_side else "REVERSE"

                used_rows += 1

                if action == "FOLLOW":
                    agg[key]["follow_total"] += 1
                    agg[key]["follow_fail"] += failure
                else:
                    agg[key]["reverse_total"] += 1
                    agg[key]["reverse_fail"] += failure

    # 최종 테이블 계산
    table: Dict[str, Any] = {}

    for key, v in agg.items():
        ft = int(v["follow_total"])
        rt = int(v["reverse_total"])
        total = ft + rt
        if total < MIN_LINE_SAMPLES_REQUIRED:
            continue

        # failure_rate 계산 (표본 0인 쪽은 1.0으로 채움: "근거 없음 = 최악" 처리)
        ff = float(v["follow_fail"]) / ft if ft > 0 else 1.0
        rf = float(v["reverse_fail"]) / rt if rt > 0 else 1.0

        preferred_action = "FOLLOW" if ff < rf else "REVERSE"

        table[key] = {
            "follow_failure_rate": round(ff, 4),
            "reverse_failure_rate": round(rf, 4),
            "samples": int(total),
            "preferred_action": preferred_action,
        }

    stats = {
        "files_scanned": int(len(files)),
        "files_used": int(used_files),
        "rows_used": int(used_rows),
        "keys_generated": int(len(table)),
        "min_line_samples_required": int(MIN_LINE_SAMPLES_REQUIRED),
    }
    return table, stats


# ----------------------------
# 실행
# ----------------------------
def main() -> None:
    # 1) 기존 risk_calibration.json 생성(그대로)
    df, files = load_excel_records(EXCEL_GLOB)
    print(f"📊 집계 완료: rows={len(df)} files={files}")

    calib = build_calibration(df, files)
    save_json(calib, OUT_JSON)

    print("\n✅ 기준값 생성 완료")
    print(f" - 저장 경로: {OUT_JSON}")
    print(f" - chaos_cut_low : {calib['chaos_cut_low']:.4f}")
    print(f" - chaos_cut_high: {calib['chaos_cut_high']:.4f}")
    print(f" - pattern_stability_min: {calib['pattern_stability_min']:.4f}")
    print(f" - flow_stability_min   : {calib['flow_stability_min']:.4f}")
    if calib.get("failure_rate_by_zone"):
        frz = calib["failure_rate_by_zone"]
        for k in ("stable", "mixed", "high_chaos"):
            if k in frz:
                print(f" - failure_rate[{k}]: {frz[k]:.4f}")
    print("")

    # 2) [추가] line_transition_table.json 생성
    line_table, st = build_line_transition_table(EXCEL_GLOB)
    save_json(line_table, OUT_LINE_JSON)

    print("✅ 줄 전이 통계 생성 완료")
    print(f" - 저장 경로: {OUT_LINE_JSON}")
    print(f" - files_used: {st['files_used']} / scanned={st['files_scanned']}")
    print(f" - rows_used : {st['rows_used']}")
    print(f" - keys      : {st['keys_generated']}")
    print("")


if __name__ == "__main__":
    main()
