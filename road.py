# -*- coding: utf-8 -*-
# road.py
"""
Road & Shoe State for Baccarat Predictor AI Engine v12.0 (RULE-ONLY · STRICT)

역할:
- Big Road / 중국점 3종 상태 저장 (전역 상태)
- P/B/T 통계 및 스트릭 계산
- Big Road / 중국점 매트릭스 / 타이 매트릭스 생성
- 로드맵 무결성 검사
- Big Road의 "논리적 블록(run)" 기반 구조 메타 제공

핵심 정책:
- STRICT · NO-FALLBACK · FAIL-FAST
- 잘못된 winner 입력 시 즉시 예외
- TIE(T)는 P/B 구조 계산에 직접 사용하지 않음
- 단, MAX_ROAD overflow 로 오래된 P/B가 잘려나가면 파생 캐시를 즉시 재계산한다
- 파생 로드/매트릭스/run cache 불일치 시 즉시 예외

변경 요약 (2026-03-14)
----------------------------------------------------
1) TIE + overflow 정합성 수정
   - update_road(winner=="T")에서도 overflow로 오래된 P/B가 제거되면 recompute_all_roads() 수행
   - stale big_road_matrix / big_road_positions / china seq / run cache 방지
2) 무결성 검증 강화
   - big_road_positions 비교 추가
   - 중국점 matrix 3종 비교 추가
   - run_sequence / logical_column_heights / recent_structure_meta 비교 추가
3) recompute_all_roads() 캐시 계산 명확화
   - logical_column_heights 를 캐시 함수 우회 없이 직접 계산
----------------------------------------------------

기존 변경 요약 (2026-01-03)
----------------------------------------------------
1) Tie(T) 입력은 Big Road/중국점 재계산 경로에서 완전 분리
   - update_road(): winner=="T" 인 경우 big_road에만 기록 후 즉시 return
   - recompute_all_roads()는 P/B 입력에서만 수행
   - tie 표시는 build_big_road_tie_matrix() 오버레이 방식 유지
----------------------------------------------------

기존 변경 요약 (2026-01-01)
----------------------------------------------------
1) 중국점 3종(Big Eye / Small / Cockroach) 판단 계산을 카지노 기준으로 정렬
   - 드래곤테일(바닥/충돌로 우측 이동)은 "같은 논리 컬럼의 연장"으로 취급
   - 중국점 계산은 Big Road의 "논리 컬럼(=P/B 변화로만 증가)" 기준으로 수행
2) 중국점 비교 로직을 "존재 비교" 중심으로 통일
3) 시작 조건을 카지노 규칙에 맞게 "논리 컬럼" 기준으로 적용
4) Big Road 구조 메타 공개 API 추가
5) 무결성 정책 강화
----------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

MAX_ROAD = 80
BIG_ROAD_ROWS = 6
VALID_WINNERS = ("P", "B", "T")
VALID_PB = ("P", "B")
VALID_RB = ("R", "B")

# Big Road
big_road: List[str] = []                        # 'P' / 'B' / 'T'
big_road_matrix: List[List[str]] = []          # 6 x N, P/B only (row-major)
big_road_positions: List[Tuple[int, int]] = []  # 각 비타이(P/B) 결과의 (col,row) 좌표

# 중국점 3종 시퀀스 (엔진/피처 호환용)
big_eye_seq: List[str] = []       # 'R' / 'B'
small_road_seq: List[str] = []    # 'R' / 'B'
cockroach_seq: List[str] = []     # 'R' / 'B'

# 중국점 3종 보드 (표시용)
big_eye_matrix: List[List[str]] = []       # 6 x N, R/B (row-major)
small_road_matrix: List[List[str]] = []    # 6 x N, R/B (row-major)
cockroach_matrix: List[List[str]] = []     # 6 x N, R/B (row-major)

# (호환용 캐시 – 실제 계산은 get_pb_sequence() 기준)
pb_sequence: List[str] = []

# ---------------- Big Road 구조 메타(논리 run) 캐시 ----------------
run_sequence: List[Tuple[str, int]] = []                 # [(side, len), ...]
logical_column_heights: List[int] = []                  # [h1, h2, ...]
recent_structure_meta: Dict[str, Any] = {}              # structure summary


# ---------------- 내부 strict helpers ----------------
def _validate_winner_symbol(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    s = value.strip().upper()
    if s not in VALID_WINNERS:
        raise ValueError(f"{name} invalid: {value!r} (allowed: {VALID_WINNERS})")
    return s


def _validate_pb_symbol(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    s = value.strip().upper()
    if s not in VALID_PB:
        raise ValueError(f"{name} invalid: {value!r} (allowed: {VALID_PB})")
    return s


def _validate_rb_symbol(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    s = value.strip().upper()
    if s not in VALID_RB:
        raise ValueError(f"{name} invalid: {value!r} (allowed: {VALID_RB})")
    return s


def _validate_row_major_matrix(
    matrix: Any,
    *,
    name: str,
    allowed_nonempty: Tuple[str, ...],
    rows: int = BIG_ROAD_ROWS,
) -> List[List[str]]:
    if not isinstance(matrix, list):
        raise TypeError(f"{name} must be list, got {type(matrix).__name__}")

    normalized: List[List[str]] = []
    width: Optional[int] = None

    for r, row in enumerate(matrix):
        if not isinstance(row, list):
            raise TypeError(f"{name}[{r}] must be list, got {type(row).__name__}")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise RuntimeError(f"{name} row width mismatch at row={r}")

        new_row: List[str] = []
        for c, cell in enumerate(row):
            if not isinstance(cell, str):
                raise TypeError(f"{name}[{r}][{c}] must be str, got {type(cell).__name__}")
            s = cell.strip().upper()
            if s == "":
                new_row.append("")
                continue
            if s not in allowed_nonempty:
                raise RuntimeError(
                    f"{name}[{r}][{c}] invalid value: {cell!r} "
                    f"(allowed empty or {allowed_nonempty})"
                )
            new_row.append(s)
        normalized.append(new_row)

    if normalized and len(normalized) != rows:
        raise RuntimeError(f"{name} must have {rows} rows, got {len(normalized)}")

    return normalized


def _validate_big_road_positions(
    positions: Any,
    *,
    pb_len: int,
    matrix: List[List[str]],
    pb_seq: List[str],
    name: str = "big_road_positions",
) -> List[Tuple[int, int]]:
    if not isinstance(positions, list):
        raise TypeError(f"{name} must be list, got {type(positions).__name__}")
    if len(positions) != pb_len:
        raise RuntimeError(f"{name} length mismatch: {len(positions)} != pb_len({pb_len})")

    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    normalized: List[Tuple[int, int]] = []

    for i, item in enumerate(positions):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{name}[{i}] must be tuple[int,int], got {type(item).__name__}")
        col, row = item
        if not isinstance(col, int) or not isinstance(row, int):
            raise TypeError(f"{name}[{i}] must be tuple[int,int]")
        if row < 0 or row >= BIG_ROAD_ROWS:
            raise RuntimeError(f"{name}[{i}].row out of range: {row}")
        if col < 0:
            raise RuntimeError(f"{name}[{i}].col out of range: {col}")
        if rows > 0 and cols > 0:
            if col >= cols:
                raise RuntimeError(f"{name}[{i}].col exceeds matrix width: {col} >= {cols}")
            cell = matrix[row][col]
            if cell != pb_seq[i]:
                raise RuntimeError(
                    f"{name}[{i}] points to mismatched cell: "
                    f"matrix[{row}][{col}]={cell!r}, expected={pb_seq[i]!r}"
                )
        normalized.append((col, row))
    return normalized


# ---------------- PB 통계 / 엔트로피 ----------------
def get_pb_sequence() -> List[str]:
    """Tie(T)를 제거한 순수 P/B 시퀀스."""
    return [r for r in big_road if r in VALID_PB]


def compute_pb_stats() -> Dict[str, Any]:
    """P/B/T 카운트 및 Shannon 엔트로피 계산."""
    total = len(big_road)
    p_cnt = big_road.count("P")
    b_cnt = big_road.count("B")
    t_cnt = big_road.count("T")

    non_tie = max(p_cnt + b_cnt, 1)

    pb_ratio = {
        "player": p_cnt / non_tie,
        "banker": b_cnt / non_tie,
    }
    tie_ratio = t_cnt / max(total, 1) if total > 0 else 0.0

    probs: List[float] = []
    if p_cnt:
        probs.append(p_cnt / total)
    if b_cnt:
        probs.append(b_cnt / total)
    if t_cnt:
        probs.append(t_cnt / total)

    entropy = 0.0
    for p in probs:
        entropy -= p * math.log2(p)
    max_entropy = math.log2(3.0) if probs else 1.0
    norm_entropy = entropy / max_entropy if probs else 0.0

    return {
        "total_rounds": total,
        "p_count": p_cnt,
        "b_count": b_cnt,
        "t_count": t_cnt,
        "pb_ratio": pb_ratio,
        "tie_ratio": tie_ratio,
        "entropy": norm_entropy,
    }


# ---------------- run(논리 블록) 시퀀스 ----------------
def _build_run_sequence(pb_seq: List[str]) -> List[Tuple[str, int]]:
    runs: List[Tuple[str, int]] = []
    if not pb_seq:
        return runs

    cur = _validate_pb_symbol(pb_seq[0], name="pb_seq[0]")
    ln = 1

    for idx, raw in enumerate(pb_seq[1:], start=1):
        v = _validate_pb_symbol(raw, name=f"pb_seq[{idx}]")
        if v == cur:
            ln += 1
        else:
            runs.append((cur, ln))
            cur = v
            ln = 1
    runs.append((cur, ln))
    return runs


def get_run_sequence(pb_seq: Optional[List[str]] = None) -> List[Tuple[str, int]]:
    """Big Road의 논리적 블록(run) 시퀀스. [(side, length), ...]"""
    if pb_seq is None:
        pb_seq = get_pb_sequence()
    return _build_run_sequence(pb_seq)


def get_recent_runs(window: int = 6, pb_seq: Optional[List[str]] = None) -> List[Tuple[str, int]]:
    """최근 run들(기본 6개)."""
    runs = get_run_sequence(pb_seq)
    if window <= 0:
        return []
    return runs[-window:]


def get_block_lengths(window: int = 6, pb_seq: Optional[List[str]] = None) -> List[int]:
    """최근 run들 중 len>=2 인 블록 길이만."""
    recent = get_recent_runs(window=window, pb_seq=pb_seq)
    return [ln for _, ln in recent if isinstance(ln, int) and ln >= 2]


def _compute_recent_structure_meta_from_runs(runs: List[Tuple[str, int]]) -> Dict[str, Any]:
    """
    runs 기반으로 구조 메타를 산출한다.
    - pingpong / blocks / mixed / streak / random
    - 마지막 1~2개 결과를 추종하지 않으며, 최소 run 근거를 요구한다.
    """
    recent = runs[-6:] if runs else []
    last4 = runs[-4:] if len(runs) >= 4 else []
    last6 = recent

    has_pingpong = bool(len(last4) >= 4 and all(ln == 1 for _, ln in last4))
    has_blocks = bool(len(runs) >= 2 and runs[-1][1] >= 2 and runs[-2][1] >= 2 and runs[-1][0] != runs[-2][0])

    has_mixed_blocks = False
    if len(last4) >= 4:
        lens = [ln for _, ln in last4]
        has_single = any(ln == 1 for ln in lens)
        has_block = any(ln >= 2 for ln in lens)
        alternating_runs = all(last4[i][0] != last4[i - 1][0] for i in range(1, len(last4)))
        has_mixed_blocks = bool(has_single and has_block and alternating_runs)

    max_len_recent = max((ln for _, ln in last6), default=0)
    has_streak = bool(max_len_recent >= 4 and not has_pingpong and not has_mixed_blocks and not has_blocks)

    is_random = bool(not has_pingpong and not has_blocks and not has_mixed_blocks and not has_streak)

    if has_pingpong:
        structure = "pingpong"
    elif has_blocks:
        structure = "blocks"
    elif has_mixed_blocks:
        structure = "mixed"
    elif has_streak:
        structure = "streak"
    else:
        structure = "random"

    return {
        "structure": structure,
        "run_sequence": list(runs),
        "recent_runs": list(recent),
        "block_lengths": [ln for _, ln in recent if ln >= 2],
        "has_pingpong": has_pingpong,
        "has_blocks": has_blocks,
        "has_mixed_blocks": has_mixed_blocks,
        "has_streak": has_streak,
        "is_random": is_random,
    }


def get_recent_structure(pb_seq: Optional[List[str]] = None) -> str:
    """pingpong / blocks / mixed / streak / random"""
    runs = get_run_sequence(pb_seq)
    meta = _compute_recent_structure_meta_from_runs(runs)
    return str(meta.get("structure") or "random")


def get_structure_meta(pb_seq: Optional[List[str]] = None) -> Dict[str, Any]:
    """Big Road 구조 메타(외부 소비용)."""
    runs = get_run_sequence(pb_seq)
    return _compute_recent_structure_meta_from_runs(runs)


# ---------------- 스트릭 / 추세 강도 ----------------
def compute_streaks(pb_seq: Optional[List[str]] = None) -> Dict[str, Any]:
    """연속 패턴 / 스트릭 / 추세 강도 계산."""
    if pb_seq is None:
        pb_seq = get_pb_sequence()

    streaks: List[Dict[str, Any]] = []
    current_symbol: Optional[str] = None
    current_len = 0

    for idx, raw in enumerate(pb_seq):
        r = _validate_pb_symbol(raw, name=f"pb_seq[{idx}]")
        if current_symbol is None:
            current_symbol = r
            current_len = 1
        elif r == current_symbol:
            current_len += 1
        else:
            streaks.append({"who": current_symbol, "len": current_len})
            current_symbol = r
            current_len = 1

    if current_symbol is not None:
        streaks.append({"who": current_symbol, "len": current_len})

    if streaks:
        current_streak = streaks[-1]
        max_p = max((s["len"] for s in streaks if s["who"] == "P"), default=0)
        max_b = max((s["len"] for s in streaks if s["who"] == "B"), default=0)
        avg_len = sum(s["len"] for s in streaks) / len(streaks)
    else:
        current_streak = {"who": None, "len": 0}
        max_p = max_b = 0
        avg_len = 0.0

    last_20 = pb_seq[-20:] if pb_seq else []

    recent_non_tie = pb_seq[-15:] if pb_seq else []
    score = 0
    for r in recent_non_tie:
        score += 1 if r == "P" else -1
    norm = score / len(recent_non_tie) if recent_non_tie else 0.0

    player_strength = max(0, min(100, round((norm + 1) / 2 * 100)))
    banker_strength = 100 - player_strength
    momentum = abs(norm)

    return {
        "streaks": streaks,
        "current_streak": current_streak,
        "max_streak_p": max_p,
        "max_streak_b": max_b,
        "avg_streak_len": avg_len,
        "last_20": last_20,
        "trend_strength": {
            "player": player_strength,
            "banker": banker_strength,
        },
        "momentum": momentum,
    }


# ---------------- Big Road 매트릭스 ----------------
def build_big_road_structure(
    pb_seq: Optional[List[str]] = None,
    max_rows: int = BIG_ROAD_ROWS,
) -> Tuple[List[List[str]], List[Tuple[int, int]]]:
    """Big Road 좌표 매트릭스(P/B만)와 각 P/B 결과의 (col,row) 좌표 리스트 생성."""
    if pb_seq is None:
        pb_seq = get_pb_sequence()
    if not pb_seq:
        return [], []

    for idx, raw in enumerate(pb_seq):
        _validate_pb_symbol(raw, name=f"pb_seq[{idx}]")

    grid: Dict[Tuple[int, int], str] = {}
    positions: List[Tuple[int, int]] = []

    for i, r in enumerate(pb_seq):
        if i == 0:
            col, row = 0, 0
        else:
            prev_r = pb_seq[i - 1]
            prev_col, prev_row = positions[-1]
            if r == prev_r:
                new_row = prev_row + 1
                if new_row < max_rows and (prev_col, new_row) not in grid:
                    col, row = prev_col, new_row
                else:
                    col, row = prev_col + 1, prev_row
            else:
                col, row = prev_col + 1, 0

        positions.append((col, row))
        grid[(col, row)] = r

    max_col = max(c for (c, _) in grid.keys()) if grid else -1
    matrix: List[List[str]] = []
    for row in range(max_rows):
        row_vals: List[str] = []
        for col in range(max_col + 1):
            row_vals.append(grid.get((col, row), ""))
        matrix.append(row_vals)
    return matrix, positions


# ---------------- 중국점 3종: 판단(시퀀스) 계산 ----------------
def _build_logical_columns_meta(
    pb_seq: List[str],
    positions: List[Tuple[int, int]],
    max_rows: int,
) -> Tuple[List[int], List[int]]:
    if not pb_seq or not positions or len(pb_seq) != len(positions):
        return [], []

    run_ids: List[int] = []
    run_max_rows: List[int] = []

    current_run = -1
    prev_r: Optional[str] = None

    for i, (_col, row) in enumerate(positions):
        r = _validate_pb_symbol(pb_seq[i], name=f"pb_seq[{i}]")
        if i == 0 or prev_r is None or r != prev_r:
            current_run += 1
            run_max_rows.append(row)
        else:
            if row > run_max_rows[current_run]:
                run_max_rows[current_run] = row

        run_ids.append(current_run)
        prev_r = r

    run_heights: List[int] = []
    for mr in run_max_rows:
        h = mr + 1
        if h < 1:
            h = 1
        if h > max_rows:
            h = max_rows
        run_heights.append(h)

    return run_ids, run_heights


def get_logical_column_heights(pb_seq: Optional[List[str]] = None) -> List[int]:
    if pb_seq is None:
        pb_seq = get_pb_sequence()

    if not pb_seq:
        return []

    if pb_sequence == pb_seq and logical_column_heights:
        return list(logical_column_heights)

    if not big_road_positions or len(big_road_positions) != len(pb_seq):
        _m, pos = build_big_road_structure(pb_seq)
    else:
        pos = big_road_positions

    max_rows = BIG_ROAD_ROWS if not big_road_matrix else len(big_road_matrix)
    _run_ids, heights = _build_logical_columns_meta(pb_seq, pos, max_rows=max_rows)
    return list(heights)


def _logical_exists(run_heights: List[int], run_id: int, row: int) -> bool:
    if run_id < 0 or run_id >= len(run_heights):
        return False
    if row < 0:
        return False
    return row <= (run_heights[run_id] - 1)


def _compute_derived_road(
    matrix: List[List[str]],
    positions: List[Tuple[int, int]],
    pb_seq: List[str],
    n: int,
) -> List[str]:
    road_seq: List[str] = []

    if not matrix or not positions or not pb_seq:
        return road_seq

    max_rows = len(matrix) if matrix else BIG_ROAD_ROWS
    run_ids, run_heights = _build_logical_columns_meta(pb_seq, positions, max_rows=max_rows)
    if not run_ids or not run_heights:
        return road_seq

    for idx, (_col, row) in enumerate(positions):
        if idx == 0:
            continue

        run_id = run_ids[idx]

        if run_id < n:
            continue
        if run_id == n and row == 0:
            continue

        prev_run_id = run_ids[idx - 1]
        is_new_column = (run_id != prev_run_id)

        if is_new_column:
            comp_run_1 = run_id - 1
            comp_run_2 = comp_run_1 - n
            if comp_run_1 < 0 or comp_run_2 < 0:
                continue

            h2 = run_heights[comp_run_2]
            has_at_h2m1 = _logical_exists(run_heights, comp_run_1, h2 - 1)
            has_at_h2 = _logical_exists(run_heights, comp_run_1, h2)
            mark = "R" if (has_at_h2m1 and not has_at_h2) else "B"
        else:
            ref_run = run_id - n
            if ref_run < 0:
                continue

            now_filled = _logical_exists(run_heights, ref_run, row)
            above_filled = _logical_exists(run_heights, ref_run, row - 1)
            mark = "R" if now_filled == above_filled else "B"

        road_seq.append(mark)

    for idx, mark in enumerate(road_seq):
        _validate_rb_symbol(mark, name=f"derived_road[{idx}]")

    return road_seq


def compute_chinese_roads(
    matrix: List[List[str]],
    positions: List[Tuple[int, int]],
    pb_seq: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    if pb_seq is None:
        pb_seq = get_pb_sequence()
    big_eye = _compute_derived_road(matrix, positions, pb_seq, n=1)
    small_road = _compute_derived_road(matrix, positions, pb_seq, n=2)
    cockroach = _compute_derived_road(matrix, positions, pb_seq, n=3)
    return big_eye, small_road, cockroach


# ---------------- 중국점 3종: 표시용 matrix 생성 (전용 커서) ----------------
def build_china_road_matrix(
    rb_seq: List[str],
    max_rows: int = BIG_ROAD_ROWS,
) -> List[List[str]]:
    if not rb_seq:
        return []

    grid: Dict[Tuple[int, int], str] = {}
    positions: List[Tuple[int, int]] = []

    for i, raw_mark in enumerate(rb_seq):
        mark = _validate_rb_symbol(raw_mark, name=f"rb_seq[{i}]")

        if i == 0:
            col, row = 0, 0
        else:
            prev_mark = rb_seq[i - 1]
            prev_col, prev_row = positions[-1]
            if mark == prev_mark:
                new_row = prev_row + 1
                if new_row < max_rows and (prev_col, new_row) not in grid:
                    col, row = prev_col, new_row
                else:
                    col, row = prev_col + 1, prev_row
            else:
                col, row = prev_col + 1, 0

        positions.append((col, row))
        grid[(col, row)] = mark

    max_col = max(c for (c, _) in grid.keys()) if grid else -1
    matrix: List[List[str]] = []
    for row in range(max_rows):
        row_vals: List[str] = []
        for col in range(max_col + 1):
            row_vals.append(grid.get((col, row), ""))
        matrix.append(row_vals)
    return matrix


def get_last_china_marks() -> Dict[str, Optional[str]]:
    return {
        "big_eye_last": big_eye_seq[-1] if big_eye_seq else None,
        "small_road_last": small_road_seq[-1] if small_road_seq else None,
        "cockroach_last": cockroach_seq[-1] if cockroach_seq else None,
    }


# ---------------- 로드맵 무결성 검증 ----------------
def validate_roadmap_integrity() -> Tuple[bool, str]:
    try:
        # 1) big_road 원본 심볼 검증
        for idx, r in enumerate(big_road):
            _validate_winner_symbol(r, name=f"big_road[{idx}]")

        pb_seq_local = get_pb_sequence()

        # 2) P/B가 있으면 matrix / positions 가 반드시 있어야 함
        if pb_seq_local and (not big_road_matrix or not big_road_positions):
            return False, "BIG_ROAD_MATRIX_OR_POSITIONS_EMPTY"

        # 3) 기대값 재계산
        matrix2, positions2 = build_big_road_structure(pb_seq_local)
        big2, small2, cock2 = compute_chinese_roads(matrix2, positions2, pb_seq_local)

        big_eye_matrix2 = build_china_road_matrix(big2, max_rows=BIG_ROAD_ROWS) if big2 else []
        small_road_matrix2 = build_china_road_matrix(small2, max_rows=BIG_ROAD_ROWS) if small2 else []
        cockroach_matrix2 = build_china_road_matrix(cock2, max_rows=BIG_ROAD_ROWS) if cock2 else []

        expected_runs = _build_run_sequence(pb_seq_local)
        _expected_run_ids, expected_heights = _build_logical_columns_meta(pb_seq_local, positions2, max_rows=BIG_ROAD_ROWS)
        expected_meta = _compute_recent_structure_meta_from_runs(expected_runs)

        # 4) 현재 matrix / positions 검증
        current_big_matrix = _validate_row_major_matrix(
            big_road_matrix,
            name="big_road_matrix",
            allowed_nonempty=VALID_PB,
            rows=BIG_ROAD_ROWS,
        ) if big_road_matrix else []

        _validate_big_road_positions(
            big_road_positions,
            pb_len=len(pb_seq_local),
            matrix=current_big_matrix,
            pb_seq=pb_seq_local,
            name="big_road_positions",
        ) if pb_seq_local else None

        # 5) 직접 비교
        if current_big_matrix != matrix2:
            return False, "BIG_ROAD_MATRIX_MISMATCH"
        if big_road_positions != positions2:
            return False, "BIG_ROAD_POSITIONS_MISMATCH"

        if big_eye_seq != big2:
            return False, "BIG_EYE_MISMATCH"
        if small_road_seq != small2:
            return False, "SMALL_ROAD_MISMATCH"
        if cockroach_seq != cock2:
            return False, "COCKROACH_MISMATCH"

        # 6) 중국점 sequence symbol 검증
        for seq, name in (
            (big_eye_seq, "BIG_EYE"),
            (small_road_seq, "SMALL_ROAD"),
            (cockroach_seq, "COCKROACH"),
        ):
            for idx, v in enumerate(seq):
                _validate_rb_symbol(v, name=f"{name}[{idx}]")

        # 7) 중국점 matrix 검증 및 비교
        current_big_eye_matrix = _validate_row_major_matrix(
            big_eye_matrix,
            name="big_eye_matrix",
            allowed_nonempty=VALID_RB,
            rows=BIG_ROAD_ROWS,
        ) if big_eye_matrix else []

        current_small_road_matrix = _validate_row_major_matrix(
            small_road_matrix,
            name="small_road_matrix",
            allowed_nonempty=VALID_RB,
            rows=BIG_ROAD_ROWS,
        ) if small_road_matrix else []

        current_cockroach_matrix = _validate_row_major_matrix(
            cockroach_matrix,
            name="cockroach_matrix",
            allowed_nonempty=VALID_RB,
            rows=BIG_ROAD_ROWS,
        ) if cockroach_matrix else []

        if current_big_eye_matrix != big_eye_matrix2:
            return False, "BIG_EYE_MATRIX_MISMATCH"
        if current_small_road_matrix != small_road_matrix2:
            return False, "SMALL_ROAD_MATRIX_MISMATCH"
        if current_cockroach_matrix != cockroach_matrix2:
            return False, "COCKROACH_MATRIX_MISMATCH"

        # 8) run cache 비교
        if run_sequence != expected_runs:
            return False, "RUN_SEQUENCE_MISMATCH"
        if logical_column_heights != expected_heights:
            return False, "LOGICAL_COLUMN_HEIGHTS_MISMATCH"
        if recent_structure_meta != expected_meta:
            return False, "RECENT_STRUCTURE_META_MISMATCH"

        return True, ""

    except Exception as e:
        return False, f"ROADMAP_VALIDATE_EXCEPTION:{type(e).__name__}:{e}"


def enforce_roadmap_integrity() -> None:
    ok, msg = validate_roadmap_integrity()
    if not ok:
        raise RuntimeError(f"ROADMAP_INTEGRITY_FAIL:{msg}")


def recompute_all_roads() -> None:
    global big_road_matrix, big_road_positions
    global big_eye_seq, small_road_seq, cockroach_seq
    global big_eye_matrix, small_road_matrix, cockroach_matrix
    global pb_sequence
    global run_sequence, logical_column_heights, recent_structure_meta

    pb_seq_local = get_pb_sequence()
    for idx, raw in enumerate(pb_seq_local):
        _validate_pb_symbol(raw, name=f"pb_seq_local[{idx}]")

    pb_sequence = list(pb_seq_local)

    big_road_matrix, big_road_positions = build_big_road_structure(pb_seq_local)

    run_sequence = _build_run_sequence(pb_seq_local)
    _run_ids, logical_column_heights_local = _build_logical_columns_meta(
        pb_seq_local,
        big_road_positions,
        max_rows=BIG_ROAD_ROWS,
    )
    logical_column_heights = list(logical_column_heights_local)
    recent_structure_meta = _compute_recent_structure_meta_from_runs(run_sequence)

    big_eye_seq, small_road_seq, cockroach_seq = compute_chinese_roads(
        big_road_matrix, big_road_positions, pb_seq_local
    )

    big_eye_matrix = build_china_road_matrix(big_eye_seq, max_rows=BIG_ROAD_ROWS) if big_eye_seq else []
    small_road_matrix = build_china_road_matrix(small_road_seq, max_rows=BIG_ROAD_ROWS) if small_road_seq else []
    cockroach_matrix = build_china_road_matrix(cockroach_seq, max_rows=BIG_ROAD_ROWS) if cockroach_seq else []

    enforce_roadmap_integrity()


def update_road(winner: str) -> None:
    winner_norm = _validate_winner_symbol(winner, name="winner")

    # Tie는 Big Road / 중국점 재계산 경로에 직접 진입하지 않는다.
    # 단, overflow 로 오래된 P/B가 제거되면 파생 캐시를 즉시 재계산한다.
    if winner_norm == "T":
        big_road.append("T")

        removed: List[str] = []
        if len(big_road) > MAX_ROAD:
            overflow = len(big_road) - MAX_ROAD
            removed = big_road[:overflow]
            del big_road[:overflow]

        removed_pb = any(x in VALID_PB for x in removed)
        runtime_pb = get_pb_sequence()

        need_recompute = False
        if removed_pb:
            need_recompute = True
        elif runtime_pb and (not big_road_matrix or not big_road_positions):
            need_recompute = True

        if need_recompute:
            recompute_all_roads()

        return

    # P/B 입력에서 전체 재계산 수행
    big_road.append(winner_norm)

    if len(big_road) > MAX_ROAD:
        overflow = len(big_road) - MAX_ROAD
        del big_road[:overflow]

    recompute_all_roads()


def build_big_road_tie_matrix() -> List[List[int]]:
    if not big_road_matrix or not big_road_positions:
        return []

    rows = len(big_road_matrix)
    cols = len(big_road_matrix[0]) if rows > 0 else 0
    if cols == 0:
        return []

    tie_matrix: List[List[int]] = [[0 for _ in range(cols)] for _ in range(rows)]

    pb_index = -1
    for idx, raw in enumerate(big_road):
        r = _validate_winner_symbol(raw, name=f"big_road[{idx}]")
        if r in VALID_PB:
            pb_index += 1
        elif r == "T" and pb_index >= 0:
            if pb_index < len(big_road_positions):
                col, row = big_road_positions[pb_index]
                if 0 <= row < rows and 0 <= col < cols:
                    tie_matrix[row][col] += 1

    return tie_matrix


def add_round(winner: str) -> None:
    update_road(winner)


def reset_all() -> None:
    global big_road, big_road_matrix, big_road_positions
    global big_eye_seq, small_road_seq, cockroach_seq
    global big_eye_matrix, small_road_matrix, cockroach_matrix
    global pb_sequence
    global run_sequence, logical_column_heights, recent_structure_meta

    big_road = []
    big_road_matrix = []
    big_road_positions = []

    big_eye_seq = []
    small_road_seq = []
    cockroach_seq = []

    big_eye_matrix = []
    small_road_matrix = []
    cockroach_matrix = []

    pb_sequence = []

    run_sequence = []
    logical_column_heights = []
    recent_structure_meta = {}