# -*- coding: utf-8 -*-
"""
future_simulator.py
====================================================
Baccarat Predictor AI Engine v12.0 전용
RULE-ONLY FUTURE CHINA ROADS 시뮬레이터

역할
------
- 현재 Big Road 상태(road.big_road)를 기준으로,
  가상의 다음 결과가 P/B(또는 2수 시퀀스)일 때
  Big Eye / Small Road / Cockroach 각 로드에
  어떤 점(R, B 또는 점 없음)이 추가되는지 계산한다.

- 내부 계산은 항상 road 모듈을 통해 이루어진다.
  즉,
    1) road.get_pb_sequence()            → 현재 P/B 시퀀스(Tie 제외)
    2) road.build_big_road_structure()   → Big Road 매트릭스/좌표
    3) road.compute_chinese_roads()      → 중국점 3종 시퀀스
  순서로 현재 상태를 재계산한 후,
  여기에 가상의 P/B 시퀀스를 1수씩 추가하면서
  각 수가 만든 중국점 3종의 마지막 점만 추출한다.

정책
------
STRICT · NO-FALLBACK · FAIL-FAST
- 잘못된 side / sequence / 타입 / 심볼은 즉시 예외
- 잘못된 문자를 조용히 무시하지 않는다
- merge 시 alias 중복(bigEye/small/smallRoad/cock) 허용하지 않는다
- canonical key만 유지한다:
    big_eye / small_road / cockroach

공개 API
--------
- simulate_future_for_side(side, max_rows=6)
- simulate_future_sequence(sequence, max_rows=6)
- build_future_scenarios(include_two_step=True, max_rows=6)
- merge_future_china_roads(base_future, include_two_step=True, max_rows=6)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import road

VALID_SIDE = ("P", "B")
VALID_MARK = ("R", "B")
TWO_STEP_SEQUENCES = ("PP", "PB", "BP", "BB")


# ---------------- 내부 strict helpers ----------------
def _require_dict(v: Any, name: str) -> Dict[str, Any]:
    if not isinstance(v, dict):
        raise TypeError(f"{name} must be dict, got {type(v).__name__}")
    return v


def _require_list(v: Any, name: str) -> List[Any]:
    if not isinstance(v, list):
        raise TypeError(f"{name} must be list, got {type(v).__name__}")
    return v


def _require_int(v: Any, name: str) -> int:
    if isinstance(v, bool):
        raise TypeError(f"{name} must be int, got bool")
    if not isinstance(v, int):
        raise TypeError(f"{name} must be int, got {type(v).__name__}")
    return v


def _validate_max_rows(max_rows: int) -> int:
    rows = _require_int(max_rows, "max_rows")
    if rows <= 0:
        raise ValueError(f"max_rows must be > 0, got {rows}")
    return rows


def _normalize_side(side: Any, name: str = "side") -> str:
    if not isinstance(side, str):
        raise TypeError(f"{name} must be str, got {type(side).__name__}")
    s = side.strip().upper()
    if s not in VALID_SIDE:
        raise ValueError(f"{name} must be 'P' or 'B', got {side!r}")
    return s


def _normalize_sequence(sequence: Any, name: str = "sequence") -> str:
    if not isinstance(sequence, str):
        raise TypeError(f"{name} must be str, got {type(sequence).__name__}")
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError(f"{name} must be non-empty sequence of P/B")
    for idx, ch in enumerate(seq):
        if ch not in VALID_SIDE:
            raise ValueError(f"{name}[{idx}] invalid: {ch!r} (allowed: 'P'/'B')")
    return seq


def _validate_pb_seq(pb_seq: Any, name: str) -> List[str]:
    seq = _require_list(pb_seq, name)
    out: List[str] = []
    for i, v in enumerate(seq):
        out.append(_normalize_side(v, name=f"{name}[{i}]"))
    return out


def _validate_rb_seq(rb_seq: Any, name: str) -> List[str]:
    seq = _require_list(rb_seq, name)
    out: List[str] = []
    for i, v in enumerate(seq):
        if not isinstance(v, str):
            raise TypeError(f"{name}[{i}] must be str, got {type(v).__name__}")
        s = v.strip().upper()
        if s not in VALID_MARK:
            raise ValueError(f"{name}[{i}] invalid: {v!r} (allowed: 'R'/'B')")
        out.append(s)
    return out


def _validate_big_road_positions(positions: Any, pb_len: int, name: str) -> List[Tuple[int, int]]:
    pos_list = _require_list(positions, name)
    if len(pos_list) != pb_len:
        raise RuntimeError(f"{name} length mismatch: {len(pos_list)} != pb_len({pb_len})")

    out: List[Tuple[int, int]] = []
    for i, item in enumerate(pos_list):
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"{name}[{i}] must be tuple[int,int], got {type(item).__name__}")
        col, row = item
        if not isinstance(col, int) or not isinstance(row, int):
            raise TypeError(f"{name}[{i}] must be tuple[int,int]")
        if col < 0 or row < 0:
            raise RuntimeError(f"{name}[{i}] invalid negative coordinate: {(col, row)}")
        out.append((col, row))
    return out


def _validate_sim_result_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_dict(payload, "sim_result")

    seq = payload.get("sequence")
    if not isinstance(seq, str):
        raise RuntimeError("sim_result.sequence must be str")

    side = payload.get("side")
    if side is not None and side not in VALID_SIDE:
        raise RuntimeError(f"sim_result.side invalid: {side!r}")

    for key in ("big_eye", "small_road", "cockroach"):
        v = payload.get(key)
        if v is not None and v not in VALID_MARK:
            raise RuntimeError(f"sim_result.{key} invalid: {v!r}")

    for key in ("big_road_col", "big_road_row"):
        v = payload.get(key)
        if v is not None and not isinstance(v, int):
            raise RuntimeError(f"sim_result.{key} must be int or None")

    steps = payload.get("steps")
    if not isinstance(steps, int) or steps <= 0:
        raise RuntimeError(f"sim_result.steps invalid: {steps!r}")

    return payload


def _assert_no_future_alias_keys(d: Dict[str, Any], name: str) -> None:
    """
    STRICT:
    - bigEye / smallRoad / small / cock 같은 alias 키가 들어오면 즉시 예외.
    - canonical key(big_eye/small_road/cockroach)만 허용한다.
    """
    alias_keys = ("bigEye", "smallRoad", "small", "cock")
    found = [k for k in alias_keys if k in d]
    if found:
        raise RuntimeError(f"{name} contains forbidden alias keys: {found} (canonical only)")


# ---------------- 내부 계산 ----------------
def _build_base_struct(
    max_rows: int = 6,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    현재 Big Road 상태를 기준으로
    - P/B 시퀀스
    - 중국점 3종 시퀀스
    를 계산해서 반환한다.
    """
    rows = _validate_max_rows(max_rows)

    pb_seq_raw = road.get_pb_sequence()
    pb_seq = _validate_pb_seq(pb_seq_raw, "road.get_pb_sequence()")
    if not pb_seq:
        return [], [], [], []

    big_matrix, positions = road.build_big_road_structure(pb_seq, max_rows=rows)
    _require_list(big_matrix, "road.build_big_road_structure().matrix")
    _validate_big_road_positions(
        positions,
        pb_len=len(pb_seq),
        name="road.build_big_road_structure().positions",
    )

    big_eye_seq, small_seq, cock_seq = road.compute_chinese_roads(
        big_matrix,
        positions,
        pb_seq,
    )

    be = _validate_rb_seq(big_eye_seq, "road.compute_chinese_roads().big_eye_seq")
    sm = _validate_rb_seq(small_seq, "road.compute_chinese_roads().small_seq")
    ck = _validate_rb_seq(cock_seq, "road.compute_chinese_roads().cock_seq")

    return pb_seq, be, sm, ck


def _simulate_sequence_internal(
    sequence: str,
    max_rows: int = 6,
) -> Dict[str, Any]:
    """
    현재 상태에서 `sequence` (예: "P", "B", "PP", "PB", "BP", "BB") 를
    순서대로 추가했을 때, 마지막 수가 만들어내는 중국점 3종의 점을 계산한다.
    """
    seq = _normalize_sequence(sequence, name="sequence")
    rows = _validate_max_rows(max_rows)

    pb_seq, be_seq, sm_seq, ck_seq = _build_base_struct(max_rows=rows)

    len_be = len(be_seq)
    len_sm = len(sm_seq)
    len_ck = len(ck_seq)

    work_pb = list(pb_seq)

    last_be: Optional[str] = None
    last_sm: Optional[str] = None
    last_ck: Optional[str] = None

    last_col: Optional[int] = None
    last_row: Optional[int] = None

    for ch in seq:
        work_pb.append(ch)

        sim_matrix, sim_pos = road.build_big_road_structure(
            work_pb,
            max_rows=rows,
        )
        _require_list(sim_matrix, "sim_matrix")
        _validate_big_road_positions(sim_pos, pb_len=len(work_pb), name="sim_pos")

        sim_be, sim_sm, sim_ck = road.compute_chinese_roads(
            sim_matrix,
            sim_pos,
            work_pb,
        )

        sim_be_v = _validate_rb_seq(sim_be, "sim_be")
        sim_sm_v = _validate_rb_seq(sim_sm, "sim_sm")
        sim_ck_v = _validate_rb_seq(sim_ck, "sim_ck")

        last_be = sim_be_v[-1] if len(sim_be_v) > len_be else None
        last_sm = sim_sm_v[-1] if len(sim_sm_v) > len_sm else None
        last_ck = sim_ck_v[-1] if len(sim_ck_v) > len_ck else None

        len_be = len(sim_be_v)
        len_sm = len(sim_sm_v)
        len_ck = len(sim_ck_v)

        if not sim_pos:
            raise RuntimeError("sim_pos empty after non-empty work_pb")
        last_col, last_row = sim_pos[-1]

    result = {
        "sequence": seq,
        "side": seq[-1],
        "big_eye": last_be,
        "small_road": last_sm,
        "cockroach": last_ck,
        "big_road_col": last_col,
        "big_road_row": last_row,
        "steps": len(seq),
    }
    return _validate_sim_result_payload(result)


# ---------------- 공개 API ----------------
def simulate_future_for_side(side: str, max_rows: int = 6) -> Dict[str, Any]:
    """
    1수 앞만 시뮬레이션.
    """
    s = _normalize_side(side, name="side")
    return _simulate_sequence_internal(s, max_rows=max_rows)


def simulate_future_sequence(sequence: str, max_rows: int = 6) -> Dict[str, Any]:
    """
    여러 수 앞(PP / PB / BP / BB 등)을 한 번에 시뮬레이션하고,
    마지막 수가 만들어내는 중국점 상태를 반환한다.
    """
    seq = _normalize_sequence(sequence, name="sequence")
    return _simulate_sequence_internal(seq, max_rows=max_rows)


def build_future_scenarios(
    include_two_step: bool = True,
    max_rows: int = 6,
) -> Dict[str, Dict[str, Any]]:
    """
    FUTURE CHINA ROADS 전체 시나리오 딕셔너리 생성.
    """
    if not isinstance(include_two_step, bool):
        raise TypeError(f"include_two_step must be bool, got {type(include_two_step).__name__}")
    rows = _validate_max_rows(max_rows)

    scenarios: Dict[str, Dict[str, Any]] = {
        "P": simulate_future_for_side("P", max_rows=rows),
        "B": simulate_future_for_side("B", max_rows=rows),
    }

    if include_two_step:
        for seq in TWO_STEP_SEQUENCES:
            scenarios[seq] = simulate_future_sequence(seq, max_rows=rows)

    return scenarios


def merge_future_china_roads(
    base_future: Optional[Dict[str, Dict[str, Any]]] = None,
    include_two_step: bool = True,
    max_rows: int = 6,
) -> Dict[str, Dict[str, Any]]:
    """
    features.build_feature_payload_v3() 가 이미 만들어둔 future_scenarios가 있다면,
    그 위에 중국점 3종 정보만 이 모듈 결과로 덮어쓴다.

    중요:
    - recommend._extract_future_scenarios_strict()와 충돌하지 않도록
      canonical key만 남긴다.
    - alias 키(bigEye/smallRoad/small/cock)는 금지한다.
    """
    if base_future is not None:
        _require_dict(base_future, "base_future")
        if "P" not in base_future or "B" not in base_future:
            raise RuntimeError("base_future must contain keys 'P' and 'B'")
        _require_dict(base_future["P"], "base_future.P")
        _require_dict(base_future["B"], "base_future.B")

    sim_future = build_future_scenarios(
        include_two_step=include_two_step,
        max_rows=max_rows,
    )

    if base_future is None:
        return sim_future

    merged: Dict[str, Dict[str, Any]] = {}

    for key, sim in sim_future.items():
        original_raw = base_future.get(key, {})
        original = _require_dict(original_raw, f"base_future[{key}]")
        _assert_no_future_alias_keys(original, name=f"base_future[{key}]")
        out = dict(original)

        # canonical key만 유지
        out["big_eye"] = sim["big_eye"]
        out["small_road"] = sim["small_road"]
        out["cockroach"] = sim["cockroach"]

        # Big Road 좌표
        out["big_road_col"] = sim["big_road_col"]
        out["big_road_row"] = sim["big_road_row"]

        # 메타
        out["sequence"] = sim["sequence"]
        out["side"] = sim["side"]
        out["steps"] = sim["steps"]

        merged[key] = out

    # base_future 에만 있는 기타 키 유지
    for key, val in base_future.items():
        if key in merged:
            continue
        obj = _require_dict(val, f"base_future[{key}]")
        _assert_no_future_alias_keys(obj, name=f"base_future[{key}]")
        merged[key] = dict(obj)

    return merged