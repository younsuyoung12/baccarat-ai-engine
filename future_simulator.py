# -*- coding: utf-8 -*-
"""
future_simulator.py

Baccarat Predictor AI Engine v10.x 전용
"FUTURE CHINA ROADS" (미래 Big Eye / Small Road / Cockroach) 시뮬레이터.

역할
------
- 현재 Big Road 상태(road.big_road)를 기준으로,
  가상의 다음 결과가 P/B(또는 2수 시퀀스)일 때
  Big Eye / Small Road / Cockroach Pig 각 로드에
  어떤 점(R, B 또는 점 없음)이 추가되는지 계산한다.

- 내부 계산은 항상 road 모듈을 통해 이루어진다.
  즉,
    1) road.get_pb_sequence()           → 현재 P/B 시퀀스(Tie 제외)
    2) road.build_big_road_structure() → Big Road 매트릭스/좌표
    3) road.compute_chinese_roads()    → 중국점 3종 시퀀스
  순서로 "현재 상태"를 재계산한 후,
  여기에 가상의 P/B 시퀀스를 1수씩 추가하면서
  각 수가 만든 중국점 3종의 마지막 점만 추출한다.

공개 API
--------
- simulate_future_for_side(side, max_rows=6)
    다음 판이 'P' 또는 'B'일 때의 미래 중국점 3종 결과 1개를 반환.

- simulate_future_sequence(sequence, max_rows=6)
    "PP", "PB", "BP", "BB" 등 여러 수를 가정했을 때,
    마지막 수가 만들어내는 중국점 3종 결과를 반환.

- build_future_scenarios(include_two_step=True, max_rows=6)
    "P", "B" (+ 필요 시 "PP", "PB", "BP", "BB") 전체 시나리오 딕셔너리 생성.

- merge_future_china_roads(base_future, include_two_step=True, max_rows=6)
    features.build_feature_payload_v3()에서 만들어둔 future_scenarios 위에
    이 모듈이 계산한 big_eye / small_road / cockroach 값을 덮어쓴다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import road


# ---------------- 내부 유틸 ----------------
def _build_base_struct(
    max_rows: int = 6,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    현재 Big Road 상태를 기준으로
    - P/B 시퀀스
    - 중국점 3종 시퀀스
    를 한 번에 계산해서 반환한다.

    항상
      get_pb_sequence() → build_big_road_structure() → compute_chinese_roads()
    순서로 road 모듈의 현재 상태를 재계산한다.
    """
    pb_seq = road.get_pb_sequence()
    if not pb_seq:
        # 아무것도 없는 경우: 모두 빈 구조
        return [], [], [], []

    big_matrix, positions = road.build_big_road_structure(pb_seq, max_rows=max_rows)

    # road.compute_chinese_roads 가 pb_seq 를 필요로 하도록 수정된 버전에 맞춤
    big_eye_seq, small_seq, cock_seq = road.compute_chinese_roads(
        big_matrix,
        positions,
        pb_seq,  # 현재 슈 P/B 시퀀스
    )
    return pb_seq, big_eye_seq, small_seq, cock_seq


def _simulate_sequence_internal(
    sequence: str,
    max_rows: int = 6,
) -> Dict[str, Any]:
    """
    현재 상태에서 `sequence` (예: "P", "B", "PP", "PB", "BP", "BB") 를
    순서대로 추가했을 때, "마지막 수"가 만들어내는 중국점 3종의 점을 계산한다.

    로직 개요:
    - 현재 pb_seq 기준으로 Big Road/중국점 3종을 1회 계산
    - 각 로드별 현재 길이를 기억해둔다 (len_be / len_sm / len_ck)
    - sequence 를 1글자씩 돌면서:
        * work_pb 에 side(P/B)를 1수 추가
        * road.build_big_road_structure() / road.compute_chinese_roads() 로
          Big Road 및 중국점 3종 전체를 다시 계산
        * 새 길이가 기존 길이보다 길어졌다면
          → 새로 늘어난 마지막 점이 바로 해당 수가 만든 점
          → 이것을 현재 iteration 의 "마지막 점"으로 기록
        * 길이가 늘어나지 않으면
          → 해당 수는 그 로드에 점을 생성하지 못한 것으로 보고 None
    - 최종적으로 sequence 의 "마지막 수"에서 만들어진 점만 반환한다.
    """
    # 현재 상태 기준 구조
    pb_seq, be_seq, sm_seq, ck_seq = _build_base_struct(max_rows=max_rows)

    # 현재 길이 기록
    len_be = len(be_seq)
    len_sm = len(sm_seq)
    len_ck = len(ck_seq)

    # 작업용 시퀀스 (실제 + 가상 PB 시퀀스)
    work_pb = list(pb_seq)

    # 마지막 수에서 만들어진 점 (초기값 None)
    last_be: Optional[str] = None
    last_sm: Optional[str] = None
    last_ck: Optional[str] = None

    # 시퀀스별 Big Road 좌표 (마지막 수가 놓일 좌표)
    last_col: Optional[int] = None
    last_row: Optional[int] = None

    if not sequence:
        return {
            "sequence": "",
            "side": None,
            "big_eye": None,
            "small_road": None,
            "cockroach": None,
            "big_road_col": None,
            "big_road_row": None,
            "steps": 0,
        }

    for ch in sequence:
        if ch not in ("P", "B"):
            # 잘못된 문자가 섞여 있으면 무시
            continue

        # 한 수 추가
        work_pb.append(ch)

        # 이 시점의 가상 Big Road/중국점 전체 재계산
        sim_matrix, sim_pos = road.build_big_road_structure(
            work_pb,
            max_rows=max_rows,
        )

        # 여기서도 반드시 work_pb 를 넘겨서 길이 mismatch 방지
        sim_be, sim_sm, sim_ck = road.compute_chinese_roads(
            sim_matrix,
            sim_pos,
            work_pb,
        )

        # 각 로드별로 길이가 늘어났는지 확인
        if len(sim_be) > len_be:
            last_be = sim_be[-1]
        else:
            last_be = None

        if len(sim_sm) > len_sm:
            last_sm = sim_sm[-1]
        else:
            last_sm = None

        if len(sim_ck) > len_ck:
            last_ck = sim_ck[-1]
        else:
            last_ck = None

        # 다음 루프를 위해 길이 갱신  🔧 sm_sm → sim_sm 로 수정
        len_be = len(sim_be)
        len_sm = len(sim_sm)
        len_ck = len(sim_ck)

        # Big Road 상의 좌표(마지막 수)
        if sim_pos:
            last_col, last_row = sim_pos[-1]
        else:
            last_col, last_row = None, None

    return {
        "sequence": sequence,
        "side": sequence[-1] if sequence else None,
        "big_eye": last_be,
        "small_road": last_sm,
        "cockroach": last_ck,
        "big_road_col": last_col,
        "big_road_row": last_row,
        "steps": len(sequence),
    }


# ---------------- 공개 API ----------------
def simulate_future_for_side(side: str, max_rows: int = 6) -> Dict[str, Any]:
    """
    1수 앞만 시뮬레이션.

    예:
        simulate_future_for_side("P")
        → {"sequence": "P", "side": "P", "big_eye": "R"/"B"/None, ...}

    side: 'P' 또는 'B'
    """
    if side not in ("P", "B"):
        raise ValueError(f"side must be 'P' or 'B', got {side!r}")
    return _simulate_sequence_internal(side, max_rows=max_rows)


def simulate_future_sequence(sequence: str, max_rows: int = 6) -> Dict[str, Any]:
    """
    여러 수 앞(PP / PB / BP / BB 등)을 한 번에 시뮬레이션하고,
    마지막 수가 만들어내는 중국점 상태를 반환한다.

    예:
        simulate_future_sequence("PP")
        simulate_future_sequence("PB")
    """
    return _simulate_sequence_internal(sequence, max_rows=max_rows)


def build_future_scenarios(
    include_two_step: bool = True,
    max_rows: int = 6,
) -> Dict[str, Dict[str, Any]]:
    """
    FUTURE CHINA ROADS 전체 시나리오 딕셔너리 생성.

    반환 예:
    {
        "P":  {...},   # 다음 판이 P일 때
        "B":  {...},   # 다음 판이 B일 때
        "PP": {...},   # 2수 앞 (옵션)
        "PB": {...},
        "BP": {...},
        "BB": {...},
    }

    프론트(index.html) FUTURE CHINA ROADS 패널은
    일반적으로 "P", "B" 두 개만 쓰면 된다.
    Excel / 디버깅 용도로 2수 시나리오까지 보고 싶으면 include_two_step=True 유지.
    """
    scenarios: Dict[str, Dict[str, Any]] = {}

    # 1수 앞 시나리오
    scenarios["P"] = simulate_future_for_side("P", max_rows=max_rows)
    scenarios["B"] = simulate_future_for_side("B", max_rows=max_rows)

    # 2수 앞 시나리오 (옵션)
    if include_two_step:
        for seq in ("PP", "PB", "BP", "BB"):
            scenarios[seq] = simulate_future_sequence(seq, max_rows=max_rows)

    return scenarios


def merge_future_china_roads(
    base_future: Optional[Dict[str, Dict[str, Any]]] = None,
    include_two_step: bool = True,
    max_rows: int = 6,
) -> Dict[str, Dict[str, Any]]:
    """
    features.build_feature_payload_v3() 가 이미 만들어둔 future_scenarios가 있다면,
    그 위에 "중국점 3종(R/B)" 정보만 이 모듈에서 계산해서 덮어쓴다.
    """
    sim_future = build_future_scenarios(
        include_two_step=include_two_step,
        max_rows=max_rows,
    )

    if base_future is None:
        # 그대로 반환
        return sim_future

    # base_future 를 수정하지 않으려면 여기서 deepcopy 를 할 수도 있지만,
    # 이미 feat 내부에서만 쓰이는 용도라면 얕은 복사만으로도 충분하다.
    merged: Dict[str, Dict[str, Any]] = {}
    for key, sim in sim_future.items():
        # 기존 값이 있으면 가져오고, 없으면 새 딕셔너리
        original: Dict[str, Any] = dict(base_future.get(key, {}))

        # 중국점 3종만 시뮬레이터 결과로 덮어쓴다.
        if sim.get("big_eye") in ("R", "B", None):
            original["big_eye"] = sim["big_eye"]
        if sim.get("small_road") in ("R", "B", None):
            original["small_road"] = sim["small_road"]
        if sim.get("cockroach") in ("R", "B", None):
            original["cockroach"] = sim["cockroach"]

        # Big Road 상 좌표도 필요하면 같이 기록 (Excel/디버깅 용)
        if sim.get("big_road_col") is not None:
            original["big_road_col"] = sim["big_road_col"]
        if sim.get("big_road_row") is not None:
            original["big_road_row"] = sim["big_road_row"]

        # 메타 정보
        original.setdefault("sequence", sim.get("sequence"))
        original.setdefault("side", sim.get("side"))
        original.setdefault("steps", sim.get("steps"))

        merged[key] = original

    # base_future 에만 있는 다른 키가 있다면 그대로 유지
    for key, val in base_future.items():
        if key not in merged:
            merged[key] = val

    return merged
