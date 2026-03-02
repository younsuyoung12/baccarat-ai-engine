# ================================================
#   Chinese Roads Test Engine (Standalone v1)
#   BigEye / Small / Cockroach 로직 검증용
# ================================================

from typing import List, Tuple

# --------------------------
# 1) Big Road Structure
# --------------------------
def build_big_road_structure(pb_seq: List[str], max_rows=6):
    """
    Macau Standard Big Road
    """
    if not pb_seq:
        return [], []

    grid = {}
    positions = []

    for i, r in enumerate(pb_seq):
        if i == 0:
            col, row = 0, 0
        else:
            prev_r = pb_seq[i-1]
            prev_col, prev_row = positions[-1]

            if r == prev_r:
                # 아래로 떨어뜨리기
                new_row = prev_row + 1
                if new_row < max_rows and (prev_col, new_row) not in grid:
                    col, row = prev_col, new_row
                else:
                    # 아래 막힘 → 오른쪽
                    col, row = prev_col + 1, prev_row
            else:
                # 색 바뀜 → 새로운 컬럼
                col, row = prev_col + 1, 0

        positions.append((col, row))
        grid[(col, row)] = r

    # 매트릭스 생성
    max_col = max(c for c, _ in positions)
    matrix = []
    for row in range(max_rows):
        row_vals = []
        for col in range(max_col + 1):
            row_vals.append(grid.get((col, row), ""))
        matrix.append(row_vals)

    return matrix, positions


# --------------------------
# 2) 안전 접근 함수
# --------------------------
def _safe_get(matrix, row, col):
    if row < 0 or col < 0:
        return ""
    if row >= len(matrix):
        return ""
    if col >= len(matrix[row]):
        return ""
    return matrix[row][col] or ""


# --------------------------
# 3) 중국점 (표준 Macau)
# --------------------------
def compute_chinese_road(matrix, positions, n):
    """
    Macau Standard (정석) 중국점 규칙
    n = 1 (Big Eye), 2 (Small), 3 (Cockroach)
    """
    road = []

    for col, row in positions:
        # n번째 앞 열이 없으면 skip
        if col < n:
            continue
        # n번째 앞 열의 첫 칩은 기준점이라 skip
        if col == n and row == 0:
            continue

        # --------------------------
        # Row == 0 (새 컬럼 시작)
        # --------------------------
        if row == 0:
            # height 비교 규칙
            col1 = col - 1
            col2 = col - (n + 1)

            h1 = sum(1 for r in range(6)
                     if _safe_get(matrix, r, col1) != "")
            h2 = sum(1 for r in range(6)
                     if _safe_get(matrix, r, col2) != "")

            road.append("R" if h1 == h2 else "B")
            continue

        # --------------------------
        # Row > 0 (패턴 일치 비교)
        #   ✔ 색(P/B) 비교가 아니라
        #   ✔ "칸이 채워져 있는지 여부"만 비교
        # --------------------------
        ref_col = col - n
        now_filled = (_safe_get(matrix, row, ref_col) != "")
        above_filled = (_safe_get(matrix, row - 1, ref_col) != "")

        road.append("R" if now_filled == above_filled else "B")

    return road


# --------------------------
# 4) 메인 테스트 함수
# --------------------------
def test_chinese_roads(pb_seq: List[str]):
    print("입력 Big Road PB:", pb_seq)

    matrix, positions = build_big_road_structure(pb_seq)

    print("\n=== Big Road Matrix ===")
    for r in matrix:
        print(r)

    big_eye = compute_chinese_road(matrix, positions, 1)
    small = compute_chinese_road(matrix, positions, 2)
    cock  = compute_chinese_road(matrix, positions, 3)

    print("\nBig Eye :", big_eye)
    print("Small   :", small)
    print("Cockroach :", cock)

    return big_eye, small, cock


if __name__ == "__main__":
    # 테스트용 샘플 PB 입력 (수영님이 원하는 걸로 바꿔 넣어 테스트 가능)
    sample = ["P","P","B","B","B","P","P","B","P"]
    test_chinese_roads(sample)
