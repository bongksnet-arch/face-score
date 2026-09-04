import cv2
import mediapipe as mp
import numpy as np
import sys
import os
from math import hypot

INPUT = "face.jpg"
OUTPUT = "face_scored.jpg"

# MediaPipe FaceMesh 랜드마크 인덱스 (478개 중 주요 지점)
IDX = {
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "left_eyebrow_outer": 70,
    "right_eyebrow_outer": 300,
    "nose_tip": 1,
    "nose_bridge": 6,
    "chin": 152,
    "forehead": 10,
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
    "left_cheek": 234,
    "right_cheek": 454,
    "face_top": 10,
    "face_bottom": 152,
    "eye_center_line": 168,
}


def dist(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_points(landmarks, w, h):
    return {name: (landmarks[idx].x * w, landmarks[idx].y * h) for name, idx in IDX.items()}


def score_symmetry(p):
    """좌우 대칭 점수 (30점 만점)"""
    # 얼굴 중심선 = 코 다리 x좌표
    cx = p["nose_bridge"][0]
    pairs = [
        ("left_eye_outer", "right_eye_outer"),
        ("left_eye_inner", "right_eye_inner"),
        ("left_eyebrow_outer", "right_eyebrow_outer"),
        ("mouth_left", "mouth_right"),
        ("left_cheek", "right_cheek"),
    ]
    face_width = dist(p["left_cheek"], p["right_cheek"])
    diffs = []
    for L, R in pairs:
        left_dx = abs(p[L][0] - cx)
        right_dx = abs(p[R][0] - cx)
        # 얼굴 폭으로 정규화한 차이
        diff = abs(left_dx - right_dx) / face_width
        diffs.append(diff)
    avg_diff = np.mean(diffs)
    # avg_diff 0 = 완벽 대칭 → 30점, 0.05 이상 → 0점
    score = max(0, 30 * (1 - avg_diff / 0.05))
    return round(score, 1), round(avg_diff * 100, 2)


def score_golden_ratio(p):
    """세로 3등분 황금비율 (30점 만점)"""
    forehead_y = p["forehead"][1]
    eyebrow_y = (p["left_eyebrow_outer"][1] + p["right_eyebrow_outer"][1]) / 2
    nose_end_y = p["nose_tip"][1]
    chin_y = p["chin"][1]

    upper = eyebrow_y - forehead_y  # 이마
    middle = nose_end_y - eyebrow_y  # 눈~코
    lower = chin_y - nose_end_y  # 코~턱

    total = upper + middle + lower
    if total <= 0:
        return 0, "0:0:0"
    ratios = [upper / total, middle / total, lower / total]
    # 이상: 1/3 씩
    deviation = sum(abs(r - 1 / 3) for r in ratios)
    # deviation 0 = 완벽 → 30점, 0.3 이상 → 0점
    score = max(0, 30 * (1 - deviation / 0.3))
    return round(score, 1), f"{ratios[0]:.2f}:{ratios[1]:.2f}:{ratios[2]:.2f}"


def score_features(p):
    """이목구비 비율 (25점 만점)"""
    face_width = dist(p["left_cheek"], p["right_cheek"])

    # 눈 크기 (좌우 평균)
    left_eye_w = dist(p["left_eye_outer"], p["left_eye_inner"])
    right_eye_w = dist(p["right_eye_inner"], p["right_eye_outer"])
    eye_w = (left_eye_w + right_eye_w) / 2

    # 눈 사이 거리
    eye_gap = dist(p["left_eye_inner"], p["right_eye_inner"])

    # 입 크기
    mouth_w = dist(p["mouth_left"], p["mouth_right"])

    # 이상 비율 (경험적):
    # 눈 폭 = 얼굴 폭의 약 20%
    # 눈 사이 = 눈 폭 1개 정도
    # 입 폭 = 얼굴 폭의 약 38%
    eye_ratio = eye_w / face_width
    gap_ratio = eye_gap / eye_w if eye_w > 0 else 0
    mouth_ratio = mouth_w / face_width

    eye_score = max(0, 10 * (1 - abs(eye_ratio - 0.20) / 0.10))
    gap_score = max(0, 8 * (1 - abs(gap_ratio - 1.0) / 0.5))
    mouth_score = max(0, 7 * (1 - abs(mouth_ratio - 0.38) / 0.15))

    total = eye_score + gap_score + mouth_score
    return round(total, 1), {
        "eye_ratio": round(eye_ratio, 3),
        "gap_ratio": round(gap_ratio, 3),
        "mouth_ratio": round(mouth_ratio, 3),
    }


def score_face_shape(p):
    """얼굴형 (15점 만점) — 폭:길이 = 1 : 1.5 근처"""
    face_width = dist(p["left_cheek"], p["right_cheek"])
    face_length = dist(p["face_top"], p["face_bottom"])
    ratio = face_length / face_width if face_width > 0 else 0
    # 이상: 1.45~1.55
    score = max(0, 15 * (1 - abs(ratio - 1.5) / 0.3))
    return round(score, 1), round(ratio, 2)


def percentile_from_score(total):
    """총점 → 대략적 상위 % (임시, 나중에 실제 데이터로 교체)
    가정: 평균 65점, 표준편차 12점의 정규분포"""
    from math import erf, sqrt
    mean, std = 65, 12
    z = (total - mean) / std
    # 상위 %  = 1 - CDF(z)
    cdf = 0.5 * (1 + erf(z / sqrt(2)))
    top_pct = (1 - cdf) * 100
    return round(top_pct, 1)


def get_advice(sym, gr, feat, shape):
    advice = []
    if sym[0] < 20:
        advice.append(f"⚠ 좌우 비대칭 감지 (편차 {sym[1]}%). 눈썹/입꼬리 정렬 케어 추천.")
    if gr[0] < 20:
        advice.append(f"⚠ 상/중/하 비율 불균형 ({gr[1]}). 이마-헤어라인 or 턱 길이 조정으로 개선 여지.")
    if feat[1]["eye_ratio"] < 0.17:
        advice.append("💡 눈이 작은 편. 아이라인/앞트임으로 인상 개선 가능.")
    elif feat[1]["eye_ratio"] > 0.24:
        advice.append("👀 눈이 큰 편 (플러스 요소).")
    if feat[1]["mouth_ratio"] < 0.33:
        advice.append("💡 입이 작은 편. 립 확장 메이크업 or 시술로 밸런스 개선.")
    if shape[1] > 1.65:
        advice.append("💡 얼굴이 긴 편. 헤어스타일로 이마 가리기 or 턱 시술 고려.")
    elif shape[1] < 1.35:
        advice.append("💡 얼굴이 둥근/짧은 편. V라인 시술 or 브이라인 메이크업.")
    if not advice:
        advice.append("✨ 전반적으로 밸런스 좋음. 큰 개선 포인트 없음.")
    return advice


# ============ 실행 ============
if not os.path.exists(INPUT):
    print(f"[!] '{INPUT}' 파일이 없습니다.")
    sys.exit(1)

from face_analysis import imread_with_exif
img = imread_with_exif(INPUT)
if img is None:
    print("[!] 이미지 로드 실패.")
    sys.exit(1)

h, w = img.shape[:2]

mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
) as face_mesh:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        print("[!] 얼굴을 찾지 못했습니다.")
        sys.exit(1)

    landmarks = result.multi_face_landmarks[0].landmark
    p = get_points(landmarks, w, h)

    sym = score_symmetry(p)
    gr = score_golden_ratio(p)
    feat = score_features(p)
    shape = score_face_shape(p)

    total = sym[0] + gr[0] + feat[0] + shape[0]
    top_pct = percentile_from_score(total)

    # 결과 이미지에 주요 지점 표시
    radius = max(3, int(min(w, h) / 300))
    for name, pt in p.items():
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, (0, 255, 0), -1)
    # 중심선
    cv2.line(img, (int(p["forehead"][0]), int(p["forehead"][1])),
             (int(p["chin"][0]), int(p["chin"][1])), (0, 200, 255), 2)

    cv2.imwrite(OUTPUT, img)

# ============ 리포트 출력 ============
print("\n" + "=" * 50)
print(f"  얼굴 점수 리포트")
print("=" * 50)
print(f"  대칭성      : {sym[0]:5.1f} / 30   (편차 {sym[1]}%)")
print(f"  황금비율    : {gr[0]:5.1f} / 30   (상:중:하 = {gr[1]})")
print(f"  이목구비    : {feat[0]:5.1f} / 25")
print(f"     - 눈 크기 비율 : {feat[1]['eye_ratio']}")
print(f"     - 눈 간격 비율 : {feat[1]['gap_ratio']}")
print(f"     - 입 크기 비율 : {feat[1]['mouth_ratio']}")
print(f"  얼굴형      : {shape[0]:5.1f} / 15   (세로/가로 = {shape[1]})")
print("-" * 50)
print(f"  총점        : {total:5.1f} / 100")
print(f"  상위 백분위 : 약 {top_pct}%  (임시 기준, 추후 실제 데이터로 교체)")
print("=" * 50)
print("\n  📌 개선 조언:")
for a in get_advice(sym, gr, feat, shape):
    print(f"     {a}")
print(f"\n[✓] 시각화 이미지 저장: {OUTPUT}\n")
