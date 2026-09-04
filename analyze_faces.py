"""faces/ 폴더의 모든 얼굴 이미지 → 점수 계산 → 분포 통계
결과: face_scores.csv + 콘솔 리포트 (평균/표준편차/백분위 테이블)
"""
import os
import glob
import cv2
import mediapipe as mp
import numpy as np
from math import hypot
from face_analysis import imread_with_exif

FACES_DIR = "faces"
OUT_CSV = "face_scores.csv"

IDX = {
    "left_eye_outer": 33, "left_eye_inner": 133,
    "right_eye_inner": 362, "right_eye_outer": 263,
    "left_eyebrow_outer": 70, "right_eyebrow_outer": 300,
    "nose_tip": 1, "nose_bridge": 6,
    "chin": 152, "forehead": 10,
    "mouth_left": 61, "mouth_right": 291,
    "left_cheek": 234, "right_cheek": 454,
    "face_top": 10, "face_bottom": 152,
}


def dist(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_points(landmarks, w, h):
    return {name: (landmarks[idx].x * w, landmarks[idx].y * h) for name, idx in IDX.items()}


def score_symmetry(p):
    cx = p["nose_bridge"][0]
    pairs = [("left_eye_outer", "right_eye_outer"), ("left_eye_inner", "right_eye_inner"),
             ("left_eyebrow_outer", "right_eyebrow_outer"), ("mouth_left", "mouth_right"),
             ("left_cheek", "right_cheek")]
    face_width = dist(p["left_cheek"], p["right_cheek"])
    diffs = [abs(abs(p[L][0] - cx) - abs(p[R][0] - cx)) / face_width for L, R in pairs]
    avg = np.mean(diffs)
    return max(0, 30 * (1 - avg / 0.05))


def score_golden(p):
    forehead_y = p["forehead"][1]
    eyebrow_y = (p["left_eyebrow_outer"][1] + p["right_eyebrow_outer"][1]) / 2
    nose_end_y = p["nose_tip"][1]
    chin_y = p["chin"][1]
    upper, middle, lower = eyebrow_y - forehead_y, nose_end_y - eyebrow_y, chin_y - nose_end_y
    total = upper + middle + lower
    if total <= 0:
        return 0
    ratios = [upper / total, middle / total, lower / total]
    deviation = sum(abs(r - 1 / 3) for r in ratios)
    return max(0, 30 * (1 - deviation / 0.3))


def score_features(p):
    face_width = dist(p["left_cheek"], p["right_cheek"])
    le = dist(p["left_eye_outer"], p["left_eye_inner"])
    re = dist(p["right_eye_inner"], p["right_eye_outer"])
    eye_w = (le + re) / 2
    eye_gap = dist(p["left_eye_inner"], p["right_eye_inner"])
    mouth_w = dist(p["mouth_left"], p["mouth_right"])
    er = eye_w / face_width
    gr = eye_gap / eye_w if eye_w > 0 else 0
    mr = mouth_w / face_width
    es = max(0, 10 * (1 - abs(er - 0.20) / 0.10))
    gs = max(0, 8 * (1 - abs(gr - 1.0) / 0.5))
    ms = max(0, 7 * (1 - abs(mr - 0.38) / 0.15))
    return es + gs + ms


def score_shape(p):
    fw = dist(p["left_cheek"], p["right_cheek"])
    fl = dist(p["face_top"], p["face_bottom"])
    ratio = fl / fw if fw > 0 else 0
    return max(0, 15 * (1 - abs(ratio - 1.5) / 0.3))


# ============ 실행 ============
mp_fm = mp.solutions.face_mesh
face_mesh = mp_fm.FaceMesh(
    static_image_mode=True, max_num_faces=1,
    refine_landmarks=True, min_detection_confidence=0.5,
)

files = sorted(glob.glob(f"{FACES_DIR}/*.jpg"))
print(f"[i] {len(files)}장 발견. 분석 시작...")

results = []
fail = 0
for i, fp in enumerate(files):
    img = imread_with_exif(fp)
    if img is None:
        fail += 1
        continue
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = face_mesh.process(rgb)
    if not r.multi_face_landmarks:
        fail += 1
        continue
    p = get_points(r.multi_face_landmarks[0].landmark, w, h)
    sym = score_symmetry(p)
    gr = score_golden(p)
    feat = score_features(p)
    shape = score_shape(p)
    total = sym + gr + feat + shape
    results.append({
        "file": os.path.basename(fp),
        "sym": round(sym, 2), "golden": round(gr, 2),
        "features": round(feat, 2), "shape": round(shape, 2),
        "total": round(total, 2),
    })
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(files)} 처리")

face_mesh.close()

if not results:
    print("[!] 유효한 얼굴 없음")
    exit(1)

# CSV 저장
with open(OUT_CSV, "w", encoding="utf-8") as f:
    f.write("file,sym,golden,features,shape,total\n")
    for r in results:
        f.write(f"{r['file']},{r['sym']},{r['golden']},{r['features']},{r['shape']},{r['total']}\n")

# 통계 계산
totals = np.array([r["total"] for r in results])
mean = totals.mean()
std = totals.std()
percentiles = {p: np.percentile(totals, p) for p in [10, 25, 50, 75, 90, 95, 99]}

print("\n" + "=" * 60)
print(f"  분석 완료: 성공 {len(results)}장 / 실패 {fail}장")
print("=" * 60)
print(f"  평균 (mean)        : {mean:.2f}")
print(f"  표준편차 (std)     : {std:.2f}")
print(f"  최소               : {totals.min():.2f}")
print(f"  최대               : {totals.max():.2f}")
print("-" * 60)
print("  백분위 (percentile) — 이 점수보다 낮은 사람이 X%")
for pct, val in percentiles.items():
    print(f"     {pct:3d}% 지점         : {val:.2f}")
print("=" * 60)
print(f"\n  → app.py 의 percentile_from_score() 함수를 아래 값으로 교체 필요:")
print(f"     mean = {mean:.2f}")
print(f"     std  = {std:.2f}")
print(f"\n[i] 상세 데이터: {OUT_CSV}")
