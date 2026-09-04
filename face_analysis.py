"""FaceScore 순수 분석 로직 (UI 무관).

app.py (Streamlit) 와 analyze_dataset.py (대량 통계) 양쪽이 재사용.
env var 잠금이 tensorflow import 전에 걸려야 하므로 최상단에서 설정.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import io
import hashlib
import cv2
import mediapipe as mp
import numpy as np
from math import hypot, erf, sqrt
from PIL import Image, ImageOps

cv2.setNumThreads(1)
np.random.seed(0)


# ============ 상수 ============
IDX_GROUPS = {
    "left_eye_outer":  [33, 130, 25, 110, 24],
    "left_eye_inner":  [133, 173, 155, 154, 153],
    "right_eye_inner": [362, 398, 382, 381, 380],
    "right_eye_outer": [263, 359, 255, 339, 254],
    "left_eyebrow_outer":  [70, 63, 105, 46, 53],
    "right_eyebrow_outer": [300, 293, 334, 276, 283],
    "glabella":     [9, 8, 168, 6, 197],
    "nose_tip":     [1, 4, 5, 195, 197],
    "nose_bridge":  [6, 197, 195, 5, 4],
    "subnasale":    [2, 94, 164, 326, 97, 98, 327, 328],
    "chin":         [152, 175, 199, 200, 18, 148, 377, 176, 400, 421],
    "forehead":     [10, 151, 108, 337, 109, 338, 67, 297, 69, 299],
    "mouth_left":   [61, 91, 146, 76, 62],
    "mouth_right":  [291, 321, 375, 306, 292],
    "mouth_top":    [13, 12, 0, 11, 302],
    "mouth_bottom": [14, 15, 17, 16, 87],
    "left_cheek":   [234, 227, 137, 93, 132, 138, 215],
    "right_cheek":  [454, 447, 366, 323, 361, 367, 435],
    "face_top":     [10, 151, 108, 337, 109, 338],
    "face_bottom":  [152, 175, 199, 200, 18, 148, 377],
}

SYMMETRY_PAIRS = [
    (33, 263), (133, 362), (7, 249), (163, 466), (144, 373), (145, 374),
    (153, 380), (154, 381), (155, 382), (173, 398),
    (161, 388), (160, 387), (159, 386), (158, 385), (157, 384),
    (70, 300), (63, 293), (105, 334), (66, 296), (107, 336),
    (46, 276), (53, 283), (52, 282), (65, 295), (55, 285),
    (234, 454), (127, 356), (162, 389), (21, 251), (54, 284),
    (103, 332), (67, 297), (109, 338), (93, 323), (132, 361),
    (58, 288), (172, 397), (136, 365), (150, 379), (149, 378),
    (176, 400), (148, 377),
    (61, 291), (146, 375), (91, 321), (181, 405), (84, 314),
    (185, 409), (40, 270), (39, 269), (37, 267),
    (78, 308), (95, 324), (88, 318), (178, 402), (87, 317),
    (191, 415), (80, 310), (81, 311), (82, 312),
    (48, 278), (219, 439), (218, 438), (44, 274), (45, 275),
    (49, 279), (131, 360), (198, 420), (209, 429),
]

MIDLINE_INDICES = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164,
                   0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]

CANONICAL_LONG_SIDE = 1024

CANONICAL_3D = {
    "nose_tip":        np.array([  0.0,   0.0,   0.0]),
    "chin":            np.array([  0.0,  63.6, -12.5]),
    "left_eye_outer":  np.array([ 43.3, -32.7, -26.0]),
    "right_eye_outer": np.array([-43.3, -32.7, -26.0]),
    "mouth_left":      np.array([ 28.9,  28.9, -24.1]),
    "mouth_right":     np.array([-28.9,  28.9, -24.1]),
}
POSE_ANCHORS = list(CANONICAL_3D.keys())


# ============ 기본 유틸 ============
def dist(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def sha12(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


# ============ 랜드마크 / 정면화 ============
def _median_point_3d(landmarks, indices, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z * w] for i in indices],
                   dtype=np.float64)
    return np.median(pts, axis=0)


def get_points_3d(landmarks, w, h):
    return {name: _median_point_3d(landmarks, idxs, w, h) for name, idxs in IDX_GROUPS.items()}


def estimate_pose(pts_3d, w, h):
    img_pts = np.array([pts_3d[n][:2] for n in POSE_ANCHORS], dtype=np.float64)
    obj_pts = np.array([CANONICAL_3D[n] for n in POSE_ANCHORS], dtype=np.float64)
    focal = float(w)
    K = np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coef = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec


def euler_deg(R):
    sy = sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    else:
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
    def wrap(a):
        a = ((a + 180.0) % 360.0) - 180.0
        if abs(a) > 90.0:
            a = a - 180.0 if a > 0 else a + 180.0
        return round(a, 1)
    return wrap(yaw), wrap(pitch), wrap(roll)


def get_points(landmarks, w, h):
    pts_3d = get_points_3d(landmarks, w, h)
    R, _ = estimate_pose(pts_3d, w, h)
    frontal = {name: R.T @ pt for name, pt in pts_3d.items()}
    p = {name: (float(pt[0]), float(pt[1])) for name, pt in frontal.items()}
    p["_pose_deg"] = euler_deg(R)
    p["_pts_2d"] = {name: (float(pt[0]), float(pt[1])) for name, pt in pts_3d.items()}
    all_3d = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float64)
    p["_all_frontal"] = (R.T @ all_3d.T).T[:, :2]
    return p


# ============ 전처리 ============
def canonicalize(img_bgr):
    h, w = img_bgr.shape[:2]
    long_side = max(h, w)
    if long_side == CANONICAL_LONG_SIDE:
        return img_bgr
    scale = CANONICAL_LONG_SIDE / long_side
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)


def denoise(img_bgr):
    x = cv2.medianBlur(img_bgr, 3)
    x = cv2.bilateralFilter(x, d=9, sigmaColor=45, sigmaSpace=9)
    return x


def _rot(img, deg):
    if deg == 0: return img
    if deg == 90: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(deg)


# ============ MediaPipe 모델 (worker-safe 싱글턴) ============
_face_mesh = None
_face_det = None


def get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.7,
        )
    return _face_mesh


def get_face_detector():
    global _face_det
    if _face_det is None:
        _face_det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.3,
        )
    return _face_det


def auto_orient(img_bgr):
    det = get_face_detector()
    best = (0.0, 0, img_bgr)
    for deg in (0, 90, 180, 270):
        cand = _rot(img_bgr, deg)
        rgb = np.ascontiguousarray(cv2.cvtColor(cand, cv2.COLOR_BGR2RGB))
        res = det.process(rgb)
        if res.detections:
            conf = float(res.detections[0].score[0])
            if conf > best[0]:
                best = (conf, deg, cand)
    return best[2], best[1], best[0]


def bytes_to_bgr(img_bytes: bytes):
    """PIL(EXIF orientation 적용) 로 읽고 BGR 로 변환. EXIF 없으면 cv2 폴백."""
    try:
        pil = Image.open(io.BytesIO(img_bytes))
        pil = ImageOps.exif_transpose(pil)
        return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imread_with_exif(path: str):
    """디스크 이미지 로드 + EXIF 회전 적용. app/CLI/dataset 공통 진입점.
    실패 시 None 반환 (cv2.imread 와 동일 시맨틱)."""
    try:
        with open(path, "rb") as f:
            return bytes_to_bgr(f.read())
    except Exception:
        return cv2.imread(path)


# ============ 원시 지표 추출 헬퍼 (compute_metrics + analyze 재사용) ============
def _extract_metrics_from_points(p):
    """정면화된 랜드마크 dict `p` 에서 18개 지표 산출. None 이면 계산 불가."""
    face_width = dist(p["left_cheek"], p["right_cheek"])
    face_length = dist(p["face_top"], p["face_bottom"])
    if face_width <= 0 or face_length <= 0:
        return None

    upper = p["glabella"][1] - p["forehead"][1]
    middle = p["subnasale"][1] - p["glabella"][1]
    lower = p["chin"][1] - p["subnasale"][1]
    total_h = upper + middle + lower
    if total_h <= 0:
        return None

    all_2d = p["_all_frontal"]
    cx = float(np.median(all_2d[MIDLINE_INDICES, 0]))
    diffs = [abs(abs(all_2d[L, 0] - cx) - abs(all_2d[R, 0] - cx)) / face_width
             for L, R in SYMMETRY_PAIRS]
    diffs = np.sort(np.array(diffs))
    trim = max(1, len(diffs) // 10)
    sym_dev = float(diffs[trim:len(diffs) - trim].mean())

    eye_w = (dist(p["left_eye_outer"], p["left_eye_inner"]) +
             dist(p["right_eye_inner"], p["right_eye_outer"])) / 2
    eye_gap = dist(p["left_eye_inner"], p["right_eye_inner"])
    mouth_w = dist(p["mouth_left"], p["mouth_right"])

    def _lm(idx):
        return all_2d[idx]

    l_eye_h = abs(_lm(159)[1] - _lm(145)[1])
    r_eye_h = abs(_lm(386)[1] - _lm(374)[1])
    eye_h_avg = (l_eye_h + r_eye_h) / 2
    eye_aspect = eye_h_avg / eye_w if eye_w > 0 else float("nan")

    def _brow_arch(peak, out_end, in_end):
        peak_y = _lm(peak)[1]
        mid_y = (_lm(out_end)[1] + _lm(in_end)[1]) / 2
        return (mid_y - peak_y) / face_width
    brow_arch = (_brow_arch(105, 70, 55) + _brow_arch(334, 300, 285)) / 2

    brow_eye_gap = (abs(_lm(105)[1] - _lm(159)[1]) +
                    abs(_lm(334)[1] - _lm(386)[1])) / (2 * face_width)

    nose_w = abs(_lm(48)[0] - _lm(278)[0])
    nose_width_ratio = nose_w / face_width
    nose_length = abs(_lm(4)[1] - _lm(9)[1])
    nose_length_ratio = nose_length / face_length

    upper_lip_h = abs(_lm(13)[1] - _lm(0)[1])
    lower_lip_h = abs(_lm(17)[1] - _lm(14)[1])
    lip_ratio = upper_lip_h / lower_lip_h if lower_lip_h > 0 else float("nan")
    upper_lip_thickness = upper_lip_h / face_width
    lower_lip_thickness = lower_lip_h / face_width

    chin = _lm(152)
    jaw_l = _lm(172)
    jaw_r = _lm(397)
    v1 = np.array([jaw_l[0] - chin[0], jaw_l[1] - chin[1]])
    v2 = np.array([jaw_r[0] - chin[0], jaw_r[1] - chin[1]])
    cos_a = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    cos_a = max(-1.0, min(1.0, cos_a))
    jaw_angle = float(np.degrees(np.arccos(cos_a)))

    cheek_bone_w = abs(_lm(234)[0] - _lm(454)[0])
    cheek_ratio = cheek_bone_w / face_width
    philtrum_ratio = abs(_lm(2)[1] - _lm(0)[1]) / face_length

    return {
        "gr_upper": upper / total_h,
        "gr_middle": middle / total_h,
        "gr_lower": lower / total_h,
        "sym_dev": sym_dev,
        "eye_ratio": eye_w / face_width,
        "gap_ratio": eye_gap / eye_w if eye_w > 0 else float("nan"),
        "mouth_ratio": mouth_w / face_width,
        "shape_ratio": face_length / face_width,
        "eye_aspect": eye_aspect,
        "brow_arch": brow_arch,
        "brow_eye_gap": brow_eye_gap,
        "nose_width_ratio": nose_width_ratio,
        "nose_length_ratio": nose_length_ratio,
        "lip_ratio": lip_ratio,
        "upper_lip_thickness": upper_lip_thickness,
        "lower_lip_thickness": lower_lip_thickness,
        "jaw_angle": jaw_angle,
        "cheek_ratio": cheek_ratio,
        "philtrum_ratio": philtrum_ratio,
    }


# ============ 원시 지표 추출 (대량 분석용) ============
def compute_metrics(img_bgr, *, do_denoise=True):
    """정면화·정렬 파이프라인 후 각 지표의 raw 값 반환. 채점 X, 통계용.

    Returns dict or None(검출 실패). 다음 필드 포함:
      gr_upper/gr_middle/gr_lower : 3분할 비율 (합=1)
      sym_dev                     : 대칭 편차 (0=완벽)
      eye_ratio/gap_ratio/mouth_ratio/shape_ratio : 이목구비·얼굴형 비율
      pose_yaw/pose_pitch/pose_roll : 촬영 자세
      det_conf                    : 얼굴 검출 신뢰도
      auto_rot                    : 자동 회전 각도
    """
    img_bgr = canonicalize(img_bgr)
    img_bgr, applied_deg, det_conf = auto_orient(img_bgr)
    img_bgr = canonicalize(img_bgr)
    if do_denoise:
        img_bgr = denoise(img_bgr)
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    mesh = get_face_mesh()
    result = mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    landmarks = result.multi_face_landmarks[0].landmark
    p = get_points(landmarks, w, h)
    metrics = _extract_metrics_from_points(p)
    if metrics is None:
        return None
    yaw, pitch, roll = p["_pose_deg"]
    metrics.update({
        "pose_yaw": float(yaw),
        "pose_pitch": float(pitch),
        "pose_roll": float(roll),
        "det_conf": det_conf,
        "auto_rot": applied_deg,
    })
    return metrics


# ============ 채점 (현재 캐논 기준 — Phase 6 에서 population 기반으로 교체) ============
def score_symmetry(p):
    all_2d = p["_all_frontal"]
    cx = float(np.median(all_2d[MIDLINE_INDICES, 0]))
    face_width = dist(p["left_cheek"], p["right_cheek"])
    diffs = [abs(abs(all_2d[L, 0] - cx) - abs(all_2d[R, 0] - cx)) / face_width
             for L, R in SYMMETRY_PAIRS]
    diffs = np.sort(np.array(diffs))
    trim = max(1, len(diffs) // 10)
    avg_diff = float(diffs[trim:len(diffs) - trim].mean())
    score = max(0, 30 * (1 - avg_diff / 0.05))
    return round(score, 1), round(avg_diff * 100, 2)


def score_golden_ratio(p):
    upper = p["glabella"][1] - p["forehead"][1]
    middle = p["subnasale"][1] - p["glabella"][1]
    lower = p["chin"][1] - p["subnasale"][1]
    total = upper + middle + lower
    if total <= 0:
        return 0, "0:0:0"
    ratios = [upper / total, middle / total, lower / total]
    deviation = sum(abs(r - 1 / 3) for r in ratios)
    score = max(0, 30 * (1 - deviation / 0.3))
    return round(score, 1), f"{ratios[0]:.2f} : {ratios[1]:.2f} : {ratios[2]:.2f}"


def score_features(p):
    face_width = dist(p["left_cheek"], p["right_cheek"])
    eye_w = (dist(p["left_eye_outer"], p["left_eye_inner"]) +
             dist(p["right_eye_inner"], p["right_eye_outer"])) / 2
    eye_gap = dist(p["left_eye_inner"], p["right_eye_inner"])
    mouth_w = dist(p["mouth_left"], p["mouth_right"])
    eye_ratio = eye_w / face_width
    gap_ratio = eye_gap / eye_w if eye_w > 0 else 0
    mouth_ratio = mouth_w / face_width
    eye_score = max(0, 10 * (1 - abs(eye_ratio - 0.20) / 0.10))
    gap_score = max(0, 8 * (1 - abs(gap_ratio - 1.0) / 0.5))
    mouth_score = max(0, 7 * (1 - abs(mouth_ratio - 0.38) / 0.15))
    return round(eye_score + gap_score + mouth_score, 1), {
        "eye_ratio": round(eye_ratio, 3),
        "gap_ratio": round(gap_ratio, 3),
        "mouth_ratio": round(mouth_ratio, 3),
    }


def score_face_shape(p):
    face_width = dist(p["left_cheek"], p["right_cheek"])
    face_length = dist(p["face_top"], p["face_bottom"])
    ratio = face_length / face_width if face_width > 0 else 0
    score = max(0, 15 * (1 - abs(ratio - 1.5) / 0.3))
    return round(score, 1), round(ratio, 2)


def percentile_from_score(total):
    mean, std = 65, 12
    z = (total - mean) / std
    cdf = 0.5 * (1 + erf(z / sqrt(2)))
    return round((1 - cdf) * 100, 1)


def get_advice(sym, gr, feat, shape):
    advice = []
    if sym[0] < 20:
        advice.append(f"⚠️ **좌우 비대칭** (편차 {sym[1]}%) — 눈썹/입꼬리 정렬 케어 추천")
    if gr[0] < 20:
        advice.append(f"⚠️ **상/중/하 비율 불균형** ({gr[1]}) — 헤어라인 or 턱 길이 조정으로 개선 여지")
    if feat[1]["eye_ratio"] < 0.17:
        advice.append("💡 **눈 크기 작은 편** — 아이라인/앞트임으로 인상 개선 가능")
    elif feat[1]["eye_ratio"] > 0.24:
        advice.append("✨ **눈 크기 큰 편** — 플러스 요소")
    if feat[1]["mouth_ratio"] < 0.33:
        advice.append("💡 **입 작은 편** — 립 확장 메이크업 or 시술로 밸런스 개선")
    if shape[1] > 1.65:
        advice.append("💡 **얼굴 긴 편** — 헤어스타일로 이마 커버 or 턱 시술 고려")
    elif shape[1] < 1.35:
        advice.append("💡 **얼굴 둥근/짧은 편** — V라인 시술 or 브이라인 메이크업")
    if not advice:
        advice.append("✨ **전반적으로 밸런스 좋음** — 큰 개선 포인트 없음")
    return advice


# ============ 전체 채점 파이프라인 (UI 용) ============
def analyze(img_bgr):
    img_bgr = canonicalize(img_bgr)
    img_bgr, applied_deg, det_conf = auto_orient(img_bgr)
    img_bgr = canonicalize(img_bgr)
    img_bgr = denoise(img_bgr)
    h, w = img_bgr.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    canon_sha = sha12(rgb.tobytes())
    mesh = get_face_mesh()
    result = mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    landmarks = result.multi_face_landmarks[0].landmark
    result2 = mesh.process(np.ascontiguousarray(rgb.copy()))
    lms2 = result2.multi_face_landmarks[0].landmark
    lm_diff = max(abs(landmarks[i].x - lms2[i].x) + abs(landmarks[i].y - lms2[i].y)
                  for i in range(len(landmarks)))
    p = get_points(landmarks, w, h)
    sym = score_symmetry(p)
    gr = score_golden_ratio(p)
    feat = score_features(p)
    shape = score_face_shape(p)
    total = sym[0] + gr[0] + feat[0] + shape[0]
    top_pct = percentile_from_score(total)

    # ---- 데이터셋 기반 채점 (population + reference 프로파일이 있을 때만) ----
    data_score = None
    try:
        import scoring
        raw_metrics = _extract_metrics_from_points(p)
        if raw_metrics is not None:
            data_score = scoring.score_metrics(raw_metrics)
    except Exception:
        data_score = None

    # ---- 벤치마킹 기반 고급 분석 (캐논/얼굴형/피부/퍼스널컬러) ----
    advanced = None
    try:
        import advanced_analysis as adv
        advanced = adv.analyze_advanced(img_bgr, p)
    except Exception as e:
        advanced = {"error": str(e)}

    # ---- SOTA AI 미모 점수 (SCUT-FBP5500 학습, ArcFace + Ridge) ----
    ai_beauty = None
    try:
        import beauty_model as bm
        ai_beauty = bm.score_face(img_bgr)
    except Exception as e:
        ai_beauty = {"error": str(e)}

    viz = img_bgr.copy()
    small_r = max(1, int(min(w, h) / 600))
    tiny_r = max(1, int(min(w, h) / 900))
    anchor_r = max(3, int(min(w, h) / 220))

    # 픽셀 좌표 캐싱
    lm_px = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.int32)

    # (1) MediaPipe FaceMesh 정식 tessellation (~900 삼각형 엣지) — 어두운 녹색 얇은 선
    tess = mp.solutions.face_mesh.FACEMESH_TESSELATION
    for (i, j) in tess:
        if i < len(lm_px) and j < len(lm_px):
            cv2.line(viz, tuple(lm_px[i]), tuple(lm_px[j]), (0, 140, 0), 1, cv2.LINE_AA)

    # (2) 각 tessellation 삼각형 내부에 barycentric 격자 점 (밀도 대폭 증가)
    tri_set = set()
    tess_list = list(tess)
    adj = {}
    for i, j in tess_list:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    for i in adj:
        neigh_i = adj[i]
        for j in neigh_i:
            if j <= i: continue
            for k in adj.get(j, ()):
                if k <= j: continue
                if k in neigh_i:
                    tri_set.add((i, j, k))
    # 5분할 barycentric 격자 → 삼각형당 (5+1)(5+2)/2 - 3 = 18개 내부점
    # (꼭짓점 제외, 엣지 중간 + 내부 점)
    SUBDIV = 5
    bary_pts = []
    for a in range(SUBDIV + 1):
        for b in range(SUBDIV + 1 - a):
            c = SUBDIV - a - b
            if a == SUBDIV or b == SUBDIV or c == SUBDIV:
                continue  # 꼭짓점 스킵 (원본 랜드마크로 이미 있음)
            bary_pts.append((a / SUBDIV, b / SUBDIV, c / SUBDIV))
    tri_dot_count = 0
    for (i, j, k) in tri_set:
        if i < len(lm_px) and j < len(lm_px) and k < len(lm_px):
            pi = lm_px[i].astype(np.float32)
            pj = lm_px[j].astype(np.float32)
            pk = lm_px[k].astype(np.float32)
            for wa, wb, wc in bary_pts:
                x = wa * pi[0] + wb * pj[0] + wc * pk[0]
                y = wa * pi[1] + wb * pj[1] + wc * pk[1]
                cv2.circle(viz, (int(x), int(y)), tiny_r, (100, 255, 100), -1)
                tri_dot_count += 1

    # (3) 478개 원본 랜드마크 위에 진한 녹색 점 겹쳐 그리기
    for pt in lm_px:
        cv2.circle(viz, tuple(pt), small_r, (0, 255, 0), -1)

    # (4) 55개 대칭 pair 를 옅은 청록 얇은 선으로
    for L, R in SYMMETRY_PAIRS:
        cv2.line(viz, tuple(lm_px[L]), tuple(lm_px[R]), (200, 180, 0), 1, cv2.LINE_AA)

    # (5) 얼굴 중심선
    cx_val = int(np.median([landmarks[i].x for i in MIDLINE_INDICES]) * w)
    cv2.line(viz, (cx_val, 0), (cx_val, h), (255, 100, 100), 1, cv2.LINE_AA)

    # (6) 20개 해부학적 앵커(클러스터 중앙값) 을 큰 노란 점으로 강조
    pts_2d = p["_pts_2d"]
    for pt in pts_2d.values():
        cv2.circle(viz, (int(pt[0]), int(pt[1])), anchor_r, (0, 220, 255), -1)
        cv2.circle(viz, (int(pt[0]), int(pt[1])), anchor_r + 1, (0, 0, 0), 1)

    # (7) 이마-턱 세로 라인
    cv2.line(viz, (int(pts_2d["forehead"][0]), int(pts_2d["forehead"][1])),
             (int(pts_2d["chin"][0]), int(pts_2d["chin"][1])), (0, 100, 255), 2)

    yaw, pitch, roll = p["_pose_deg"]
    tri_n = len(tri_set)
    cv2.putText(viz, f"yaw {yaw:+.1f} pitch {pitch:+.1f} roll {roll:+.1f}  |  {len(landmarks)} + {tri_n} tri centroids = {len(landmarks)+tri_n} pts",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return {
        "sym": sym, "gr": gr, "feat": feat, "shape": shape,
        "total": total, "top_pct": top_pct,
        "pose": p["_pose_deg"],
        "auto_rotated_deg": applied_deg,
        "detect_conf": det_conf,
        "canon_sha": canon_sha,
        "canon_size": (w, h),
        "determinism_diff": float(lm_diff),
        "viz": cv2.cvtColor(viz, cv2.COLOR_BGR2RGB),
        "advice": get_advice(sym, gr, feat, shape),
        "data_score": data_score,  # 데이터셋 기반 채점 (없으면 None)
        "advanced": advanced,  # 캐논/얼굴형/피부/퍼스널컬러
        "ai_beauty": ai_beauty,  # SOTA AI 미모 점수 (SCUT-FBP5500 학습)
    }
