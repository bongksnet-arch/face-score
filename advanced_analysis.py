"""벤치마킹 기반 고급 분석 모듈.

경쟁 서비스 (Face++, Beauty Scanner, FaceScore TW, Fotor 등) 알고리즘 정리:
1. Farkas Neoclassical Canons (9개 중 계산 가능한 4개 + Phi 비율)
2. 얼굴형 분류 (oval/round/heart/square/long/oblong)
3. 피부 분석 (매끄러움/톤 균일도/밝기/언더톤) — 뺨 영역 샘플링
4. 퍼스널 컬러 4계절 (봄웜/여름쿨/가을웜/겨울쿨)

face_analysis.py 의 랜드마크·정면화 결과를 재사용한다.
"""
import cv2
import numpy as np
from math import hypot


# ---------------- 유틸 ----------------
def _d(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def _match_score(actual: float, target: float, tol: float) -> float:
    """|actual-target|/tol 이 0이면 100, 1이면 0. 0~100 clip."""
    if tol <= 0:
        return 0.0
    dev = abs(actual - target) / tol
    return float(max(0.0, min(100.0, 100.0 * (1.0 - dev))))


# ---------------- Farkas Neoclassical Canons ----------------
def neoclassical_canons(p) -> dict:
    """계산 가능한 4개 캐논 + Phi 3개.

    Farkas 1994 정리 기준. 정면화된 랜드마크 dict `p` 사용.
    각 항목: {value, target, score(0-100)}
    """
    all_2d = p["_all_frontal"]

    face_w = _d(p["left_cheek"], p["right_cheek"])
    face_h = _d(p["face_top"], p["face_bottom"])
    if face_w <= 0 or face_h <= 0:
        return {}

    # 3-section (이미 있음, 참조용 재계산)
    upper = p["glabella"][1] - p["forehead"][1]
    middle = p["subnasale"][1] - p["glabella"][1]
    lower = p["chin"][1] - p["subnasale"][1]
    total = upper + middle + lower
    thirds_dev = (abs(upper/total - 1/3) + abs(middle/total - 1/3) + abs(lower/total - 1/3)) if total > 0 else 1.0

    # Orbital canon: 눈 폭 = 눈 사이 간격 = 코 폭
    eye_w_l = _d(p["left_eye_outer"], p["left_eye_inner"])
    eye_w_r = _d(p["right_eye_inner"], p["right_eye_outer"])
    eye_w = (eye_w_l + eye_w_r) / 2
    eye_gap = _d(p["left_eye_inner"], p["right_eye_inner"])
    nose_w = abs(all_2d[48, 0] - all_2d[278, 0])

    orbital_gap_ratio = eye_gap / eye_w if eye_w > 0 else 0
    orbital_nose_ratio = nose_w / eye_w if eye_w > 0 else 0

    # Naso-oral canon: 입 폭 = 1.5 × 코 폭 (Ricketts)
    mouth_w = _d(p["mouth_left"], p["mouth_right"])
    mouth_nose_ratio = mouth_w / nose_w if nose_w > 0 else 0

    # Phi (1.618) 비율 3종
    face_shape_ratio = face_h / face_w  # 얼굴 길이/폭 (target 1.618)
    mouth_face_ratio = face_w / mouth_w if mouth_w > 0 else 0  # 얼굴폭/입폭 (target 1.618)
    lip_h_upper = abs(all_2d[13, 1] - all_2d[0, 1])
    lip_h_lower = abs(all_2d[17, 1] - all_2d[14, 1])
    lip_phi = lip_h_lower / lip_h_upper if lip_h_upper > 0 else 0  # 아랫/윗 (target 1.618)

    return {
        "three_section": {
            "label": "3분할 캐논 (이마=중안=하관)",
            "value": round(1 - thirds_dev, 3),
            "target": 1.0,
            "score": _match_score(thirds_dev, 0.0, 0.15),
        },
        "orbital_gap": {
            "label": "안와 캐논 (눈 간격 = 눈 폭)",
            "value": round(orbital_gap_ratio, 3),
            "target": 1.0,
            "score": _match_score(orbital_gap_ratio, 1.0, 0.35),
        },
        "orbital_nose": {
            "label": "안와 캐논 (코 폭 = 눈 폭)",
            "value": round(orbital_nose_ratio, 3),
            "target": 1.0,
            "score": _match_score(orbital_nose_ratio, 1.0, 0.35),
        },
        "naso_oral": {
            "label": "구비 캐논 (입 폭 = 1.5 × 코 폭)",
            "value": round(mouth_nose_ratio, 3),
            "target": 1.5,
            "score": _match_score(mouth_nose_ratio, 1.5, 0.5),
        },
        "phi_face": {
            "label": "얼굴 Phi (길이/폭 = 1.618)",
            "value": round(face_shape_ratio, 3),
            "target": 1.618,
            "score": _match_score(face_shape_ratio, 1.618, 0.35),
        },
        "phi_mouth": {
            "label": "입 Phi (얼굴폭/입폭 = 1.618)",
            "value": round(mouth_face_ratio, 3),
            "target": 1.618,
            "score": _match_score(mouth_face_ratio, 1.618, 0.45),
        },
        "phi_lip": {
            "label": "입술 Phi (아랫/윗 = 1.618)",
            "value": round(lip_phi, 3),
            "target": 1.618,
            "score": _match_score(lip_phi, 1.618, 0.7),
        },
    }


# ---------------- 얼굴형 분류 ----------------
_SHAPE_LABEL = {
    "oval":    ("계란형 (Oval)",   "황금 균형. 대부분 헤어스타일이 어울림"),
    "round":   ("둥근형 (Round)",  "귀여운 인상. V라인 컷·긴 헤어로 세로감 강조"),
    "long":    ("긴 얼굴 (Long)",  "성숙한 인상. 앞머리·볼륨 옆머리로 가로감 보완"),
    "heart":   ("하트형 (Heart)",  "이마 넓고 턱 좁음. 사이드 뱅으로 이마 커버"),
    "square":  ("각진형 (Square)", "카리스마. 레이어드 컷으로 각진 라인 완화"),
    "oblong":  ("긴타원형 (Oblong)","길고 좁음. 볼륨 사이드로 균형"),
}


def classify_face_shape(p) -> dict:
    """3개 폭(이마/광대/턱) + 길이 비율로 얼굴형 분류."""
    all_2d = p["_all_frontal"]

    face_w = _d(p["left_cheek"], p["right_cheek"])
    face_h = _d(p["face_top"], p["face_bottom"])
    if face_w <= 0 or face_h <= 0:
        return {"key": "unknown", "label": "판정 불가", "hint": "", "ratios": {}}

    forehead_w = abs(all_2d[54, 0] - all_2d[284, 0])
    cheek_w = face_w
    jaw_w = abs(all_2d[172, 0] - all_2d[397, 0])
    length_ratio = face_h / face_w

    # 비율 정규화 (광대=1 기준)
    fh_r = forehead_w / cheek_w
    jw_r = jaw_w / cheek_w

    # 턱 각도 (좁을수록 뾰족)
    chin = all_2d[152]
    jl = all_2d[172]
    jr = all_2d[397]
    v1 = np.array([jl[0] - chin[0], jl[1] - chin[1]])
    v2 = np.array([jr[0] - chin[0], jr[1] - chin[1]])
    cos_a = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    cos_a = max(-1.0, min(1.0, cos_a))
    jaw_angle = float(np.degrees(np.arccos(cos_a)))

    # 규칙 기반 (우선순위 순)
    if length_ratio >= 1.75:
        key = "oblong"
    elif length_ratio <= 1.3:
        key = "round"
    elif fh_r > jw_r + 0.08 and jaw_angle < 110:
        key = "heart"
    elif abs(fh_r - jw_r) < 0.05 and jw_r > 0.85 and jaw_angle > 115:
        key = "square"
    elif length_ratio >= 1.6:
        key = "long"
    else:
        key = "oval"

    label, hint = _SHAPE_LABEL[key]
    return {
        "key": key,
        "label": label,
        "hint": hint,
        "ratios": {
            "length": round(length_ratio, 2),
            "forehead_w": round(fh_r, 2),
            "cheek_w": 1.0,
            "jaw_w": round(jw_r, 2),
            "jaw_angle": round(jaw_angle, 1),
        },
    }


# ---------------- 피부 분석 ----------------
def _sample_skin_pixels(img_bgr, p) -> np.ndarray:
    """양쪽 뺨 중심 근처 영역에서 픽셀 샘플링. shape=(N,3) BGR."""
    h, w = img_bgr.shape[:2]
    pts_2d = p["_pts_2d"]
    samples = []
    for name in ("left_cheek", "right_cheek"):
        cx, cy = pts_2d[name]
        cx, cy = int(cx), int(cy)
        r = max(6, int(min(w, h) * 0.03))
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w, cx + r), min(h, cy + r)
        patch = img_bgr[y0:y1, x0:x1]
        if patch.size > 0:
            samples.append(patch.reshape(-1, 3))
    if not samples:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(samples, axis=0)


def skin_analysis(img_bgr, p) -> dict:
    """뺨 픽셀 → 매끄러움/톤 균일도/밝기/언더톤 산출.

    - smoothness: 100 - normalize(Laplacian variance). 낮을수록 매끄러움 = 높은 점수.
    - tone_uniformity: 100 - normalize(L 채널 std). 표준편차 낮으면 톤 균일.
    - brightness: L 채널 평균 (0-100 in LAB → 0-100 그대로).
    - undertone: LAB a/b 비율 → warm(양수) / cool(음수) 방향.
    """
    pixels = _sample_skin_pixels(img_bgr, p)
    if pixels.shape[0] < 50:
        return {"available": False}

    # LAB 변환 (톤 균일도 + 언더톤)
    lab_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    # OpenCV LAB: L 0-255, a/b 0-255 (128 중심)
    L = lab_pixels[:, 0] * (100.0 / 255.0)
    a = lab_pixels[:, 1] - 128.0
    b = lab_pixels[:, 2] - 128.0

    brightness = float(np.mean(L))
    tone_std = float(np.std(L))
    tone_uniformity = max(0.0, min(100.0, 100.0 - tone_std * 4.0))

    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))
    # 언더톤: b(황) - a(적) 조합. b가 크면 warm, a가 상대적으로 크면 cool.
    # 실무 근사: undertone_index = b_mean - 0.6*a_mean. 양수=warm, 음수=cool
    undertone_index = b_mean - 0.6 * a_mean
    undertone = "warm" if undertone_index > 0 else "cool"
    undertone_kr = "웜톤" if undertone == "warm" else "쿨톤"
    undertone_strength = min(100.0, abs(undertone_index) * 8.0)

    # 매끄러움: 뺨 패치 원본에서 Laplacian variance
    patch_gray_list = []
    pts_2d = p["_pts_2d"]
    h_img, w_img = img_bgr.shape[:2]
    for name in ("left_cheek", "right_cheek"):
        cx, cy = pts_2d[name]
        cx, cy = int(cx), int(cy)
        r = max(6, int(min(w_img, h_img) * 0.03))
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w_img, cx + r), min(h_img, cy + r)
        patch = img_bgr[y0:y1, x0:x1]
        if patch.size > 0:
            patch_gray_list.append(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
    if patch_gray_list:
        lap_vars = [float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in patch_gray_list]
        lap_var_avg = float(np.mean(lap_vars))
    else:
        lap_var_avg = 100.0
    # Laplacian var 는 대략 10~500 범위. 낮을수록 매끈함. 100 이하 = 좋음.
    smoothness = max(0.0, min(100.0, 100.0 - (lap_var_avg - 20.0) * 0.35))

    return {
        "available": True,
        "smoothness": round(smoothness, 1),
        "tone_uniformity": round(tone_uniformity, 1),
        "brightness": round(brightness, 1),
        "undertone": undertone,
        "undertone_kr": undertone_kr,
        "undertone_strength": round(undertone_strength, 1),
        "raw": {
            "lap_var": round(lap_var_avg, 1),
            "L_mean": round(brightness, 2),
            "L_std": round(tone_std, 2),
            "a_mean": round(a_mean, 2),
            "b_mean": round(b_mean, 2),
            "undertone_index": round(undertone_index, 2),
        },
    }


# ---------------- 퍼스널 컬러 4계절 ----------------
_SEASON_ADVICE = {
    "spring": ("봄 웜톤 (Spring Warm)",
               "코랄, 피치, 아이보리, 웜 베이지. 골드 액세서리. 밝고 채도 높은 색."),
    "summer": ("여름 쿨톤 (Summer Cool)",
               "라벤더, 로즈 핑크, 파우더 블루, 라이트 그레이. 실버 액세서리. 파스텔·뮤트 톤."),
    "autumn": ("가을 웜톤 (Autumn Warm)",
               "카멜, 머스타드, 올리브, 벽돌색. 골드/브론즈. 딥하고 채도 낮은 웜."),
    "winter": ("겨울 쿨톤 (Winter Cool)",
               "블랙, 화이트, 로얄 블루, 마젠타. 실버/플래티넘. 선명한 원색·모노톤."),
}


def personal_color_season(skin: dict) -> dict:
    """언더톤(warm/cool) × 명도(light/deep) → 4계절."""
    if not skin.get("available"):
        return {"available": False}

    undertone = skin["undertone"]  # warm / cool
    brightness = skin["brightness"]  # LAB L 0-100
    # 임계값: L>=60 은 light, <60 은 deep (동아시아 평균 ~55-70)
    depth = "light" if brightness >= 60 else "deep"

    key = {
        ("warm", "light"): "spring",
        ("cool", "light"): "summer",
        ("warm", "deep"):  "autumn",
        ("cool", "deep"):  "winter",
    }[(undertone, depth)]

    label, advice = _SEASON_ADVICE[key]
    return {
        "available": True,
        "key": key,
        "label": label,
        "advice": advice,
        "depth": depth,
        "undertone": undertone,
    }


# ---------------- 통합 진입점 ----------------
def analyze_advanced(img_bgr, p) -> dict:
    """face_analysis.analyze() 에서 호출. 랜드마크·정면화 결과 재사용."""
    canons = neoclassical_canons(p)
    shape = classify_face_shape(p)
    skin = skin_analysis(img_bgr, p)
    season = personal_color_season(skin)
    # 캐논 종합 점수 (평균)
    canon_avg = round(np.mean([c["score"] for c in canons.values()]), 1) if canons else 0.0
    return {
        "canons": canons,
        "canon_avg": canon_avg,
        "face_shape": shape,
        "skin": skin,
        "personal_color": season,
    }
