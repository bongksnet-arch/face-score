"""Gold-standard 기반 채점 (2026-09-04 재작성).

이전: 아이돌 프로파일 근접도 70% + 일반 평균 근접도 30%
     → 평범한 얼굴도 pop_score 로 30점 공짜 + z-decay 완만 → 변별력 없음

현재: 사용자 지정 gold 얼굴 (images.jpeg = 100점 기준) 과의 거리로 채점
     - 지표별 z = (x - gold) / pop_std
     - 지표별 점수 = 100 * exp(-z² / 2)  (가우시안 감쇠, z=1이면 61점, z=2면 14점)
     - 가중치 = 아이돌 vs 일반 판별력 (|ref_mean - pop_mean| / pop_std)
     - 총점 = 가중 평균
     - Gold 얼굴 자체 → 모든 z=0 → 100 만점
"""
import json
import math
from pathlib import Path
from typing import Optional

_DIR = Path(__file__).parent
POP_JSON = _DIR / "population_stats.json"
REF_JSON = _DIR / "reference_profile.json"
GOLD_FEMALE_JSON = _DIR / "gold_face_female.json"
GOLD_MALE_JSON = _DIR / "gold_face_male.json"
GOLD_JSON = _DIR / "gold_face.json"  # legacy fallback

METRIC_COLS = [
    "gr_upper", "gr_middle", "gr_lower", "sym_dev",
    "eye_ratio", "gap_ratio", "mouth_ratio", "shape_ratio",
    "eye_aspect", "brow_arch", "brow_eye_gap",
    "nose_width_ratio", "nose_length_ratio",
    "lip_ratio", "upper_lip_thickness", "lower_lip_thickness",
    "jaw_angle", "cheek_ratio", "philtrum_ratio",
]

METRIC_LABEL = {
    "gr_upper":    "이마 비율 (상단 1/3)",
    "gr_middle":   "중안부 비율",
    "gr_lower":    "하관 비율 (턱 부분)",
    "sym_dev":     "좌우 대칭 편차",
    "eye_ratio":   "눈 크기 / 얼굴폭",
    "gap_ratio":   "눈 간격 / 눈 폭",
    "mouth_ratio": "입 크기 / 얼굴폭",
    "shape_ratio": "얼굴 길이 / 폭 (갸름도)",
    "eye_aspect":  "눈 두께 / 폭 (눈꺼풀 열림)",
    "brow_arch":   "눈썹 아치 높이",
    "brow_eye_gap": "눈-눈썹 거리",
    "nose_width_ratio":  "코 폭 / 얼굴폭",
    "nose_length_ratio": "코 길이 / 얼굴 길이",
    "lip_ratio":         "윗입술 / 아랫입술 두께",
    "upper_lip_thickness": "윗입술 두께 / 얼굴폭",
    "lower_lip_thickness": "아랫입술 두께 / 얼굴폭",
    "jaw_angle":   "턱 각도 (좁을수록 V라인)",
    "cheek_ratio": "광대뼈 폭 / 얼굴폭",
    "philtrum_ratio": "인중 길이 / 얼굴 길이",
}

_pop_cache = None
_ref_cache = None
_golds_cache = None  # dict: {"female": {...}, "male": {...}}


def load_profiles(force_reload: bool = False):
    global _pop_cache, _ref_cache, _golds_cache
    if force_reload or _pop_cache is None:
        _pop_cache = json.loads(POP_JSON.read_text(encoding="utf-8")) if POP_JSON.exists() else None
    if force_reload or _ref_cache is None:
        _ref_cache = json.loads(REF_JSON.read_text(encoding="utf-8")) if REF_JSON.exists() else None
    if force_reload or _golds_cache is None:
        golds = {}
        if GOLD_FEMALE_JSON.exists():
            golds["female"] = json.loads(GOLD_FEMALE_JSON.read_text(encoding="utf-8"))
        if GOLD_MALE_JSON.exists():
            golds["male"] = json.loads(GOLD_MALE_JSON.read_text(encoding="utf-8"))
        # legacy fallback
        if not golds and GOLD_JSON.exists():
            golds["female"] = json.loads(GOLD_JSON.read_text(encoding="utf-8"))
        _golds_cache = golds or None
    return _pop_cache, _ref_cache, _golds_cache


def compute_weights(pop: dict, ref: dict) -> dict:
    """지표별 가중치 = |아이돌 평균 - 일반 평균| / 일반 std."""
    weights = {}
    for m in METRIC_COLS:
        if m not in pop or m not in ref:
            weights[m] = 0.0
            continue
        p_std = pop[m]["std"] or 1e-6
        weights[m] = abs(ref[m]["mean"] - pop[m]["mean"]) / p_std
    return weights


def _gaussian_score(z: float) -> float:
    """z=0 → 100, z=1 → 88, z=2 → 61, z=3 → 32, z=4 → 14.

    σ² = 4 (기존 1 대비 완화). 같은 인물 다른 사진의 자연스러운 지표 편차
    (pose/expression/lighting) 를 30~40점 감점이 아닌 10~20점으로 흡수.
    실제로 다른 얼굴은 여전히 z 가 크게 벌어져 낮은 점수.
    또한 단일 metric outlier 방지: |z| > 4 는 4로 캡.
    """
    z = min(4.0, abs(z))
    return float(100.0 * math.exp(-(z * z) / 8.0))


def _score_against_gold(metrics: dict, pop: dict, gold: dict, weights: dict):
    """단일 gold 대비 채점. (total, per_metric) 튜플 반환."""
    w_sum = sum(weights.values()) or 1.0
    per_metric = []
    total_weighted = 0.0
    for m in METRIC_COLS:
        if m not in metrics or m not in pop or m not in gold:
            continue
        x = float(metrics[m])
        p_mean = pop[m]["mean"]
        p_std = pop[m]["std"] or 1e-6
        g_val = float(gold[m])
        z_gold = (x - g_val) / p_std
        z_pop = (x - p_mean) / p_std
        m_score = _gaussian_score(z_gold)
        w = weights[m]
        total_weighted += w * m_score
        per_metric.append({
            "key": m,
            "label": METRIC_LABEL.get(m, m),
            "value": round(x, 4),
            "gold_value": round(g_val, 4),
            "pop_mean": round(p_mean, 4),
            "z_gold": round(z_gold, 2),
            "z_pop": round(z_pop, 2),
            "score": round(m_score, 1),
            "weight": round(w, 3),
        })
    per_metric.sort(key=lambda x: -x["weight"])
    return total_weighted / w_sum, per_metric


def score_metrics(metrics: dict,
                  pop: Optional[dict] = None,
                  ref: Optional[dict] = None,
                  golds: Optional[dict] = None):
    """metrics(compute_metrics 결과) → 채점 결과 dict.

    여자/남자 gold 둘 다 대비 채점 후 **더 높은 점수** 를 채택 (자동 성별 매칭).
    Gold 얼굴 자체를 입력하면 해당 gold 에 대해 100점.
    """
    if pop is None or ref is None or golds is None:
        pop, ref, golds = load_profiles()

    if pop is None or not golds:
        return None

    weights = compute_weights(pop, ref) if ref is not None else {m: 1.0 for m in METRIC_COLS}

    # 각 gold 대비 채점
    candidates = {}
    for gender, gold in golds.items():
        total, per_metric = _score_against_gold(metrics, pop, gold, weights)
        candidates[gender] = {
            "gender": gender,
            "total": round(total, 1),
            "per_metric": per_metric,
            "gold_source": gold.get("_meta", {}).get("source", "unknown"),
        }

    # 더 높은 점수의 gold 를 채택 (자동 성별 매칭)
    best = max(candidates.values(), key=lambda c: c["total"])

    return {
        "total": best["total"],
        "matched_gender": best["gender"],
        "matched_gold": best["gold_source"],
        "per_metric": best["per_metric"],
        "all_genders": {g: {"total": c["total"], "gold_source": c["gold_source"]}
                        for g, c in candidates.items()},
        "meta": {
            "pop_n": pop.get("meta", {}).get("n_samples", 0),
            "ref_n": ref.get("meta", {}).get("n_samples", 0) if ref else 0,
            "gold_source": best["gold_source"],
        },
    }
