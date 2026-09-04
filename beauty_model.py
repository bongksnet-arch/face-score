"""SOTA 얼굴 미모 점수 (SCUT-FBP5500 + ArcFace embedding 학습 모델).

- insightface buffalo_l 로 512-d embedding 추출
- 학습된 회귀 모델로 1~5 점수 예측
- 0~100 점으로 변환하여 반환

배포용 파일:
  - beauty_model.pkl: 학습된 sklearn 모델 (~수 MB)
  - buffalo_l/*.onnx: insightface 다운로드 (~280MB, 첫 실행 자동 다운로드)
"""
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# insightface/sklearn 은 무거워서 Streamlit Cloud 배포엔 포함하지 않음.
# 로컬 실행 시에만 활성 (pip install -r requirements-full.txt).
try:
    from insightface.app import FaceAnalysis  # noqa: F401
    import sklearn  # noqa: F401
    _AVAILABLE = True
except Exception:
    _AVAILABLE = False

_DIR = Path(__file__).parent
MODEL_PKL = _DIR / "beauty_model.pkl"

_app = None
_beauty = None


def get_face_app():
    """buffalo_l 중 detection + recognition 만 로드 (3d68/2d106/genderage 제외).

    Streamlit Cloud 1GB RAM 한도 대응: 5개 모델(343MB) → 2개(192MB) 로 절감.
    """
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    return _app


def get_beauty_model():
    global _beauty
    if _beauty is None and MODEL_PKL.exists():
        with open(MODEL_PKL, "rb") as f:
            _beauty = pickle.load(f)
    return _beauty


def _score_to_100(raw: float, meta: dict) -> float:
    """모델 예측값(1~5) → 0~100 변환.

    학습셋 관측 범위 [y_min, y_max] 기준 min-max 정규화 후 100 스케일.
    y_min → 0, y_max → 100.
    """
    lo, hi = meta["y_min"], meta["y_max"]
    if hi - lo < 1e-6:
        return 50.0
    v = (raw - lo) / (hi - lo)
    return float(max(0.0, min(100.0, v * 100.0)))


def score_face(img_bgr) -> Optional[dict]:
    """이미지(BGR) → {score, raw, meta} or None (얼굴 미검출/모델 없음/미설치)."""
    if not _AVAILABLE:
        return None  # 클라우드 배포 등 insightface 미설치 환경
    beauty = get_beauty_model()
    if beauty is None:
        return None
    app = get_face_app()
    faces = app.get(img_bgr)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
               reverse=True)
    emb = faces[0].embedding.astype(np.float32)
    emb /= (np.linalg.norm(emb) + 1e-9)
    raw = float(beauty["model"].predict(emb[None, :])[0])
    score = _score_to_100(raw, beauty)
    return {
        "score": round(score, 1),
        "raw": round(raw, 3),
        "model_name": beauty.get("model_name", "?"),
        "cv_pc": beauty.get("cv_pc", 0.0),
        "y_range": [beauty["y_min"], beauty["y_max"]],
        "n_train": beauty.get("n_train", 0),
    }
