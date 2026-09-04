---
title: FaceScore
emoji: ✨
colorFrom: pink
colorTo: purple
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
license: mit
python_version: 3.11
---

# 얼굴 점수 AI ✨

사진 한 장으로 얼굴 밸런스를 100점 만점으로 분석합니다. (재미용 서비스)

## 기능
- 🤖 **AI 미모 점수** (SCUT-FBP5500 + ArcFace, Pearson 0.864)
- 🏛️ Farkas 고전 캐논 (안와/구비/Phi 비율)
- 👤 얼굴형 분류 (계란형/둥근/긴/하트/각진/긴타원)
- 🧴 피부 분석 (매끄러움/톤/밝기/언더톤)
- 🎨 퍼스널 컬러 4계절 진단
- 좌우 대칭성 + 황금비율 (상/중/하 3등분)
- 이목구비 비율 + Gold 얼굴 유사도 (김태희/원빈 기준)

## 기술 스택
- Python 3.11
- Streamlit (웹 UI)
- MediaPipe (얼굴 랜드마크 478개)
- InsightFace ArcFace (512-d embedding)
- scikit-learn Ridge (SCUT-FBP5500 학습)
- OpenCV, NumPy

## 로컬 실행

**전체 기능 (AI 미모 점수 포함)**
```bash
pip install -r requirements-full.txt
streamlit run app.py
```

**슬림 (클라우드 배포판, AI 미모 없음)**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 배포
- Streamlit Cloud: 1GB RAM 한도로 AI 미모 오프 (기하 분석 + gold 유사도 + 캐논 + 얼굴형 + 피부만)
- 로컬: `requirements-full.txt` 로 SCUT-FBP5500 학습 모델 (Pearson 0.864) 활성

## 면책
본 서비스는 재미용이며, 성형 권유가 아닙니다. 시술은 반드시 전문의 상담 후 결정하세요.
