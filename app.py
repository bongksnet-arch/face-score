"""Streamlit UI. 분석 로직은 face_analysis 에 있음."""
import face_analysis as fa  # env var 잠금이 여기서 걸림 (다른 import 이전 필수)

import streamlit as st

st.set_page_config(page_title="얼굴 점수 AI", page_icon="✨", layout="centered")


# Streamlit 캐시 래퍼 (모듈 싱글턴 초기화만 트리거)
@st.cache_resource
def _prime():
    fa.get_face_mesh()
    fa.get_face_detector()
    return True


@st.cache_data(show_spinner=False)
def analyze_cached(img_bytes: bytes):
    _prime()
    img_bgr = fa.bytes_to_bgr(img_bytes)
    if img_bgr is None:
        return None
    r = fa.analyze(img_bgr)
    if r is None:
        return None
    r["file_sha"] = fa.sha12(img_bytes)
    r["file_bytes"] = len(img_bytes)
    return r


# ============ UI ============
st.title("✨ 얼굴 점수 AI")
st.caption("사진 한 장으로 얼굴 밸런스를 100점 만점으로 분석합니다. (재미용 서비스)")

uploaded = st.file_uploader("얼굴 사진 업로드 (정면, 밝은 곳 권장)", type=["jpg", "jpeg", "png"])

if uploaded:
    img_bytes = uploaded.getvalue()

    with st.spinner("AI가 얼굴을 분석 중..."):
        r = analyze_cached(img_bytes)

    if r is None:
        st.error("얼굴을 찾지 못했습니다. 정면 얼굴 사진으로 다시 시도해주세요.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(r["viz"], caption="분석 결과 시각화", use_column_width=True)
        with col2:
            ai = r.get("ai_beauty")
            if ai and not ai.get("error"):
                st.metric("🤖 AI 미모 점수", f"{ai['score']:.1f} / 100",
                          f"raw {ai['raw']:.2f}/5")
                st.caption(
                    f"SCUT-FBP5500 ({ai['n_train']:,}장 인간 평가) 학습 · "
                    f"ArcFace + {ai['model_name']} · 인간 상관계수 **PC={ai['cv_pc']:.3f}**"
                )
                st.progress(min(ai["score"] / 100, 1.0))
            else:
                st.info(
                    "🤖 **AI 미모 점수 (SCUT-FBP5500 학습, PC=0.864)** 는 "
                    "클라우드 메모리 제약으로 배포판에서 비활성입니다. "
                    "로컬 실행 시 활성됩니다 (`pip install -r requirements-full.txt`)."
                )
            ds = r.get("data_score")
            if ds is not None:
                gender_kr = {"female": "여자 (김태희)", "male": "남자 (원빈)"}.get(ds.get("matched_gender"), "?")
                with st.expander(f"↳ Gold 얼굴 유사도 (참고)"):
                    st.write(f"매칭: **{gender_kr}**, 총점 {ds['total']:.1f} / 100")
                    if ds.get("all_genders"):
                        cols = st.columns(len(ds["all_genders"]))
                        for i, (g, info) in enumerate(ds["all_genders"].items()):
                            gl = {"female": "여자 기준(김태희)", "male": "남자 기준(원빈)"}.get(g, g)
                            cols[i].metric(gl, f"{info['total']:.1f}")
                with st.expander("↳ 이전 캐논(비트루비우스) 기준 점수 (참고)"):
                    st.write(f"총점 {r['total']:.1f} / 100 (임시 캐논)")
            else:
                st.metric("총점", f"{r['total']:.1f} / 100")
                st.metric("상위 백분위 (임시)", f"약 {r['top_pct']}%")
                st.progress(min(r["total"] / 100, 1.0))
                st.caption("⚠️ 데이터셋 기반 채점 미준비 (임시 캐논 점수)")

        st.divider()
        st.subheader("📊 세부 점수")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("대칭성", f"{r['sym'][0]}/30", f"편차 {r['sym'][1]}%")
        c2.metric("황금비율", f"{r['gr'][0]}/30", r['gr'][1])
        c3.metric("이목구비", f"{r['feat'][0]}/25")
        c4.metric("얼굴형", f"{r['shape'][0]}/15", f"비율 {r['shape'][1]}")

        with st.expander("🔍 이목구비 상세 지표"):
            st.write(f"- 눈 크기 비율 (얼굴폭 대비): **{r['feat'][1]['eye_ratio']}**")
            st.write(f"- 눈 간격 비율 (눈 폭 대비): **{r['feat'][1]['gap_ratio']}**")
            st.write(f"- 입 크기 비율 (얼굴폭 대비): **{r['feat'][1]['mouth_ratio']}**")
            yaw, pitch, roll = r["pose"]
            st.write(f"- 촬영 자세 (보정 전): yaw **{yaw:+.1f}°** / pitch **{pitch:+.1f}°** / roll **{roll:+.1f}°**")
            if r["auto_rotated_deg"]:
                st.write(f"- 자동 회전 보정: **{r['auto_rotated_deg']}°** (검출 신뢰도 {r['detect_conf']:.2f})")
            st.caption("※ 3D solvePnP 로 정면 자세로 역회전 후 측정합니다. |yaw|<20°, |pitch|<15° 권장.")

        with st.expander("🧪 재현성 진단 (같은 사진인데 점수 달라질 때 확인)"):
            st.write(f"- **파일 SHA-256(앞12자리)**: `{r['file_sha']}` · 바이트 {r['file_bytes']:,}")
            st.write(f"- **전처리 후 픽셀 SHA(앞12자리)**: `{r['canon_sha']}` · 크기 {r['canon_size']}")
            st.write(f"- **자기 결정성 편차** (같은 입력 2회 실행 랜드마크 차): `{r['determinism_diff']:.2e}`")
            st.caption(
                "두 사진의 **파일 SHA**가 다르면 서로 다른 바이트 → 갤러리 재저장한 것. "
                "**전처리 후 픽셀 SHA**가 같으면 이후 파이프라인은 100% 동일 값 보장. "
                "**자기 결정성 편차**가 0.0 이 아니면 스레드 결정론이 꺼진 것 → Streamlit 완전 종료 후 재기동."
            )

        ds = r.get("data_score")
        if ds is not None and ds.get("per_metric"):
            with st.expander("🎯 지표별 Gold 대비 상세"):
                import pandas as pd
                rows = []
                for pm in ds["per_metric"]:
                    rows.append({
                        "지표": pm["label"],
                        "내 값": f"{pm['value']:.3f}",
                        "Gold": f"{pm['gold_value']:.3f}",
                        "일반 평균": f"{pm['pop_mean']:.3f}",
                        "z (vs Gold)": f"{pm['z_gold']:+.2f}",
                        "점수": f"{pm['score']:.0f}",
                        "가중치": f"{pm['weight']:.2f}",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.caption("각 지표 점수 = 100 × exp(-z²/2). z=1 → 61점, z=2 → 14점. 가중치 = 아이돌 vs 일반 판별력.")

        # ---- 벤치마킹 기반 고급 분석 ----
        adv = r.get("advanced")
        if adv and not adv.get("error"):
            st.divider()
            st.subheader("🏛️ Farkas 고전 캐논 (해외 서비스 벤치마킹)")
            st.caption(f"캐논 종합 점수: **{adv['canon_avg']} / 100**  ·  Beauty Scanner / 金比容 등에서 사용")
            import pandas as pd
            canon_rows = []
            for k, c in adv["canons"].items():
                canon_rows.append({
                    "캐논": c["label"],
                    "내 값": c["value"],
                    "이상값": c["target"],
                    "매치도": f"{c['score']:.0f} / 100",
                })
            st.dataframe(pd.DataFrame(canon_rows), hide_index=True, use_container_width=True)

            st.divider()
            st.subheader("👤 얼굴형 분류 (Fotor / AILab 스타일)")
            fs = adv["face_shape"]
            colA, colB = st.columns([1, 2])
            colA.metric("얼굴형", fs["label"])
            colB.info(fs["hint"])
            fr = fs["ratios"]
            st.caption(
                f"길이/폭 **{fr.get('length','-')}** · 이마 폭 **{fr.get('forehead_w','-')}** · "
                f"광대 폭 **1.00** · 턱 폭 **{fr.get('jaw_w','-')}** · 턱 각도 **{fr.get('jaw_angle','-')}°**"
            )

            skin = adv.get("skin", {})
            if skin.get("available"):
                st.divider()
                st.subheader("🧴 피부 분석 (Face++ / Media.io 스타일)")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("매끄러움", f"{skin['smoothness']}/100")
                s2.metric("톤 균일도", f"{skin['tone_uniformity']}/100")
                s3.metric("밝기 (L)", f"{skin['brightness']}/100")
                s4.metric("언더톤", skin["undertone_kr"], f"강도 {skin['undertone_strength']:.0f}")
                with st.expander("↳ 원시 값"):
                    st.json(skin["raw"])

                pc = adv.get("personal_color", {})
                if pc.get("available"):
                    st.divider()
                    st.subheader("🎨 퍼스널 컬러 진단 (FaceScore TW 스타일)")
                    st.success(f"**{pc['label']}**")
                    st.write(pc["advice"])
                    st.caption(f"판정 근거: 언더톤 **{pc['undertone']}** × 명도 **{pc['depth']}**")

        st.divider()
        st.subheader("💬 개선 조언")
        for a in r["advice"]:
            st.write(a)

        st.divider()
        st.caption("⚠️ 상위 % 는 임시 기준입니다. 실측 데이터셋 기반 재보정 진행 중.")
        st.caption("본 서비스는 재미용이며, 성형 권유가 아닙니다. 시술은 반드시 전문의 상담 후 결정하세요.")
else:
    st.info("👆 위쪽에 얼굴 사진을 업로드하세요")
