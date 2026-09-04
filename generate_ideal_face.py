"""만점(100/100) 기준 얼굴 비율 도식 생성.

app.py 의 채점 기준을 정확히 반영:
- 3분할: 1/3 : 1/3 : 1/3
- 좌우 대칭: 편차 0%
- 눈 폭 / 얼굴 폭 = 0.20 (Five-Eye Rule)
- 눈 간격 / 눈 폭 = 1.00
- 입 폭 / 얼굴 폭 = 0.38
- 얼굴 길이 / 폭 = 1.50
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyBboxPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches
import numpy as np

FW = 100.0                  # 얼굴 폭 (기준 단위)
FL = FW * 1.5               # 얼굴 길이 (=150) — 얼굴형 비율 1.5
EYE_W = FW * 0.20           # 눈 폭 =20
EYE_GAP = EYE_W * 1.00      # 눈 간격 =20
MOUTH_W = FW * 0.38         # 입 폭 =38

# Y 좌표: 이마 top = 0, 턱 bottom = 150 (아래 방향 +)
Y_FOREHEAD = 0.0
Y_GLABELLA = FL / 3.0       # 50  — 상/중 경계
Y_SUBNASALE = 2 * FL / 3.0  # 100 — 중/하 경계
Y_CHIN = FL                 # 150

# 얼굴 내부 요소 y 위치
Y_EYE = 62.0                # 중단 상부 (미간 아래)
Y_BROW = 47.0               # 미간 라인 살짝 위
Y_NOSE_TIP = 96.0           # 코밑선 근처
Y_MOUTH = 120.0             # 하단 상부 (2/5 지점)


def draw_face(ax, gender):
    """gender = 'male' | 'female'. 비율은 완전 동일, 외형 스타일만 차이."""
    # ---- 얼굴 윤곽 (타원) ----
    # 남자: 아래쪽 각지게, 여자: 아래쪽 부드럽게
    if gender == "male":
        face_color = "#e8d5b7"
        jaw_squareness = 0.92   # 아래쪽 폭 유지 (각진 턱)
        chin_pointy = 0.05
        hair_style = "short"
    else:
        face_color = "#f0d9c4"
        jaw_squareness = 0.78   # 아래쪽 좁게 (계란형)
        chin_pointy = 0.15
        hair_style = "medium"

    # 얼굴 외곽선을 상반부(원형)+하반부(테이퍼)로 구성
    theta = np.linspace(0, 2 * np.pi, 200)
    y = -FL / 2 * np.cos(theta) + FL / 2
    x = FW / 2 * np.sin(theta)
    # 아래쪽(y > FL/2) 폭을 gender 스타일로 축소
    lower_ratio = np.clip((y - FL / 2) / (FL / 2), 0.0, 1.0)
    exponent = 1.5 if gender == "female" else 1.0
    scale = 1.0 - (1.0 - jaw_squareness) * (lower_ratio ** exponent)
    x = x * scale
    # 턱 끝 뾰족하게
    tip_mask = y > FL * 0.9
    x[tip_mask] *= (1.0 - chin_pointy * ((y[tip_mask] - FL * 0.9) / (FL * 0.1)))

    ax.fill(x, y, color=face_color, zorder=1, edgecolor="#333", linewidth=1.5)

    # ---- 헤어 (외형용, 채점 무관) ----
    hair_theta = np.linspace(np.pi, 2 * np.pi, 80)
    if hair_style == "short":
        # 남자: 얼굴 위 반원 헤어캡
        hair_rx = FW / 2 * 0.95
        hair_ry = 22
        hair_x = hair_rx * np.cos(hair_theta)
        hair_y = hair_ry * np.sin(hair_theta) + 3
        ax.fill(hair_x, hair_y, color="#3a2418", zorder=2)
    else:
        # 여자: 반원 헤어캡 + 양옆으로 길게 내려오는 머리카락
        hair_rx = FW / 2 * 1.02
        hair_ry = 24
        cap_x = hair_rx * np.cos(hair_theta)
        cap_y = hair_ry * np.sin(hair_theta) + 3
        # 양옆 세로 흘림 (얼굴 뒤로 지나감)
        side_left_x = [-FW/2*1.02, -FW/2*1.05, -FW/2*0.95, -FW/2*0.85, -FW/2*0.65]
        side_left_y = [3, FL*0.35, FL*0.6, FL*0.8, FL*0.9]
        side_right_x = [-x for x in side_left_x][::-1]
        side_right_y = list(side_left_y)[::-1]
        # 전체 폐곡선: 좌측 아래 → 좌측 위 → 캡 → 우측 위 → 우측 아래
        px = side_left_x[::-1] + list(cap_x) + side_right_x
        py = side_left_y[::-1] + list(cap_y) + side_right_y
        ax.fill(px, py, color="#3a2418", zorder=0)  # 얼굴 뒤에 배치

    # ---- 3분할 라인 (황금비 만점 기준선) ----
    for y_line, label in [(Y_FOREHEAD, "이마 top"),
                          (Y_GLABELLA, "미간 (1/3)"),
                          (Y_SUBNASALE, "코밑 (2/3)"),
                          (Y_CHIN, "턱 bottom")]:
        ax.axhline(y_line, color="#d63384", linestyle="--", linewidth=0.8, alpha=0.6, zorder=3)
        ax.text(FW/2 + 8, y_line, label, fontsize=7, color="#d63384", va="center")

    # ---- 수직 중심선 (좌우 대칭 기준) ----
    ax.axvline(0, color="#0d6efd", linestyle=":", linewidth=0.7, alpha=0.5, zorder=3)

    # ---- 눈썹 ----
    for sign in [-1, 1]:
        cx = sign * (EYE_GAP / 2 + EYE_W / 2)
        brow_x = np.linspace(cx - EYE_W/2*1.1, cx + EYE_W/2*1.1, 20)
        brow_y = Y_BROW - 1.5 * np.sin(np.linspace(0, np.pi, 20)) * (1.3 if gender == "male" else 0.8)
        thickness = 3.0 if gender == "male" else 2.0
        ax.plot(brow_x, brow_y, color="#2a1810", linewidth=thickness, zorder=5)

    # ---- 눈 (아몬드 형태) ----
    for sign in [-1, 1]:
        cx = sign * (EYE_GAP / 2 + EYE_W / 2)
        eye = Ellipse((cx, Y_EYE), width=EYE_W, height=EYE_W * 0.42,
                      facecolor="white", edgecolor="#222", linewidth=1.3, zorder=5)
        ax.add_patch(eye)
        # 홍채
        iris = plt.Circle((cx, Y_EYE), EYE_W * 0.19, facecolor="#3a2818", zorder=6)
        ax.add_patch(iris)
        # 동공
        pupil = plt.Circle((cx, Y_EYE), EYE_W * 0.08, facecolor="black", zorder=7)
        ax.add_patch(pupil)

    # ---- 코 ----
    nose_x = [0, -3, -5, 0, 5, 3, 0]
    nose_y = [Y_GLABELLA + 5, Y_EYE + 8, Y_NOSE_TIP - 4, Y_NOSE_TIP, Y_NOSE_TIP - 4, Y_EYE + 8, Y_GLABELLA + 5]
    ax.plot(nose_x, nose_y, color="#8a6a52", linewidth=1.0, zorder=4)
    # 콧구멍
    for sign in [-1, 1]:
        ax.add_patch(plt.Circle((sign * 3, Y_NOSE_TIP + 1.5), 1.2, facecolor="#5a3a2a", zorder=5))

    # ---- 입 ----
    mouth_lip_h = 4.5 if gender == "female" else 3.2
    lip_top_x = np.linspace(-MOUTH_W/2, MOUTH_W/2, 30)
    lip_top_y = Y_MOUTH - mouth_lip_h * np.sin(np.linspace(0, np.pi, 30)) * 0.35
    lip_bot_x = lip_top_x[::-1]
    lip_bot_y = Y_MOUTH + mouth_lip_h * np.sin(np.linspace(0, np.pi, 30)) * 0.6
    lip_color = "#c94f6d" if gender == "female" else "#a05555"
    ax.fill(np.r_[lip_top_x, lip_bot_x], np.r_[lip_top_y, lip_bot_y],
            color=lip_color, edgecolor="#5a1f2f", linewidth=0.8, zorder=5)
    ax.plot([-MOUTH_W/2, MOUTH_W/2], [Y_MOUTH, Y_MOUTH], color="#5a1f2f", linewidth=0.6, zorder=6)

    # ---- 치수 주석 (왼쪽) ----
    # 3분할 편차 표시 (완전 균등)
    ax.annotate('', xy=(-FW/2 - 20, Y_FOREHEAD), xytext=(-FW/2 - 20, Y_GLABELLA),
                arrowprops=dict(arrowstyle='<->', color='#d63384'))
    ax.text(-FW/2 - 26, Y_GLABELLA / 2, "1/3", fontsize=9, color='#d63384',
            rotation=90, va='center', ha='center', fontweight='bold')
    ax.annotate('', xy=(-FW/2 - 20, Y_GLABELLA), xytext=(-FW/2 - 20, Y_SUBNASALE),
                arrowprops=dict(arrowstyle='<->', color='#d63384'))
    ax.text(-FW/2 - 26, (Y_GLABELLA + Y_SUBNASALE) / 2, "1/3", fontsize=9, color='#d63384',
            rotation=90, va='center', ha='center', fontweight='bold')
    ax.annotate('', xy=(-FW/2 - 20, Y_SUBNASALE), xytext=(-FW/2 - 20, Y_CHIN),
                arrowprops=dict(arrowstyle='<->', color='#d63384'))
    ax.text(-FW/2 - 26, (Y_SUBNASALE + Y_CHIN) / 2, "1/3", fontsize=9, color='#d63384',
            rotation=90, va='center', ha='center', fontweight='bold')

    # 눈 폭·간격 표시
    y_eye_dim = Y_EYE + 16
    for sign in [-1, 1]:
        cx = sign * (EYE_GAP / 2 + EYE_W / 2)
        ax.annotate('', xy=(cx - EYE_W/2, y_eye_dim), xytext=(cx + EYE_W/2, y_eye_dim),
                    arrowprops=dict(arrowstyle='<->', color='#198754', lw=0.8))
    ax.annotate('', xy=(-EYE_GAP/2, y_eye_dim), xytext=(EYE_GAP/2, y_eye_dim),
                arrowprops=dict(arrowstyle='<->', color='#198754', lw=0.8))
    ax.text(0, y_eye_dim + 3, "간격=눈폭 (1.00)", fontsize=7, color='#198754', ha='center')
    ax.text(-(EYE_GAP/2 + EYE_W/2), y_eye_dim + 3, "눈 폭\n=얼굴폭×0.20", fontsize=7,
            color='#198754', ha='center')

    # 입 폭 표시
    ax.annotate('', xy=(-MOUTH_W/2, Y_MOUTH + 12), xytext=(MOUTH_W/2, Y_MOUTH + 12),
                arrowprops=dict(arrowstyle='<->', color='#fd7e14', lw=0.8))
    ax.text(0, Y_MOUTH + 15, "입 폭 = 얼굴폭×0.38", fontsize=7, color='#fd7e14', ha='center')

    # 얼굴 폭·길이 (오른쪽)
    ax.annotate('', xy=(FW/2 + 42, 0), xytext=(FW/2 + 42, FL),
                arrowprops=dict(arrowstyle='<->', color='#6f42c1', lw=1.0))
    ax.text(FW/2 + 46, FL/2, f"길이 = 폭×1.50", fontsize=8, color='#6f42c1',
            rotation=90, va='center', fontweight='bold')

    # ---- 축 설정 ----
    ax.set_xlim(-FW/2 - 40, FW/2 + 70)
    ax.set_ylim(FL + 15, -30)  # y 뒤집기 (위가 이마)
    ax.set_aspect('equal')
    ax.axis('off')

    title = "만점 기준 남성 얼굴" if gender == "male" else "만점 기준 여성 얼굴"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)


# ---- 그리기 ----
plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
draw_face(axes[0], "male")
draw_face(axes[1], "female")

fig.suptitle("얼굴 점수 AI — 100/100 만점 기준 비율 도식",
             fontsize=15, fontweight='bold', y=0.98)
fig.text(0.5, 0.02,
         "※ 실존 인물 사진이 아닌 채점 기준(3분할·Five-Eye·황금비 등)을 정확히 반영한 기하학적 도식입니다. "
         "이 비율은 고전 미술 캐논이며 실제 미의 실증 데이터는 아닙니다.",
         ha='center', fontsize=8, color='#666', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 0.95])
out = "ideal_face_reference.png"
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
print(f"저장 완료: {out}")
