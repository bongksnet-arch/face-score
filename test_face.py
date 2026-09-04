import cv2
import mediapipe as mp
import sys
import os

INPUT = "face.jpg"
OUTPUT = "face_result.jpg"

if not os.path.exists(INPUT):
    print(f"[!] '{INPUT}' 파일이 없습니다. 얼굴 사진을 이 폴더에 face.jpg 이름으로 넣어주세요.")
    sys.exit(1)

from face_analysis import imread_with_exif
img = imread_with_exif(INPUT)
if img is None:
    print("[!] 이미지를 읽을 수 없습니다. 파일 형식을 확인하세요.")
    sys.exit(1)

h, w = img.shape[:2]
print(f"[i] 이미지 로드 성공: {w}x{h}")

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
        print("[!] 얼굴을 찾지 못했습니다. 정면 얼굴 사진으로 다시 시도하세요.")
        sys.exit(1)

    landmarks = result.multi_face_landmarks[0].landmark
    print(f"[✓] 얼굴 랜드마크 {len(landmarks)}개 검출 성공!")

    radius = max(2, int(min(w, h) / 400))
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (x, y), radius, (0, 255, 0), -1)

cv2.imwrite(OUTPUT, img)
print(f"[✓] 결과 저장: {OUTPUT}")
print("[i] 폴더에서 face_result.jpg 열어보세요.")
