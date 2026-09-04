"""AI 생성 얼굴 다운로드 (thispersondoesnotexist.com)
- 실존 인물 아님 → 초상권/개인정보 문제 없음
- N장 다운로드해서 faces/ 폴더에 저장
"""
import os
import time
import sys

try:
    import requests
except ImportError:
    print("[!] requests 라이브러리가 필요합니다. 설치 중...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

N = 500  # 다운로드 개수 (원하면 조정)
OUT_DIR = "faces"
URL = "https://thispersondoesnotexist.com/random-person.jpeg"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

os.makedirs(OUT_DIR, exist_ok=True)

# 이미 다운받은 것 개수 세기
existing = len([f for f in os.listdir(OUT_DIR) if f.endswith('.jpg')])
if existing > 0:
    print(f"[i] 기존 {existing}장 발견. {N - existing}장 추가 다운로드.")
start = existing

session = requests.Session()
session.headers.update(HEADERS)

success = 0
fail = 0
for i in range(start, N):
    try:
        r = session.get(URL, timeout=15)
        if r.status_code != 200 or len(r.content) < 10000:
            fail += 1
            print(f"  [!] {i+1}/{N} 실패 (status={r.status_code}, size={len(r.content)})")
            time.sleep(2)
            continue
        fname = f"{OUT_DIR}/face_{i:04d}.jpg"
        with open(fname, "wb") as f:
            f.write(r.content)
        success += 1
        print(f"  [✓] {i+1}/{N} 저장 ({len(r.content)//1024}KB)")
        # 서버에 부담 안 주기 위해 대기 (매 요청마다 새 이미지 생성)
        time.sleep(1.2)
    except Exception as e:
        fail += 1
        print(f"  [!] {i+1}/{N} 에러: {e}")
        time.sleep(3)

print(f"\n[완료] 성공 {success}장 / 실패 {fail}장")
print(f"[i] {OUT_DIR}/ 폴더에 저장됨")
