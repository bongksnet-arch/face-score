"""AAF (All-Age-Faces Dataset) 성인만 필터링.
- 소스: Dropbox mirror. 13,322 mostly Asian faces, age 2-80.
- 파일명 형식: %05dA%02d.jpg (person_id + age)
- 필터: age >= 18
- 출력: faces/aaf/aaf_XXXXX.jpg
"""
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "https://www.dropbox.com/s/a0lj1ddd54ns8qy/All-Age-Faces%20Dataset.zip?dl=1"
OUT_DIR = "faces/aaf"
MIN_AGE = 18
NAME_RE = re.compile(r"^\d{5}A(\d{2})\.jpg$", re.IGNORECASE)

os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()
tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
tmp_path = tmp.name
try:
    print(f"[1/3] downloading: {URL}")
    with requests.get(URL, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        total = 0
        for chunk in r.iter_content(1024 * 1024):
            tmp.write(chunk)
            total += len(chunk)
            if total % (50 * 1024 * 1024) < 1024 * 1024:
                print(f"    downloaded {total/1024/1024:.0f} MB...")
    tmp.close()
    print(f"    total {total/1024/1024:.1f} MB (elapsed {time.time()-t0:.0f}s)")

    print("[2/3] scanning zip contents")
    saved = 0
    scanned = 0
    with zipfile.ZipFile(tmp_path) as zf:
        names = zf.namelist()
        print(f"    zip entries: {len(names)}")
        for name in names:
            base = os.path.basename(name)
            m = NAME_RE.match(base)
            if not m:
                continue
            scanned += 1
            age = int(m.group(1))
            if age < MIN_AGE:
                continue
            data = zf.read(name)
            out = os.path.join(OUT_DIR, f"aaf_{saved:05d}.jpg")
            with open(out, "wb") as f:
                f.write(data)
            saved += 1
            if saved % 500 == 0:
                print(f"    saved {saved} (scanned {scanned})")
    print(f"[3/3] done. saved {saved} from {scanned} matched entries")
finally:
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

print(f"\n[DONE] AAF saved {saved} adult faces (elapsed {time.time()-t0:.0f}s)")
print(f"[OUT ] {OUT_DIR}/")
