"""AFAD (Asian Face Aging Database) 성인만 필터링.
- 소스: github.com/John-niu-07/tarball (52 pieces of AFAD-Full.tar.xz)
- 165,501 Asian (Chinese-heavy) faces, age 15-40
- 폴더 구조 (extract 후): AFAD-Full/{age}/{gender}/{img.jpg}  (gender: 111=male, 112=female)
- 필터: age >= 18
- 출력: faces/afad/afad_XXXXXX.jpg
"""
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "https://raw.githubusercontent.com/John-niu-07/tarball/master/"
# Generate names xzaa..xzbt
def gen_pieces():
    # aa..az, ba..bt
    out = []
    for c1 in "ab":
        for c2 in "abcdefghijklmnopqrstuvwxyz":
            out.append(f"AFAD-Full.tar.xz{c1}{c2}")
            if c1 == "b" and c2 == "t":
                return out
    return out

PIECES = gen_pieces()  # 52 files: aa..az, ba..bt
OUT_DIR = "faces/afad"
MIN_AGE = 18

os.makedirs(OUT_DIR, exist_ok=True)

work = tempfile.mkdtemp(prefix="afad_")
combined = os.path.join(work, "AFAD-Full.tar.xz")

t0 = time.time()
try:
    print(f"[1/3] downloading {len(PIECES)} pieces to {combined}")
    with open(combined, "wb") as out:
        for i, name in enumerate(PIECES):
            url = BASE + name
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            n = 0
            for chunk in r.iter_content(1024 * 1024):
                out.write(chunk)
                n += len(chunk)
            print(f"    [{i+1}/{len(PIECES)}] {name}  {n/1024/1024:.1f} MB  (elapsed {time.time()-t0:.0f}s)")
    total = os.path.getsize(combined)
    print(f"    combined size {total/1024/1024:.1f} MB")

    print(f"[2/3] extracting + filtering age>={MIN_AGE}")
    saved = 0
    scanned = 0
    # tarfile can read xz transparently in Py3
    with tarfile.open(combined, mode="r:xz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            scanned += 1
            # path like AFAD-Full/25/111/12345-0.jpg
            parts = member.name.replace("\\", "/").split("/")
            if len(parts) < 3:
                continue
            try:
                age = int(parts[1])
            except ValueError:
                continue
            if age < MIN_AGE:
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            data = f.read()
            if not data:
                continue
            fname = os.path.join(OUT_DIR, f"afad_{saved:06d}.jpg")
            with open(fname, "wb") as out:
                out.write(data)
            saved += 1
            if saved % 2000 == 0:
                print(f"    saved {saved} (scanned {scanned}, elapsed {time.time()-t0:.0f}s)")
    print(f"[3/3] done. saved {saved} from {scanned} scanned entries")
finally:
    try:
        shutil.rmtree(work)
    except OSError:
        pass

print(f"\n[DONE] AFAD saved {saved} adult faces (elapsed {time.time()-t0:.0f}s)")
print(f"[OUT ] {OUT_DIR}/")
