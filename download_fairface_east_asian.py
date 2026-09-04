"""FairFace 데이터셋에서 East Asian 성인만 추출.
- 소스: HuggingFace HuggingFaceM4/FairFace, config 0.25 (tight crop)
- 필터: race == "East Asian" AND age not in {"0-2","3-9","10-19"}
- 출력: faces/east_asian/east_asian_XXXXX.jpg
"""
import os
import sys
import tempfile
import time

import requests
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URLS = [
    "https://huggingface.co/api/datasets/HuggingFaceM4/FairFace/parquet/0.25/train/0.parquet",
    "https://huggingface.co/api/datasets/HuggingFaceM4/FairFace/parquet/0.25/train/1.parquet",
    "https://huggingface.co/api/datasets/HuggingFaceM4/FairFace/parquet/0.25/validation/0.parquet",
]
OUT_DIR = "faces/east_asian"
TARGET_RACE = "East Asian"
EXCLUDE_AGE = {"0-2", "3-9", "10-19"}

RACE_MAP = {0: "East Asian", 1: "Indian", 2: "Black", 3: "White",
            4: "Middle Eastern", 5: "Latino_Hispanic", 6: "Southeast Asian"}
AGE_MAP = {0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39",
           5: "40-49", 6: "50-59", 7: "60-69", 8: "more than 70"}

os.makedirs(OUT_DIR, exist_ok=True)

def to_str(val, mapping):
    if isinstance(val, int):
        return mapping.get(val, str(val))
    return val

saved = 0
scanned = 0
t0 = time.time()

for pi, url in enumerate(URLS):
    print(f"[{pi+1}/{len(URLS)}] downloading: {url}")
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = 0
            for chunk in r.iter_content(1024 * 1024):
                tmp.write(chunk)
                total += len(chunk)
            print(f"    downloaded {total/1024/1024:.1f} MB")
        tmp.close()

        tbl = pq.read_table(tmp.name, columns=["image", "race", "age"])
        n = tbl.num_rows
        race_col = tbl.column("race").to_pylist()
        age_col = tbl.column("age").to_pylist()
        img_col = tbl.column("image")
        print(f"    rows={n}, filtering race=={TARGET_RACE!r} AND age not in {EXCLUDE_AGE}")

        for i in range(n):
            scanned += 1
            race = to_str(race_col[i], RACE_MAP)
            age = to_str(age_col[i], AGE_MAP)
            if race != TARGET_RACE:
                continue
            if age in EXCLUDE_AGE:
                continue
            item = img_col[i].as_py()
            img_bytes = item.get("bytes") if isinstance(item, dict) else None
            if not img_bytes:
                continue
            fname = os.path.join(OUT_DIR, f"east_asian_{saved:05d}.jpg")
            with open(fname, "wb") as f:
                f.write(img_bytes)
            saved += 1
            if saved % 500 == 0:
                print(f"    saved {saved} (scanned {scanned}, elapsed {time.time()-t0:.0f}s)")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

print(f"\n[DONE] saved {saved} East Asian adult faces from {scanned} scanned rows")
print(f"[OUT ] {OUT_DIR}/ (elapsed {time.time()-t0:.0f}s)")
