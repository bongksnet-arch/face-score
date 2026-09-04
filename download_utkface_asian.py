"""UTKFace 데이터셋에서 Asian만 필터링하여 faces/asian/ 에 저장.
- 소스: HuggingFace nu-delta/utkface (auto-parquet, no auth)
- 이미지는 200x200 RGB
"""
import io
import os
import sys
import tempfile
import time

import requests
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URLS = [
    "https://huggingface.co/api/datasets/nu-delta/utkface/parquet/default/train/0.parquet",
    "https://huggingface.co/api/datasets/nu-delta/utkface/parquet/default/train/1.parquet",
    "https://huggingface.co/api/datasets/nu-delta/utkface/parquet/default/train/2.parquet",
]
OUT_DIR = "faces/asian"
TARGET = "Asian"
MIN_AGE = 18

os.makedirs(OUT_DIR, exist_ok=True)

saved = 0
scanned = 0
t0 = time.time()

for pi, url in enumerate(URLS):
    print(f"[{pi+1}/{len(URLS)}] downloading parquet: {url}")
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

        tbl = pq.read_table(tmp.name, columns=["image", "ethnicity", "age"])
        n = tbl.num_rows
        eth_col = tbl.column("ethnicity").to_pylist()
        age_col = tbl.column("age").to_pylist()
        img_col = tbl.column("image")
        print(f"    rows={n}, filtering ethnicity=={TARGET!r} AND age>={MIN_AGE}")

        for i in range(n):
            scanned += 1
            if eth_col[i] != TARGET:
                continue
            if age_col[i] is None or age_col[i] < MIN_AGE:
                continue
            item = img_col[i].as_py()
            img_bytes = item.get("bytes") if isinstance(item, dict) else None
            if not img_bytes:
                continue
            fname = os.path.join(OUT_DIR, f"asian_{saved:05d}.jpg")
            with open(fname, "wb") as f:
                f.write(img_bytes)
            saved += 1
            if saved % 200 == 0:
                print(f"    saved {saved} (scanned {scanned}, elapsed {time.time()-t0:.0f}s)")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

print(f"\n[DONE] saved {saved} Asian faces from {scanned} scanned rows")
print(f"[OUT ] {OUT_DIR}/ (elapsed {time.time()-t0:.0f}s)")
