"""SCUT-FBP5500 parquet → ArcFace 512-d embedding + beauty score 추출.

HuggingFace 데이터셋(train/test parquet) 로드 → insightface buffalo_l 로
각 얼굴 embedding + beauty_score 저장 (scut_embeddings.npz).
"""
import io
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from insightface.app import FaceAnalysis

DATA_DIR = Path("scut_fbp5500/data")
OUT = "scut_embeddings.npz"


def main():
    parquets = sorted(DATA_DIR.glob("*.parquet"))
    print(f"[data] parquet files: {[p.name for p in parquets]}")

    print("[model] loading insightface buffalo_l ...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    X, y, names, races, genders, splits = [], [], [], [], [], []
    fails = 0
    total_rows = sum(pq.read_metadata(p).num_rows for p in parquets)
    print(f"[data] total rows: {total_rows}")
    t0 = time.time()
    idx = 0
    for pq_path in parquets:
        split = "test" if "test" in pq_path.name else "train"
        table = pq.read_table(pq_path)
        img_col = table.column("image").to_pylist()
        score_col = table.column("beauty_score").to_pylist()
        name_col = table.column("image_name").to_pylist()
        race_col = table.column("race").to_pylist()
        gender_col = table.column("gender").to_pylist()
        n = len(img_col)
        print(f"[data] {pq_path.name}: {n} rows")
        for i in range(n):
            idx += 1
            img_dict = img_col[i]
            img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                fails += 1
                continue
            faces = app.get(img)
            if not faces:
                fails += 1
                continue
            faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                       reverse=True)
            emb = faces[0].embedding.astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-9)
            X.append(emb)
            y.append(float(score_col[i]))
            names.append(str(name_col[i]))
            races.append(str(race_col[i]))
            genders.append(str(gender_col[i]))
            splits.append(split)
            if idx % 100 == 0 or idx == total_rows:
                el = time.time() - t0
                rate = idx / el if el > 0 else 0
                eta = (total_rows - idx) / rate if rate > 0 else 0
                print(f"[{idx:5d}/{total_rows}] ok={len(X)} fail={fails} "
                      f"rate={rate:.1f}/s eta={eta/60:.1f}min")

    X = np.stack(X)
    y = np.array(y, dtype=np.float32)
    np.savez_compressed(OUT, X=X, y=y, names=np.array(names),
                        races=np.array(races), genders=np.array(genders),
                        splits=np.array(splits))
    print(f"[save] {OUT}: X={X.shape}, y range [{y.min():.2f},{y.max():.2f}], "
          f"fails={fails}, time={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
