"""faces/ 폴더 전체를 파이프라인으로 처리해 원시 지표 CSV 생성.

사용:
  python analyze_dataset.py --pilot                    # 각 폴더 500장 파일럿
  python analyze_dataset.py --all                      # 전수 (수 시간)
  python analyze_dataset.py --folder east_asian        # 특정 폴더만
  python analyze_dataset.py --workers 8 --limit 5000   # 병렬 8개, 5000장

출력: metrics.csv (append 모드, 재실행 시 이미 처리한 파일 skip)
"""
import os
import sys
import csv
import time
import argparse
import random
import multiprocessing as mp
from pathlib import Path

FACES_DIR = Path(__file__).parent / "faces"
OUT_CSV = Path(__file__).parent / "metrics.csv"

CSV_FIELDS = [
    "path", "folder",
    # 기본 8개
    "gr_upper", "gr_middle", "gr_lower", "sym_dev",
    "eye_ratio", "gap_ratio", "mouth_ratio", "shape_ratio",
    # 확장 11개
    "eye_aspect", "brow_arch", "brow_eye_gap",
    "nose_width_ratio", "nose_length_ratio",
    "lip_ratio", "upper_lip_thickness", "lower_lip_thickness",
    "jaw_angle", "cheek_ratio", "philtrum_ratio",
    # 메타
    "pose_yaw", "pose_pitch", "pose_roll", "det_conf", "auto_rot",
    "status",
]


def _process_one(path_str):
    """워커 함수: 이미지 하나 처리 → dict 리턴."""
    import cv2
    import numpy as np
    import face_analysis as fa
    path = Path(path_str)
    folder = path.parent.name
    try:
        # Windows 한글/유니코드 경로 + EXIF 회전 통일 (bytes_to_bgr 경유)
        with open(path, "rb") as f:
            data = f.read()
        img = fa.bytes_to_bgr(data)
        if img is None:
            return {"path": str(path), "folder": folder, "status": "read_error"}
        m = fa.compute_metrics(img)
        if m is None:
            return {"path": str(path), "folder": folder, "status": "no_face"}
        # 큰 pose 편차는 통계 오염 → 표시하되 status로 별도 마킹
        status = "ok"
        if abs(m["pose_yaw"]) > 25 or abs(m["pose_pitch"]) > 20:
            status = "bad_pose"
        return {"path": str(path), "folder": folder, "status": status, **m}
    except Exception as e:
        return {"path": str(path), "folder": folder, "status": f"error:{type(e).__name__}"}


def _load_done_paths(out_csv=None):
    """이미 처리된 path 셋 로드 (재실행 시 중복 방지)."""
    if out_csv is None:
        out_csv = OUT_CSV
    if not out_csv.exists():
        return set()
    done = set()
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            done.add(row["path"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="각 폴더 랜덤 500장")
    ap.add_argument("--all", action="store_true", help="전수 처리")
    ap.add_argument("--folder", default=None, help="특정 폴더만 (aaf/afad/east_asian)")
    ap.add_argument("--limit", type=int, default=None, help="총 처리 개수 제한")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--reset", action="store_true", help="출력 CSV 초기화 후 시작")
    ap.add_argument("--out", default=str(OUT_CSV), help="출력 CSV 경로")
    args = ap.parse_args()

    out_csv = Path(args.out)
    if args.reset and out_csv.exists():
        out_csv.unlink()
        print(f"[i] {out_csv.name} 초기화")

    # 대상 파일 수집
    folders = [args.folder] if args.folder else ["aaf", "afad", "east_asian"]
    all_paths = []
    for fname in folders:
        fdir = FACES_DIR / fname
        if not fdir.exists():
            print(f"[!] {fdir} 없음, skip")
            continue
        files = sorted([str(p) for p in fdir.glob("*.jpg")])
        if args.pilot:
            random.Random(42).shuffle(files)
            files = files[:500]
        all_paths.extend(files)

    print(f"[i] 총 대상: {len(all_paths):,}장 ({len(folders)}개 폴더)")

    # 이미 처리한 것 skip
    done = _load_done_paths(out_csv)
    if done:
        print(f"[i] 기존 처리 완료: {len(done):,}장 → skip")
    todo = [p for p in all_paths if p not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[i] 처리 대상: {len(todo):,}장 · 워커: {args.workers}")

    if not todo:
        print("[i] 처리할 파일 없음. 종료.")
        return

    # CSV 열기 (append 모드)
    is_new = not out_csv.exists()
    fout = open(out_csv, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if is_new:
        writer.writeheader()

    # 병렬 실행
    t0 = time.time()
    n_ok = n_fail = 0
    status_count = {}
    try:
        with mp.Pool(args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_process_one, todo, chunksize=8), 1):
                writer.writerow(result)
                st = result.get("status", "?")
                status_count[st] = status_count.get(st, 0) + 1
                if st == "ok":
                    n_ok += 1
                else:
                    n_fail += 1
                if i % 100 == 0 or i == len(todo):
                    fout.flush()
                    elapsed = time.time() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(todo) - i) / rate if rate > 0 else 0
                    print(f"[{i:>7,}/{len(todo):,}] "
                          f"ok={n_ok:,} fail={n_fail:,} · "
                          f"{rate:.1f}장/s · ETA {eta/60:.1f}분")
    finally:
        fout.close()

    print(f"\n=== 완료 ({(time.time()-t0)/60:.1f}분) ===")
    for st, c in sorted(status_count.items(), key=lambda x: -x[1]):
        print(f"  {st:20s}: {c:>7,}")


if __name__ == "__main__":
    main()
