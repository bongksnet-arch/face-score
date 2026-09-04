"""metrics.csv → population_stats.json 산출.

각 지표의 mean/std + 5/25/50/75/95 백분위 계산.
Phase 6에서 face_analysis 의 score_* 함수가 이 파일을 로드해 백분위 기반 채점.

사용:
  python compute_stats.py                     # metrics.csv 전체
  python compute_stats.py --folder east_asian # 특정 폴더만
  python compute_stats.py --exclude-bad-pose  # bad_pose 제외 (권장)
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS_CSV = Path(__file__).parent / "metrics.csv"
OUT_JSON = Path(__file__).parent / "population_stats.json"

# 기본 (in) 및 (out) CLI 로 오버라이드 가능

METRIC_COLS = [
    # 기본 8
    "gr_upper", "gr_middle", "gr_lower", "sym_dev",
    "eye_ratio", "gap_ratio", "mouth_ratio", "shape_ratio",
    # 확장 11
    "eye_aspect", "brow_arch", "brow_eye_gap",
    "nose_width_ratio", "nose_length_ratio",
    "lip_ratio", "upper_lip_thickness", "lower_lip_thickness",
    "jaw_angle", "cheek_ratio", "philtrum_ratio",
]

PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=None, help="특정 폴더만 (aaf/afad/east_asian)")
    ap.add_argument("--exclude-bad-pose", action="store_true",
                    help="pose가 큰 사진 제외 (권장, 통계 오염 방지)")
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--in", dest="input", default=str(METRICS_CSV))
    args = ap.parse_args()

    in_csv = Path(args.input)
    if not in_csv.exists():
        print(f"[!] {in_csv} 없음. analyze_dataset.py 를 먼저 실행하세요.")
        return

    df = pd.read_csv(in_csv)
    print(f"[i] 총 {len(df):,} 행 로드")
    print(f"[i] status 분포:")
    for s, c in df["status"].value_counts().items():
        print(f"    {s:20s}: {c:>7,}")

    # 필터링
    ok_mask = df["status"] == "ok"
    if args.exclude_bad_pose:
        keep = ok_mask
    else:
        keep = ok_mask | (df["status"] == "bad_pose")
    if args.folder:
        keep &= (df["folder"] == args.folder)
    dfk = df[keep].copy()
    print(f"[i] 통계 대상: {len(dfk):,} 행 (folder={args.folder or 'all'}, exclude_bad_pose={args.exclude_bad_pose})")

    if len(dfk) == 0:
        print("[!] 유효 데이터 없음.")
        return

    # 각 지표 통계
    stats = {
        "meta": {
            "n_samples": int(len(dfk)),
            "folder": args.folder or "all",
            "exclude_bad_pose": bool(args.exclude_bad_pose),
        }
    }
    for col in METRIC_COLS:
        v = dfk[col].dropna().values
        if len(v) == 0:
            continue
        s = {
            "n": int(len(v)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }
        for p in PERCENTILES:
            s[f"p{p:02d}"] = float(np.percentile(v, p))
        stats[col] = s

    # 저장
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[i] 저장: {args.out}")

    # 콘솔 리포트
    print("\n=== 지표별 분포 요약 ===")
    print(f"{'지표':<15} {'mean':>8} {'std':>8} {'p05':>8} {'p50':>8} {'p95':>8}")
    print("-" * 60)
    for col in METRIC_COLS:
        if col not in stats:
            continue
        s = stats[col]
        print(f"{col:<15} {s['mean']:>8.4f} {s['std']:>8.4f} "
              f"{s['p05']:>8.4f} {s['p50']:>8.4f} {s['p95']:>8.4f}")


if __name__ == "__main__":
    main()
