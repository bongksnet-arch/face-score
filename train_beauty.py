"""ArcFace embedding → beauty score 회귀 학습.

- Ridge / MLP 두 모델 5-fold CV 비교
- 최종 모델을 전체 데이터로 재학습해 beauty_model.pkl 저장
- Streamlit Cloud 배포용 최소 파일: beauty_model.pkl, 정규화 params
"""
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

NPZ = "scut_embeddings.npz"
OUT = "beauty_model.pkl"


def eval_model(model_name, get_model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mae_list, rmse_list, pc_list = [], [], []
    for fold, (tr, te) in enumerate(kf.split(X)):
        m = get_model()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        mae = mean_absolute_error(y[te], pred)
        rmse = np.sqrt(mean_squared_error(y[te], pred))
        pc, _ = pearsonr(y[te], pred)
        mae_list.append(mae); rmse_list.append(rmse); pc_list.append(pc)
        print(f"  [fold{fold+1}] MAE={mae:.4f} RMSE={rmse:.4f} PC={pc:.4f}")
    print(f"[{model_name}] avg MAE={np.mean(mae_list):.4f} "
          f"RMSE={np.mean(rmse_list):.4f} PC={np.mean(pc_list):.4f}")
    return np.mean(pc_list)


def main():
    data = np.load(NPZ, allow_pickle=True)
    X, y = data["X"], data["y"]
    print(f"[data] X={X.shape} y range [{y.min():.2f},{y.max():.2f}] mean={y.mean():.2f}")

    print("\n[Ridge]")
    ridge_pc = eval_model("Ridge",
        lambda: Ridge(alpha=1.0), X, y)

    print("\n[MLP-64]")
    mlp_pc = eval_model("MLP-64",
        lambda: MLPRegressor(hidden_layer_sizes=(64,), max_iter=500,
                             random_state=42, early_stopping=True), X, y)

    print("\n[MLP-128-64]")
    mlp2_pc = eval_model("MLP-128-64",
        lambda: MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                             random_state=42, early_stopping=True), X, y)

    # 최고 성능 모델을 전체 데이터로 재학습
    best_pc = max(ridge_pc, mlp_pc, mlp2_pc)
    if best_pc == mlp2_pc:
        final = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                             random_state=42, early_stopping=False)
        name = "MLP-128-64"
    elif best_pc == mlp_pc:
        final = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500,
                             random_state=42, early_stopping=False)
        name = "MLP-64"
    else:
        final = Ridge(alpha=1.0)
        name = "Ridge"
    print(f"\n[final] {name} on full data (PC={best_pc:.4f})")
    final.fit(X, y)

    # 저장: 모델 + 학습 데이터 통계 (score 정규화용)
    with open(OUT, "wb") as f:
        pickle.dump({
            "model": final,
            "model_name": name,
            "cv_pc": float(best_pc),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "y_mean": float(y.mean()),
            "y_std": float(y.std()),
            "n_train": int(len(y)),
        }, f)
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
