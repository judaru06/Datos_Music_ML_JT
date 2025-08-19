# rf_uppertri.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("COV/ML_data")
NPZ_PATH = DATA_DIR / "dataset_cov_4x4_segments50_deglabels.npz"
CSV_PATH = DATA_DIR / "dataset_cov_4x4_segments50_deglabels.csv"

def load_dataset():
    if NPZ_PATH.exists():
        z = np.load(NPZ_PATH, allow_pickle=True)
        X_real = z["X_real"]  # (n,16)
        X_imag = z["X_imag"]  # (n,16)
        y_deg  = z["y_deg"].astype(float)
        return X_real, X_imag, y_deg
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        cols_re = [c for c in df.columns if c.endswith("_re")]
        cols_im = [c for c in df.columns if c.endswith("_im")]
        X_real = df[cols_re].to_numpy(dtype=np.float32)
        X_imag = df[cols_im].to_numpy(dtype=np.float32)
        y_deg  = df["angle_deg"].to_numpy(dtype=float)
        return X_real, X_imag, y_deg
    else:
        raise FileNotFoundError("No se encontró NPZ/CSV en COV/ML_data")

def uppertri_features(X_real, X_imag, M=4):
    """
    De (n,16)+(n,16) que representan R_real y R_imag 'row-major' (4x4),
    construye features únicas: diag(real) + upper_offdiag(real) + upper_offdiag(imag).
    Devuelve (n, 16).
    """
    n = X_real.shape[0]
    Rr = X_real.reshape(n, M, M)
    Ri = X_imag.reshape(n, M, M)
    # índices de triangular superior
    iu = np.triu_indices(M, k=1)  # off-diagonal
    diag = np.arange(M)

    diag_re = Rr[:, diag, diag]                  # (n,4)
    off_re  = Rr[:, iu[0], iu[1]]                # (n,6)
    off_im  = Ri[:, iu[0], iu[1]]                # (n,6)

    X_ut = np.hstack([diag_re, off_re, off_im])  # (n,16)
    return X_ut

def balance_by_angle(X, y):
    # undersampling al mínimo por ángulo
    rng = np.random.default_rng(42)
    uniq, counts = np.unique(y, return_counts=True)
    k = counts.min()
    idx_all = []
    for a in uniq:
        idx = np.where(y == a)[0]
        if len(idx) > k:
            idx = rng.choice(idx, size=k, replace=False)
        idx_all.append(idx)
    idx_all = np.concatenate(idx_all)
    return X[idx_all], y[idx_all]

# === Cargar y construir features ===
X_real, X_imag, y_deg = load_dataset()
X_ut = uppertri_features(X_real, X_imag)  # (n,16)

print("X_ut shape:", X_ut.shape, "| y_deg shape:", y_deg.shape)
# Comprobación rápida: imag(diagonal) ~ 0
# print(np.abs(X_imag.reshape(-1,4,4)[:, range(4), range(4)]).max())

# === Balancear por ángulo (opcional pero recomendado) ===
Xb, yb = balance_by_angle(X_ut, y_deg)
uniq, cnt = np.unique(yb, return_counts=True)
print("\nDistribución balanceada (ángulo -> n):")
print(pd.DataFrame({"angle_deg": uniq, "count": cnt}).to_string(index=False))

# === Split estratificado (usando ángulos como categorías) ===
X_tr, X_te, y_tr, y_te = train_test_split(
    Xb, yb, test_size=0.2, random_state=42, stratify=yb.astype(str)
)

# === RandomForestRegressor ===
rf = RandomForestRegressor(
    n_estimators=500, n_jobs=-1, random_state=42,
    min_samples_leaf=2
)
rf.fit(X_tr, y_tr)

pred_tr = rf.predict(X_tr)
pred_te = rf.predict(X_te)

rmse = lambda y,p: np.sqrt(((y-p)**2).mean())
print("\n== Métricas con triangular superior ==")
print(f"Train: MAE={mean_absolute_error(y_tr, pred_tr):.3f} | RMSE={rmse(y_tr, pred_tr):.3f}")
print(f"Test : MAE={mean_absolute_error(y_te, pred_te):.3f} | RMSE={rmse(y_te, pred_te):.3f}")

# Importancias (mapeo de nombres)
feat_names = (
    [f"R{i}{i}_re" for i in range(4)] +                     # diag real (4)
    [f"R{i}{j}_re" for i in range(4) for j in range(i+1,4)] +  # off real (6)
    [f"R{i}{j}_im" for i in range(4) for j in range(i+1,4)]    # off imag (6)
)
imp = pd.DataFrame({"feature": feat_names, "importance": rf.feature_importances_}) \
        .sort_values("importance", ascending=False)
print("\nTop 10 features:")
print(imp.head(10).to_string(index=False))
    