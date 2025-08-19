# plot_distribucion_angulos.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("COV/ML_data")
NPZ_PATH = DATA_DIR / "dataset_cov_4x4_segments50_deglabels.npz"
CSV_PATH = DATA_DIR / "dataset_cov_4x4_segments50_deglabels.csv"

def cargar():
    if NPZ_PATH.exists():
        z = np.load(NPZ_PATH, allow_pickle=True)
        y = z["y_deg"].astype(float)
        return y
    df = pd.read_csv(CSV_PATH)
    return df["angle_deg"].to_numpy(dtype=float)

y = cargar()

# --- Distribución original ---
u, c = np.unique(y, return_counts=True)
df1 = pd.DataFrame({"angle_deg": u, "count": c}).sort_values("angle_deg")

plt.figure(figsize=(10,4))
plt.bar(df1["angle_deg"].astype(str), df1["count"])
plt.title("Distribución por ángulo (antes de balancear)")
plt.xlabel("Ángulo (°)")
plt.ylabel("N muestras")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(DATA_DIR/"dist_angulos_before.png", dpi=200)
plt.show()  # ¡importante para que aparezca!

# --- Distribución balanceada por undersampling ---
rng = np.random.default_rng(42)
min_per_class = c.min()
idx_all = []
for ang in u:
    idx = np.where(y == ang)[0]
    if len(idx) > min_per_class:
        idx = rng.choice(idx, size=min_per_class, replace=False)
    idx_all.append(idx)
idx_all = np.concatenate(idx_all)
y_bal = y[idx_all]

u2, c2 = np.unique(y_bal, return_counts=True)
df2 = pd.DataFrame({"angle_deg": u2, "count": c2}).sort_values("angle_deg")

plt.figure(figsize=(10,4))
plt.bar(df2["angle_deg"].astype(str), df2["count"])
plt.title(f"Distribución por ángulo (balanceada a {min_per_class} c/u)")
plt.xlabel("Ángulo (°)")
plt.ylabel("N muestras")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(DATA_DIR/"dist_angulos_after.png", dpi=200)
plt.show()
print("Figuras guardadas en:", DATA_DIR)
