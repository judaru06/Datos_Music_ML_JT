from pathlib import Path
import re
import numpy as np
from collections import defaultdict

# ===== CONFIG =====
BASE_DIR = Path(r"IQ_samps")
OUT_DIR  = Path(r"COV")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_SELECT   = 3          # nº de tomas a conservar por ángulo
SELECTION  = "power"    # "power" | "oldest" | "newest"
SEGMENTS   = 50         # dividir cada archivo en 50 partes -> 3*50 = 150 muestras/ángulo
REMOVE_MEAN = True      # quitar media por canal antes de cov (recomendado)

# patrón: IQ_ch{ch}_{YYYYMMDD}_{HHMMSS}_{ANGLE}.npy
pat = re.compile(r"^IQ_ch(?P<ch>\d+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<angle>[A-D]\d+)\.npy$", re.IGNORECASE)

# ===== 1) Indexar archivos → shots[(angle, stamp)][ch] = Path =====
shots = {}
for p in BASE_DIR.glob("*.npy"):
    m = pat.match(p.name)
    if not m:
        continue
    ch = int(m.group("ch"))
    angle = m.group("angle").upper()
    stamp = f"{m.group('date')}_{m.group('time')}"
    shots.setdefault((angle, stamp), {})
    shots[(angle, stamp)][ch] = p

# Agrupar por ángulo y filtrar tomas completas (4ch)
per_angle = defaultdict(list)
for (angle, stamp), chmap in shots.items():
    if len(chmap) == 4:
        per_angle[angle].append((stamp, chmap))

# ===== 2) Helpers =====
def load_X(chmap: dict):
    """Carga los 4 canales ch0..ch3 → X shape (4, N)."""
    X = np.vstack([np.load(chmap[ch]) for ch in sorted(chmap.keys())])
    return X

def cov_from_X(X: np.ndarray):
    """Rxx = X X^H / N → shape (M, M)."""
    return (X @ X.conj().T) / X.shape[1]

def power_metric(X: np.ndarray):
    """Potencia media ≈ mean(|X|^2)."""
    return float(np.mean(np.abs(X)**2))

def select_shots(items, k=3, mode="power"):
    """items = [(stamp, chmap), ...] → selecciona k tomas."""
    if mode == "oldest":
        return sorted(items, key=lambda t: t[0])[:k]
    if mode == "newest":
        return sorted(items, key=lambda t: t[0], reverse=True)[:k]
    if mode == "power":
        scored = []
        for stamp, chmap in items:
            X = load_X(chmap)
            scored.append((power_metric(X), stamp, chmap))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [(stamp, chmap) for _, stamp, chmap in scored[:k]]
    raise ValueError("SELECTION inválido")

# ===== 3) Construir dataset: dividir en 50 segmentos, cov por segmento, aplanar =====
rows_real = []     # lista de filas reales (N_total x 16)
rows_imag = []     # lista de filas imaginarias (N_total x 16)
labels    = []     # ángulo como texto (e.g., "D6")
meta      = []     # (angle, stamp, seg_idx) por si quieres rastrear

angles_sorted = sorted(per_angle.keys(), key=lambda a: (a[0], int(a[1:])))
angle_to_idx  = {ang:i for i, ang in enumerate(angles_sorted)}  # clase numérica

for angle in angles_sorted:
    items = per_angle[angle]
    if not items:
        continue
    selected = select_shots(items, k=min(K_SELECT, len(items)), mode=SELECTION)

    for stamp, chmap in selected:
        X = load_X(chmap)                 # (4, N)
        N = X.shape[1]
        L = N // SEGMENTS                 # largo por segmento
        if L == 0:
            continue
        # usar exactamente SEGMENTS segmentos no solapados
        for seg in range(SEGMENTS):
            s = seg * L
            e = s + L
            if e > N: break
            Xb = X[:, s:e].copy()         # (4, L)
            if REMOVE_MEAN:
                Xb -= Xb.mean(axis=1, keepdims=True)
            R = cov_from_X(Xb)            # (4,4)

            r_flat = R.reshape(-1)        # 16 complejos en orden fila
            rows_real.append(np.real(r_flat))
            rows_imag.append(np.imag(r_flat))
            labels.append(angle)
            meta.append((angle, stamp, seg))

# ===== 4) Empaquetar y guardar =====
X_real = np.vstack(rows_real).astype(np.float32)   # (samples, 16)
X_imag = np.vstack(rows_imag).astype(np.float32)   # (samples, 16)
y_txt  = np.array(labels)                          # (samples,)
y_idx  = np.array([angle_to_idx[a] for a in labels], dtype=np.int32)

DATA_DIR = OUT_DIR / "ML_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Formato NPZ (recomendado para ML en Python)
np.savez_compressed(
    DATA_DIR / "dataset_cov_4x4_segments50.npz",
    X_real=X_real, X_imag=X_imag,
    y_txt=y_txt, y_idx=y_idx,
    angle_to_idx=angle_to_idx,
    meta=np.array(meta, dtype=object)
)

# Opcional: CSV con columnas reales e imaginarias separadas
# Columnas: R00_re, R01_re, ..., R33_re, R00_im, ..., R33_im, label
col_re = [f"R{i}{j}_re" for i in range(4) for j in range(4)]
col_im = [f"R{i}{j}_im" for i in range(4) for j in range(4)]
header = ",".join(col_re + col_im + ["label"])
csv_path = DATA_DIR / "dataset_cov_4x4_segments50.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write(header + "\n")
    for k in range(X_real.shape[0]):
        row = list(X_real[k].tolist()) + list(X_imag[k].tolist()) + [y_txt[k]]
        f.write(",".join(map(str, row)) + "\n")

# ===== 5) Resumen =====
samples_per_angle = {}
for a in labels:
    samples_per_angle[a] = samples_per_angle.get(a, 0) + 1

print("\n=== RESUMEN DATASET ===")
for a in angles_sorted:
    print(f"{a}: {samples_per_angle.get(a, 0)} muestras")
print(f"\nTotal muestras: {len(labels)}")
print(f"NPZ: {DATA_DIR/'dataset_cov_4x4_segments50.npz'}")
print(f"CSV: {csv_path}")
