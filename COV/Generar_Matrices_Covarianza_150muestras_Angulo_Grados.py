from pathlib import Path
import re
import numpy as np
from collections import defaultdict
import pandas as pd

# ===== CONFIG =====
BASE_DIR = Path(r"IQ_samps")
OUT_DIR  = Path(r"COV")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_SELECT    = 3          # nº de tomas a conservar por ángulo
SELECTION   = "power"    # "power" | "oldest" | "newest"
SEGMENTS    = 50         # dividir cada archivo en 50 partes -> 3*50 = 150 muestras/ángulo
REMOVE_MEAN = True       # quitar media por canal antes de cov (recomendado)

# === Mapa de etiquetas -> grados (desde Angulos.csv) ===
ANG_CSV = Path("Angulos.csv")  # pon la ruta si está en otro sitio
df = pd.read_csv(ANG_CSV)
# poner "Unnamed: 0" como índice ("col 0"..."col 8")
if "Unnamed: 0" in df.columns:
    df = df.set_index("Unnamed: 0")
df.index = df.index.astype(str)  # asegurar strings tipo "col 0"

def label_to_degrees(label: str) -> float:
    """
    Convierte 'A0'..'D8' al ángulo en grados usando Angulos.csv.
    'A'..'D' = fila en tu croquis (columnas del CSV),
    número 0..8 = columna (filas 'col 0'..'col 8' en el CSV).
    """
    label = label.strip().upper()
    letter = label[0]           # A/B/C/D
    idx = int(label[1:])        # 0..8
    key = f"col {idx}"
    try:
        return float(df.loc[key, letter])
    except Exception as e:
        raise ValueError(f"No encuentro ángulo para etiqueta {label} (busqué df.loc['{key}','{letter}'])") from e

# patrón: IQ_ch{ch}_{YYYYMMDD}_{HHMMSS}_{ANGLE}.npy
pat = re.compile(r"^IQ_ch(?P<ch>\d+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<angle>[A-D]\d+)\.npy$", re.IGNORECASE)

# ===== 1) Indexar archivos → shots[(angle_txt, stamp)][ch] = Path =====
shots = {}
for p in BASE_DIR.glob("*.npy"):
    m = pat.match(p.name)
    if not m:
        continue
    ch = int(m.group("ch"))
    angle_txt = m.group("angle").upper()         # 'A0'..'D8'
    stamp = f"{m.group('date')}_{m.group('time')}"
    shots.setdefault((angle_txt, stamp), {})
    shots[(angle_txt, stamp)][ch] = p

# Agrupar por etiqueta y filtrar tomas completas (4ch)
per_angle = defaultdict(list)  # clave: 'A0'..'D8'
for (angle_txt, stamp), chmap in shots.items():
    if len(chmap) == 4:
        per_angle[angle_txt].append((stamp, chmap))

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

# ===== 3) Construir dataset: 50 segmentos por toma, cov por segmento, aplanar =====
rows_real = []     # (n_samples, 16)
rows_imag = []     # (n_samples, 16)
y_deg     = []     # etiqueta numérica en grados
y_txt     = []     # 'A0'..'D8'
meta      = []     # (angle_txt, stamp, seg)

angles_sorted_txt = sorted(per_angle.keys(), key=lambda a: (a[0], int(a[1:])))  # 'A0'..'D8' ordenadas

for angle_txt in angles_sorted_txt:
    items = per_angle[angle_txt]
    if not items:
        continue
    selected = select_shots(items, k=min(K_SELECT, len(items)), mode=SELECTION)

    # convertir etiqueta textual a grados una sola vez
    angle_deg_value = label_to_degrees(angle_txt)

    for stamp, chmap in selected:
        X = load_X(chmap)                 # (4, N)
        N = X.shape[1]
        L = N // SEGMENTS                 # largo por segmento
        if L == 0:
            continue
        for seg in range(SEGMENTS):
            s = seg * L
            e = s + L
            if e > N:
                break
            Xb = X[:, s:e].copy()         # (4, L)
            if REMOVE_MEAN:
                Xb -= Xb.mean(axis=1, keepdims=True)
            R = cov_from_X(Xb)            # (4,4)

            r_flat = R.reshape(-1)        # 16 complejos
            rows_real.append(np.real(r_flat))
            rows_imag.append(np.imag(r_flat))
            y_deg.append(angle_deg_value) # etiqueta en grados
            y_txt.append(angle_txt)       # por si quieres rastrear
            meta.append((angle_txt, stamp, seg))

# ===== 4) Empaquetar y guardar =====
X_real = np.vstack(rows_real).astype(np.float32)   # (samples, 16)
X_imag = np.vstack(rows_imag).astype(np.float32)   # (samples, 16)
y_deg  = np.array(y_deg, dtype=np.float32)         # (samples,)
y_txt  = np.array(y_txt)

DATA_DIR = OUT_DIR / "ML_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# NPZ con todo
np.savez_compressed(
    DATA_DIR / "dataset_cov_4x4_segments50_deglabels.npz",
    X_real=X_real, X_imag=X_imag,
    y_deg=y_deg, y_txt=y_txt,
    meta=np.array(meta, dtype=object)
)

# CSV: columnas reales + imaginarias + ángulo (grados)
col_re = [f"R{i}{j}_re" for i in range(4) for j in range(4)]
col_im = [f"R{i}{j}_im" for i in range(4) for j in range(4)]
header = ",".join(col_re + col_im + ["angle_deg"])
csv_path = DATA_DIR / "dataset_cov_4x4_segments50_deglabels.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write(header + "\n")
    for k in range(X_real.shape[0]):
        row = list(X_real[k].tolist()) + list(X_imag[k].tolist()) + [str(y_deg[k])]
        f.write(",".join(map(str, row)) + "\n")

# Resumen
samples_per_angle = {}
for a in y_txt:
    samples_per_angle[a] = samples_per_angle.get(a, 0) + 1

print("\n=== RESUMEN DATASET ===")
for a in angles_sorted_txt:
    print(f"{a} ({label_to_degrees(a):6.2f}°): {samples_per_angle.get(a, 0)} muestras")
print(f"\nTotal muestras: {len(y_deg)}")
print(f"NPZ: {DATA_DIR/'dataset_cov_4x4_segments50_deglabels.npz'}")
print(f"CSV: {csv_path}")
