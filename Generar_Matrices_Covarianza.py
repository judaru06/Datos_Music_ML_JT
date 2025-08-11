from pathlib import Path
import re
import numpy as np
from collections import defaultdict

# ===== CONFIG =====
BASE_DIR = Path(r"IQ_samps")
OUT_DIR  = Path(r"COV")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_SELECT = 3                      # nº de tomas a conservar por ángulo
SELECTION = "power"              # "power" | "oldest" | "newest"
SAVE_AVERAGE = True

# patrón de nombre: IQ_ch{ch}_{YYYYMMDD}_{HHMMSS}_{ANGLE}.npy
pat = re.compile(r"^IQ_ch(?P<ch>\d+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<angle>[A-D]\d+)\.npy$", re.IGNORECASE)

# ===== 1) Indexar archivos → shots[(angle, stamp)][ch] = Path =====
shots = {}
for p in BASE_DIR.glob("*.npy"):
    m = pat.match(p.name)
    if not m:
        continue
    ch = int(m.group("ch"))
    date = m.group("date")
    time = m.group("time")
    angle = m.group("angle").upper()
    stamp = f"{date}_{time}"
    shots.setdefault((angle, stamp), {})
    shots[(angle, stamp)][ch] = p

# Agrupar por ángulo
per_angle = defaultdict(list)
for (angle, stamp), chmap in shots.items():
    if len(chmap) == 4:  # solo tomas completas
        per_angle[angle].append((stamp, chmap))

# ===== 2) Helpers =====
def load_X(chmap: dict):
    """Carga los 4 canales en orden ch0..ch3 → X de shape (4, N)."""
    X = np.vstack([np.load(chmap[ch]) for ch in sorted(chmap.keys())])
    return X

def cov_from_X(X: np.ndarray):
    """Rxx = X X^H / N  → shape (M, M)."""
    return (X @ X.conj().T) / X.shape[1]

def power_metric(X: np.ndarray):
    """Potencia media ≈ trace(Rxx) = mean(|X|^2) * M."""
    # Más rápido: media de |X|^2 (equivalente a trace/N canales)
    return float(np.mean(np.abs(X)**2))

def select_shots(angle: str, items, k=3, mode="power"):
    """
    items = lista de (stamp, chmap) para un ángulo.
    Devuelve las k tomas seleccionadas según `mode`.
    """
    if mode == "oldest":
        items = sorted(items, key=lambda t: t[0])[:k]
    elif mode == "newest":
        items = sorted(items, key=lambda t: t[0], reverse=True)[:k]
    elif mode == "power":
        scored = []
        for stamp, chmap in items:
            X = load_X(chmap)
            scored.append((power_metric(X), stamp, chmap))
        scored.sort(reverse=True, key=lambda x: x[0])  # mayor potencia primero
        items = [(stamp, chmap) for _, stamp, chmap in scored[:k]]
    else:
        raise ValueError("SELECTION inválido")
    return items

# ===== 3) Pipeline principal =====
summary = []
for angle, items in per_angle.items():
    if len(items) == 0:
        continue

    # Seleccionar K tomas
    selected = select_shots(angle, items, k=min(K_SELECT, len(items)), mode=SELECTION)

    R_list = []
    for stamp, chmap in selected:
        X = load_X(chmap)                   # (4, 256000)
        Rxx = cov_from_X(X)                 # (4, 4)
        R_list.append(Rxx)

        # Guardar Rxx por toma
        out_path = OUT_DIR / f"Rxx_{angle}_{stamp}.npy"
        np.save(out_path, Rxx)



    summary.append((angle, len(items), len(selected)))

# ===== 4) Resumen final =====
angles_sorted = sorted(summary, key=lambda a: (a[0][0], int(a[0][1:])))
print("\n=== RESUMEN COV ===")
for angle, total, used in angles_sorted:
    print(f"{angle}: tomas_completas={total} | usadas={used} | guardado: Rxx por toma")
print(f"\nRxx guardadas en: {OUT_DIR}")

