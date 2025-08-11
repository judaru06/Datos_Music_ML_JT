from pathlib import Path
import re
import numpy as np
from collections import defaultdict

# ===== CONFIG =====
BASE_DIR = Path(r"IQ_samps")
OUT_DIR  = Path(r"COV")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_SELECT = 3                      # nº de tomas a conservar por ángulo
SELECTION = "power"               # "power" | "oldest" | "newest"

# Segmentación: o defines N_SEGMENTS o defines L (tamaño de ventana)
N_SEGMENTS = 50                   # nº de segmentos no solapados por toma
L = None                          # si quieres fijar tamaño de ventana, pon un entero aquí y N_SEGMENTS=None
REMOVE_MEAN_PER_SEG = True        # quitar media por canal en cada segmento (recomendado)

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

# Agrupar por ángulo (solo tomas con los 4 canales)
per_angle = defaultdict(list)
for (angle, stamp), chmap in shots.items():
    if len(chmap) == 4:
        per_angle[angle].append((stamp, chmap))

# ===== 2) Helpers =====
def load_X(chmap: dict):
    """Carga los 4 canales en orden ch0..ch3 → X de shape (4, N)."""
    X = np.vstack([np.load(chmap[ch]) for ch in sorted(chmap.keys())])  # (4, T)
    return X

def cov_from_X(X: np.ndarray):
    """Rxx = X X^H / N  → shape (M, M)."""
    return (X @ X.conj().T) / X.shape[1]

def power_metric(X: np.ndarray):
    """Potencia media ≈ mean(|X|^2)."""
    return float(np.mean(np.abs(X)**2))

def select_shots(angle: str, items, k=3, mode="power"):
    """Selecciona k tomas según criterio."""
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

def iter_segments(X: np.ndarray, n_segments=None, L=None):
    """
    Genera segmentos (submatrices) de X con tamaño (4, L).
    Si se da n_segments: divide en ese número de partes no solapadas.
    Si se da L: usa ventanas de largo L (no solapadas) tantas como quepan.
    """
    T = X.shape[1]
    if n_segments is not None:
        L_eff = T // n_segments
        n_eff = n_segments
    elif L is not None:
        L_eff = int(L)
        n_eff = T // L_eff
    else:
        raise ValueError("Debes especificar n_segments o L")

    start = 0
    for i in range(n_eff):
        s = start + i * L_eff
        e = s + L_eff
        if e <= T:
            Xi = X[:, s:e]
            if REMOVE_MEAN_PER_SEG:
                Xi = Xi - Xi.mean(axis=1, keepdims=True)
            yield i, Xi

# ===== 3) Pipeline principal =====
summary = []
for angle, items in per_angle.items():
    if len(items) == 0:
        continue

    # Seleccionar K tomas por ángulo
    selected = select_shots(angle, items, k=min(K_SELECT, len(items)), mode=SELECTION)

    total_saved = 0
    for stamp, chmap in selected:
        X = load_X(chmap)  # (4, T)

        # Segmentar X
        for seg_idx, Xseg in iter_segments(X, n_segments=N_SEGMENTS, L=L):
            Rxx_seg = cov_from_X(Xseg)  # (4, 4)
            out_path = OUT_DIR / f"Rxx_{angle}_{stamp}_seg{seg_idx:02d}.npy"
            np.save(out_path, Rxx_seg)
            total_saved += 1

    summary.append((angle, len(items), len(selected), total_saved))

# ===== 4) Resumen final =====
angles_sorted = sorted(summary, key=lambda a: (a[0][0], int(a[0][1:])))
print("\n=== RESUMEN COV SEGMENTADO ===")
for angle, total, used, saved in angles_sorted:
    print(f"{angle}: tomas_completas={total} | usadas={used} | Rxx guardadas={saved} (segmentos)")
print(f"\nRxx segmentadas guardadas en: {OUT_DIR}")
