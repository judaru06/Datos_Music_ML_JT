from pathlib import Path
import re
import numpy as np

# === 1) Config ===
BASE_DIR = Path("IQ_samps")

# Patrón esperado: IQ_ch{ch}_{YYYYMMDD}_{HHMMSS}_{ANGLE}.npy  (p. ej. IQ_ch2_20250729_233216_D6.npy)
pat = re.compile(
    r"^IQ_ch(?P<ch>\d+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<angle>[A-D]\d+)\.npy$",
    re.IGNORECASE
)

# === 2) Indexar archivos → dict[(angle, timestamp)][ch] = Path ===
shots = {}  # p.ej. shots[("D6","20250729_233216")] = {0: Path(...), 1: Path(...), 2:..., 3:...}

for p in BASE_DIR.glob("*.npy"):
    m = pat.match(p.name)
    if not m:
        # Si hay archivos que no cumplen el patrón, los ignoramos (o log)
        # print("Nombre no reconocido:", p.name)
        continue
    ch = int(m.group("ch"))
    date = m.group("date")
    time = m.group("time")
    angle = m.group("angle").upper()
    stamp = f"{date}_{time}"

    shots.setdefault((angle, stamp), {})
    shots[(angle, stamp)][ch] = p

# === 3) Resumen por ángulo ===
from collections import defaultdict
per_angle = defaultdict(list)
for (angle, stamp), chmap in shots.items():
    per_angle[angle].append((stamp, chmap))

print("\n=== Resumen por ángulo ===")
angles_sorted = sorted(per_angle.keys(), key=lambda a: (a[0], int(a[1:])))  # orden A1..D8
for ang in angles_sorted:
    total_tomas = len(per_angle[ang])
    completas = sum(1 for _, chmap in per_angle[ang] if len(chmap) == 4)
    print(f"{ang}: tomas={total_tomas} | completas(4ch)={completas}")

# === 4) Utilidades ===
def iter_shots_for_angle(angle:str, require_full=True):
    """
    Itera las tomas (timestamp) de un ángulo.
    yield: (stamp, chmap) con chmap: dict[ch] = Path
    - require_full=True: solo devuelve tomas con los 4 canales presentes.
    """
    angle = angle.upper()
    for (ang, stamp), chmap in shots.items():
        if ang != angle:
            continue
        if require_full and len(chmap) != 4:
            continue
        yield stamp, chmap

def load_X_from_chmap(chmap:dict):
    """
    Carga las 4 señales en orden de canal (0..3) y devuelve X (4 x N).
    Si falta algún canal, se omite (y la dimensión será <4); úsalo solo tras filtrar require_full=True.
    """
    chans = sorted(chmap.keys())
    data = [np.load(chmap[ch]) for ch in chans]
    X = np.vstack([d[np.newaxis, :] for d in data])  # (M x N)
    return X, chans

# === 5) Ejemplo de uso: preparar X para un ángulo (p.ej. 'D6') ===
target_angle = "D6"
print(f"\n=== Ejemplo: primeras 2 tomas completas para {target_angle} ===")
count = 0
for stamp, chmap in iter_shots_for_angle(target_angle, require_full=True):
    X, chans = load_X_from_chmap(chmap)
    print(f"Stamp {stamp} | canales={chans} | X.shape={X.shape}")
    # Aquí ya puedes calcular Rxx = X @ X.conj().T / X.shape[1]
    # Rxx = X @ X.conj().T / X.shape[1]
    # print("Rxx shape:", Rxx.shape)
    count += 1
    if count == 2:
        break
