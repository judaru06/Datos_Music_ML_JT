# === Visualizar y analizar una matriz de covarianza ===
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualizar_cov(fpath):
    # 1) Cargar matriz
    R = np.load(fpath)
    print(f"\nArchivo: {fpath.name}")
    print(f"Shape: {R.shape} (compleja)")

    # 2) Imprimir en consola
    print("\nParte real:\n", np.real(R))
    print("\nParte imaginaria:\n", np.imag(R))
    print("\nMagnitud |R|:\n", np.abs(R))
    print("\nRxx:\n",(R))
    # 3) Graficar en una sola ventana
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axs[0].imshow(np.real(R)); axs[0].set_title("Parte real"); fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(np.imag(R)); axs[1].set_title("Parte imaginaria"); fig.colorbar(im1, ax=axs[1])
    im2 = axs[2].imshow(np.abs(R));  axs[2].set_title("|R| Magnitud"); fig.colorbar(im2, ax=axs[2])
    fig.suptitle(f"Matriz de covarianza: {fpath.stem}", fontsize=14)
    plt.tight_layout()
    plt.show()

# === Ruta de ejemplo (puedes cambiarla) ===
COV_DIR = Path("COV")
fpath = sorted(COV_DIR.glob("Rxx_*.npy"))[1]  # primera matriz
visualizar_cov(fpath)
