import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path

# ===== CONFIG =====
IQ_FILE = Path("IQ_samps/IQ_ch0_20250729_224921_A0.npy")  # <-- cámbialo por tu archivo real
fs = 1e6  # Frecuencia de muestreo en Hz (ajústala según tu captura real)

# ===== 1. Cargar muestras IQ =====
x = np.load(IQ_FILE)  # (N,) complejo64 si fue guardado como IQ
print("Shape:", x.shape, "| dtype:", x.dtype)

# ===== 2. Calcular PSD con Welch =====
f, Pxx = welch(x, fs=fs, nperseg=4096, return_onesided=False)
# Centrar espectro en 0 Hz
fshift = np.fft.fftshift(f)
Pxx_shift = np.fft.fftshift(Pxx)

# ===== 3. Graficar =====
plt.figure(figsize=(8,4))
plt.semilogy(fshift/1e3, Pxx_shift)  # frecuencia en kHz
plt.xlabel("Frecuencia [kHz]")
plt.ylabel("PSD [V²/Hz] (escala log)")
plt.title(f"Densidad espectral de potencia - {IQ_FILE.name}")
plt.grid(True)
plt.show()