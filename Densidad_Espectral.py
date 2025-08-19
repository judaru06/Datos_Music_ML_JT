import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

# Señal de ejemplo: tono puro a 50 Hz + ruido
fs = 1000  # frecuencia de muestreo [Hz]
t = np.arange(0, 1, 1/fs)
x = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(len(t))

# Welch: estima la densidad espectral de potencia
f, Pxx = welch(x, fs=fs, nperseg=256)

# Graficar
plt.semilogy(f, Pxx)   # eje Y logarítmico (dB)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [V^2/Hz]")
plt.title("Densidad espectral de potencia (Welch)")
plt.show()
