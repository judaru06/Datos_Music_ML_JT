import numpy as np
import matplotlib.pyplot as plt
file_path = "IQ_samps/IQ_ch3_20250729_233257_D8.npy"  # ✅ si estás en la misma carpeta
# O también:
# file_path = r"E:\OneDrive - INSTITUTO TECNOLOGICO METROPOLITANO - ITM\Maestria\Semestre_5\Resultados_Obje_2\IQ_ch0_20250729_224757_A4.npy"

data = np.load(file_path)

print("Forma:", data.shape)
print("Tipo de datos:", data.dtype)
print("Ejemplo de datos:", data[:5])


plt.figure(figsize=(6, 6))
plt.plot(np.real(data), np.imag(data), '.', alpha=0.2)
plt.title("Nube de puntos IQ")
plt.xlabel("In-phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.gca().set_aspect('equal')
plt.show()

########

fase = np.angle(data)

plt.plot(fase[:100])
plt.title("Fase de la señal (primeros 1000 puntos)")
plt.xlabel("Muestra")
plt.ylabel("Fase (radianes)")
plt.grid()
plt.show()
#####
fft = np.fft.fft(data)
frequencies = np.fft.fftfreq(len(data))

plt.plot(np.fft.fftshift(frequencies), np.fft.fftshift(np.abs(fft)))
plt.title("Espectro de frecuencia de la señal")
plt.xlabel("Frecuencia (normalizada)")
plt.ylabel("Magnitud")
plt.grid()
plt.show()
