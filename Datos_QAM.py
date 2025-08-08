import numpy as np
import matplotlib.pyplot as plt

# 1. Generar bits aleatorios
num_bits = 1000
bits = np.random.randint(0, 2, num_bits)

# 2. Agrupar en símbolos de 4 bits (16-QAM → 4 bits/símbolo)
bits_reshaped = bits.reshape(-1, 4)

# 3. Mapear bits a coordenadas I y Q (Gray coding simple)
def bits_to_symbol(b):
    # Gray-coded 16-QAM (puedes personalizarlo)
    I = 2*(2*b[0] + b[1]) - 3
    Q = 2*(2*b[2] + b[3]) - 3
    return I + 1j*Q

symbols = np.array([bits_to_symbol(b) for b in bits_reshaped])

# 4. Normalizar potencia (opcional pero estándar)
symbols /= np.sqrt((np.mean(np.abs(symbols)**2)))

# 5. Mostrar constelación ideal
plt.figure(figsize=(6, 6))
plt.plot(np.real(symbols), np.imag(symbols), 'bo')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.title('Constelación 16-QAM (sin ruido)')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.grid(True)
plt.gca().set_aspect('equal')
plt.show()
# 6. Agregar ruido (AWGN)
SNR_dB = 15  # relación señal/ruido en decibelios
SNR_linear = 10**(SNR_dB/10)
noise_std = np.sqrt(1 / (2 * SNR_linear))

ruido = noise_std * (np.random.randn(*symbols.shape) + 1j*np.random.randn(*symbols.shape))
received = symbols + ruido

# 7. Mostrar constelación con ruido
plt.figure(figsize=(6, 6))
plt.plot(np.real(received), np.imag(received), 'r.', label='Recibidos')
plt.plot(np.real(symbols), np.imag(symbols), 'bo', alpha=0.4, label='Originales')
plt.title(f'Constelación 16-QAM con ruido (SNR={SNR_dB} dB)')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal')
plt.show()
