import numpy as np
import matplotlib.pyplot as plt
X = np.array([
    [1+1j, 2+2j, 3+3j],
    [1-1j, 2-2j, 3-3j]
])  # Forma (2x3)

Rxx = X @ X.conj().T / X.shape[1]

print(np.round(Rxx, 2))
