import numpy as np


def generateSByGeometricBrownianMotion(R, N, S0, r, T, volatily, K):
    """
    Genera múltiples trayectorias del precio de la
    acción bajo un modelo log-normal.

    Parámetros:
    - S0: Precio inicial de la acción
    - r: Tasa de interés anual efectiva constante
    - sigma: Volatilidad logarítmica (30% en este caso)
    - N: Número de pasos en el tiempo
    - R: Número de trayectorias simuladas

    Retorna:
    - Un array con las trayectorias simuladas del precio.
    """
    sigma = volatily    # Volatilidad logarítmica

    dt = T / (N - 1)

    # interestRate = (1 + r) ** (1 / 4) - 1

    # Media ajustada
    mu = np.log(1 + r) - 0.5 * sigma**2

    # Inicializamos las trayectorias del precio
    S = np.zeros((R, N))
    S[:, 0] = S0  # Precio inicial en t=0

    # Simulación de las trayectorias mediante un proceso Browniano Geométrico
    # Variable aleatoria normal estándar
    Z = np.random.normal(0, 1, R * (N - 1))
    c = sigma * np.sqrt(dt)
    u = mu * dt
    low = 0
    high = R
    for i in range(1, N):
        S[:, i] = S[:, i - 1] * np.exp(u + c * Z[low:high])
        low = high
        high += R

    return S
