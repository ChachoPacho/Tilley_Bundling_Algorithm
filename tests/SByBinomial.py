import pyop3
import numpy as np


def generateSByBinomial(R, N, S0, r, T, volatily, K):
    tree = pyop3.binomial_tree(
        S0, r, T, N - 1, sigma=volatily, tree_type='CRR')
    ST = tree.underlying_asset_tree()

    S = np.zeros((R, N), dtype=float)

    # Variable aleatoria normal estándar
    Z = np.random.normal(0, 1, R * (N - 1) + 1)
    for i in range(R):
        S[i][0] = S0

        k = 0
        offset = i * (N - 1)
        for j in range(1, N):
            if Z[offset + j] < 0:
                k += 1

            S[i][j] = ST[k][j]

    return S
