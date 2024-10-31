import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tests.SByBinomial import generateSByBinomial, generatePrimeByBinomial
from tests.SByGeometricBrownianMotion import generateSByGeometricBrownianMotion


class TilleyBundlingAlgorithm:

    ALLOW_INMEDIATE_EXERCISE = False

    def __init__(self, P, Q, T, N, X, r, isCall):
        # Tipo de opción
        self.isCall = isCall

        # Tiempo de simulación total
        self.N = N

        # Tiempo de expiración de la opción
        self.T = T

        # Precio del subyascente del camino i en tiempo j
        self.S = None

        if X.shape[0] != N:
            raise ValueError("Invalid shape for X")

        # Precio de ejercicio de la opción en tiempo i
        self.X = X

        # Número de armados
        self.Q = Q

        # Número de caminos en cada armado
        self.P = P

        # Caminos
        self.R = int(P * Q)

        # Interests vars
        if type(r) is float:
            self.r = r
            self.rs = None

        elif type(r) is np.array:
            if r.shape[0] != N:
                raise ValueError("Invalid shape for r")

            self.r = None
            self.rs = r

        else:
            raise ValueError("Invalid type for r")

        # Reindex of S
        self.reindex = np.zeros(R, dtype=int)
        for i in range(R):
            self.reindex[i] = i

    def __prepare(self):
        R = self.R
        N = self.N

        # Indicadora temporal de ejercicio de la opción en
        # tiempo i en el camino k
        self.x = np.zeros((R, N), dtype=int)

        # Indicadora temporal de ejercicio de la opción en
        # tiempo i en el camino k
        self.y = np.zeros((R, N), dtype=int)

        # Valor de retención de la opción en tiempo i en el camino k
        self.H = np.zeros((R, N), dtype=float)

        # Valor en el camino k en tiempo t de un pago
        # con madurez en t + 1
        self.d = np.zeros((R, N), dtype=float)

        # Valor presente en tiempo 0 de un pago en tiempo
        # t del camino k, calculado mediante el producto
        # de los factores de descuento d(k, s)
        self.D = np.full((R, N), 1, dtype=float)
        self.__calcDs()

        # Valor intrínseco de la opción en tiempo i en el camino k
        self.Intrinsic = np.zeros((R, N), dtype=float)
        self.__calcIntrinsic()

        # Valor actual de la opción en tiempo i en el camino k
        self.V = np.zeros((R, N + 1), dtype=float)
        self.V[:, N] = self.Intrinsic[:, N - 1]

        # Indicadora de ejercicio de la opción en tiempo i en el camino k
        self.z = np.zeros((R, N), dtype=int)
        self.__estimateExerciseOrHold()

        return

    def __calcDsWithFloatRate(self):
        dt = self.T / (self.N - 1)
        r = 1 / (1 + self.r) ** dt

        self.d.fill(r)

        it = 1
        for t in range(1, self.N):
            it *= r
            self.D[:, t] = it

        return

    def __calcDsWithArrayRate(self):
        for k in range(self.R):
            r = 1 + self.rs[0]

            self.d[k][0] = 1 / r

        for t in range(1, self.N):
            for k in range(self.R):
                r = 1 + self.rs[t]

                self.d[k][t] = 1 / r
                self.D[k][t] = self.D[k][t - 1] / r

        return

    def __calcDs(self):
        if self.r is not None:
            self.__calcDsWithFloatRate()
        elif self.rs is not None:
            self.__calcDsWithArrayRate()
        else:
            raise ValueError("No interest rate set")

        return

    def __calcIntrinsic(self):
        if self.isCall:
            for k in range(self.R):
                self.Intrinsic[k, :] = np.maximum(0, self.S[k, :] - self.X)
        else:
            for k in range(self.R):
                self.Intrinsic[k, :] = np.maximum(0, self.X - self.S[k, :])

    def __step1(self, t):
        """_summary_
        Reordenar las rutas de precios de las acciones por precio de
        las acciones, desde el precio más bajo hasta el precio más alto
        para una opción de compra o desde el precio más alto hasta el precio
        más bajo para una opción de venta.
        Reindexar las rutas de 1 a R según el reordenamiento.
        """

        self.reindex = np.argsort(self.S[:, t])
        if not self.isCall:
            self.reindex = np.flip(self.reindex)

    def __step2(self, t):
        """_summary_
        Para cada ruta k, calcule el valor intrínseco I(k, t) de la opción.
        """

        # Los hicimos todos juntos en __prepare

        return

    def __step3(self, t):
        """_summary_
        Dividir el conjunto de R caminos ordenados en Q distintos armados de
        P caminos cada uno. Asignar los primeros P caminos al primer haz,
        los segundos P caminos al segundo haz, y así sucesivamente,
        y finalmente los últimos P camino al Q-ésimo haz.
        """

        # Creamos grupos de [t * p, (t + 1) * p]
        # Esos van a ser nuestros bundles

        return

    def __step4(self, t):
        """_summary_
        Para cada ruta k, el “valor de retención” de la opción H(k, t)
        se calcula como la siguiente expectativa matemática tomada
        sobre todas las rutas en el paquete que contiene la ruta k
        """

        low = 0
        for _ in range(self.Q):
            end = low + self.P

            # Ésto es un bundle
            indexes = self.reindex[low:end]
            sumTotal = np.sum(self.V[indexes, t + 1])
            self.H[indexes, t] = self.d[indexes, t] * sumTotal / self.P

            low = end

        return

    def __step5(self, t):
        """_summary_
        Para cada ruta, compare el valor de retención H(k, t) con el valor
        intrínseco I(k, t) y decida “provisionalmente” si ejercer o mantener.
        """

        indexes = self.reindex
        self.x[indexes, t] = (self.Intrinsic[indexes, t] > self.H[indexes, t])

        return

    def __step6(self, t, indexes):
        """_summary_
        Examine la secuencia de 0's y 1's {x(k, t); k = 1, 2, ..., R}.
        Determine un límite entre los Hold y el Exercise como
        el inicio de la primera cadena de 1's cuya longitud exceda
        la longitud de cada cadena posterior de 0. Sea k*(t) el
        índice de ruta (en la muestra, tal como se ordenó en el subpaso 1
        anterior) del 1 principal en dicha cadena. La “zona de transición”
        entre la espera y el ejercicio se define como la secuencia de 0's y
        1's que comienza con el primer 1 y termina con el último 0.
        """

        out = len(indexes)

        q0s = 0
        q1s = 0
        maxQ0s = 0
        ks = out
        for idx, k in enumerate(reversed(indexes)):
            i = out - idx - 1

            if self.x[k][t] == 0:
                q0s += 1

                if q1s >= maxQ0s and q0s == 1 and i != (out - 1):
                    ks = i + 1

                q1s = 0
            else:
                q1s += 1

                if q0s > maxQ0s:
                    maxQ0s = q0s

                q0s = 0

        return ks

    def __step7(self, t, indexes, ks):
        """_summary_
        Defina una nueva variable indicadora de ejercicio o retención y(k, t)
        que incorpore el límite de la siguiente manera:
        """

        for i, k in enumerate(indexes):
            self.y[k][t] = (i >= ks)

        return

    def __step8(self, t):
        """_summary_
        Para cada camino k, se define el valor actual
        de V(k, t) de la opción como
        """

        self.V[:, t] = np.where(
            self.y[:, t], self.Intrinsic[:, t], self.H[:, t])

        return

    def __estimateExerciseOrHold(self):
        lowBoundary = 0 if self.ALLOW_INMEDIATE_EXERCISE else 1

        # t = Tiempo de simulación
        for t in reversed(range(lowBoundary, self.N)):
            self.__step1(t)
            # self.__step2(t)
            # self.__step3(t)
            self.__step4(t)
            self.__step5(t)

            indexes = self.reindex

            # Indicadora de sharp boundary
            ks = self.__step6(t, indexes)
            self.__step7(t, indexes, ks)

            self.__step8(t)

        for k in range(self.R):
            for t in range(self.N):
                if self.y[k][t]:
                    self.z[k][t] = 1
                    break

        return

    def estimatePremiumEstimator(self, S):
        if S.shape[0] != self.R or S.shape[1] != self.N:
            raise ValueError("Invalid shape for S")

        # Precio del subyascente del camino i en tiempo j
        self.S = S

        self.__prepare()

        totalSum = np.sum(self.z * self.Intrinsic * self.D)

        return totalSum / self.R


if __name__ == "__main__":
    np.random.seed(0)

    # Datos de prueba
    r = 0.07
    K = 45
    S0 = 40
    volatily = 0.3
    T = 3
    N_sim = 1000

    N = int(4 * T) + 1              # Tiempo de simulación total
    P = 72                          # Número de caminos en cada armado
    Q = 70                          # Número de armados
    R = int(P * Q)                  # Número de caminos
    isCall = False                  # Tipo de opción

    # Datos generados
    X = np.full(N, K)
    args = (R, N, S0, r, T, volatily, K)

    TBA = TilleyBundlingAlgorithm(P, Q, T, N, X, r, isCall)

    t0 = time.time()
    PremiumEstimator = 0
    Premiums = []
    for _ in range(N_sim):
        S = generateSByGeometricBrownianMotion(*args)
        Premiums.append(TBA.estimatePremiumEstimator(S))
    t1 = time.time()

    # print(PremiumEstimator / N_sim, t1 - t0)

    # t0 = time.time()
    premiumBinomial, tree = generatePrimeByBinomial(*args, isCall=isCall)
    # t1 = time.time()
    # print(premiumBinomial, t1 - t0)

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Primer subplot en la esquina superior izquierda
    ax1 = fig.add_subplot(gs[0, 0])
    for x in S:
        ax1.plot(x)

    ax1.set_title("Dist. of stock price movements de CRR")

    # Segundo subplot en la esquina superior derecha
    ax2 = fig.add_subplot(gs[0, 1])
    for x in tree:
        ax2.plot(x)
    ax2.set_title("Binomial Lattice de CRR")

    # Tercer subplot que ocupa la fila inferior completa
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Comparación de primas")

    ax3.plot(Premiums, label="Tilley")
    ax3.set_ylim([7.6, 8.4])
    
    ax3.axhline(y=premiumBinomial, color='r', linestyle='--',
                label="Binomial CRR")

    ax3.axhline(y=np.mean(Premiums), color='g', linestyle='--',
                label="Tilley (mean)")

    ax3.legend()

    # Tercer subplot que ocupa la fila inferior completa
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Comparación de primas")

    ax4.axvline(x=premiumBinomial, color='r', linestyle='--', label="Binomial CRR")
        
    num_grupos = 100
    conteos, bordes = np.histogram(Premiums, bins=num_grupos)

    # Posiciones en el eje X para las barras (usar los centros de los bordes)
    x = (bordes[:-1] + bordes[1:]) / 2

    ax4.hist(x, bins=num_grupos, weights=conteos, label="Tilley")

    ax4.set_ylim([0, 40])
    ax4.set_xlim([7.6, 8.4])

    ax4.legend()

    plt.savefig('output.png', dpi=300, bbox_inches='tight')
