import numpy as np
import time
import matplotlib.pyplot as plt
from tests.SByBinomial import generateSByBinomial
from tests.SByGeometricBrownianMotion import generateSByGeometricBrownianMotion


def mergeSort(arr, comp):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    leftHalf = arr[:mid]
    rightHalf = arr[mid:]

    sortedLeft = mergeSort(leftHalf, comp)
    sortedRight = mergeSort(rightHalf, comp)

    return merge(sortedLeft, sortedRight, comp)


def merge(left, right, comp):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if comp(left[i], right[j]):  # left[i] <= right[j]
            result.append(left[i])
            i += 1
        else:                        # left[i] > right[j]
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


class TilleyBundlingAlgorithm:

    ALLOW_INMEDIATE_EXERCISE = False

    def __init__(self, isCall, N):
        # Tipo de opción
        self.isCall = isCall

        # Tiempo de simulación total
        self.N = N

        # Precio del subyascente del camino i en tiempo j
        self.S = None

        # Precio de ejercicio de la opción en tiempo i
        self.X = None

        # Interests vars
        self.r = None
        self.rs = None

    def setPQ(self, P, Q):
        # Número de armados
        self.Q = Q

        # Número de caminos en cada armado
        self.P = P

        # Caminos
        self.R = int(P * Q)

        # Reindex of S
        self.reindex = np.zeros(self.R, dtype=int)
        for i in range(self.R):
            self.reindex[i] = i

    def setS(self, S: np.array):
        if S.shape[0] != self.R or S.shape[1] != self.N:
            raise ValueError("Invalid shape for S")

        # Precio del subyascente del camino i en tiempo j
        self.S = S

    def setX(self, X: np.array):
        if X.shape[0] != self.N:
            raise ValueError("Invalid shape for X")

        # Precio de ejercicio de la opción en tiempo i
        self.X = X

    def setInterestRate(self, r):
        if type(r) is float:
            self.r = r
            self.rs = None

        elif type(r) is np.array:
            if r.shape[0] != self.N:
                raise ValueError("Invalid shape for r")

            self.r = None
            self.rs = r

        else:
            raise ValueError("Invalid type for r")

    def __prepare(self):
        R = self.R
        N = self.N

        # Indicadora temporal de ejercicio de la opción en
        # tiempo i en el camino k
        self.x = np.zeros((R, N), dtype=int)

        # Indicadora temporal de ejercicio de la opción en
        # tiempo i en el camino k
        self.y = np.zeros((R, N), dtype=int)

        # Indicadora de sharp boundary
        self.ks = np.zeros(self.N, dtype=int)

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

        t = N - 1
        if self.isCall:
            for k in range(R):
                self.V[k][t + 1] = self.Intrinsic[k][t]
        else:
            for k in range(R):
                self.V[k][t + 1] = self.Intrinsic[k][t]

        # Indicadora de ejercicio de la opción en tiempo i en el camino k
        self.z = np.zeros((R, N), dtype=int)
        self.__estimateExerciseOrHold()

        return

    def __calcDsWithFloatRate(self):
        r = 1 + self.r

        self.d.fill(1 / r)

        it = 1
        for t in range(1, self.N):
            it /= r

            for k in range(self.R):
                self.D[k][t] = it

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
                for t in range(self.N):
                    self.Intrinsic[k][t] = max(0, self.S[k][t] - self.X[t])
        else:
            for k in range(self.R):
                for t in range(self.N):
                    self.Intrinsic[k][t] = max(0, self.X[t] - self.S[k][t])

    def __step1(self, t):
        """_summary_
        Reordenar las rutas de precios de las acciones por precio de
        las acciones, desde el precio más bajo hasta el precio más alto
        para una opción de compra o desde el precio más alto hasta el precio
        más bajo para una opción de venta.
        Reindexar las rutas de 1 a R según el reordenamiento.
        """

        if self.isCall:
            self.reindex = mergeSort(
                self.reindex,
                lambda x, y: self.S[x][t] < self.S[y][t]
            )
        else:
            self.reindex = mergeSort(
                self.reindex,
                lambda x, y: self.S[x][t] > self.S[y][t]
            )

    def __step2(self, t):
        """_summary_
        Para cada ruta k, calcule el valor intrínseco I(k, t) de la opción.
        """

        return

    def __step3(self, t):
        """_summary_
        Dividir el conjunto de R caminos ordenados en Q distintos armados de
        P caminos cada uno. Asignar los primeros P caminos al primer haz,
        los segundos P caminos al segundo haz, y así sucesivamente,
        y finalmente los últimos P camino al Q-ésimo haz.
        """

        return

    def __step4(self, t):
        """_summary_
        Para cada ruta k, el “valor de retención” de la opción H(k, t)
        se calcula como la siguiente expectativa matemática tomada
        sobre todas las rutas en el paquete que contiene la ruta k
        """

        low = -self.P
        totalsSumsV = np.zeros((self.Q, self.N), dtype=float)
        for i in range(self.Q):
            low += self.P

            for index in range(self.P):
                k = self.reindex[low + index]
                totalsSumsV[i][t] += self.V[k][t + 1]

        low = -self.P
        for i in range(self.Q):
            low += self.P

            for index in range(self.P):
                k = self.reindex[low + index]
                self.H[k][t] = self.d[k][t] * totalsSumsV[i][t] / self.P

        return

    def __step5(self, t):
        """_summary_
        Para cada ruta, compare el valor de retención H(k, t) con el valor
        intrínseco I(k, t) y decida “provisionalmente” si ejercer o mantener.
        """

        for k in range(self.R):
            self.x[k][t] = (self.Intrinsic[k][t] > self.H[k][t])

        return

    def __step6(self, t):
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
        
        print("STEP 6")

        q0s = 0
        q1s = 0
        maxQ0s = 0
        iOfSharp1 = self.R
        for i in reversed(range(self.R)):
            k = self.reindex[i]
            if self.x[k][t] == 0:
                q0s += 1

                if q1s >= maxQ0s and q0s == 1 and i != (R - 1):
                    iOfSharp1 = i + 1
                    print("new iOfSharp1", iOfSharp1, i)
                    print("x", self.x)

                q1s = 0
            else:
                q1s += 1

                if q0s > maxQ0s:
                    maxQ0s = q0s

                q0s = 0

        self.ks[t] = iOfSharp1
        
        print(iOfSharp1, self.reindex)

        return

    def __step7(self, t):
        """_summary_
        Defina una nueva variable indicadora de ejercicio o retención y(k, t)
        que incorpore el límite de la siguiente manera:
        """

        for i in range(self.R):
            k = self.reindex[i]
            self.y[k][t] = (i >= self.ks[t])
            
        print("y\n", self.y)

        return

    def __step8(self, t):
        """_summary_
        Para cada camino k, se define el valor actual
        de V(k, t) de la opción como
        """

        for k in range(self.R):
            if self.y[k][t]:
                self.V[k][t] = self.Intrinsic[k][t]
            else:
                self.V[k][t] = self.H[k][t]

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
            self.__step6(t)
            self.__step7(t)
            self.__step8(t)

        for k in range(self.R):
            for t in range(self.N):
                if self.y[k][t]:
                    self.z[k][t] = 1
                    break
                
        print("z", self.z)
        print("Intrinsic", self.Intrinsic)

        return

    def estimatePremiumEstimator(self):
        if self.S is None:
            raise ValueError("S is not set")

        if self.X is None:
            raise ValueError("X is not set")

        self.__prepare()

        total = 0
        for k in range(self.R):
            for t in range(self.N):
                total += self.z[k][t] * self.Intrinsic[k][t] * self.D[k][t]

        return total / self.R


if __name__ == "__main__":
    np.random.seed(0)

    # N = int((3 * 12) / 4) + 1       # Tiempo de simulación total
    # P = 72                          # Número de caminos en cada armado
    # Q = 70                          # Número de armados
    N = 6                           # Tiempo de simulación total
    P = 2                           # Número de caminos en cada armado
    Q = 3                           # Número de armados
    R = int(P * Q)                  # Número de caminos
    isCall = False                  # Tipo de opción

    # Datos de prueba
    TEA = 0.07
    K = 45
    S0 = 40
    volatily = 0.3
    T = 3

    # Datos generados
    X = np.full(N, K)
    interestRate = (1 + TEA) ** (1 / 4) - 1
    args = (R, N, S0, TEA, T, volatily, K)

    # Test
    # S = generateSByBinomial(*args)
    S = generateSByGeometricBrownianMotion(*args)

    t0 = time.time()

    # Algoritmo de Tilley
    TBA = TilleyBundlingAlgorithm(isCall, N)
    TBA.setPQ(P, Q)
    TBA.setX(X)
    TBA.setInterestRate(interestRate)
    TBA.setS(S)

    PremiumEstimator = TBA.estimatePremiumEstimator()
    t1 = time.time()

    print(PremiumEstimator)
    print(t1 - t0)

    # # Graph S
    # for i in range(R):
    #     plt.plot(S[i])

    # plt.show()
