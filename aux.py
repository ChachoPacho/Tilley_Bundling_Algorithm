import numpy as np
import time
from scipy import interpolate
import matplotlib.pyplot as plt
#from tests.SByBinomial import generateSByBinomial
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


class TilleyBundlingAlgorithm_withoutSharpBoundary:

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
        self.R = int(5040)

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
        R = int(5040)
        self.reindex = np.zeros(R, dtype=int)
        for i in range(R):
            self.reindex[i] = i

    def __prepare(self,):
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

        low = 0
        for _ in range(self.Q):
            end = low + self.P

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

    

    def __step7(self, t, indexes):
        """_summary_
        Defina una nueva variable indicadora de ejercicio o retención y(k, t)
        que incorpore el límite de la siguiente manera:
        """
        self.y = self.x

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

            # for q in range(self.Q):
            #     low = q * self.P
            #     end = low + self.P

            #     indexes = self.reindex[low:end]

            # Indicadora de sharp boundary
            self.__step7(t, indexes)

            self.__step8(t)

        for k in range(self.R):
            for t in range(self.N):
                if self.y[k][t]:
                    self.z[k][t] = 1
                    break

        return

    def estimatePremiumEstimator(self, S, ):
        if S.shape[0] != self.R or S.shape[1] != self.N:
            raise ValueError("Invalid shape for S")

        # Precio del subyascente del camino i en tiempo j
        self.S = S

        self.__prepare()

        totalSum = np.sum(self.z * self.Intrinsic * self.D)

        return totalSum / self.R
