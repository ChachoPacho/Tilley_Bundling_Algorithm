import numpy as np


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
        if comp(left[i], right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


def makeCallComp(t):
    return lambda x, y: x[t] > y[t]


def makePutComp(t):
    return lambda x, y: x[t] < y[t]


class TilleyBundlingAlgorithm:

    ALLOW_INMEDIATE_EXERCISE = False

    def __init__(self, isCall, N, R):
        # Tipo de opción
        self.isCall = isCall

        # Tiempo de simulación
        self.t = 0

        # Tiempo de simulación total
        self.N = N

        # Caminos
        self.R = R

        # Precio del subyascente del camino i en tiempo j
        self.S = np.zeros((R, N))

        # VAlor intrínseco de la opción en tiempo i en el camino k
        self.Intrinsic = np.zeros((R, N))

        # Valor en el camino k en tiempo t de un pago
        # con madurez en t+1
        self.d = np.zeros((R, N))

        # Valor presente en tiempo 0 de un pago en tiempo
        # t del camino 𝑘, calculado mediante el producto
        # de los factores de descuento 𝑑(𝑘,𝑠)
        self.D = np.zeros((R, N))

        # Precio de ejercicio de la opción en tiempo i
        self.X = np.zeros(self.N)

        # Número de armados
        self.Q = 10

        # Número de caminos en cada armado
        self.P = R / self.Q

        # Indicadora de ejercicio de la opción en tiempo i en el camino k
        self.z = np.zeros((R, N))

        # Indicadora temporal de ejercicio de la opción en
        # tiempo i en el camino k
        self.x = np.zeros((R, N))

        # Indicadora temporal de ejercicio de la opción en
        # tiempo i en el camino k
        self.y = np.zeros((R, N))

        # Indicadora de sharp boundary
        self.ks = np.zeros(self.N)

        # Valor de retención de la opción en tiempo i en el camino k
        self.H = np.zeros((R, N))

        # Valor actual de la opción en tiempo i en el camino k
        self.V = np.zeros((R, N + 1))
        t = N - 1
        if isCall:
            for k in range(R):
                self.V[k][self.N] = max(0, self.S[k, t] - self.X[t])
        else:
            for k in range(R):
                self.V[k][self.N] = max(0, self.X[t] - self.S[k, t])

    def step1(self, t):
        """_summary_
        Reordenar las rutas de precios de las acciones por precio de
        las acciones, desde el precio más bajo hasta el precio más alto
        para una opción de compra o desde el precio más alto hasta el precio
        más bajo para una opción de venta.
        Reindexar las rutas de 1 a R según el reordenamiento.
        """

        comp = makeCallComp(t) if self.isCall else makePutComp(t)
        return mergeSort(self.S, comp)

    def step2(self, t):
        """_summary_
        Para cada ruta 𝑘, calcule el valor intrínseco I(k, t) de la opción.
        """

        if self.isCall:
            for k in range(self.R):
                self.Intrinsic[k][t] = max(0, self.S[k, t] - self.X[t])
        else:
            for k in range(self.R):
                self.Intrinsic[k][t] = max(0, self.X[t] - self.S[k, t])

        return

    def step3(self, t):
        """_summary_
        Dividir el conjunto de R caminos ordenados en Q distintos armados de
        P caminos cada uno. Asignar los primeros P caminos al primer haz,
        los segundos P caminos al segundo haz, y así sucesivamente,
        y finalmente los últimos P camino al Q-ésimo haz.
        """

        return

    def step4(self, t):
        """_summary_
        Para cada ruta k, el “valor de retención” de la opción H(k, t)
        se calcula como la siguiente expectativa matemática tomada
        sobre todas las rutas en el paquete que contiene la ruta k
        """

        totalsSumsV = np.zeros((self.Q, self.N))
        for i in range(self.Q):
            low = i * self.P

            for j in range(low, low + self.P):
                totalsSumsV[i][t] += self.V[j][t + 1]

        for i in range(self.Q):
            low = i * self.P

            for k in range(low, low + self.P):
                self.H[k][t] = self.d[k][t] * totalsSumsV[i][t] / self.P

        return

    def step5(self, t):
        """_summary_
        Para cada ruta, compare el valor de retención H(k, t) con el valor
        intrínseco I(k, t) y decida “provisionalmente” si ejercer o mantener.
        """

        for k in range(self.R):
            self.x[k][t] = (self.Intrinsic[k][t] > self.H[k][t])

        return

    def step6(self, t):
        """_summary_
        Examine la secuencia de 0’s y 1’s {𝑥(𝑘, 𝑡); 𝑘 = 1, 2, ..., 𝑅}.
        Determine un límite entre los Hold y el Exercise como
        el inicio de la primera cadena de 1’s cuya longitud exceda
        la longitud de cada cadena posterior de 0. Sea 𝑘∗(𝑡)(t) el
        índice de ruta (en la muestra, tal como se ordenó en el subpaso 1
        anterior) del 1 principal en dicha cadena. La “zona de transición”
        entre la espera y el ejercicio se define como la secuencia de 0’s y
        1’s que comienza con el primer 1 y termina con el último 0.
        """

        q0s = 0
        q1s = 0
        maxQ0s = 0
        iOfSharp1 = -1
        for k in range(self.R - 1, -1, -1):
            if self.x[k][t] == 0:
                q0s += 1

                if q1s >= maxQ0s and q0s == 1:
                    iOfSharp1 = k + 1

                q1s = 0
            else:
                q1s += 1

                if q0s > maxQ0s:
                    maxQ0s = q0s

                q0s = 0

        self.ks[t] = iOfSharp1

        return

    def step7(self, t):
        """_summary_
        Defina una nueva variable indicadora de ejercicio o retención y(k, t)
        que incorpore el límite de la siguiente manera:
        """

        for k in range(self.R):
            self.y[k][t] = (k >= self.ks[t])

        return

    def step8(self, t):
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

    def estimateExerciseOrHold(self):
        lowBoundary = -1 if self.ALLOW_INMEDIATE_EXERCISE else 0

        for t in range(self.N - 1, lowBoundary, -1):
            self.step1(t)
            self.step2(t)
            self.step3(t)
            self.step4(t)
            self.step5(t)
            self.step6(t)
            self.step7(t)
            self.step8(t)

        return

    def estimatePremiumEstimator(self):
        self.estimateExerciseOrHold()

        total = 0
        for k in range(self.R):
            for t in range(self.N):
                total += self.z[k][t] * self.Intrinsic[k][t] * self.D[k][t]

        return total / self.R
