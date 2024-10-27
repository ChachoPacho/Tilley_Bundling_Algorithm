import numpy as np

ALLOW_INMEDIATE_EXERCISE = False

# Tipo de opción
isCall = False

# Tiempo de simulación
t = 0

# Tiempo de simulación total
N = 100

# Caminos
R = 1000

# Precio del subyascente del camino i en tiempo j
S = np.zeros((R, N))

# Valor en el camino k en tiempo t de un pago
# con madurez en t+1
d = np.zeros((R, N))

# Valor presente en tiempo 0 de un pago en tiempo
# t del camino 𝑘, calculado mediante el producto
# de los factores de descuento 𝑑(𝑘,𝑠)
D = np.zeros((R, N))

# Precio de ejercicio de la opción en tiempo i
X = np.zeros(N)

# Número de armados
Q = 10

# Número de caminos en cada armado
P = R / Q

# Indicadora de ejercicio de la opción en tiempo i en el camino k
z = np.zeros((R, N))

# Indicadora temporal de ejercicio de la opción en tiempo i en el camino k
x = np.zeros((R, N))

# Indicadora temporal de ejercicio de la opción en tiempo i en el camino k
y = np.zeros((R, N))

# Indicadora de sharp boundary
ks = np.zeros(N)


def Intrinsic(k, i, isCall):
    """
    Valor intrínseco de la opción en tiempo i en el camino k
    """
    if isCall:
        return max(0, S[k, i] - X[i])
    else:
        return max(0, X[i] - S[k, i])


# Valor de retención de la opción en tiempo i en el camino k
H = np.zeros((R, N))

# Valor actual de la opción en tiempo i en el camino k
V = np.zeros((R, N + 1))
for k in range(R):
    V[k][N] = Intrinsic(k, N, isCall)

# Estimación de Ejercicio-o-Retención: 𝑧
for k in range(R):
    z[k][N - 1] = (Intrinsic(k, N - 1, isCall) > 0)


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


def isCallComp(x, y):
    return x[N - 1] > y[N - 1]


def isPutComp(x, y):
    return x[N - 1] < y[N - 1]


def step1(S):
    """_summary_
    Reordenar las rutas de precios de las acciones por precio de las acciones,
    desde el precio más bajo hasta el precio más alto para una opción de compra
    o desde el precio más alto hasta el precio más bajo para una opción de
    venta.
    Reindexar las rutas de 1 a 𝑅 según el reordenamiento.
    """

    comp = isCallComp if isCall else isPutComp
    S = mergeSort(S, comp)

    return S


def step2():
    """_summary_
    Para cada ruta 𝑘, calcule el valor intrínseco 𝐼(𝑘, 𝑡) de la opción.
    """

    return


def step3():
    """_summary_
    Dividir el conjunto de 𝑅 caminos ordenados en 𝑄 distintos armados de
    𝑃 caminos cada uno. Asignar los primeros 𝑃 caminos al primer haz,
    los segundos 𝑃 caminos al segundo haz, y así sucesivamente,
    y finalmente los últimos 𝑃 camino al 𝑄-ésimo haz.
    """

    return


def step4(P):
    """_summary_
    Para cada ruta 𝑘, el “valor de retención” de la opción 𝐻(𝑘, 𝑡)
    se calcula como la siguiente expectativa matemática tomada
    sobre todas las rutas en el paquete que contiene la ruta 𝑘
    """

    totalsSumsV = np.zeros((Q, N))
    for t in range(N):
        for i in range(Q):
            low = i * P

            for j in range(low, low + P):
                totalsSumsV[i][t] += V[j][t + 1]

    for t in range(N):
        for i in range(Q):
            low = i * P

            for k in range(low, low + P):
                H[k][t] = d[k][t] * totalsSumsV[i][t] / P

    return


def step5(x):
    """_summary_
    Para cada ruta, compare el valor de retención 𝐻(𝑘, 𝑡) con el valor
    intrínseco 𝐼(𝑘, 𝑡) y decida “provisionalmente” si ejercer o mantener.
    """

    for k in range(R):
        for t in range(N):
            x[k][t] = (Intrinsic(k, t, isCall) > H[k][t])

    return x


def step6():
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

    return


def step7():
    """_summary_
    Defina una nueva variable indicadora de ejercicio o retención 𝑦(𝑘, 𝑡)
    que incorpore el límite de la siguiente manera:
    """
    
    for k in range(R):
        for t in range(N):
            y[k][t] = (k >= ks[t])

    return


def step8(y):
    """_summary_
    Para cada camino 𝑘, se define el valor actual
    de 𝑉(𝑘, 𝑡) de la opción como
    """

    for k in range(R):
        for t in range(N):
            if y[k][t]:
                V[k][t] = Intrinsic(k, t, isCall)
            else:
                V[k][t] = H[k][t]

    return


def estimateExerciseOrHold():
    lowBoundary = -1 if ALLOW_INMEDIATE_EXERCISE else 0
    
    for t in range(N - 1, lowBoundary, -1):
        S = step1(S)
        step2()
        step3()
        step4(P)
        x = step5(x)
        step6()
        step7()
        step8(y)



# Estimador de la Prima
# Primero hay que calcular todos los 𝐷(𝑘, 𝑡) para luego poder hacer el promedio
PremiumEstimator = 0
