import numpy as np
import time
from scipy import interpolate
import matplotlib.pyplot as plt
#from tests.SByBinomial import generateSByBinomial
from tests.SByGeometricBrownianMotion import generateSByGeometricBrownianMotion
from aux import TilleyBundlingAlgorithm_withoutSharpBoundary

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

    def __step6(self, t, indexes ):
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

    def __step7(self, t, indexes, ks, ):
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

            # for q in range(self.Q):
            #     low = q * self.P
            #     end = low + self.P

            #     indexes = self.reindex[low:end]

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
    N_sim = 10

    N = int(4 * T) + 1              # Tiempo de simulación total
    R = int(5040)                  # Número de caminos

    alpha_array = [0.   , 0.015, 0.03 , 0.045, 0.06 , 0.075, 0.09 , 0.105, 0.12 ,
       0.135, 0.15 , 0.165, 0.18 , 0.195, 0.21 , 0.225, 0.24 , 0.255,
       0.27 , 0.285, 0.3  , 0.315, 0.33 , 0.345, 0.36 , 0.375, 0.39 ,
       0.405, 0.42 , 0.435, 0.45 , 0.465, 0.48 , 0.495, 0.51 , 0.525,
       0.54 , 0.555, 0.57 , 0.585, 0.6  , 0.615, 0.63 , 0.645, 0.66 ,
       0.675, 0.69 , 0.705, 0.72 , 0.735, 0.75 , 0.765, 0.78 , 0.795,
       0.81 , 0.825, 0.84 , 0.855, 0.87 , 0.885, 0.9  , 0.915, 0.93 ,
       0.945, 0.96 , 0.975 , 0.99]
    resultOfEstimations_shrapBoundary = []
    resultOfEstimations_withoutSharpBoundary = []

    P = 72                          # Número de caminos en cada armado
    Q = 70                          # Número de armados
    # N = 6                           # Tiempo de simulación total
    # P = 2                           # Número de caminos en cada armado
    # Q = 3                           # Número de armados
    isCall = False                  # Tipo de opción

    # Datos generados
    X = np.full(N, K)
    args = (R, N, S0, r, T, volatily, K)
    # Test
    # S = generateSByBinomial(*args)

    #for i in range(len(alpha_array)):
    #    P = int(R **(1-alpha_array[i]))
    #    Q = int(R**(alpha_array[i]))
    #    TBA = TilleyBundlingAlgorithm(P, Q, T, N, X, r, isCall)
    #    TBA_withoutSharpBoundary = TilleyBundlingAlgorithm_withoutSharpBoundary(P, Q, T, N, X, r, isCall)
#
#
    #    t0 = time.time()
    #    PremiumEstimator_shrapBoundary = 0
    #    PremiumEstimator_withoutSharpBoundary = 0
    #    for _ in range(N_sim):
    #        S = generateSByGeometricBrownianMotion(*args)
    #        PremiumEstimator_shrapBoundary += TBA.estimatePremiumEstimator(S)
    #        PremiumEstimator_withoutSharpBoundary += TBA_withoutSharpBoundary.estimatePremiumEstimator(S)
#
    #    resultOfEstimations_shrapBoundary.append( (alpha_array[i],PremiumEstimator_shrapBoundary / N_sim) )
    #    resultOfEstimations_withoutSharpBoundary.append( (alpha_array[i],PremiumEstimator_withoutSharpBoundary / N_sim) )
#
    #    t1 = time.time()

    print("resultOfEstimations_shrapBoundary", resultOfEstimations_shrapBoundary)
    print("resultOfEstimations_withoutSharpBoundary",resultOfEstimations_withoutSharpBoundary)
    resultOfEstimations_shrapBoundary = [(0.0, 7.108610304531927), (0.015, 7.806940224727137), (0.03, 7.822186208675385), (0.045, 7.630289557933153), (0.06, 7.349914539631195), (0.075, 7.013435895331317), (0.09, 7.213258935675816), (0.105, 7.720488643710981), (0.12, 7.6710563154750435), (0.135, 6.5891600975328775), (0.15, 7.490673469583621), (0.165, 6.864527929793953), (0.18, 7.754864270313922), (0.195, 7.3889382055363715), (0.21, 7.727787567266617), (0.225, 7.8527620963599745), (0.24, 7.825634137777729), (0.255, 7.859689306051203), (0.27, 7.867162672159973), (0.285, 7.950597652303371), (0.3, 7.8714308122785495), (0.315, 7.906496658416328), (0.33, 7.912883518128249), (0.345, 7.935725194877139), (0.36, 7.982805779460563), (0.375, 7.902235882944258), (0.39, 7.913840266524701), (0.405, 7.909516956719915), (0.42, 7.942643613765692), (0.435, 7.966172591494105), (0.45, 8.00576780830539), (0.465, 7.939665739123832), (0.48, 7.930585954549654), (0.495, 7.99412137643417), (0.51, 8.014378581162827), (0.525, 7.98441121054482), (0.54, 7.9479692981769805), (0.555, 7.981442458611509), (0.57, 7.971308048706126), (0.585, 8.012875268287797), (0.6, 7.9497780275328775), (0.615, 7.938694127742063), (0.63, 8.019318744240902), (0.645, 7.952980712479686), (0.66, 7.9382718807979185), (0.675, 7.961171155500123), (0.69, 7.955416524028685), (0.705, 7.960825887558381), (0.72, 7.985282396835825), (0.735, 7.9516094255689564), (0.75, 7.951397898514294), (0.765, 7.954931329514605), (0.78, 7.949760207319106), (0.795, 7.939699813625843), (0.81, 7.959267550262071), (0.825, 7.896236402440609), (0.84, 7.820461532795538), (0.855, 7.874673703021512), (0.87, 7.89947510296207), (0.885, 7.796660975234898), (0.9, 7.8892337878839145), (0.915, 7.951923418349128), (0.93, 7.1166927198804135), (0.945, 7.474633434000802), (0.96, 7.661203336026344), (0.975, 7.780137381576258), (0.99, 7.478848399644195)]
    resultOfEstimations_withoutSharpBoundary = [(0.0, 7.108610304531927), (0.015, 7.806940224727137), (0.03, 7.822186208675385), (0.045, 7.630289557933153), (0.06, 7.349914539631195), (0.075, 7.013435895331317), (0.09, 7.213677550625809), (0.105, 7.720488643710981), (0.12, 7.6710563154750435), (0.135, 7.268220593397231), (0.15, 7.632850237827938), (0.165, 7.136482601998137), (0.18, 7.629766104503355), (0.195, 7.446967384393457), (0.21, 7.676420193034493), (0.225, 7.639273055385142), (0.24, 7.625236408122047), (0.255, 7.6157254143725694), (0.27, 7.651714051176623), (0.285, 7.615833247451503), (0.3, 7.630881186839217), (0.315, 7.640650211960514), (0.33, 7.668802854334986), (0.345, 7.722230515429691), (0.36, 7.7858755677552836), (0.375, 7.765115876867775), (0.39, 7.755780968572717), (0.405, 7.785960934226452), (0.42, 7.81973409720875), (0.435, 7.871127523025952), (0.45, 7.909677563319522), (0.465, 7.848942435605113), (0.48, 7.874807480201769), (0.495, 7.950767275231139), (0.51, 7.990401997635037), (0.525, 7.983438842897743), (0.54, 7.974365888446672), (0.555, 8.00566988274058), (0.57, 8.028057769904933), (0.585, 8.067232413622548), (0.6, 8.057126896686004), (0.615, 8.061714421325414), (0.63, 8.158590092726843), (0.645, 8.141823253914243), (0.66, 8.168790646481673), (0.675, 8.188419170769482), (0.69, 8.226331325511598), (0.705, 8.28563241753027), (0.72, 8.307895236288843), (0.735, 8.371715661359376), (0.75, 8.381368319603599), (0.765, 8.442188520930857), (0.78, 8.444640008413971), (0.795, 8.430084315297075), (0.81, 8.676975181287688), (0.825, 8.564909424048851), (0.84, 8.223637576928073), (0.855, 8.63165901479656), (0.87, 9.043946455333195), (0.885, 8.36054037449188), (0.9, 8.921982204670458), (0.915, 9.572984649665878), (0.93, 7.481045973961114), (0.945, 8.18610781120905), (0.96, 8.956051873828867), (0.975, 9.994707268645064), (0.99, 11.163914411536172)]
    # Extract data for sharp boundary
    x_values_sharp = [point[0] for point in resultOfEstimations_shrapBoundary]
    y_values_sharp = [point[1] for point in resultOfEstimations_shrapBoundary]

    # Extract data for without sharp boundary
    x_values_without = [point[0] for point in resultOfEstimations_withoutSharpBoundary]
    y_values_without = [point[1] for point in resultOfEstimations_withoutSharpBoundary]

    # Create interpolation for both datasets
    x_new_sharp = np.linspace(min(x_values_sharp), max(x_values_sharp), 200)
    f_sharp = interpolate.interp1d(x_values_sharp, y_values_sharp, kind='cubic', fill_value='extrapolate')
    y_new_sharp = f_sharp(x_new_sharp)

    x_new_without = np.linspace(min(x_values_without), max(x_values_without), 200)
    f_without = interpolate.interp1d(x_values_without, y_values_without, kind='cubic', fill_value='extrapolate')
    y_new_without = f_without(x_new_without)

    # Set style
    plt.style.use('seaborn')
    fig, (ax1) = plt.subplots(1, 1, figsize=(16, 9))

    # Top plot (without sharp boundary) 
    #bars1 = ax1.bar(x_values_without, y_values_without, width=0.01, color='#3498db', alpha=0.7)
    ax1.plot(x_new_without, y_new_without, '-', color='#ff6961', linewidth=2, alpha=0.7, label='withoutSharpBoundary')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(6.5, 9.5)

    # Bottom plot (sharp boundary)
    #bars2 = ax2.bar(x_values_sharp, y_values_sharp, width=0.01, color='#3498db', alpha=0.7)
    ax1.plot(x_new_sharp, y_new_sharp, '-', color='#3498db', linewidth=2, alpha=0.7, label='SharpBoundary')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=10)


    # Common styling for both plots
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#bdc3c7')
    ax1.spines['bottom'].set_color('#bdc3c7')
    ax1.tick_params(colors='#7f8c8d')
    ax1.set_facecolor('#f8f9fa')
    ax1.set_xlabel('α', fontsize=10, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Prime Estimations', fontsize=10, fontweight='bold', labelpad=10)
    # 1Add more y-axis ticks for better readability
    ax1.yaxis.set_ticks(np.arange(6.5, 10.6, 0.5))
    ax1.tick_params(axis='y', labelsize=8)
    ax1.xaxis.set_ticks(np.arange(0, 1, 0.03))
    ax1.tick_params(axis='x', labelsize=8)

    # Main title for entire figure
    fig.suptitle('Comparison of Estimation Methods',
                 fontsize=16,
                 fontweight='bold',
                 color='#2c3e50',
                 y=0.95)

    # Add subtitle
    fig.text(0.5, 0.91,
             'Sharp Boundary vs Without Sharp Boundary Estimations',
             ha='center',
             fontsize=12,
             fontstyle='italic',
             color='#7f8c8d')

    # Set background color
    fig.patch.set_facecolor('white')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plt.show()
    # # Graph S
    # for i in range(R):
    #     plt.plot(S[i])

    # plt.show()
