# Consigna

El `Método de Tilley` es un método de `Monte Carlo` que utiliza una estrategia de agrupamiento de trayectorias, y puede ser utilizado para valorar opciones americanas como también opciones sobre multiactivos.

Investigar sobre este algoritmo y realizar una implementación.

## Referencias

[Tilley, J. (1993) Valuing American Options in a Path Simulation Model. Transactions of the Society of Actuaries, 45, 83-104.](https://drive.google.com/file/d/1pBzLWTC4jJBOj7R3El49gZ-0IPZU9pdu/view?pli=1)

# Resumen

Sección 2: A little background on options is provided. 

Sección 3: The algrithm for valuing American options is described

Sección 4: tested by means of an example. 

Sección 5: The issue of bias in the estimator of the option premium is examined.

Sección 6: The example is revisited. 

Sección 7: summarizes the paper.

## Algoritmo de valuación de opciones Americanas

- $t_i$ con $i \in \N_0$, donde $t_0$ es el tiempo de origen.
- $S(i)$ con $i \in \N_0$, que sería el precio del subyacente en el tiempo $i$.
- Se requiere una muestra finita de $R$ caminos.
- $S(0), S(k, 1), S(k, 2), \dots, S(k, N)$ es la secuencia del $k$-ésimo camino.
- $d(k, t)$ es el valor presente en tiempo $t$ del camino $k$ de un pago que ocurre en $t + 1$.
- $D(k, t)$ es el valor presente en tiempo $0$ de un pago en tiempo $t$ del camino $k$, calculado mediante el producto de los factores de descuento $d(k, s)$ desde $s= 0$ hasta $s= t - 1$.
- $X(i)$ con $i \in \N$, que sería el precio de ejercicio en tiempo $i$.
    
    > Por lo general es una constante.
    > 
- Valor intrínseco
    
    $$
    \begin{darray}c
    
    I(k, t) &=& \begin{cases}
    \max[0, S(k, t) - X(t)] & call \\
    
    \max[0, X(t) - S(k, t)] & put \\
    
    \end{cases}
    
    \end{darray}
    $$
    
- $z(k, t)$ booleano que indica si se ejerció la acción ($1$) o no ($0$) en el camino $k$ en tiempo $t$.
    - $z(k, t_*) = 1 \Longrightarrow z(k, t) = 0, \forall t \ne t_*$.

### Estimador de la Prima

Primero hay que calcular todos los $D(k, t)$ para luego poder hacer el promedio

$$
\begin{darray}c

Premium\ Estimator &=& R^{-1} \sum_{k} \sum_{t} z(k, t) D(k, t) I(k, t)

\end{darray}
$$

### Estimación de Ejercicio-o-Retención: $z$

1. Reordenar las rutas de precios de las acciones por precio de las acciones, desde el precio más bajo hasta el precio más alto para una opción de compra o desde el precio más alto hasta el precio más bajo para una opción de venta. Reindexar las rutas de $1$ a $R$ según el reordenamiento.
2. Para cada ruta $k$, calcule el valor intrínseco $I(k, t)$ de la opción.
3. Dividir el conjunto de $R$ caminos ordenados en $Q$ distintos armados de $P$ caminos cada uno. Asignar los primeros $P$ caminos al primer haz, los segundos $P$ caminos al segundo haz, y así sucesivamente, y finalmente los últimos $P$ camino al $Q$-ésimo haz. 
    
    > Se supone que $P$ y $Q$ son factores enteros de $R$.
    > 
4. Para cada ruta $k$, el “valor de retención” de la opción $H(k, t)$ se calcula como la siguiente expectativa matemática tomada sobre todas las rutas en el paquete que contiene la ruta $k$:
    
    $$
    \begin{darray}c
    
    H(k, t) &=& d(k, t) P^{-1}  \sum_{j \in B_k} V(j, t + 1)
    
    \end{darray}
    $$
    
    > $B_k$ es el armado que contiene el camino $k$.
    > 
    
    > $V(k, t)$ se define más adelante y se tiene $\forall k, V(k, N) = I(k, N)$.
    > 
5. Para cada ruta, compare el valor de retención $H(k, t)$ con el valor intrínseco $I(k, t)$ y decida “provisionalmente” si ejercer o mantener. 
    
    Defina una variable indicadora $x(b,t)$ de la siguiente manera:
    
    $$
    \begin{darray}c
    
    x(k, t) &=& \begin{cases}
    1 &,& I(k, t) > H(k, t)  \\
    
    0 &,& cc  \\
    \end{cases}
    
    \end{darray}
    $$
    
    > Es un $z$ provisorio.
    > 
6. Examine la secuencia de $0$’s y $1$’s $\{x(k, t); k = 1,2..., R\}$. Determine un límite entre los `Hold` y el `Exercise` como el inicio de la primera cadena de $1$’s cuya longitud exceda la longitud de cada cadena posterior de $0$. Sea $k_*(t)$ el índice de ruta (en la muestra, tal como se ordenó en el subpaso $1$ anterior) del $1$ principal en dicha cadena. La “zona de transición” entre la espera y el ejercicio se define como la secuencia de $0$’s y $1$’s que comienza con el primer $1$ y termina con el último $0$.
7. Defina una nueva variable indicadora de ejercicio o retención $y(k, t)$ que incorpore el límite de la siguiente manera:
    
    $$
    \begin{darray}c
    
    y(k, t) &=& \begin{cases}
    1 &,& k \ge k_*(t)  \\
    
    0 &,& cc  \\
    \end{cases}
    
    \end{darray}
    $$
    
8. Para cada camino $k$, se define el valor actual de $V(k, t)$ de la opción como:

    $$
    \begin{darray}c

    V(k, t) &=& \begin{cases}
    I(k t) &,& y(k, t) = 1 \\
    H(k t) &,& y(k, t) = 0
    \end{cases}

    \end{darray}
    $$

Luego de que el algoritmo ha sido procesado hacia atrás desde el tiempo $N$ hasta el tiempo $1$ (o tiempo $0$ si se permite el ejercicio inmediato), la variable indicadora $z(k, t)$ para $t < N$ se estima de la siguiente manera:

$$
\begin{darray}c

z(k, t) &=& \begin{cases}
1 &,& y(k, t) = 1 \land y(k, s) = 0\ \forall s < t \\
0 &,& cc
\end{cases}

\end{darray}
$$
