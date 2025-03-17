import numpy as np
import matplotlib.pyplot as plt

'''Ejercicio 1: Integración indirecta'''

random = np.random.default_rng()
def g(x, n=10, alpha=4/5):
    return sum(np.exp(-(x - k)**2 * k) / k**alpha for k in range(1, n+1))

def metropolis(f,x0=5,n=700_000, sigma=0.5):
    """Tries to sample a distribution with density function proportional to the function `f`, which needs not to be normalized.
    The initial condition `x0` may not appear in the resulting array.
    The `sigma` argument is used as the standard deviation of the perturbation on each step.
    The size of this sigma should depend on the charactersitic size of the dunction to be sampled.

    Returns an array of length `n` with random samples from `f`."""
    samples = np.zeros(n)
    samples[-1] = x0
    for i in range(n):
        sample_new = samples[i-1] + random.normal(0,sigma)
        if random.random() < f(sample_new)/f(samples[i-1]):
            samples[i] = sample_new
        else:
            samples[i] = samples[i-1]
    return samples


# Generar muestras
samples = metropolis(g)

# Crear histograma
plt.figure(figsize=(10, 5))
plt.hist(samples, bins=200, density=True, alpha=0.75, color='blue')
plt.xlabel('x')
plt.ylabel('Densidad de probabilidad')
plt.title('Histograma de muestras generadas con Metrópolis')
plt.grid()
plt.savefig("Tarea4/1.a.pdf")

# Calcular A mediante la estimación con f(x) = exp(-x^2) lo vamos a deplazar 4 unidades a la derecha para que
#tenga pesos similares para todos los valores
def f(x):
    return np.exp(-(x-4)**2)

sqrt_pi = np.sqrt(np.pi)

#algebra y definiendo variables para que sea igual que la ecuacion en el pdf
ratios = f(samples) / g(samples)
prom = np.sum(ratios) / len(samples)
A_est = sqrt_pi/(prom)

#propagacion de error
std_A = A_est * np.sqrt(np.var(ratios)/len(samples)) / prom

# Imprimir resultado
print(f"1.b) {A_est} ± {std_A}")

'''Ejercicio 2: Integral de camino para difracción de Fresnel'''

# Parámetros
N = 100000  # Número de muestras
D1 = D2 = .50  # Distancias fuente-rendija y rendija-pantalla
lambda_ = 670e-9  # Longitud de onda
A = 0.4e-3  # Ancho de la apertura principal
a = 0.1e-3  # Ancho de la rendija
d = 0.1e-2  # Separación entre las rendijas

# Generar puntos aleatorios en la apertura de la fuente
x_fuente = np.random.uniform(-A/2, A/2, N)

# Generar puntos aleatorios en las rendijas
y_rendija1 = np.random.uniform(d/2 - a/2, d/2 + a/2, N//2)
y_rendija2 = np.random.uniform(-d/2 - a/2, -d/2 + a/2, N//2)
y_rendijas = np.concatenate([y_rendija1, y_rendija2])

# Puntos en la pantalla
z_vals = np.linspace(-0.4e-2, 0.4e-2, 500)  # En metros (-0.4 cm a 0.4 cm)
 
 #Función de la integral de camino de fresnel
def f(x, z, y, D_1, lambda_):
    return np.exp((np.pi*1j / lambda_) * (4 * D_1 + (x - y)**2 / D_1+ (y-z)**2/ D_1))

intensidad = []
#Evalua la función y obtiene la intensidad con la formula de np.abs(1/N* np.sum(f/p)
#P es constante por lo que se toma el promedio. 
for z in z_vals:
    valores_f = f(x_fuente, z, y_rendijas, D1, lambda_)
    I = (np.abs(np.mean(valores_f)))**2
    intensidad.append(I)


intensidad /= np.max(intensidad)


theta = np.arctan(z_vals / D2)
intensidad_classic = (np.cos(np.pi * d / lambda_ * np.sin(theta))**2 *
                     np.sinc(a / lambda_ * np.sin(theta))**2)
intensidad_classic /= np.max(intensidad_classic)


plt.figure(figsize=(8, 5))
plt.plot(z_vals * 100, intensidad, label='Intensidad', lw=2, color = 'red')
plt.plot(z_vals * 100, intensidad_classic, label='Modelo Clásico', lw=2, linestyle='dashed', color = 'orange')
plt.xlabel("Posición en la pantalla (cm)")
plt.ylabel("Intensidad")
plt.legend()
plt.grid()
plt.savefig('Tarea4/2.pdf')

'''Ejercicio 3: Modelo de Ising con Hetrópolis Hastings'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros del sistema
N = 150
J = 0.2
beta = 10

# Inicialización del sistema con valores aleatorios de -1 y 1
espines = np.random.choice([-1, 1], size=(N, N))

def energia_sitio(espines, i, j):
    # Calcula la energía de interacción de un sitio (i, j) con sus vecinos.
    arriba = espines[(i - 1) % N, j]
    abajo = espines[(i + 1) % N, j]
    izquierda = espines[i, (j - 1) % N]
    derecha = espines[i, (j + 1) % N]
    return -J * espines[i, j] * (arriba + abajo + izquierda + derecha)

def metropolis_step(espines, beta):
    # Realiza un paso del algoritmo de Metropolis-Hastings.
    i, j = np.random.randint(0, N, size=2)  # Seleccionar un espín aleatorio
    E_old = energia_sitio(espines, i, j)
    espines[i, j] *= -1  # Voltear el espín
    E_new = energia_sitio(espines, i, j)

    dE = E_new - E_old
    if dE > 0 and np.random.rand() >= np.exp(-beta * dE):
        espines[i, j] *= -1  # Revertir cambio si no se acepta

def update(frame):
    # Actualiza el estado del sistema para la animación.
    for _ in range(400):
        metropolis_step(espines, beta)
    im.set_array(espines)
    return [im]

# Configuración de la animación
fig, ax = plt.subplots()
im = ax.imshow(espines, cmap='gray', animated=True)
ani = animation.FuncAnimation(fig, update, frames=500, interval=50, blit=True)

# Guardar animación
ani.save("Tarea4/3.mp4", writer="ffmpeg", fps=30)
#plt.show()

'''Ejercicio 5: Evolución temporal de procesos estocásticos'''

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
import time
from numba import njit

# Datos del problema
A = 1000  # Producción diaria de U
B = 20     # Extracción diaria de Pu

# Tiempos de vida media (en días)
t_half_U = 23.4 / (24 * 60)  # Convertido a días
t_half_Np = 2.36

# Cálculo de las constantes de decaimiento lambda
lambda_U = np.log(2) / t_half_U
lambda_Np = np.log(2) / t_half_Np

# Sistema de ecuaciones diferenciales
def sistema(t, y):
    U, Np, Pu = y
    dU_dt = A - lambda_U * U
    dNp_dt = lambda_U * U - lambda_Np * Np
    dPu_dt = lambda_Np * Np - B * Pu
    return [dU_dt, dNp_dt, dPu_dt]

# Condiciones iniciales
U0, Np0, Pu0 = 1, 1, 1

# Tiempo de simulación (30 días)
tiempo_simulacion = 30

# Medición del tiempo de ejecución
start_time = time.time()

# Resolviendo el sistema de ecuaciones diferenciales
sol = solve_ivp(sistema, [0, tiempo_simulacion], [U0, Np0, Pu0], t_eval=np.linspace(0, tiempo_simulacion, 300))

# Graficamos los resultados
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='Uranio-239 (U)')
plt.plot(sol.t, sol.y[1], label='Neptunio-239 (Np)')
plt.plot(sol.t, sol.y[2], label='Plutonio-239 (Pu)')
plt.xlabel('Tiempo (días)')
plt.ylabel('Cantidad de material')
plt.yscale('log')
plt.legend()
plt.title('Evolución de las cantidades de U, Np y Pu en 30 días')
plt.grid()

# Imprimir los valores finales de cada átomo
print(f"Cantidad final de U: {sol.y[0, -1]:.2f}")
print(f"Cantidad final de Np: {sol.y[1, -1]:.2f}")
print(f"Cantidad final de Pu: {sol.y[2, -1]:.2f}")

# Método basado en resolver_sistema para verificar estabilidad
def verificar_estabilidad(sol):
    """Evalúa si las soluciones han alcanzado estabilidad."""
    ultima_seccion = sol.y[:, -10:]  # Últimos 10 valores
    cambios = np.abs(np.diff(ultima_seccion, axis=1))
    estabilidad = np.all(cambios < 1e-1, axis=1)
    return estabilidad

estabilidad = verificar_estabilidad(sol)

print("\nAnálisis de estabilidad:")
print(f"¿El Uranio-239 ha alcanzado estabilidad? {'Sí' if estabilidad[0] else 'No'}")
print(f"¿El Neptunio-239 ha alcanzado estabilidad? {'Sí' if estabilidad[1] else 'No'}")
print(f"¿El Plutonio-239 ha alcanzado estabilidad? {'Sí' if estabilidad[2] else 'No'}")

R = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1], [0, 0, -1]])

@njit
def simulacion_gillespie(tiempo_simulacion, max_pasos=115000):
    t = 0
    U, Np, Pu = 1, 1, 1
    tiempos = np.zeros(max_pasos)
    valores_pu = np.zeros(max_pasos)
    indice = 0

    while t < tiempo_simulacion and indice < max_pasos - 1:
        tasas = np.array([A, U * lambda_U, Np * lambda_Np, B * Pu])
        tasa_total = tasas.sum()
        if tasa_total == 0:
            break

        tau = np.random.exponential(1 / tasa_total)
        probabilidades = tasas / tasa_total
        probabilidades_acumuladas = np.cumsum(probabilidades)
        r_index = np.searchsorted(probabilidades_acumuladas, np.random.rand())
        
        U, Np, Pu = U + R[r_index, 0], Np + R[r_index, 1], Pu + R[r_index, 2]
        
        t += tau
        tiempos[indice] = t
        valores_pu[indice] = Pu
        indice += 1
    
    return tiempos[:indice], valores_pu[:indice]

num_simulaciones = 100
tiempo_simulacion = 30

resultados_pu = []
trayectorias_tiempo = []
trayectorias_pu = []

for _ in range(num_simulaciones):
    tiempos, valores_pu = simulacion_gillespie(tiempo_simulacion)
    trayectorias_tiempo.append(tiempos)
    trayectorias_pu.append(valores_pu)
    resultados_pu.append(valores_pu[-1])

resultados_pu = np.array(resultados_pu)
num_critico = np.sum(resultados_pu >= 80)
probabilidad_critica = num_critico / num_simulaciones * 100
print(f"\n 5) Probabilidad de que Pu llegue a 80 o más: {probabilidad_critica:.3f}%")
print(resultados_pu)

plt.figure(figsize=(10, 6))
for tiempos, valores_pu in zip(trayectorias_tiempo, trayectorias_pu):
    plt.plot(tiempos, valores_pu, color='red', alpha=0.1)
plt.plot(sol.t, sol.y[2], label="Pu - Solución Determinista", color="blue", linewidth=2)
plt.xlabel("Tiempo (días)")
plt.ylabel("Cantidad de Plutonio-239 (Pu)")
plt.title("Comparación entre la solución determinista y las simulaciones estocásticas de Pu")
plt.legend()
plt.grid()
plt.savefig("Tarea4/5.pdf")

plt.figure(figsize=(10, 6))
plt.hist(resultados_pu, bins=101, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel("Cantidad final de Pu")
plt.ylabel("Frecuencia")
plt.title("Histograma de los valores finales de Pu en las simulaciones de Gillespie")
plt.grid()

#plt.show()

print("Para disminuir la probabilidad de alcanzar la criticalidad se puede retirar más plutonio por día, "
"así como se puede insertar menos uranio a la planta, de manera a generar menos neptunio y plutonio.")