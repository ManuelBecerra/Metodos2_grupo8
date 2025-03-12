import numpy as np
import matplotlib.pyplot as plt

'''Ejercicio 1: Integración indirecta'''

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

'''Ejercicio 4: Generación de lenguaje natural con Cadenas de Markov'''