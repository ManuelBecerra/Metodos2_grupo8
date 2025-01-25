import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import pandas as pd
from scipy.integrate import simps

'''Ejercicio 1'''


'''1a'''
data = pd.read_csv("Rhodium.csv")
x_data = data['Wavelength (pm)']
y_data = data['Intensity (mJy)']


def filtered_data(data):
    filtered_intensity = medfilt(data['Intensity (mJy)'], kernel_size=5)
    corrupt_data_mask = data['Intensity (mJy)'] != filtered_intensity
    num_corrupt_data = corrupt_data_mask.sum()
    
    return filtered_intensity, num_corrupt_data

y_data_filtered = filtered_data(data)[0]

plt.plot(x_data, y_data, label="Original data", color = "darkturquoise")
plt.plot(x_data, y_data_filtered, label="Filtered data", color = "coral")
plt.xlabel('Wavelength (pm)')
plt.ylabel('Intensity (mJy)')
plt.title('Intensity vs Wavelength')
plt.legend()
plt.tight_layout()


output_path = "limpieza.pdf"
plt.savefig(output_path)

print(f"1.a) Número de datos eliminados : {filtered_data(data)[1]}")

'''1b'''


print("1.b) Método: ")

#1.c

print("1.c) ")

#1.d
#para este problema usamos un metodo de monte carlo donde simulamos n samples de los datos
#con valores de y random calculados desde una distribucion gaussiana con desviacion estandar 0.02*y
incertidumbre = 0.02
n = 10000

integral_samples = []

#calcular cada integral para cada sample usando simps y agregarlos a una lista
for _ in range(n):
    ruido = y_data_filtered + np.random.normal(0, incertidumbre * y_data_filtered)  # agregando ruido
    integral = simps(ruido, x_data)  
    integral_samples.append(integral)

#define los resultados de las integrals calculando el promedio de los samples 
#y luego la desviacion estandar para la incertidumbre
integral_samples = np.array(integral_samples)
prom_integral = np.mean(integral_samples)
incert_integral = np.std(integral_samples)

print("1.d) Integral:", round(prom_integral, 3), "\u00B1", round(incert_integral,3), "W/m\u00b2")