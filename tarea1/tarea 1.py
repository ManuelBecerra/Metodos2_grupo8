import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import pandas as pd
from scipy.integrate import simpson
from scipy.stats import linregress
import re
'''Ejercicio 1'''


'''1a'''
#Para limpiar los datos corruptos se hizo un filtro de mediana (mefilt) el cual calcula la medianta de una ventana de tamaño 5, es decir que compara los dos vecinos de ambos lados.
#Si los datos difieren notoriamente se reemplazan por la mediana
data = pd.read_csv("tarea1/Rhodium.csv")
x_data = data['Wavelength (pm)']
y_data = data['Intensity (mJy)']


def filtered_data(data):
    filtered_intensity = medfilt(data['Intensity (mJy)'], kernel_size=5)
    corrupt_data_mask = data['Intensity (mJy)'] != filtered_intensity
    num_corrupt_data = corrupt_data_mask.sum()
    
    return filtered_intensity, num_corrupt_data

y_data_filtered = filtered_data(data)[0]
plt.figure(1)
plt.plot(x_data, y_data, label="Original data", color = "darkturquoise")
plt.plot(x_data, y_data_filtered, label="Filtered data", color = "coral")
plt.xlabel('Wavelength (pm)')
plt.ylabel('Intensity (mJy)')
plt.title('Intensity vs Wavelength')
plt.legend()
plt.tight_layout()


output_path = "tarea1/limpieza.pdf"
plt.savefig(output_path)

print(f"1.a) Número de datos eliminados : {filtered_data(data)[1]}")

'''1b'''
new_data = pd.DataFrame({'Wavelength (pm)': x_data, 'Intensity (mJy)': y_data_filtered})

new_data.drop(range(260,429), axis=0, inplace=True)

x = np.array([72.8857 , 114.0033])
y = np.array([0.1116 , 0.0547])

slope, intercept, r_value, p_value, std_err = linregress(x, y)

def lineal_regression(x):
  return slope*x + intercept

x_reg = np.linspace(72.8857, 114.0033, 169)
y_reg = lineal_regression(x_reg)

data_reg = pd.DataFrame({'Wavelength (pm)': x_reg, 'Intensity (mJy)': y_reg})
new_data_part1 = new_data.iloc[:260]
new_data_part2 = new_data.iloc[260:]

new_data = pd.concat([new_data_part1, data_reg, new_data_part2], ignore_index=True)

y_data_peaks = y_data_filtered - new_data['Intensity (mJy)']
plt.figure(2)
plt.plot(x_data, y_data_peaks, color= 'lightpink')
plt.xlabel('Wavelength (pm)')
plt.ylabel('Intensity (mJy)')
plt.title('Intensity vs Wavelength Peaks')
plt.legend()
plt.tight_layout()

output_path = "tarea1/picos.pdf"
plt.savefig(output_path)

print("1.b) Método: ")

'''1c'''

print("1.c) ")

'''1d'''
#para este problema usamos un metodo de monte carlo donde simulamos n samples de los datos
#con valores de y random calculados desde una distribucion gaussiana con desviacion estandar 0.02*y
incertidumbre = 0.02
n = 10000

integral_samples = []

#calcular cada integral para cada sample usando simps y agregarlos a una lista
for _ in range(n):
    ruido = y_data_filtered + np.random.normal(0, incertidumbre * y_data_filtered)  # agregando ruido
    integral = simpson(ruido, x_data)  
    integral_samples.append(integral)

#define los resultados de las integrals calculando el promedio de los samples 
#y luego la desviacion estandar para la incertidumbre
integral_samples = np.array(integral_samples)
prom_integral = np.mean(integral_samples)
incert_integral = np.std(integral_samples)

print("1.d) Integral:", round(prom_integral, 3), "\u00B1", round(incert_integral,3), "W/m\u00b2")



'''Ejercicio 2'''

'''2a'''
def procesar_linea(linea):
    numeros = re.findall(r'-?\d+\.?\d*', linea)
    return [float(n) for n in numeros]

datos = []

#procesando la info
with open('tarea1/hysteresis.dat', 'r') as archivo:
    for linea in archivo:
        procesado = procesar_linea(linea.strip())
        if len(procesado) == 3:  
            datos.append(procesado)

data = pd.DataFrame(datos, columns=['t', 'B', 'H'])

for i in range(len(data['H'])):
    if data['H'][i] > 1:
        data.at[i, 'H'] = data['H'][i] / 1000

#graficando la primera 
plt.figure(figsize=(10, 6))
plt.plot(data['t'], data['H'], marker='o', linestyle='-', label='H vs t', color='blue')
plt.plot(data['t'], data['B'], marker='o', linestyle='None', label='B vs t', color='red')
plt.title('Gráfica de H y B en función de t')
plt.xlabel('t')
plt.ylabel('Valores de H y B')
plt.legend()
plt.grid()

plt.savefig('histérico.pdf')

'''2b'''

t = data['t']  
B = data['B'] * 1e-3
H = data['H']

delta_t = np.mean(np.diff(t))  

#transformada de fourier
frecuencias = np.fft.fftfreq(len(B), d=delta_t) 
espectro = np.fft.fft(B)  

#determinando las frecuencias positivas y el espectro entero
frecuencias_positivas = frecuencias[frecuencias > 0]
espectro_positivo = 2.0 / len(B) * np.abs(espectro[frecuencias > 0])

#encontrando la mayor frecuencia positiva
frecuencia_dominante = frecuencias_positivas[np.argmax(espectro_positivo)]
print(f" 2.b) La frecuencia dominante es {frecuencia_dominante:.4f} Hz")

print("Se utilizó una transformada de fourier discreta para pasar los datos de B de un espacio de tiempo a un espacio de frecuencias y despues se encontró la frecuencia positiva mayor")

'''2c'''

energia_perdida = np.trapezoid(H, B)
print(f"2.c) La energía perdida por unidad de volumen es {energia_perdida:.4e} J/m³")

plt.figure(figsize=(10, 6))
plt.plot(B, H, marker='o', linestyle='-', label='Ciclo de Histéresis')
plt.title('Ciclo de Histéresis (H vs B)')
plt.xlabel('B (T)')  
plt.ylabel('H (A/m)')  
plt.grid()
plt.legend()

plt.savefig('energy.pdf')