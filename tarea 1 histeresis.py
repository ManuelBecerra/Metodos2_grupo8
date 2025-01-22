import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import pandas as pd
import re

def procesar_linea(linea):
    numeros = re.findall(r'-?\d+\.?\d*', linea)
    return [float(n) for n in numeros]

datos = []
with open('hysteresis.dat', 'r') as archivo:
    for linea in archivo:
        procesado = procesar_linea(linea.strip())
        if len(procesado) == 3:  
            datos.append(procesado)

data = pd.DataFrame(datos, columns=['t', 'B', 'H'])

for i in range(len(data['H'])):
    if data['H'][i] > 1:
        data.at[i, 'H'] = data['H'][i] / 1000

plt.figure(figsize=(10, 6))
plt.plot(data['t'], data['H'], marker='o', linestyle='-', label='H vs t', color='blue')
plt.plot(data['t'], data['B'], marker='o', linestyle='None', label='B vs t', color='red')
plt.title('Gráfica de H y B en función de t')
plt.xlabel('t')
plt.ylabel('Valores de H y B')
plt.legend()
plt.grid()

plt.savefig('histérico.pdf')

t = data['t']  
B = data['B'] * 1e-3
H = data['H']

delta_t = np.mean(np.diff(t))  

frecuencias = np.fft.fftfreq(len(B), d=delta_t) 
espectro = np.fft.fft(B)  

frecuencias_positivas = frecuencias[frecuencias > 0]
espectro_positivo = 2.0 / len(B) * np.abs(espectro[frecuencias > 0])

frecuencia_dominante = frecuencias_positivas[np.argmax(espectro_positivo)]
print(f"La frecuencia dominante es {frecuencia_dominante:.4f} Hz")

energia_perdida = np.trapezoid(H, B)
print(f"La energía perdida por unidad de volumen es {energia_perdida:.4e} J/m³")

plt.figure(figsize=(10, 6))
plt.plot(B, H, marker='o', linestyle='-', label='Ciclo de Histéresis')
plt.title('Ciclo de Histéresis (H vs B)')
plt.xlabel('B (T)')  
plt.ylabel('H (A/m)')  
plt.grid()
plt.legend()

plt.savefig('energy.pdf')