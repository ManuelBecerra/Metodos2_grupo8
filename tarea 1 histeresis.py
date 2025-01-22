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

plt.figure(figsize=(10, 6))
plt.plot(data['t'], data['H'], marker='o', linestyle='-', label='H vs t', color='blue')
plt.plot(data['t'], data['B'], marker='s', linestyle='None', label='B vs t', color='red')
plt.title('Gráfica de H y B en función de t')
plt.xlabel('t')
plt.ylabel('Valores de H y B')
plt.legend()
plt.grid()

plt.savefig('grafica_H_B_vs_t.pdf')

plt.show()