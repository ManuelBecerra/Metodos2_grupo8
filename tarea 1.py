import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import pandas as pd

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

print(f"1.a) Número de datos eliminados : {filtered_data(data)[1]}")
plt.plot(x_data, y_data, label="Original data", color = "darkturquoise")
plt.plot(x_data, y_data_filtered, label="Filtered data", color = "coral")
plt.xlabel('Wavelength (pm)')
plt.ylabel('Intensity (mJy)')
plt.title('Intensity vs Wavelength')
plt.legend()
plt.tight_layout()


output_path = "limpieza.pdf"
plt.savefig(output_path)
plt.show()

print(f"1.a) Número de datos eliminados : {filtered_data(data)[1]}")

'''1b'''
