import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.optimize import curve_fit
import pandas as pd

data = pd.read_excel("Rhodium.csv")
x_data2 = data['Wavelength (pm)']
y_data2 = data['Intensity (mJy)']

def model_function(x, a, b, c):
    # Handling the sinc function behavior at x=0 to avoid division by zero
    sinc_x = np.sin(c* np.sin(x)) / np.where(x == 0, 1, c* np.sin(x))
    return a * (np.cos(b * np.sin(x)))**2 * (sinc_x **2)

popt2, pcov1 = curve_fit(model_function, x_data2, y_data2, p0=[650,1500,1500], maxfev=10000)
perr2 = np.sqrt(np.diag(pcov1))

plt.figure(1)

plt.scatter(x_data2, y_data2, label="Data")
plt.xlabel('\u03b8 (rad)')
plt.ylabel('V (mV)')
plt.title('\u03b8 vs Voltaje inducido')
plt.plot([], [], ' ', label=f'a = {popt2[0]:.2f} ± {perr2[0]:.2f} [mV]')
plt.plot([], [], ' ', label=f'b = {popt2[1]:.2f} ± {perr2[1]:.2f} [rad]')
plt.plot([], [], ' ', label=f'c = {popt2[2]:.2f} ± {perr2[2]:.2f} [rad]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=1)
plt.tight_layout()
plt.show()
