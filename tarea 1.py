import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.optimize import curve_fit
import pandas as pd

data = pd.read_csv("Rhodium.csv")
x_data = data['Wavelength (pm)']
y_data = data['Intensity (mJy)']


plt.figure(1)

plt.scatter(x_data, y_data, label="Data")
plt.xlabel('Wavelength (pm)')
plt.ylabel('Intensity (mJy)')
plt.title('Intensity vs Wavelength')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=1)
plt.tight_layout()
plt.show()

