import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.optimize import curve_fit
import pandas as pd

data = pd.read_excel("lab moderna - Doble rendija.xlsx")
x_data2 = data['Teta_D (rad)']
y_data2 = data['V_D (mv)']
