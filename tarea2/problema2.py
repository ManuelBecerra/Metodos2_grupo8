import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import pandas as pd
from scipy.optimize import curve_fit

#PROBLEMA 2
#2.a
data = pd.read_csv("H_field.csv", usecols=['t', 'H'])

t = data['t']
H = data['H']

delta_t = np.mean(np.diff(t))

#transformada de fourier
frecuencias = np.fft.rfftfreq(len(H),delta_t)
espectro = np.fft.fft(H)

#aislando las frecuencias positivas 
frecuencias_positivas = frecuencias[frecuencias > 0]
espectro_positivo = 2.0 / len(H) * np.abs(espectro[:len(frecuencias)][frecuencias > 0])

#calculando las frecuencias 
f_fast = frecuencias_positivas[np.argmax(espectro_positivo)]
f_general = 0.4873
print(f"2.a) {f_fast = :.5f}; {f_general = :.5f}")

#calcular las fases
phi_fast = np.mod(f_fast * t, 1)
phi_general = np.mod(f_general * t, 1)

#Graficar H vs φ_fast y φ_general
plt.figure(figsize=(10, 5))
plt.scatter(phi_fast, H, label=r"$H$ vs $\varphi_{\text{fast}}$", s=5, alpha=0.7)
plt.scatter(phi_general, H, label=r"$H$ vs $\varphi_{\text{general}}$", s=5, alpha=0.7)
plt.xlabel("Fase")
plt.ylabel("H")
plt.legend()
plt.grid(True)

plt.savefig("2.a.pdf")

#2.b

#descargar info y skip la primera linea porque esta mal
df = pd.read_csv("data_dates.txt", delim_whitespace=True, skiprows=1)

#concentrar la informacion de fechas a una sola columba
df["date"] = pd.to_datetime(df[['Year', 'Month', 'Day']])

#quitar la informacion desde 2012
cutoff = pd.to_datetime("2012-01-01")
df = df[df["date"] < cutoff]
df.drop(columns=['Year', 'Month', 'Day'], inplace=True)

#2.b.a.

#organizando datos para que no hayan duplicados
M = df['SSN']
t2 = df['date'].sort_values().reset_index(drop=True)
t2 = t2.drop_duplicates()

#convertir las fechas a dias numericos y organizar
t_numerico = (t2 - t2.min()).dt.total_seconds() / (24 * 3600)
t_numerico = np.sort(t_numerico)
M = df['SSN'].dropna()

#computar deltat
deltat2 = np.mean(np.diff(t_numerico))

#transformacion de fourier 
frequencies = np.fft.rfftfreq(len(M), deltat2)
spectrum = np.fft.rfft(M)
amplitudes = np.abs(spectrum)

#ignorando las frecuencias que no nos sirve, como las cero
indices = frequencies > 0  
dominant_freq = frequencies[indices][np.argmax(amplitudes[indices])]

#calculando el periodo
P_solar = 1 / dominant_freq 
P_solar = P_solar/365

print(f"2.b.a) P_solar = {P_solar:.2f}")


#2.b.b.

#hacer la transformada inversa
inversa_M = np.fft.irfft(spectrum, n=len(M)) 
import datetime

#definir el dia actual
today = pd.Timestamp(datetime.datetime.today())

#extender los datos hasta mas de 2025
extension = 365 * 13 
extendido_t = pd.date_range(start=t2.min(), periods=len(M) + extension, freq='D')

#calcular los valores corresponientes para los tiempos
extendido_M = np.fft.irfft(spectrum, n=len(extendido_t))

#convertir los dias extendidos a valores numericos
extendido_t_numerico = (extendido_t - extendido_t.min()).days

#interpolar para predecir
n_manchas_hoy = np.interp(today.timestamp(), extendido_t_numerico, extendido_M)
print(f'2.b.b) {n_manchas_hoy = :.1f}')

#graficar
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['SSN'], label="Observado", linestyle='-', marker='o', markersize=3, color='blue')
plt.plot(extendido_t, extendido_M, label="SSN Predecido (FFT)", linestyle='--', color='red')
plt.xlabel("Fecha")
plt.ylabel("SSN)")
plt.legend()
plt.xticks(rotation=45)
plt.savefig("2.b.pdf")