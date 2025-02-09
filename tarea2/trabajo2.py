import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.signal import peak_widths
import pandas as pd
'''Ejercicio 1'''

# Función para generar datos de prueba
def datos_prueba(t_max: float, dt: float, amplitudes: NDArray[float], # type: ignore
                  frecuencias: NDArray[float], ruido: float = 0.0) -> tuple[NDArray[float], NDArray[float]]: # type: ignore
    ts = np.arange(0., t_max, dt)
    ys = np.zeros_like(ts, dtype=float)
    for A, f in zip(amplitudes, frecuencias):
        ys += A * np.sin(2 * np.pi * f * ts)
    ys += np.random.normal(loc=0, size=len(ys), scale=ruido) if ruido else 0
    return ts, ys

'''1a.'''
from numba import njit
@njit
# Implementación de la transformada de Fourier
def Fourier_multiple(t: NDArray[float], y: NDArray[float], f: NDArray[float]) -> NDArray[complex]:
    N = len(t)
    resultados = np.zeros(len(f))+0.0j
    for i, freq in enumerate(f):
        exp_term = np.exp(-2j * np.pi * t * freq)
        resultados[i] = np.sum(y * exp_term) / N
    return resultados
# Generación de señales
t_max = 1.0
dt = 0.001
amplitudes = np.array([1.0, 0.5, 0.3])
frecuencias = np.array([5.0, 15.0, 30.0])
ruido = 0.5

# Señales con y sin ruido
t, y_sin_ruido = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=0.0)
_, y_con_ruido = datos_prueba(t_max, dt, amplitudes, frecuencias, ruido=ruido)

# Frecuencias para analizar la transformada
frecuencias_analisis = np.linspace(0, 50, 1000)

# Cálculo de las transformadas
transformada_sin_ruido = np.abs(Fourier_multiple(t, y_sin_ruido, frecuencias_analisis))
transformada_con_ruido = np.abs(Fourier_multiple(t, y_con_ruido, frecuencias_analisis))

# Gráficas
plt.figure(1,figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(frecuencias_analisis, transformada_sin_ruido, label="Sin ruido", color = 'orange')
plt.title("Transformada de Fourier (sin ruido)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(frecuencias_analisis, transformada_con_ruido, label="Con ruido", color='yellowgreen')
plt.title("Transformada de Fourier (con ruido)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.legend()

# Guardar gráfico en PDF
plt.savefig("tarea2/1.a.pdf")
print('1.a) Picos similares a los de la transformada sin ruido')

'''1b.'''
# Generar señales de prueba
amplitudesb = np.array([1.0])
frecuenciasb = np.array([10.0])
dtb = 0.005

# Definir rango de tiempos y frecuencias
t_max_values = np.linspace(10, 300, 10)  
frecuencias_eval = np.linspace(5, 15, 5000)

fwhm_values = []

from tqdm import tqdm
for t_max in tqdm(t_max_values):
    t, y = datos_prueba(t_max, dtb, amplitudesb, frecuenciasb)
    transformada = Fourier_multiple(t, y, frecuencias_eval)

    # Calcular el FWHM del pico
    amplitud_transformada = np.abs(transformada)
    pico_indice = np.argmax(amplitud_transformada)
    resultados_fwhm = peak_widths(amplitud_transformada, [pico_indice], rel_height=0.5)
    ancho = resultados_fwhm[0][0] * (frecuencias_eval[1] - frecuencias_eval[0])
    fwhm_values.append(ancho)
    
from scipy.optimize import curve_fit
def modelo_potencia(x, a, b):
    return a * x**b

# Ajuste del modelo
parametros, _ = curve_fit(modelo_potencia, t_max_values, fwhm_values)
a_fit, b_fit = parametros

# Crear gráfica log-log
plt.figure(figsize=(8, 6))
plt.loglog(t_max_values, fwhm_values, 'o', label="FWHM del pico", color="magenta")
plt.loglog(t_max_values, modelo_potencia(t_max_values, a_fit, b_fit), '-', 
           label=f"Ajuste: FWHM = {a_fit:.2} * t_max^{b_fit:.2f}", color="orange")

# Configuración de la gráfica
plt.title("Relación entre FWHM y duración de la señal")
plt.xlabel("Duración de la señal (t_max) [s]")
plt.ylabel("FWHM del pico [Hz]")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

# Exportar la gráfica
plt.savefig("tarea2/1.b.pdf")


'''1c.'''
# Cargar datos del archivo experimental
file_path = 'tarea2/OGLE-LMC-CEP-0001.dat'
data = np.loadtxt(file_path)
t, y, sigma_y = data[:, 0], data[:, 1], data[:, 2]

# Histograma de diferencias temporales
dt_dif = np.diff(t)
plt.figure(figsize=(10, 6))
plt.hist(dt_dif, bins=50, alpha=0.7, color='c', edgecolor='k')
plt.xlabel("Intervalos de tiempo [días]")
plt.ylabel("Frecuencia")
plt.title("Histograma de diferencias temporales")
plt.grid(True, linestyle="--", linewidth=0.5)
#plt.show()

# Transformada de Fourier con señal centrada en promedio cero
y_centrada = y - np.mean(y)
frecuencias_eval = np.linspace(0, 16, 20000)  # Rango 0 a 8 ciclos/día con alta densidad
transformada = Fourier_multiple(t, y_centrada, frecuencias_eval)
amplitud_transformada = np.abs(transformada)

# Gráfico transformada
plt.figure(figsize=(12, 6))
plt.plot(frecuencias_eval, amplitud_transformada, label="Transformada de Fourier")
plt.xlabel("Frecuencia [ciclos/día]")
plt.ylabel("Amplitud")
plt.title("Transformada de Fourier)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
#plt.show()

dt_medio = np.median(np.diff(t))
f_nyquist = 1 / (2 * dt_medio)
print(f"1.c) f Nyquist: {f_nyquist:.3f} ciclos/día")

# Encontrar la frecuencia dominante (f_true)
pico_indice = np.argmax(amplitud_transformada)
f_true = frecuencias_eval[pico_indice]
print(f"1.c) f true: {f_true:.3f} ciclos/día")

plt.figure(figsize=(12, 6))
plt.plot(frecuencias_eval, amplitud_transformada, label="Transformada de Fourier", color='b')
plt.axvline(f_nyquist, color='r', linestyle='--', label=f"f Nyquist: {f_nyquist:.3f}")
plt.axvline(f_true, color='g', linestyle='--', label=f"f True: {f_true:.3f}")
plt.xlabel("Frecuencia [ciclos/día]")
plt.ylabel("Amplitud")
plt.title("Transformada de Fourier con frecuencias destacadas")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
#plt.show()

# Cálculo de la fase y graficación
fase = np.mod(f_true * t, 1)

plt.figure(figsize=(10, 6))
plt.scatter(fase, y_centrada, color='plum')
plt.xlabel("Fase (mod(f_true * t, 1))")
plt.ylabel("Intensidad (y)")
plt.title("Datos en función de la fase")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.savefig('tarea2/1.c.pdf')

'''Ejercicio 2'''


'''2a.'''
data = pd.read_csv("tarea2/H_field.csv", usecols=['t', 'H'])

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

plt.savefig("tarea2/2.a.pdf")

'''2b.'''

#descargar info y skip la primera linea porque esta mal
df = pd.read_csv("tarea2/data_dates.txt", sep='\s+', skiprows=1)

#concentrar la informacion de fechas a una sola columba
df["date"] = pd.to_datetime(df[['Year', 'Month', 'Day']])

#quitar la informacion desde 2012
cutoff = pd.to_datetime("2012-01-01")
df = df[df["date"] < cutoff]
df.drop(columns=['Year', 'Month', 'Day'], inplace=True)

'''2b.a'''

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


'''2b.b'''

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
plt.savefig("tarea2/2.b.pdf")

'''Ejercicio 3'''