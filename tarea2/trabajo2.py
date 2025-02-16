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
plt.hist(dt_dif, bins=np.linspace(0,5,300), alpha=0.7, color='c', edgecolor='k')
plt.xlabel("Intervalos de tiempo [días]")
plt.ylabel("Frecuencia")
plt.title("Histograma de diferencias temporales")
plt.grid(True, linestyle="--", linewidth=0.5)
#plt.show()

# Transformada de Fourier con señal centrada en promedio cero
y_centrada = y - np.mean(y)
frecuencias_eval = np.linspace(0, 8, 40000)  # Rango 0 a 8 ciclos/día con alta densidad
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

t = data['t'].values
H = data['H'].values

delta_t = np.mean(np.diff(t))

#transformada de fourier
frecuencias = np.fft.rfftfreq(len(H),delta_t)
espectro = np.fft.fft(H)

#aislando las frecuencias positivas 
frecuencias_positivas = frecuencias[frecuencias > 0]
espectro_positivo = 2.0 / len(H) * np.abs(espectro[:len(frecuencias)][frecuencias > 0])

#calculando las frecuencias 
f_fast = frecuencias_positivas[np.argmax(espectro_positivo)]

frecuencias2a =np.linspace(0, frecuencias[-1], 1000)
transf_general = Fourier_multiple(t, H, frecuencias2a)
transf_general *= 2.0 / len(H)
amplitud_transformada2a = np.abs(transf_general)
pico_indice2a = np.argmax(amplitud_transformada2a)
f_general = frecuencias2a[pico_indice2a]
plt.figure()


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

import datetime

def fourier_inverse_manual(t, X_k, f_k, N):
    """Reconstrucción manual de la transformada inversa de Fourier."""
    reconstruccion = np.zeros_like(t, dtype=np.float64)
    for k in range(len(f_k)):  # Iteramos sobre las frecuencias
        reconstruccion += (2.0 / N) * np.real(X_k[k] * np.exp(2j * np.pi * f_k[k] * t))
    return reconstruccion

# Parámetros
n = 50  # Número de armónicos a considerar
N = len(M)  # Número total de datos

# Filtrar los primeros M armónicos
spectrum[n:] = 0  # Anulamos los armónicos superiores a M

# Transformada inversa con los primeros M armónicos
inversa_M = fourier_inverse_manual(t_numerico, spectrum, frequencies, N)

# Definir el día actual
today = pd.Timestamp(datetime.datetime.today())

# Extender los datos hasta más allá de 2025
extension = 365 * 15  # Extender 15 años
extendido_t = pd.date_range(start=df["date"].min(), periods=N + extension, freq='D')
extendido_t_numerico = (extendido_t - extendido_t.min()).days

# Transformada inversa extendida
extendido_M = fourier_inverse_manual(extendido_t_numerico, spectrum, frequencies, N)

# Interpolación para predecir el número de manchas solares hoy
n_manchas_hoy = np.interp(today.timestamp(), extendido_t_numerico, extendido_M)
print(f'2.b.b) {n_manchas_hoy = :.1f}')

# Graficar los datos observados y la predicción
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["SSN"], label="Observado", linestyle='-', marker='o', markersize=3, color='blue')
plt.plot(extendido_t, extendido_M, label="SSN Predecido (FFT)", linestyle='--', color='red')
plt.axvline(today, color="black", linestyle=":", label="Hoy")
plt.xlabel("Fecha")
plt.ylabel("Número de manchas solares (SSN)")
plt.title("Predicción de manchas solares usando FFT")
plt.legend()
plt.xticks(rotation=45)
plt.grid()

# Guardar la gráfica en 2.b.pdf
plt.savefig("2.b.pdf")
plt.show()

'''Ejercicio 3'''

'''3a.'''

#descargar info y skip la primera linea porque esta mal
df = pd.read_csv("tarea2/data_dates.txt", sep='\s+', skiprows=1)

#concentrar la informacion de fechas a una sola columba
df["date"] = pd.to_datetime(df[['Year', 'Month', 'Day']])

#quitar la informacion del último mes de 2017 ya que no contiene datos de SSN
cutoff = pd.to_datetime("2017-07-01")
df = df[df["date"] < cutoff]
df.drop(columns=['Year', 'Month', 'Day'], inplace=True)

#Obtener y utilizar solamente los datos de SSN
signal = df.iloc[:, 0].values
N = len(signal)
t = np.arange(N)

# Transformada de Fourier y frecuencias
dft_signal = np.fft.fft(signal)
freqs = np.fft.fftfreq(N)

# Diferentes valores de alfa para el filtro
alphas = [1, 10, 100]

fig, axes = plt.subplots(len(alphas), 2, figsize=(10, 8))

for i, alpha in enumerate(alphas):
    # Filtro gaussiano
    gaussian_filter = np.exp(- (freqs * alpha) ** 2)
    filtered_signal = np.fft.ifft(dft_signal * gaussian_filter).real

    # Graficar señal original y filtrada
    axes[i, 0].plot(t, signal, label='Original')
    axes[i, 0].plot(t, filtered_signal, label=f'Filtrada (α={alpha})')
    axes[i, 0].legend()
    axes[i, 0].set_title(f'Señal - α={alpha}')

    # Graficar transformada
    axes[i, 1].plot(freqs, np.abs(dft_signal), label='Original')
    axes[i, 1].plot(freqs, np.abs(dft_signal * gaussian_filter), label='Filtrada')
    axes[i, 1].legend()
    axes[i, 1].set_title(f'Transformada - α={alpha}')

    # Indicar el valor de alpha
    axes[i, 0].text(0.45 * N, max(signal) * 0.9, f'α={alpha}', fontsize=12)

plt.tight_layout()
plt.savefig("tarea2/3.1.pdf")

'''3b.'''

from PIL import Image
'''Figura castillo'''
#Se abre la imagen y se saca la matriz de la imagen
castle_im = Image.open("tarea2/Noisy_Smithsonian_Castle.jpg")
castle_mat = np.array(castle_im)

#Se hace la transformada de fourier en 2d de la imagen y se le hace shift para
#que quede centralizada
castle_FFT = np.fft.fftshift(np.fft.fft2(castle_mat))

#Se elimina el ruido horizontal, el cual se ve principalmente en la vertical
#centralizada de la imagen de frecuencias
castle_FFT[380:384,412] = 0
castle_FFT[406:409,412] = 0
castle_FFT[432,412] = 0
castle_FFT[356:359,412] = 0
castle_FFT[332,412] = 0
castle_FFT[0:358,511:513] = 0
castle_FFT[407:765,511:513] = 0
castle_FFT[382,612] = 0
castle_FFT[406:409,612] = 0
castle_FFT[432,612] = 0
castle_FFT[356:359,612] = 0
castle_FFT[332,612] = 0

#Se realiza el proceso inverso de transformadas para volver a la imagen original
#corregida
castle_im_new = np.fft.ifft2(np.fft.ifftshift(castle_FFT)).real

#Visualización de la FFT de la imagen
#plt.figure(figsize=[10,15])
#plt.imshow(abs(castle_FFT), norm="log")

plt.figure()
plt.matshow(castle_im_new, cmap="gray")
plt.savefig("tarea2/3.b.a.png")

'''Figura gato'''
cat_im = Image.open("tarea2/catto.png")
cat_mat = np.array(cat_im)

cat_FFT = np.fft.fftshift(np.fft.fft2(cat_mat))
for i in [-1,0,1]:
  cat_FFT[i+391,350:373] = 0
  cat_FFT[i+403,330:369] = 0
  cat_FFT[i+415,320:365] = 0
  cat_FFT[i+427,317:362] = 0
  cat_FFT[i+439,312:357] = 0
  cat_FFT[i+451:453,310:355] = 0
  cat_FFT[i+463:465,305:350] = 0
  cat_FFT[i+475:477,300:345] = 0
  cat_FFT[i+488,298:343] = 0
  cat_FFT[i+500,295:340] = 0
  cat_FFT[i+512,294:339] = 0
  cat_FFT[i+524,290:335] = 0
  cat_FFT[i+536,286:331] = 0
  cat_FFT[i+548,282:327] = 0
  cat_FFT[i+560,278:323] = 0
  cat_FFT[i+572,274:319] = 0

  cat_FFT[i+367,378:401] = 0
  cat_FFT[i+355,382:421] = 0
  cat_FFT[i+343,386:431] = 0
  cat_FFT[i+331,389:434] = 0
  cat_FFT[i+319,394:439] = 0
  cat_FFT[i+306:308,396:441] = 0
  cat_FFT[i+294:296,401:446] = 0
  cat_FFT[i+282:284,404:449] = 0
  cat_FFT[i+270,406:451] = 0
  cat_FFT[i+258,409:454] = 0
  cat_FFT[i+246,411:456] = 0
  cat_FFT[i+234,415:460] = 0
  cat_FFT[i+222,419:464] = 0
  cat_FFT[i+210,423:468] = 0
  cat_FFT[i+198,427:472] = 0
  cat_FFT[i+186,431:476] = 0

cat_im_new = np.fft.ifft2(np.fft.ifftshift(cat_FFT)).real

#Visualización de las FFT de la imagen
#plt.figure(figsize=[10,10])
#plt.imshow(abs(cat_FFT), norm="log")

plt.figure()
plt.matshow(cat_im_new, cmap="gray")
plt.savefig("tarea2/3.b.b.png")