import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

'''Ejercicio 1'''

'''1a.'''
# Función para generar datos de prueba
def datos_prueba(t_max: float, dt: float, amplitudes: NDArray[float],
                  frecuencias: NDArray[float], ruido: float = 0.0) -> tuple[NDArray[float], NDArray[float]]:
    ts = np.arange(0., t_max, dt)
    ys = np.zeros_like(ts, dtype=float)
    for A, f in zip(amplitudes, frecuencias):
        ys += A * np.sin(2 * np.pi * f * ts)
    ys += np.random.normal(loc=0, size=len(ys), scale=ruido) if ruido else 0
    return ts, ys

# Implementación de la transformada de Fourier
def Fourier_multiple(t: NDArray[float], y: NDArray[float], f: NDArray[float]) -> NDArray[complex]:
    t = t[:, np.newaxis]  # Expande t para hacer operaciones vectorizadas
    exponentes = np.exp(-2j * np.pi * f * t)  # Exponentes vectorizados
    return np.dot(y, exponentes) / len(t)  # Transformada discreta de Fourier

# Generación de señales
t_max = 1.0
dt = 0.001
amplitudes = np.array([1.0, 0.5, 0.3])
frecuencias = np.array([5.0, 15.0, 30.0])
ruido = 1

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
plt.plot(frecuencias_analisis, transformada_sin_ruido, label="Sin ruido")
plt.title("Transformada de Fourier (sin ruido)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(frecuencias_analisis, transformada_con_ruido, label="Con ruido", color='orange')
plt.title("Transformada de Fourier (con ruido)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True)
plt.legend()

# Guardar gráfico en PDF
plt.savefig("1.a.pdf")
#plt.show()

# Respuesta a la pregunta sobre el efecto del ruido
print("1.a) El ruido disminuye la amplitud de la frecuencia fundamental.")

'''1b.'''
# Parámetros
amplitud = np.array([5])
frecuencia = np.array([20.0])
dt = 0.01

# Valores de t_max a analizar
t_max_values = np.linspace(10, 300, 10)
fwhm_values = []

# Análisis para distintos t_max
for t_max in t_max_values:
    t, y_sin_ruido = datos_prueba(t_max, dt, amplitud, frecuencia)
    frecuencias_analisis = np.fft.fftfreq(len(t), d=dt)
    transformada_sin_ruido = np.abs(Fourier_multiple(t, y_sin_ruido, frecuencias_analisis))
    
    # Encontrar el ancho a media altura (FWHM) sin usar scipy.signal
    max_index = np.argmax(transformada_sin_ruido)
    half_max = transformada_sin_ruido[max_index] / 2
    
    left_idx = np.where(transformada_sin_ruido[:max_index] <= half_max)[0]
    right_idx = np.where(transformada_sin_ruido[max_index:] <= half_max)[0]
    
    if len(left_idx) > 0 and len(right_idx) > 0:
        left_idx = left_idx[-1]
        right_idx = max_index + right_idx[0]
        fwhm = frecuencias_analisis[right_idx] - frecuencias_analisis[left_idx]
    else:
        fwhm = np.nan
    
    fwhm_values.append(fwhm)

# Graficar FWHM vs t_max en escala log-log
plt.figure(figsize=(6,6))
plt.loglog(t_max_values, fwhm_values, 'o-', label="FWHM")
plt.xlabel("t_max (s)")
plt.ylabel("FWHM (Hz)")
plt.title("Ancho a media altura vs t_max")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.savefig("1.b.pdf")
plt.show()
