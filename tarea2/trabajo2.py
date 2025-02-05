import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.signal import peak_widths

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
# Implementación de la transformada de Fourier
def Fourier_multiple(t: NDArray[float], y: NDArray[float], f: NDArray[float]) -> NDArray[complex]:
    N = len(t)
    resultados = np.zeros(len(f), dtype=complex)
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


'''1b.'''
# Generar señales de prueba
amplitudesb = np.array([1.0])
frecuenciasb = np.array([10.0])
dtb = 0.001

# Definir rango de tiempos y frecuencias
t_max_values = np.linspace(10, 300, 10)  
frecuencias_eval = np.linspace(0, 50, 500)

fwhm_values = []

for t_max in t_max_values:
    t, y = datos_prueba(t_max, dtb, amplitudesb, frecuenciasb)
    transformada = Fourier_multiple(t, y, frecuencias_eval)

    # Calcular el FWHM del pico
    amplitud_transformada = np.abs(transformada)
    pico_indice = np.argmax(amplitud_transformada)
    resultados_fwhm = peak_widths(amplitud_transformada, [pico_indice], rel_height=0.5)
    ancho = resultados_fwhm[0][0] * (frecuencias_eval[1] - frecuencias_eval[0])
    fwhm_values.append(ancho)

# Graficar el FWHM en escala log-log
plt.figure(figsize=(10, 6))
plt.loglog(t_max_values, fwhm_values, marker="o", label="FWHM del pico", color= 'magenta')
plt.xlabel("Duración de la señal (t_max) [s]")
plt.ylabel("FWHM del pico [Hz]")
plt.title("Relación entre FWHM y duración de la señal")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.savefig("tarea2/1.b.pdf")

print("1.b: La relación entre FWHM y t_max sigue una tendencia aproximadamente proporcional a 1/t_max.")