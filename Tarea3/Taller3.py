import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.cm as cm


'''Ejercicio 4: Cuantización de la energía'''

def schrodinger(x, y, E): #Se define la ecuación
    """
    y[0] = f(x), y[1] = f'(x).
    """
    return [y[1], (x**2 - 2*E)*y[0]]


def resolver_sistema(E, x_span, condiciones_iniciales): #Se implementa scipy.solve_ivp()

    sol = solve_ivp(
        schrodinger,
        x_span,
        condiciones_iniciales,
        args=(E,),
        t_eval=np.linspace(*x_span, 1000),
        method='RK45',
        max_step=0.1
    )
    return sol

def encontrar_energias(E_values, x_span, condiciones_iniciales):
    energias = []
    soluciones = []
    for E in E_values:
        sol = resolver_sistema(E, x_span, condiciones_iniciales)
        # Evaluamos en la frontera
        f0_final = sol.y[0, -1]
        f1_final = sol.y[1, -1]
        if np.square(f0_final**2 + f1_final**2) < 0.5: #Condición
            energias.append(E)
            soluciones.append(sol)
    return energias, soluciones


x_span = (0, 6)            # Integramos de 0 a 6
E_values = np.arange(0, 12, 0.01)  # Valores de energía a explorar

# Condiciones iniciales para estados simétricos y antisimétricos
cond_simetricas = [1, 0]
cond_antisimetricas = [0, 1]

# Obtenemos energías y soluciones
energias_simetricas, sols_simetricas = encontrar_energias(E_values, x_span, cond_simetricas)
energias_antisimetricas, sols_antisimetricas = encontrar_energias(E_values, x_span, cond_antisimetricas)


#Función auxiliar para 'reflejar' la solución x>0 a x<0 ---

def reflejar_solucion(sol, es_antisimetra=False):
    """
    Dada una solución 'sol' de solve_ivp en [0, x_max],
    la reflejamos a x<0 para obtener la forma completa en [-x_max, x_max].
    es_antisimetra=True => cambia el signo de la parte negativa (cond. de frontera).
    """
    x_neg = -sol.t[::-1]
    f_neg = sol.y[0][::-1]
    if es_antisimetra:
        f_neg = -f_neg

    x_total = np.concatenate([x_neg, sol.t[1:]])
    f_total = np.concatenate([f_neg, sol.y[0][1:]])
    return x_total, f_total

# Graficar

plt.figure(figsize=(6, 6))


x_pot = np.linspace(-5, 5, 300)
V = 0.5 * x_pot**2
plt.plot(x_pot, V, '--', color='gray')


escala = 0.3

num_estados_a_graficar = 5


for i, (E, sol) in enumerate(zip(energias_simetricas[:num_estados_a_graficar],
                                 sols_simetricas[:num_estados_a_graficar])):
    x_full, f_full = reflejar_solucion(sol, es_antisimetra=False)

    max_f = np.max(np.abs(f_full)) or 1.0
    f_norm = f_full / max_f
    f_plot = E + escala * f_norm

    plt.plot(x_full, f_plot)



for i, (E, sol) in enumerate(zip(energias_antisimetricas[:num_estados_a_graficar],
                                 sols_antisimetricas[:num_estados_a_graficar])):
    x_full, f_full = reflejar_solucion(sol, es_antisimetra=True)

    max_f = np.max(np.abs(f_full)) or 1.0
    f_norm = f_full / max_f
    f_plot = E + escala * f_norm
    plt.plot(x_full, f_plot)

plt.xlabel("x")
plt.ylabel("Energía")

plt.figsize=(6, 10)
plt.xlim(-6, 6)
plt.ylim(0, 11)
plt.grid(True)
plt.savefig("Tarea3/4.pdf")

#PROBLEMA 3

import matplotlib.animation as animation

#3a
# Constantes
mu = 39.4234021 
a = 0.38709893  # UA (semieje mayor)
e = 0.20563069  # Excentricidad
alpha = 0.01  # Factor relativista exagerado para efectos visuales

# Condiciones iniciales
x0 = a * (1 + e)
y0 = 0
vx0 = 0
vy0 = np.sqrt(mu / a) * np.sqrt((1 - e) / (1 + e))

# Ecuaciones de movimiento
def equations(t, Y):
    x, y, vx, vy = Y
    r = np.sqrt(x**2 + y**2)
    factor = 1 + alpha / r**2  # Corrección relativista
    ax = -mu * x / r**3 * factor
    ay = -mu * y / r**3 * factor
    return [vx, vy, ax, ay]

# Simulación con solve_ivp
t_span = (0, 10)  # Simular 10 años
t_eval = np.linspace(0, 10, 1000)  # Puntos de evaluación
sol = solve_ivp(equations, t_span, [x0, y0, vx0, vy0], method='RK45', t_eval=t_eval, max_step=1e-3)

x_sol = np.array(sol.y[0])
y_sol = np.array(sol.y[1])

# Crear la figura
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("x (UA)")
ax.set_ylabel("y (UA)")
ax.set_title("Órbita de Mercurio")
ax.plot(0, 0, 'yo', markersize=10, label="Sol")  # Representar el Sol
line, = ax.plot([], [], 'r-', lw=1.5)
point, = ax.plot([], [], 'bo', markersize=5)
ax.legend()

# Función de actualización de la animación
def update(frame):
    line.set_data(x_sol[:frame+1], y_sol[:frame+1])
    point.set_data([x_sol[frame]], [y_sol[frame]])
    return line, point


# Crear la animación
ani = animation.FuncAnimation(fig, update, frames=len(x_sol), interval=10, blit=False)

# Guardar animación en MP4
ani.save("3.a.mp4", writer="ffmpeg", fps=30)

#3.b
from scipy.signal import argrelextrema

# Calcular la distancia radial al Sol
r_sol = np.sqrt(x_sol**2 + y_sol**2)

# Encontrar periastros y apoastros en toda la simulación
peri_idxs_all = argrelextrema(r_sol, np.less)[0]
apo_idxs_all = argrelextrema(r_sol, np.greater)[0]

# Definir el nuevo rango de tiempo en años (4 a 6 años) para que aparezcan solo un par de lineas
#si no hacemos esto se repiten las lineas ya que los datos son periodicos
t_min, t_max = 4, 5.2

# Filtrar periastros dentro del rango [4, 6] años
peri_idxs = peri_idxs_all[(sol.t[peri_idxs_all] >= t_min) & (sol.t[peri_idxs_all] <= t_max)]
time_peri_cent = sol.t[peri_idxs] / 100  # Convertir a siglos
theta_peri = np.arctan2(y_sol[peri_idxs], x_sol[peri_idxs])
theta_peri = np.mod(theta_peri, 2 * np.pi)
theta_peri_arcsec = np.degrees(theta_peri) * 3600

# Filtrar apoastros dentro del rango [4, 6] años
apo_idxs = apo_idxs_all[(sol.t[apo_idxs_all] >= t_min) & (sol.t[apo_idxs_all] <= t_max)]
time_apo_cent = sol.t[apo_idxs] / 100  # Convertir a siglos
theta_apo = np.arctan2(y_sol[apo_idxs], x_sol[apo_idxs])
theta_apo = np.mod(theta_apo, 2 * np.pi)
theta_apo_arcsec = np.degrees(theta_apo) * 3600

# Ajuste lineal solo con los datos filtrados
p_peri = np.polyfit(time_peri_cent, theta_peri_arcsec, 1)
p_apo = np.polyfit(time_apo_cent, theta_apo_arcsec, 1)

precesion_peri_arcsec_por_siglo = p_peri[0] * 100
precesion_apo_arcsec_por_siglo = p_apo[0] * 100

# Graficar precesión con el nuevo rango de tiempo
plt.figure(figsize=(8, 5))
plt.plot(time_peri_cent, theta_peri_arcsec, 'bo', label='Periastros')
plt.plot(time_peri_cent, np.polyval(p_peri, time_peri_cent), 'b-', label=f'Peri: {precesion_peri_arcsec_por_siglo:.2f} arcsec/siglo')

plt.plot(time_apo_cent, theta_apo_arcsec, 'ro', label='Apoastros')
plt.plot(time_apo_cent, np.polyval(p_apo, time_apo_cent), 'r-', label=f'Apo: {precesion_apo_arcsec_por_siglo:.2f} arcsec/siglo')

plt.xlabel("Tiempo (siglos)")
plt.ylabel("Precesión (arcosegundos)")
plt.legend()
plt.xlim(t_min / 100, t_max / 100)  # Ajustar a siglos (4 a 6 años = 0.04 a 0.06 siglos)
plt.savefig("3.b.pdf")

print(f"Precesión periastros: {precesion_peri_arcsec_por_siglo:.2f} arcsec/siglo")
print(f"Precesión apoastros: {precesion_apo_arcsec_por_siglo:.2f} arcsec/siglo")