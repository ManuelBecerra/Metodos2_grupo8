import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.cm as cm
from numba import njit


'''Ejercicio 1: Balística'''
# Constantes
g = 9.773  # Gravedad en Bogotá (m/s^2)
m = 10     # Masa del proyectil (kg)
v0 = 10    # Velocidad inicial (m/s)

@njit
def equations(t, y, beta):
    x, vx, y_pos, vy = y
    v = np.sqrt(vx**2 + vy**2)
    ax = -beta * v * vx / m
    ay = -g - (beta * v * vy / m)
    return np.array([vx, ax, vy, ay])

# Función de evento para detener la integración cuando el proyectil toca el suelo
def hit_ground(t, y, *args):
    return y[2]  # Se activa cuando y[2] = 0
hit_ground.terminal = True  # Detiene la integración cuando se cumple la condición
hit_ground.direction = -1   # Solo detecta cuando está cayendo

# Función para calcular el alcance horizontal
def calculate_range(beta, theta0):
    v0x, v0y = v0 * np.cos(theta0), v0 * np.sin(theta0)
    y0 = [0, v0x, 0, v0y]
    sol = solve_ivp(equations, [0, 5], y0, args=(beta,), method='RK45', max_step=0.01, 
                    events=hit_ground)  # Usar la función de evento definida
    return sol.t_events[0][0], sol.y[0, -1]  # Retorna tiempo y alcance

# Exploración del ángulo óptimo
def find_optimal_angle(beta):
    thetas = np.linspace(0, np.pi/2, 500)
    ranges = [calculate_range(beta, theta)[1] for theta in thetas]
    return thetas[np.argmax(ranges)]

# Energía perdida
def energy_lost(beta, theta0):
    v0x, v0y = v0 * np.cos(theta0), v0 * np.sin(theta0)
    y0 = [0, v0x, 0, v0y]
    sol = solve_ivp(equations, [0, 5], y0, args=(beta,), method='RK45', max_step=0.01)
    E0 = 0.5 * m * v0**2
    Ef = max(0, 0.5 * m * (sol.y[1, -1]**2 + sol.y[3, -1]**2))  # Asegurar Ef >= 0
    return max(0, E0 - Ef)  # Asegurar que la energía perdida no sea negativa

# Generación de valores de beta en escala logarítmica dentro del nuevo rango, incluyendo beta = 0
betas = np.concatenate(([0], np.logspace(-4, 1, 20)))
theta_max_values = [find_optimal_angle(beta) for beta in betas]
energy_losses = [energy_lost(beta, find_optimal_angle(beta)) for beta in betas]

# Gráfico 1: Ángulo de alcance máximo vs beta
plt.figure(figsize=(8, 5))
plt.plot(betas, np.degrees(theta_max_values), marker='o')
plt.xscale('log')
plt.xlabel(r'$\beta$ (kg/m)')
plt.ylabel(r'$\theta_{max}$ (grados)')
plt.title('Ángulo de alcance máximo vs $beta$')
plt.grid()
plt.savefig("Tarea3/1.a.pdf")
plt.close()

# Gráfico 2: Energía perdida vs beta
plt.figure(figsize=(8, 5))
plt.plot(betas, energy_losses, marker='s', color='r')
plt.xscale('log')
plt.xlabel(r'$\beta$ (kg/m)')
plt.ylabel(r'$\Delta E$ (J)')
plt.title('Energía perdida vs $beta$')
plt.grid()
plt.savefig("Tarea3/1.b.pdf")
plt.close()

'''Ejercicio 2: Paradoja en la física clásica'''
'''2a'''
# Definir condiciones iniciales en unidades atómicas
x = 1.0
y = 0.0
vx = 0.0
vy = 1.0

# Paso de tiempo y número de pasos
dt = 0.01
n_steps = 10000

# Listas para almacenar la trayectoria
x_vals, y_vals = [x], [y]
t_vals = [0]

def f_coulomb(x, y):
    r = np.sqrt(x**2 + y**2)
    f_x = -x / r**3
    f_y = -y / r**3
    return f_x, f_y

def f_coulomb_prima (x, y, vx, vy):
  fx, fy = f_coulomb(x, y)
  return vx, vy, fx, fy # Deriv. posición es velocidad, deriv. velocidad es aceleración pero como masa = 1 y F=ma, se puede igualar a fuerza

def kinetic_energy(vx, vy):
    return 0.5 * (vx**2 + vy**2)

def radio(x, y):
    return np.sqrt(x**2 + y**2)

# Implementación método Runge-Kutta
def rk4_step(f, x, y, vx, vy, dt):

    k1x, k1y, k1vx, k1vy = f(x, y, vx, vy)
    k2x, k2y, k2vx, k2vy = f(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, vx + 0.5 * dt * k1vx, vy + 0.5 * dt * k1vy)
    k3x, k3y, k3vx, k3vy = f(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, vx + 0.5 * dt * k2vx, vy + 0.5 * dt * k2vy)
    k4x, k4y, k4vx, k4vy = f(x + dt * k3x, y + dt * k3y, vx + dt * k3vx, vy + dt * k3vy)

    x_new = x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
    y_new = y + (dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
    vx_new = vx + (dt / 6) * (k1vx + 2 * k2vx + 2 * k3vx + k4vx)
    vy_new = vy + (dt / 6) * (k1vy + 2 * k2vy + 2 * k3vy + k4vy)

    return x_new, y_new, vx_new, vy_new

# Variables para el período
total_time = 0
t_cross = []  # Para medir el período cuando cruza el eje y positivo

# Valores de energía cinética y radio para comprobar que sean constantes en el tiempo
k_energy_vals = [0.5]
rad_vals = [1]

# Iterar con el método de Runge-Kutta
for step in range(n_steps):
    x, y, vx, vy = rk4_step(f_coulomb_prima, x, y, vx, vy, dt)

    # Guardar valores
    x_vals.append(x)
    y_vals.append(y)
    total_time += dt
    t_vals.append(total_time)
    k_energy_vals.append(kinetic_energy(vx, vy))
    rad_vals.append(radio(x, y))

    # Detectar cruces por y > 0 (para medir período)
    if y_vals[-2] < 0 and y > 0:
        t_cross.append(total_time)
        if len(t_cross) > 1:
            break  # Solo necesitamos un ciclo completo

# Calcular período simulado
P_sim = (t_cross[1] - t_cross[0])*24.2
P_teo = (2 * np.pi)*24.2  # Teórico en attosegundos

# Mostrar resultados
print(f'2.a) {P_teo = :.5f}; {P_sim = :.5f}')

# Demostración energía cinética y radio constantes en el tiempo
plt.plot(t_vals, k_energy_vals, label='Energía cinética')
plt.plot(t_vals, rad_vals, label='Radio')
plt.xlabel('Tiempo')
plt.legend()
#plt.show()

# Trayectoria de electrón
plt.figure(figsize=(6,6))
plt.plot(x_vals, y_vals, label='Órbita del electrón')
plt.scatter([0], [0], color='red', label='Protón (núcleo)')
#plt.show()

'''2b'''
# Definir condiciones iniciales en unidades atómicas
x, y = 1.0, 0.0
vx, vy = 0.0, 1.0

# Paso de tiempo y parámetros
alpha = 1  # Constante de estructura fina
dt = 0.001  # Paso de tiempo pequeño
n_steps = 100000  # Número máximo de pasos

# Listas para almacenar datos
x_vals, y_vals, t_vals = [x], [y], [0]
r_vals, KE_vals, E_vals = [1], [0.5], [-0.5]

def force(x, y):
    r = np.sqrt(x**2 + y**2)
    f_x, f_y = -x / r**3, -y / r**3
    return f_x, f_y

def f_coulomb_prima (x, y, vx, vy):
  fx, fy = f_coulomb(x, y)
  return vx, vy, fx, fy

def kinetic_energy(vx, vy):
    return 0.5 * (vx**2 + vy**2)

def radio(x, y):
    return np.sqrt(x**2 + y**2)

def total_energy(x, y, vx, vy):
    return kinetic_energy(vx, vy) - (1/radio(x, y))

# Implementación método Runge-Kutta
def rk4_step(f, x, y, vx, vy, dt):

    k1x, k1y, k1vx, k1vy = f(x, y, vx, vy)
    k2x, k2y, k2vx, k2vy = f(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, vx + 0.5 * dt * k1vx, vy + 0.5 * dt * k1vy)
    k3x, k3y, k3vx, k3vy = f(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, vx + 0.5 * dt * k2vx, vy + 0.5 * dt * k2vy)
    k4x, k4y, k4vx, k4vy = f(x + dt * k3x, y + dt * k3y, vx + dt * k3vx, vy + dt * k3vy)

    x_new = x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
    y_new = y + (dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
    vx_new = vx + (dt / 6) * (k1vx + 2 * k2vx + 2 * k3vx + k4vx)
    vy_new = vy + (dt / 6) * (k1vy + 2 * k2vy + 2 * k3vy + k4vy)

    return x_new, y_new, vx_new, vy_new

total_time = 0
t_fall = None  # Tiempo de caída del electrón

for step in range(n_steps):
    x, y, vx, vy = rk4_step(f_coulomb_prima, x, y, vx, vy, dt)

    # Aplicar corrección por pérdida de energía (Larmor)
    v = np.sqrt(vx**2 + vy**2)
    v_corr = np.sqrt(v - ((4/3) * alpha**3 * dt))
    factor = v_corr / v
    vx *= factor
    vy *= factor

    # Guardar valores
    x_vals.append(x)
    y_vals.append(y)
    t_vals.append(total_time)
    r_vals.append(radio(x, y))
    KE_vals.append(kinetic_energy(vx, vy))
    E_vals.append(total_energy(x, y, vx, vy))
    total_time += dt

    # Condición de caída al núcleo
    if r_vals[step] < 0.01:
        t_fall = total_time
        break

# Convertir tiempo de caída a attosegundos
if t_fall:
    t_fall_as = t_fall * 24.18  # Conversión de unidades atómicas a attosegundos
else:
    t_fall_as = None

print(f'2.b) {t_fall_as = :.5f} as')

# Gráfica de la órbita
plt.figure(figsize=(6,6))
plt.plot(x_vals, y_vals, label='Órbita del electrón')
plt.scatter([0], [0], color='red', label='Protón (núcleo)')
plt.xlabel('x (a.u.)')
plt.ylabel('y (a.u.)')
plt.title('Órbita del electrón con radiación de Larmor')
plt.legend(loc = 'upper right')
plt.axis('equal')
plt.savefig('Tarea3/2.b.XY.pdf')
#plt.show()

# Gráficas de diagnóstico
fig, axs = plt.subplots(3, 1, figsize=(6, 10))
axs[0].plot(t_vals[:len(E_vals)], E_vals, label='Energía total')
axs[0].set_ylabel('E (a.u.)')
axs[0].legend()
axs[1].plot(t_vals[:len(KE_vals)], KE_vals, label='Energía cinética')
axs[1].set_ylabel('KE (a.u.)')
axs[1].legend()
axs[2].plot(t_vals[:len(r_vals)], r_vals, label='Radio')
axs[2].set_xlabel('Tiempo (a.u.)')
axs[2].set_ylabel('r (a.u.)')
axs[2].legend()
plt.savefig('Tarea3/2.b.diagnostics.pdf')
#plt.show()


"""Ejercicio 3: Comprobación de la relatividad general"""

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
ani.save("Tarea3/3.a.mp4", writer="ffmpeg", fps=30)

#3.b
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constantes
mu = 39.4234021 
a = 0.38709893  # UA (semieje mayor)
e = 0.20563069  # Excentricidad
alpha = 1.0977e-8  # Factor relativista exagerado para efectos visuales

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

def event(t,Y):
  x, y, vx, vy = Y
  return x * vx + y * vy

event.direction = 1

# Simulación con solve_ivp
t_span = (0, 10)  # Simular 10 años
t_eval = np.linspace(0, 10, 1000)  # Puntos de evaluación
sol = solve_ivp(equations, t_span, [x0, y0, vx0, vy0], method='Radau', t_eval=t_eval, max_step=1e-3, events=[event])

if len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
    t_sol = sol.t_events[0]  # Tiempos de los eventos
    x_sol = sol.y_events[0][:, 0]  # Coordenada x en eventos
    y_sol = sol.y_events[0][:, 1]  # Coordenada y en eventos

# Calcular ángulos
angles = np.arctan2(y_sol, x_sol)
angles = np.mod(angles, 2 * np.pi)  # Ajustar ángulos al rango [0, 2π]

# Convertir ángulos a segundos de arco
angles_deg = np.degrees(angles)
angles_arcsec = angles_deg * 3600

# Ajuste lineal a la precesión
fit = np.polyfit(t_sol, angles_arcsec, 1)  # Ajuste lineal
precesion = fit[0]  # Pendiente (arcsec/año)
precesion_por_siglo = precesion * 100  # Arcsec/siglo

# Gráfica de precesión
plt.figure(figsize=(8, 5))
plt.plot(t_sol, angles_arcsec, 'bo', label="Datos")
plt.plot(t_sol, np.polyval(fit, t_sol), 'r-', label=f"Ajuste lineal: {precesion:.4f} arcsec/año")
plt.xlabel("Tiempo (años)")
plt.ylabel("Ángulo del periastro (arcsec)")
plt.title(f"Precesión de Mercurio")
plt.legend()
plt.grid()
plt.savefig("Tarea3/3.b.pdf")  # Guardar la gráfica
#plt.show()


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