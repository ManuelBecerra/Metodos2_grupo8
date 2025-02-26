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
