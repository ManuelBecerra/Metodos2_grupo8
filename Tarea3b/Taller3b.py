
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''Ejercicio 1: Poisson en un disco'''
# Parámetros
N = 70  # Tamaño de la malla (ajustar según precisión requerida)
L = 1.1  # Extensión de la malla (ligeramente mayor al círculo unitario)
dx = 2 * L / (N - 1)
dy = dx
tol = 1e-4  # Tolerancia para la convergencia
max_iter = 15000  # Máximo número de iteraciones
omega = 1.8  # Factor de sobre-relajación (ajustar para acelerar)

# Crear malla
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)
phi = np.random.rand(N, N) * 0.1  # Condiciones iniciales aleatorias

# Función rho(x, y) = - (x + y)
rho = - (X + Y)

# Condiciones de frontera
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)
phi[R >= 1] = np.sin(7 * Theta[R >= 1])  # φ en el borde del círculo

# Iteraciones usando Gauss-Seidel con SOR, este es una forma particular del método de FINITAS DIFERENCIAS con unos cambios para hacerlo más efectivo
#tiene sobre-relajación que acelera la convergencia a valores buenos introduciendo un factor de relajación ω para actualizar más rápido los valores
#y Gauss-Seidel es que apenas calcula un valor nuevo para un punto en la amtriz, ovlida el valor viejo.
for _ in range(max_iter):
    phi_old = phi.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):
            if R[i, j] < 1:  # Solo actualizar dentro del disco
                phi[i, j] = (1 - omega) * phi[i, j] + omega * 0.25 * (
                    phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - dx**2 * rho[i, j]
                )
    
    # Criterio de convergencia, asegura que la solución pueda converger a algo, si no pongo esto entonces 1. se demoraba mucho y 2. puede que no converge nunca
    if np.linalg.norm(phi - phi_old) < tol:
        break

# Gráfica 3D de la solución
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, phi, cmap='jet')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('ϕ(x, y)')
ax.set_title('Solución de la ecuación de Poisson')

plt.savefig("Tarea3b/1.png")  # Guardar la imagen



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

'''Ejercicio 2: Ondas 1D y reflexiones'''

def wave1D(boundarycond, L=2):
    dt = 8e-3       # paso temporal
    dx = 1e-2       # paso espacial
    D = 1.0         # velocidad de la onda
    Tmax = 10.1     # tiempo total de simulación

    xs = np.arange(0, L, dx)
    ts = np.arange(0, Tmax, dt)
    U = np.zeros((len(ts), len(xs)))

    U[0, :] = np.exp(-125*(xs - 0.5)**2)
    U[1, :] = np.exp(-125*(xs - 0.5)**2)

    for j in range(1, len(ts)-1):
        for i in range(1, len(xs)-1):
            U[j+1, i] = 2*U[j, i] - U[j-1, i] + (D**2 * dt**2/dx**2) * \
                        (U[j, i+1] - 2*U[j, i] + U[j, i-1])
        if boundarycond == 'Dirichlet':
            U[j+1, 0]  = 0
            U[j+1, -1] = 0
        elif boundarycond == 'Neumann':
            i = 0
            U[j+1, 0]  = 2*U[j, i] - U[j-1, i] + (D**2 * dt**2/dx**2) * \
                        (U[j, i+1] - 2*U[j, i] + U[j, i+1])
            i = -1
            U[j+1, -1] = 2*U[j, i] - U[j-1, i] + (D**2 * dt**2/dx**2) * \
                        (U[j, i-1] - 2*U[j, i] + U[j, i-1])
        elif boundarycond == 'Periodic':
            U[j+1, 0] = 2*U[j, 0] - U[j-1, 0] + (D**2 * dt**2/dx**2) * \
                        (U[j, 1] - 2*U[j, 0] + U[j, -1])
            U[j+1, -1] = 2*U[j, -1] - U[j-1, -1] + (D**2 * dt**2/dx**2) * \
                         (U[j, 0] - 2*U[j, -1] + U[j, -2])
    return xs, ts, U

# Animación
xs_d, ts, U_dir = wave1D("Dirichlet")
xs_n, ts, U_neu = wave1D("Neumann")
xs_p, ts, U_per = wave1D("Periodic")

ymin = min(U_dir.min(), U_neu.min(), U_per.min())
ymax = max(U_dir.max(), U_neu.max(), U_per.max())


fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
ax1, ax2, ax3 = axes

ax1.text(0.05, 0.9, "Dirichlet", transform=ax1.transAxes, fontsize=12, color='red')
ax2.text(0.05, 0.9, "Neumann", transform=ax2.transAxes, fontsize=12, color='blue')
ax3.text(0.05, 0.9, "Periodic", transform=ax3.transAxes, fontsize=12, color='green')

line1, = ax1.plot(xs_d, U_dir[0, :], 'r-')
line2, = ax2.plot(xs_n, U_neu[0, :], 'b-')
line3, = ax3.plot(xs_p, U_per[0, :], 'g-')


for ax in axes:
    ax.set_xlim(0, 2)
    ax.set_ylim(ymin - 0.1, ymax + 0.1)
    ax.set_ylabel("U(x,t)")
ax3.set_xlabel("x")


num_frames = 100
frame_indices = np.linspace(0, len(ts) - 1, num_frames, dtype=int)

def update(frame):
    idx = frame_indices[frame]
    line1.set_ydata(U_dir[idx, :])
    line2.set_ydata(U_neu[idx, :])
    line3.set_ydata(U_per[idx, :])
    return line1, line2, line3

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=40, blit=True)

# Guardar la animación como video MP4
Writer = animation.writers['ffmpeg']
writer = Writer(fps=50, metadata=dict(artist='Maria Paula'), bitrate=1800)
ani.save("Tarea3b/2.mp4", writer=writer)


'''Ejercicio 3: Ondas no lineales: Plasma y fluidos'''


'''Ejercicio 4: Simulación'''

# Parámetros del problema
Lx, Ly = 2.0, 1.0  # Dimensiones del tanque (m)
dx = dy = 0.01  # Resolución espacial (m)
dt = 0.001  # Paso temporal (s)
t_max = 2.0  # Tiempo total de simulación (s)
f = 10  # Frecuencia de la fuente (Hz)
A = 0.01  # Amplitud de la onda (m)

# Definir mallas
nx, ny = int(Lx/dx), int(Ly/dy)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Definir velocidad de onda c(x, y)
c0 = 0.5  # Velocidad base (m/s)
c = np.full((ny, nx), c0)

# Pared central con apertura
wy = 0.04  # Ancho de la pared
wx = 0.4   # Apertura
middle_wall = (np.abs(Y - Lx / 2) < wy / 2) & (np.abs(X - Lx / 4) >= wx / 2)
c[middle_wall] = 0  # Las paredes bloquean las ondas

# Definir la lente elíptica con la ecuación dada en la imagen
lens_mask = ((((Y - Lx / 4) ** 2) + 3 * ((X - Lx / 2) ** 2)) <= (1 / 25)) & (X > 1)
c[lens_mask & (X > Lx / 2)] = c0 / 5  # Reducir velocidad en la lente

# Paredes superior e inferior de la lente
lens_walls = ((X > 0.98) & (X < 1.02)) & ((Y < 0.4) | (Y > 0.6))
c[lens_walls] = 0  # Bloquear el paso de la onda

# Calcular el coeficiente de Courant
courant_number = np.max(c) * dt / dx
print(f"Coeficiente de Courant ejercicio 4: {courant_number:.3f}")

# Condiciones iniciales
u = np.zeros((ny, nx))  # Estado en t
u_prev = np.zeros((ny, nx))  # Estado en t - dt

# Ubicación de la fuente (en la parte inferior del tanque)
src_x, src_y = 0.5, 0.5
src_ix, src_iy = int(src_x / dx), int(src_y / dy)

# Inicializar figura
fig, ax = plt.subplots()
img = ax.imshow(np.zeros((ny, nx)), extent=[0, Lx, 0, Ly], origin='lower', cmap='seismic', vmin=-A/2, vmax=A/2)
ax.set_facecolor('white')
ax.set_title("Simulación de Onda 2D")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

# Actualizar función
def update(frame):
    global u, u_prev
    
    # Aplicar diferencias finitas
    u_next = 2*u - u_prev + (dt**2 * c**2) * (
        (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) - 2*u) / dy**2 +
        (np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 2*u) / dx**2
    )
    
    # Reducir velocidad dentro de la región de la lente
    u_next[lens_mask] = 2*u[lens_mask] - u_prev[lens_mask] + (dt**2 * (c0/5)**2) * (
        (np.roll(u, 1, axis=0)[lens_mask] + np.roll(u, -1, axis=0)[lens_mask] - 2*u[lens_mask]) / dy**2 +
        (np.roll(u, 1, axis=1)[lens_mask] + np.roll(u, -1, axis=1)[lens_mask] - 2*u[lens_mask]) / dx**2
    )

    # Aplicar condición de frontera (paredes externas, centrales y de la lente)
    u_next[0, :] = u_next[-1, :] = 0
    u_next[:, 0] = u_next[:, -1] = 0
    u_next[middle_wall] = 0  # Pared central sin apertura
    u_next[lens_walls] = 0  # Paredes superior e inferior de la lente

    # Aplicar la fuente
    u_next[src_iy, src_ix] += A * np.sin(2 * np.pi * f * frame * dt)
    
    # Actualizar estados
    u_prev = u.copy()
    u = u_next.copy()
    
    # Actualizar imagen
    img.set_array(u)
    return [img]

# Crear animación
frames = int(t_max / dt)
ani = animation.FuncAnimation(fig, update, frames=frames, interval=dt*1000, blit= False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Bruno Abello'), bitrate=1800)
ani.save("Tarea3b/4.mp4", writer=writer)

