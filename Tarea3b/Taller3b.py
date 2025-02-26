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
