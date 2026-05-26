import torch
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# =========================
# PARAMÈTRES
# =========================
CPU_POWER_W = 45
CARBON_FACTOR = 0.4

device = "cpu"
num_experiments = 15

sizes = [32 * (2 ** i) for i in range(num_experiments)]

times = []
co2 = []

# =========================
# EXPÉRIENCES
# =========================
for n in sizes:
    if n > 1024:
        break

    A = torch.rand(n, n, device=device)
    B = torch.rand(n, n, device=device)

    start = time.perf_counter()
    torch.matmul(A, B)
    end = time.perf_counter()

    t = end - start
    times.append(t)

    energy = CPU_POWER_W * (t / 3600)
    co2.append(energy * CARBON_FACTOR)

# =========================
# NORMALISATION (meilleur rendu visuel)
# =========================
x = np.arange(len(times))
y = np.array(times)
z = np.zeros(len(times))
dz = np.array(co2)

# couleurs progressives (gradient)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(times)))

# =========================
# FIGURE PROPRE
# =========================
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(
    x, z, y,
    dx=0.6, dy=0.6, dz=dz,
    color=colors,
    shade=True,
    edgecolor="black",
    linewidth=0.3
)

# =========================
# STYLE VISUEL
# =========================
ax.set_title("Impact du calcul matriciel : Taille vs Temps vs CO₂", fontsize=14, fontweight="bold")

ax.set_xlabel("Expérience (taille matrice)", labelpad=10)
ax.set_ylabel("Base")
ax.set_zlabel("CO₂ estimé (g)", labelpad=10)

ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in sizes[:len(times)]], rotation=45, ha="right")

# fond plus propre
ax.xaxis.pane.set_alpha(0.0)
ax.yaxis.pane.set_alpha(0.0)
ax.zaxis.pane.set_alpha(0.0)

ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()