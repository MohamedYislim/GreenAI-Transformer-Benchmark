import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from codecarbon import OfflineEmissionsTracker

# =========================
# PARAMÈTRES
# =========================
SIZES = [16, 24, 32, 48, 64, 96, 128]
K_REPEATS = 15
NUM_THREADS = 4
COUNTRY_CODE = "CAN"
OUTPUT_FIG = "ges_3d.png"

torch.set_num_threads(NUM_THREADS)
device = torch.device("cpu")


# =========================
# MÉTHODES DE CALCUL
# =========================
def scaled_dot_product_attention(N):
    q = torch.randn(N, N, device=device)
    k = torch.randn(N, N, device=device)
    v = torch.randn(N, N, device=device)

    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(N)
    attention = F.softmax(attn_logits, dim=-1)
    output = torch.matmul(attention, v)

    return output


def multithread_matmul(N):
    A = torch.randn(N, N, device=device)
    B = torch.randn(N, N, device=device)

    C = torch.matmul(A, B.transpose(-2, -1))

    return C


# =========================
# MESURE TEMPS + GES
# =========================
def measure_method(method_func, N, k=15):
    times = []
    emissions = []

    for _ in range(k):
        tracker = OfflineEmissionsTracker(
            country_iso_code=COUNTRY_CODE,
            log_level="error",
            save_to_file=False
        )

        tracker.start()

        start = time.perf_counter()
        method_func(N)
        end = time.perf_counter()

        emission = tracker.stop()

        times.append(end - start)
        emissions.append(emission if emission is not None else 0.0)

    return {
        "time_mean": np.mean(times),
        "time_std": np.std(times),
        "ges_mean": np.mean(emissions),
        "ges_std": np.std(emissions)
    }


# =========================
# EXPÉRIENCES
# =========================
results_attention = []
results_matmul = []

print("\n===== BENCHMARK ATTENTION VS MATMUL MULTITHREAD =====\n")
print(f"Device utilisé : {device}")
print(f"Nombre de threads PyTorch : {torch.get_num_threads()}")
print(f"Répétitions par taille : {K_REPEATS}\n")

for N in SIZES:
    print(f"--- Taille N = {N} ---")

    att_result = measure_method(scaled_dot_product_attention, N, K_REPEATS)
    mat_result = measure_method(multithread_matmul, N, K_REPEATS)

    results_attention.append(att_result)
    results_matmul.append(mat_result)

    print(
        f"Attention : "
        f"temps = {att_result['time_mean']:.6f} ± {att_result['time_std']:.6f} s/op | "
        f"GES = {att_result['ges_mean']:.12f} ± {att_result['ges_std']:.12f} kg CO2eq"
    )

    print(
        f"MatMul    : "
        f"temps = {mat_result['time_mean']:.6f} ± {mat_result['time_std']:.6f} s/op | "
        f"GES = {mat_result['ges_mean']:.12f} ± {mat_result['ges_std']:.12f} kg CO2eq"
    )

    print()


# =========================
# PRÉPARATION DES DONNÉES
# =========================
x = np.array(SIZES)

att_time_mean = np.array([r["time_mean"] for r in results_attention])
att_time_std = np.array([r["time_std"] for r in results_attention])
att_ges_mean = np.array([r["ges_mean"] for r in results_attention])
att_ges_std = np.array([r["ges_std"] for r in results_attention])

mat_time_mean = np.array([r["time_mean"] for r in results_matmul])
mat_time_std = np.array([r["time_std"] for r in results_matmul])
mat_ges_mean = np.array([r["ges_mean"] for r in results_matmul])
mat_ges_std = np.array([r["ges_std"] for r in results_matmul])


# =========================
# GRAPHE 3D
# =========================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot(
    x,
    att_time_mean,
    att_ges_mean,
    marker="o",
    label="Attention scaled dot-product"
)

ax.plot(
    x,
    mat_time_mean,
    mat_ges_mean,
    marker="^",
    label="MatMul multithread"
)

# Barres d'erreur 3D : temps Y et GES Z
for i in range(len(x)):
    # Attention : erreur sur le temps
    ax.plot(
        [x[i], x[i]],
        [att_time_mean[i] - att_time_std[i], att_time_mean[i] + att_time_std[i]],
        [att_ges_mean[i], att_ges_mean[i]]
    )

    # Attention : erreur sur GES
    ax.plot(
        [x[i], x[i]],
        [att_time_mean[i], att_time_mean[i]],
        [att_ges_mean[i] - att_ges_std[i], att_ges_mean[i] + att_ges_std[i]]
    )

    # MatMul : erreur sur le temps
    ax.plot(
        [x[i], x[i]],
        [mat_time_mean[i] - mat_time_std[i], mat_time_mean[i] + mat_time_std[i]],
        [mat_ges_mean[i], mat_ges_mean[i]]
    )

    # MatMul : erreur sur GES
    ax.plot(
        [x[i], x[i]],
        [mat_time_mean[i], mat_time_mean[i]],
        [mat_ges_mean[i] - mat_ges_std[i], mat_ges_mean[i] + mat_ges_std[i]]
    )

ax.set_title("Comparaison 3D : Attention vs MatMul multithread")
ax.set_xlabel("Taille de matrice N")
ax.set_ylabel("Temps moyen (s/op)")
ax.set_zlabel("GES moyen (kg CO2eq)")

ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.show()

print(f"\nFigure sauvegardée : {OUTPUT_FIG}")