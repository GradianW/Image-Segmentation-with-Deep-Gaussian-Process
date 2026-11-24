import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(base_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

# 1. Dataset
X = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.1, 0.5],
    [0.2, 0.6, 0.1],
    [0.5, 0.3, 0.2],
    [0.3, 0.4, 0.4]
], dtype=np.float64)
N = X.shape[0]
samples = np.arange(1, N + 1)

y1 = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.8 * X[:, 2]
y2 = -0.4 * X[:, 0] + 0.7 * X[:, 1] + 0.2 * X[:, 2]
Y_r1 = y1.reshape(-1, 1)
Y_r2 = np.column_stack([y1, y2])

def activation(h):
    return np.maximum(0.0, h)

# Load trained params
params_r1 = np.load(os.path.join(base_dir, "trained_params_v4_r3_r1.npz"))
params_r2 = np.load(os.path.join(base_dir, "trained_params_v4_r3_r2.npz"))

num_mc_samples = 1000
percentiles_list = [2.5, 5.0, 25.0, 50.0, 75.0, 95.0, 97.5]

import math

def hitung_mean_scratch(data_list):
    total = 0.0
    for val in data_list:
        total += float(val)
    return total / len(data_list)

def hitung_std_scratch(data_list, mean_val):
    sum_sq = 0.0
    for val in data_list:
        d = float(val) - mean_val
        sum_sq += d * d
    return math.sqrt(sum_sq / (len(data_list) - 1))

def hitung_persentil_sekolah_scratch(sorted_list, q):
    n_data = len(sorted_list)
    letak = (q / 100.0) * (n_data + 1)
    m = int(math.floor(letak))
    d = letak - m
    if m < 1:
        return float(sorted_list[0])
    if m >= n_data:
        return float(sorted_list[-1])
    return float(sorted_list[m - 1]) + d * (float(sorted_list[m]) - float(sorted_list[m - 1]))

def run_mc_inference(params, n3):
    M1_opt, Sigma1_opt, b1_opt = params["M1"], params["Sigma1"], params["b1"]
    M2_opt, Sigma2_opt, b2_opt = params["M2"], params["Sigma2"], params["b2"]
    M3_opt, Sigma3_opt, b3_opt = params["M3"], params["Sigma3"], params["b3"]
    
    n0, n1, n2 = 3, 4, 6
    np.random.seed(1000)
    Y_hat_mc = np.zeros((num_mc_samples, N, n3), dtype=np.float64)
    Z1_batch = np.zeros((num_mc_samples, n0, n1), dtype=np.float64)
    W1_batch = np.zeros((num_mc_samples, n0, n1), dtype=np.float64)
    
    for s in range(num_mc_samples):
        Z1_s = np.random.randn(n0, n1)
        Z2_s = np.random.randn(n1, n2)
        Z3_s = np.random.randn(n2, n3)
        
        W1_s = M1_opt + Sigma1_opt * Z1_s
        H1_s = X @ W1_s + b1_opt
        X1_s = activation(H1_s)
        
        W2_s = M2_opt + Sigma2_opt * Z2_s
        H2_s = X1_s @ W2_s + b2_opt
        X2_s = activation(H2_s)
        
        W3_s = M3_opt + Sigma3_opt * Z3_s
        H3_s = X2_s @ W3_s + b3_opt
        
        Y_hat_mc[s] = H3_s
        Z1_batch[s] = Z1_s
        W1_batch[s] = W1_s
        
    perc_values = np.zeros((len(percentiles_list), N, n3), dtype=np.float64)
    mean_pred = np.zeros((N, n3), dtype=np.float64)
    std_pred = np.zeros((N, n3), dtype=np.float64)
    for i in range(N):
        for k in range(n3):
            s1000 = [float(Y_hat_mc[s, i, k]) for s in range(num_mc_samples)]
            mean_pred[i, k] = hitung_mean_scratch(s1000)
            std_pred[i, k] = hitung_std_scratch(s1000, mean_pred[i, k])
            s_urut = sorted(s1000)
            for p_idx, q in enumerate(percentiles_list):
                perc_values[p_idx, i, k] = hitung_persentil_sekolah_scratch(s_urut, q)
    return Y_hat_mc, perc_values, mean_pred, std_pred, Z1_batch, W1_batch

Y_hat_r1, perc_r1, mean_r1, std_r1, Z1_r1, W1_r1 = run_mc_inference(params_r1, 1)
Y_hat_r2, perc_r2, mean_r2, std_r2, Z1_r2, W1_r2 = run_mc_inference(params_r2, 2)

# Print Sample s=1 explicit matrices for LaTeX worked example
print("=== SAMPLE s=1 EXPLICIT MATRICES (R3 -> R1) ===")
np.random.seed(1000)
Z1_1 = np.random.randn(3, 4)
Z2_1 = np.random.randn(4, 6)
Z3_1 = np.random.randn(6, 1)
M1_1, S1_1, b1_1 = params_r1["M1"], params_r1["Sigma1"], params_r1["b1"]
M2_1, S2_1, b2_1 = params_r1["M2"], params_r1["Sigma2"], params_r1["b2"]
M3_1, S3_1, b3_1 = params_r1["M3"], params_r1["Sigma3"], params_r1["b3"]
W1_1 = M1_1 + S1_1 * Z1_1
H1_1 = X @ W1_1 + b1_1
X1_1 = activation(H1_1)
W2_1 = M2_1 + S2_1 * Z2_1
H2_1 = X1_1 @ W2_1 + b2_1
X2_1 = activation(H2_1)
W3_1 = M3_1 + S3_1 * Z3_1
Yhat_1 = X2_1 @ W3_1 + b3_1
print("Z1^(1):\n", np.round(Z1_1, 4))
print("W1^(1):\n", np.round(W1_1, 4))
print("X1^(1):\n", np.round(X1_1, 4))
print("Z3^(1):", np.round(Z3_1.ravel(), 4))
print("W3^(1):", np.round(W3_1.ravel(), 4))
print("Yhat^(1):", np.round(Yhat_1.ravel(), 4))
print("Yhat^(2):", np.round(Y_hat_r1[1].ravel(), 4))
print("Yhat^(1000):", np.round(Y_hat_r1[999].ravel(), 4))

# 1. Generate 3D Tensor Block Figure
def save_tensor_3d_fig(M1_opt, Z1_batch, W1_batch, filename):
    S_vis = 5
    M1_3d = M1_opt[None, :, :]
    Z1_3d_sub = Z1_batch[:S_vis, :, :]
    W1_3d_sub = W1_batch[:S_vis, :, :]

    def draw_tensor_block(ax, tensor, title, cmap_name='coolwarm'):
        S_dim, R_dim, C_dim = tensor.shape
        filled = np.ones((S_dim, C_dim, R_dim), dtype=bool)
        norm = mcolors.Normalize(vmin=np.min(tensor) - 0.1, vmax=np.max(tensor) + 0.1)
        cmap = plt.get_cmap(cmap_name)
        facecolors = np.empty(filled.shape + (4,), dtype=float)
        for s in range(S_dim):
            for r in range(R_dim):
                for c in range(C_dim):
                    r_plot = (R_dim - 1) - r
                    rgba = list(cmap(norm(tensor[s, r, c])))
                    rgba[3] = 0.82
                    facecolors[s, c, r_plot] = rgba

        ax.voxels(filled, facecolors=facecolors, edgecolors='#222222', linewidth=0.9)
        for r in range(R_dim):
            for c in range(C_dim):
                r_plot = (R_dim - 1) - r
                val = tensor[0, r, c]
                ax.text(0.0, c + 0.5, r_plot + 0.5, f"{val:.2f}",
                        color='black', fontweight='bold', fontsize=7.5,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=0.8, edgecolor='none'))

        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel('\nSampel (s)', fontsize=8.5)
        ax.set_ylabel('\nKolom (n1=4)', fontsize=8.5)
        ax.set_zlabel('\nBaris (n0=3)', fontsize=8.5)
        ax.set_xticks(np.arange(S_dim) + 0.5)
        ax.set_xticklabels([f"s={i+1}" for i in range(S_dim)], fontsize=7.5)
        ax.set_yticks(np.arange(C_dim) + 0.5)
        ax.set_yticklabels([f"k={j+1}" for j in range(C_dim)], fontsize=7.5)
        ax.set_zticks(np.arange(R_dim) + 0.5)
        ax.set_zticklabels([f"b={R_dim-i}" for i in range(R_dim)], fontsize=7.5)
        ax.set_box_aspect((max(S_dim * 0.75, 1.4), C_dim, R_dim))
        ax.view_init(elev=22, azim=-38)

    fig = plt.figure(figsize=(16, 5.2))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    draw_tensor_block(ax1, M1_3d, f"1. Tensor 3D M^(1)* [1, 3, 4]\n(Balok Tipis 1 Lapis)", cmap_name='Blues')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    draw_tensor_block(ax2, Z1_3d_sub, f"2. Tensor 3D Z^(1) [1000, 3, 4]\n(Irisan 5 Lapis Pertama)", cmap_name='coolwarm')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    draw_tensor_block(ax3, W1_3d_sub, f"3. Tensor 3D W^(1) [1000, 3, 4]\n(Hasil Broadcasting M + Sigma * Z)", cmap_name='YlOrRd')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, filename), dpi=220, bbox_inches='tight')
    plt.close()

save_tensor_3d_fig(params_r1["M1"], Z1_r1, W1_r1, "tensor_3d_broadcasting.png")

# 2. Generate 4 Figures for R3 -> R1 and R3 -> R2
def save_4_figures(Y_true, Y_hat_mc, perc_values, mean_pred, out_idx, prefix, y_label_tex, c_targ, c_pred):
    y_zoom_min = min(np.min(Y_hat_mc[:50, 3:5, out_idx]), np.min(perc_values[0, 3:5, out_idx]), np.min(Y_true[3:5, out_idx])) - 0.012
    y_zoom_max = max(np.max(Y_hat_mc[:50, 3:5, out_idx]), np.max(perc_values[6, 3:5, out_idx]), np.max(Y_true[3:5, out_idx])) + 0.012

    # Figure 1: Violin / Empirical distribution
    fig1, ax1 = plt.subplots(figsize=(8.5, 4.6))
    parts = ax1.violinplot([Y_hat_mc[:, i, out_idx] for i in range(N)], positions=samples,
                           showmeans=False, showmedians=True, widths=0.55)
    for pc in parts['bodies']:
        pc.set_facecolor('#ff7f0e')
        pc.set_alpha(0.35)

    ax1.scatter(samples, Y_true[:, out_idx], color=c_targ, s=70, zorder=5, label=f'Target Asli ${y_label_tex}$')
    ax1.scatter(samples, mean_pred[:, out_idx], color=c_pred, marker='s', s=55, zorder=5, label=rf'Mean Prediksi ($\bar{{\hat{{{y_label_tex}}}}}$)')
    ax1.errorbar(samples, mean_pred[:, out_idx],
                 yerr=[mean_pred[:, out_idx] - perc_values[0, :, out_idx], perc_values[6, :, out_idx] - mean_pred[:, out_idx]],
                 fmt='none', ecolor=c_pred, elinewidth=1.5, capsize=5, label=r'Rentang $P_{2.5} - P_{97.5}$')

    ax1.set_title(f'Sebaran Distribusi 1000 Prediksi per Sampel Data (${y_label_tex}$)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Sampel Data ($i$)', fontsize=10)
    ax1.set_ylabel(f'Nilai Output (${y_label_tex}$)', fontsize=10)
    ax1.set_xticks(samples)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=8.5, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}_fig1_distribusi.png"), dpi=220, bbox_inches='tight')
    plt.close()

    # Helper for Figures 2, 3, 4
    def make_pair_fig(fname, band_type=None):
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))
        for idx, is_zoom in enumerate([False, True]):
            ax = axes[idx]
            for s in range(50):
                ax.plot(samples, Y_hat_mc[s, :, out_idx], color='#555555', alpha=0.35 if not band_type else 0.28,
                        linewidth=0.9, label=rf'50 Lintasan Sampel Acak $\hat{{{y_label_tex}}}^{{(s)}}$' if s == 0 else "")
            if band_type == '50':
                ax.fill_between(samples, perc_values[2, :, out_idx], perc_values[4, :, out_idx],
                                color='#ff7f0e', alpha=0.40, label=r'Pita Ketidakpastian 50% ($P_{25} - P_{75}$)')
            elif band_type == '95':
                ax.fill_between(samples, perc_values[0, :, out_idx], perc_values[6, :, out_idx],
                                color='#ff7f0e', alpha=0.30, label=r'Pita Ketidakpastian 95% ($P_{2.5} - P_{97.5}$)')

            ax.plot(samples, Y_true[:, out_idx], 'o-', color=c_targ, linewidth=2.2, markersize=7.5, zorder=5, label=f'Target Asli ${y_label_tex}$')
            ax.plot(samples, mean_pred[:, out_idx], 's--', color=c_pred, linewidth=2.0, markersize=6, zorder=5, label=rf'Mean Prediksi ($\bar{{\hat{{{y_label_tex}}}}}$)')
            ax.plot(samples, perc_values[3, :, out_idx], '^:', color='#2ca02c' if c_targ != '#2ca02c' else 'black',
                    linewidth=1.7, markersize=5.5, zorder=5, label=r'Median Prediksi ($P_{50}$)')

            if not is_zoom:
                ax.set_xticks(samples)
                ax.set_title('Tampilan Penuh (Sampel $i = 1, 2, 3, 4, 5$)', fontsize=10.5, fontweight='bold')
            else:
                ax.set_xlim(3.8, 5.2)
                ax.set_ylim(y_zoom_min, y_zoom_max)
                ax.set_xticks([4, 5])
                ax.set_xticklabels(['Sampel 4 ($i=4$)', 'Sampel 5 ($i=5$)'])
                ax.set_title('Close-Up Sekitar Titik $i=4$ dan $i=5$', fontsize=10.5, fontweight='bold')

            ax.set_xlabel('Sampel Data ($i$)', fontsize=9.5)
            ax.set_ylabel(f'Nilai Output (${y_label_tex}$)', fontsize=9.5)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(fontsize=8, loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, fname), dpi=220, bbox_inches='tight')
        plt.close()

    make_pair_fig(f"{prefix}_fig2_lintasan.png", band_type=None)
    make_pair_fig(f"{prefix}_fig3_pita50.png", band_type='50')
    make_pair_fig(f"{prefix}_fig4_pita95.png", band_type='95')

save_4_figures(Y_r1, Y_hat_r1, perc_r1, mean_r1, 0, "r3_r1", "y", "#1f77b4", "#d62728")
save_4_figures(Y_r2, Y_hat_r2, perc_r2, mean_r2, 0, "r3_r2_y1", "y_1", "#1f77b4", "#d62728")
save_4_figures(Y_r2, Y_hat_r2, perc_r2, mean_r2, 1, "r3_r2_y2", "y_2", "#2ca02c", "#9467bd")
print("Seluruh 13 gambar untuk kredit_bimbingan_11.tex berhasil disimpan di folder figures/!")
