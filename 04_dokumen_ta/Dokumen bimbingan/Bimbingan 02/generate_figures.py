import os
import numpy as np
import matplotlib.pyplot as plt

# Buat folder figures jika belum ada
figures_dir = r"c:\Users\MyBook Z Series\Desktop\TA\04_dokumen_ta\figures"
os.makedirs(figures_dir, exist_ok=True)

# Set style publikasi
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.autolayout': True,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.8
})

# ==========================================
# Gambar 1: Prior vs Posterior GP
# ==========================================
np.random.seed(42)
X_test = np.linspace(-4, 4, 200)[:, None]

def rbf_kernel(x1, x2, l=1.0, sigma_f=1.0):
    dist_sq = (x1 - x2.T)**2
    return (sigma_f**2) * np.exp(-0.5 * dist_sq / (l**2))

K_prior = rbf_kernel(X_test, X_test) + 1e-6 * np.eye(len(X_test))
prior_samples = np.random.multivariate_normal(np.zeros(len(X_test)), K_prior, size=4)

# Observasi contoh
X_train_ex = np.array([[-3.0], [-1.0], [1.5], [2.5]])
y_train_ex = np.array([[-0.8], [1.2], [-0.5], [0.8]])
sigma_n_ex = 0.1

K_train_ex = rbf_kernel(X_train_ex, X_train_ex) + (sigma_n_ex**2) * np.eye(len(X_train_ex))
K_s_ex = rbf_kernel(X_train_ex, X_test)
K_ss_ex = rbf_kernel(X_test, X_test) + 1e-6 * np.eye(len(X_test))

L_ex = np.linalg.cholesky(K_train_ex)
alpha_ex = np.linalg.solve(L_ex.T, np.linalg.solve(L_ex, y_train_ex))
mu_post = K_s_ex.T @ alpha_ex
v_ex = np.linalg.solve(L_ex, K_s_ex)
cov_post = K_ss_ex - v_ex.T @ v_ex
var_post = np.diag(cov_post)
std_post = np.sqrt(np.maximum(1e-8, var_post))[:, None]

post_samples = np.random.multivariate_normal(mu_post.ravel(), cov_post + 1e-6*np.eye(len(X_test)), size=4)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

# Prior plot
axes[0].fill_between(X_test.ravel(), -2, 2, color='#3b82f6', alpha=0.15, label='Prior 95% CI (±2σ)')
for i, sample in enumerate(prior_samples):
    axes[0].plot(X_test.ravel(), sample, alpha=0.8, label=f'Sampel {i+1}' if i < 2 else None)
axes[0].axhline(0, color='black', linestyle='--', alpha=0.5, label='Mean Prior (m(x)=0)')
axes[0].set_title("(a) Prior Gaussian Process (Tanpa Data)")
axes[0].set_xlabel("Input x")
axes[0].set_ylabel("f(x)")
axes[0].set_ylim(-3, 3)
axes[0].legend(loc='lower left', fontsize=9)

# Posterior plot
axes[1].fill_between(X_test.ravel(), (mu_post - 2*std_post).ravel(), (mu_post + 2*std_post).ravel(),
                     color='#10b981', alpha=0.2, label='Posterior 95% CI (±2σ)')
axes[1].plot(X_test.ravel(), mu_post.ravel(), color='#047857', linewidth=2.2, label='Mean Prediktif f*(x)')
for sample in post_samples:
    axes[1].plot(X_test.ravel(), sample, alpha=0.5, linestyle=':')
axes[1].scatter(X_train_ex, y_train_ex, color='#dc2626', s=50, zorder=5, label='Training Data (y)')
axes[1].set_title("(b) Posterior Gaussian Process (Setelah Observasi)")
axes[1].set_xlabel("Input x")
axes[1].set_ylim(-3, 3)
axes[1].legend(loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "fig1_prior_posterior.pdf"))
plt.savefig(os.path.join(figures_dir, "fig1_prior_posterior.png"), dpi=300)
plt.close()

# ==========================================
# Gambar 2: Profil Kernel RBF vs Matérn
# ==========================================
r = np.linspace(0, 4, 300)
l = 1.0
sigma_f = 1.0

# RBF
k_rbf = (sigma_f**2) * np.exp(-0.5 * (r/l)**2)
# Matérn 3/2
k_mat32 = (sigma_f**2) * (1 + np.sqrt(3)*r/l) * np.exp(-np.sqrt(3)*r/l)
# Matérn 5/2
k_mat52 = (sigma_f**2) * (1 + np.sqrt(5)*r/l + 5*(r**2)/(3*l**2)) * np.exp(-np.sqrt(5)*r/l)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].plot(r, k_rbf, label='RBF / Squared Exp (C^∞)', color='#2563eb')
axes[0].plot(r, k_mat52, label='Matérn 5/2 (C^2)', color='#10b981', linestyle='--')
axes[0].plot(r, k_mat32, label='Matérn 3/2 (C^1)', color='#f59e0b', linestyle='-.')
axes[0].set_title("(a) Profil Kovariansi vs Jarak r = ||x - x'||")
axes[0].set_xlabel("Jarak r")
axes[0].set_ylabel("Nilai Kovariansi k(r)")
axes[0].set_ylim(-0.05, 1.05)
axes[0].legend(fontsize=9)

# Pengaruh lengthscale
for l_val, col, ls in zip([0.5, 1.0, 2.0], ['#ef4444', '#2563eb', '#8b5cf6'], ['-.', '-', '--']):
    k_l = (sigma_f**2) * np.exp(-0.5 * (r/l_val)**2)
    axes[1].plot(r, k_l, label=f'RBF (l = {l_val})', color=col, linestyle=ls)
axes[1].set_title("(b) Pengaruh Parameter Lengthscale l pada RBF")
axes[1].set_xlabel("Jarak r")
axes[1].set_ylabel("Nilai Kovariansi k(r)")
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "fig2_kernel_profiles.pdf"))
plt.savefig(os.path.join(figures_dir, "fig2_kernel_profiles.png"), dpi=300)
plt.close()

# ==========================================
# Gambar 4: Visualisasi Occam's Razor
# ==========================================
complexity = np.linspace(0.1, 5, 200)
# Kurva model
data_fit = 3.5 - 2.8 * np.exp(-0.7 * complexity)       # semakin kompleks, data fit makin tinggi
penalty = 0.6 * complexity**1.3                       # semakin kompleks, penalti makin berat
log_marginal_likelihood = data_fit - penalty          # LML = data_fit - complexity_penalty

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(complexity, data_fit, label=r'Data-fit term $-\frac{1}{2}\mathbf{y}^T K_y^{-1}\mathbf{y}$', color='#10b981', linestyle='--')
ax.plot(complexity, -penalty, label=r'Complexity penalty $-\frac{1}{2}\log|K_y|$', color='#ef4444', linestyle=':')
ax.plot(complexity, log_marginal_likelihood, label=r'Log Marginal Likelihood $\log p(\mathbf{y}|X,\theta)$', color='#1e3a8a', linewidth=2.4)

idx_opt = np.argmax(log_marginal_likelihood)
opt_c = complexity[idx_opt]
opt_lml = log_marginal_likelihood[idx_opt]

ax.scatter([opt_c], [opt_lml], color='#b91c1c', s=70, zorder=6)
ax.axvline(opt_c, color='#b91c1c', linestyle='--', alpha=0.6)
ax.annotate('Kompleksitas Optimal\n(Prinsip Occam\'s Razor)',
            xy=(opt_c, opt_lml), xytext=(opt_c + 0.5, opt_lml - 1.2),
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
            fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#f59e0b", lw=1))

ax.text(0.3, -2.5, "Underfitting\n(Model terlalu kaku)", color='#4b5563', fontsize=9, ha='center')
ax.text(4.2, -2.5, "Overfitting\n(Model terlalu fleksibel)", color='#4b5563', fontsize=9, ha='center')

ax.set_title("Trade-off Otomatis pada Log Marginal Likelihood")
ax.set_xlabel("Kompleksitas Model / Ruang Fungsi (via Hyperparameter θ)")
ax.set_ylabel("Nilai Komponen Log-Likelihood")
ax.legend(loc='lower left', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "fig4_occams_razor.pdf"))
plt.savefig(os.path.join(figures_dir, "fig4_occams_razor.png"), dpi=300)
plt.close()

# ==========================================
# Gambar 5: Plot Eksak Hasil Homework Example (N=3, N*=2)
# ==========================================
X_train = np.array([[0.0], [1.0], [2.0]])
y_train = np.array([[0.00], [0.84], [0.91]])
sigma_n_sq = 0.01

# Hitung kurva kontinu evaluasi
x_dense = np.linspace(-0.5, 3.2, 300)[:, None]

K_dense_train = rbf_kernel(x_dense, X_train)
K_train_exact = rbf_kernel(X_train, X_train) + sigma_n_sq * np.eye(3)
K_dense_dense = rbf_kernel(x_dense, x_dense)

L_exact = np.linalg.cholesky(K_train_exact)
alpha_exact = np.linalg.solve(L_exact.T, np.linalg.solve(L_exact, y_train))
mu_dense = K_dense_train @ alpha_exact

v_dense = np.linalg.solve(L_exact, K_dense_train.T)
cov_dense = K_dense_dense - v_dense.T @ v_dense
std_dense = np.sqrt(np.maximum(1e-8, np.diag(cov_dense)))[:, None]

# Nilai titik uji manual
x_test_pts = np.array([0.5, 2.5])
y_test_mu = np.array([0.4263, 0.6426])
y_test_std = np.array([0.1375, 0.3918])

fig, ax = plt.subplots(figsize=(8.5, 4.8))

# Shaded confidence band
ax.fill_between(x_dense.ravel(), (mu_dense - 2*std_dense).ravel(), (mu_dense + 2*std_dense).ravel(),
                color='#60a5fa', alpha=0.25, label=r'Pita Ketidakpastian Posterior $\pm 2\sigma(x)$')
ax.plot(x_dense.ravel(), mu_dense.ravel(), color='#1d4ed8', linewidth=2.2, label=r'Mean Prediktif $\bar{f}(x)$')

# Titik data latih
ax.scatter(X_train, y_train, color='#dc2626', s=70, zorder=6, label=r'Training Data $N=3$ $(x_i, y_i)$')

# Titik data uji beserta error bar
ax.errorbar(x_test_pts, y_test_mu, yerr=2*y_test_std, fmt='o', color='#047857', ecolor='#047857',
            elinewidth=2, capsize=6, capthick=2, markersize=8, zorder=7,
            label=r'Test Points $N_*=2$ $(x_{*j}, \bar{f}_{*j} \pm 2\sigma_{*j})$')

# Anotasi
ax.annotate(r'$x_{*1} = 0.5$ (Interpolation Test Point)' + '\n' + r'$\mathbb{V}[f_{*1}] = 0.0189$ (Kecil)',
            xy=(0.5, 0.4263), xytext=(0.5, -0.4),
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
            fontsize=9, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="#d1fae5", ec="#10b981", lw=1))

ax.annotate(r'$x_{*2} = 2.5$ (Extrapolation Test Point)' + '\n' + r'$\mathbb{V}[f_{*2}] = 0.1535$ (Membesar)',
            xy=(2.5, 0.6426), xytext=(2.5, 1.4),
            arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
            fontsize=9, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#ef4444", lw=1))

ax.set_title("Hasil Eksak Regresi Gaussian Process pada Soal Latihan ($N=3, N_*=2$)")
ax.set_xlabel("Input x")
ax.set_ylabel("Nilai Fungsi f(x)")
ax.set_ylim(-0.8, 1.8)
ax.legend(loc='upper left', fontsize=9.5)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "fig5_worked_example.pdf"))
plt.savefig(os.path.join(figures_dir, "fig5_worked_example.png"), dpi=300)
plt.close()

print("Semua grafik visualisasi berhasil dibuat dengan sukses di folder figures/!")
