# Visualisasi Karakteristik Kovariansi: Peluruhan k(r) vs Sampel Prior f(x) (Gambar 1)
import numpy as np
import matplotlib.pyplot as plt
import os

r = np.linspace(0, 4.0, 500)
x_grid = np.linspace(-3.0, 3.0, 300)

# Vektor noise tetap (Rasmussen style: noise sama memperjelas murni dampak kernel)
np.random.seed(42)
u_fixed_1 = np.random.randn(len(x_grid))
u_fixed_2 = np.random.randn(len(x_grid))

# 1. Kovariansi Matern
k_m12 = kernel_exponential(r, np.array([0.0]), l=1.0, sigma_f=1.0).flatten()
k_m32 = kernel_matern32(r, np.array([0.0]), l=1.0, sigma_f=1.0).flatten()
k_m52 = kernel_matern52(r, np.array([0.0]), l=1.0, sigma_f=1.0).flatten()
k_se  = kernel_squared_exponential(r, np.array([0.0]), l=1.0, sigma_f=1.0).flatten()

# Sampel Matern (u identik)
_, _, L_m12 = sample_gp_prior(x_grid, kernel_exponential, l=1.0, sigma_f=1.0)
f_m12 = L_m12 @ u_fixed_1

_, _, L_m32 = sample_gp_prior(x_grid, kernel_matern32, l=1.0, sigma_f=1.0)
f_m32 = L_m32 @ u_fixed_1

_, _, L_m52 = sample_gp_prior(x_grid, kernel_matern52, l=1.0, sigma_f=1.0)
f_m52 = L_m52 @ u_fixed_1

_, _, L_se = sample_gp_prior(x_grid, kernel_squared_exponential, l=1.0, sigma_f=1.0)
f_se = L_se @ u_fixed_1

# 2. Kovariansi Rational Quadratic
k_rq_02 = kernel_rational_quadratic(r, np.array([0.0]), l=1.0, sigma_f=1.0, alpha=0.2).flatten()
k_rq_10 = kernel_rational_quadratic(r, np.array([0.0]), l=1.0, sigma_f=1.0, alpha=1.0).flatten()
k_rq_50 = kernel_rational_quadratic(r, np.array([0.0]), l=1.0, sigma_f=1.0, alpha=5.0).flatten()

# Sampel Rational Quadratic (u identik)
_, _, L_rq_02 = sample_gp_prior(x_grid, kernel_rational_quadratic, l=1.0, sigma_f=1.0, alpha=0.2)
f_rq_02 = L_rq_02 @ u_fixed_2

_, _, L_rq_10 = sample_gp_prior(x_grid, kernel_rational_quadratic, l=1.0, sigma_f=1.0, alpha=1.0)
f_rq_10 = L_rq_10 @ u_fixed_2

_, _, L_rq_50 = sample_gp_prior(x_grid, kernel_rational_quadratic, l=1.0, sigma_f=1.0, alpha=5.0)
f_rq_50 = L_rq_50 @ u_fixed_2

f_rq_se = L_se @ u_fixed_2

# Plotting Gambar 1
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))

# Panel (a): Peluruhan Kovariansi Matern
ax1 = axs[0, 0]
ax1.plot(r, k_m12, label=r'Exponential ($\nu=1/2$)', color='#d95f02', linestyle='--')
ax1.plot(r, k_m32, label=r'Matérn ($\nu=3/2$)', color='#7570b3', linestyle='-.')
ax1.plot(r, k_m52, label=r'Matérn ($\nu=5/2$)', color='#1b9e77', linestyle=':')
ax1.plot(r, k_se, label=r'Squared Exponential ($\nu \to \infty$)', color='#2b83ba', linewidth=2.0)
ax1.set_title(r'(a) Peluruhan Kovariansi Matérn $k(r)$', fontsize=10.5)
ax1.set_xlabel(r'Jarak Euclidean $r = \|\mathbf{x} - \mathbf{x}^\prime\|$')
ax1.set_ylabel(r'Nilai Kovariansi $k(r)$')
ax1.set_xlim([0, 4.0])
ax1.set_ylim([-0.05, 1.05])
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', framealpha=0.95)

# Panel (b): Sampel Matern (Legend di KANAN BAWAH tanpa penjelasan kurung)
ax2 = axs[0, 1]
ax2.plot(x_grid, f_m12, label=r'$\nu=1/2$', color='#d95f02', linestyle='--', alpha=0.9)
ax2.plot(x_grid, f_m32, label=r'$\nu=3/2$', color='#7570b3', linestyle='-.', alpha=0.9)
ax2.plot(x_grid, f_m52, label=r'$\nu=5/2$', color='#1b9e77', linestyle=':', alpha=0.9)
ax2.plot(x_grid, f_se, label=r'$\nu \to \infty$', color='#2b83ba', linewidth=2.0, alpha=0.9)
ax2.set_title(r'(b) Sampel Fungsi Prior Matérn ($f \sim \mathcal{GP}(0, K)$)', fontsize=10.5)
ax2.set_xlabel(r'Domain Input $x$')
ax2.set_ylabel(r'Nilai Fungsi $f(x)$')
ax2.set_xlim([-3.0, 3.0])
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower right', framealpha=0.95)

# Panel (c): Peluruhan Kovariansi RQ vs SE
ax3 = axs[1, 0]
ax3.plot(r, k_rq_02, label=r'Rational Quadratic ($\alpha=0.2$)', color='#e7298a', linestyle='--')
ax3.plot(r, k_rq_10, label=r'Rational Quadratic ($\alpha=1.0$)', color='#984ea3', linestyle='-.')
ax3.plot(r, k_rq_50, label=r'Rational Quadratic ($\alpha=5.0$)', color='#4daf4a', linestyle=':')
ax3.plot(r, k_se, label=r'Squared Exponential ($\alpha \to \infty$)', color='#2b83ba', linewidth=2.0)
ax3.set_title(r'(c) Peluruhan Kovariansi Rational Quadratic vs SE', fontsize=10.5)
ax3.set_xlabel(r'Jarak Euclidean $r = \|\mathbf{x} - \mathbf{x}^\prime\|$')
ax3.set_ylabel(r'Nilai Kovariansi $k(r)$')
ax3.set_xlim([0, 4.0])
ax3.set_ylim([-0.05, 1.05])
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper right', framealpha=0.95)

# Panel (d): Sampel RQ (Legend tanpa penjelasan kurung)
ax4 = axs[1, 1]
ax4.plot(x_grid, f_rq_02, label=r'$\alpha=0.2$', color='#e7298a', linestyle='--', alpha=0.9)
ax4.plot(x_grid, f_rq_10, label=r'$\alpha=1.0$', color='#984ea3', linestyle='-.', alpha=0.9)
ax4.plot(x_grid, f_rq_50, label=r'$\alpha=5.0$', color='#4daf4a', linestyle=':', alpha=0.9)
ax4.plot(x_grid, f_rq_se, label=r'$\alpha \to \infty$', color='#2b83ba', linewidth=2.0, alpha=0.9)
ax4.set_title(r'(d) Sampel Fungsi Prior Rational Quadratic ($f \sim \mathcal{GP}(0, K)$)', fontsize=10.5)
ax4.set_xlabel(r'Domain Input $x$')
ax4.set_ylabel(r'Nilai Fungsi $f(x)$')
ax4.set_xlim([-3.0, 3.0])
ax4.grid(True, linestyle=':', alpha=0.6)
ax4.legend(loc='lower left', framealpha=0.95)

fig.suptitle(r'Karakteristik Fungsi Kovariansi: Peluruhan $k(r)$ vs Sampel Fungsi Prior', fontsize=11.5)
plt.tight_layout()

# Simpan luaran gambar berkualitas tinggi
fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_kernel_profiles.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_kernel_profiles.png'), bbox_inches='tight')
plt.show()
