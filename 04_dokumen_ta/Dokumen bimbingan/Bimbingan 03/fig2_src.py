# Visualisasi Sampel Fungsi Prior (Gambar 2)
import numpy as np
import matplotlib.pyplot as plt
import os

x_grid = np.linspace(-3.0, 3.0, 300)

kernel_configs = [
    ("(a) Squared Exponential (RBF)\n" + r"($l=1.0, \sigma_f=1.0$)", 
     kernel_squared_exponential, {'l': 1.0, 'sigma_f': 1.0}),
    ("(b) Exponential / Matérn 1/2\n" + r"($l=1.0, \sigma_f=1.0$)", 
     kernel_exponential, {'l': 1.0, 'sigma_f': 1.0}),
    ("(c) Matérn 3/2\n" + r"($l=1.0, \sigma_f=1.0$)", 
     kernel_matern32, {'l': 1.0, 'sigma_f': 1.0}),
    ("(d) Matérn 5/2\n" + r"($l=1.0, \sigma_f=1.0$)", 
     kernel_matern52, {'l': 1.0, 'sigma_f': 1.0}),
    ("(e) Rational Quadratic (RQ)\n" + r"($l=1.0, \sigma_f=1.0, \alpha=0.5$)", 
     kernel_rational_quadratic, {'l': 1.0, 'sigma_f': 1.0, 'alpha': 0.5}),
    ("(f) Linear (Non-Stasioner)\n" + r"($c=0.0, \sigma_b=0.2, \sigma_v=1.0$)", 
     kernel_linear, {'sigma_b': 0.2, 'sigma_v': 1.0, 'c': 0.0})
]

fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
axs = axs.flatten()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, (title, k_func, kwargs) in enumerate(kernel_configs):
    ax = axs[i]
    np.random.seed(150 + i)
    f_samples, _, _ = sample_gp_prior(x_grid, k_func, n_samples=4, jitter=1e-5, **kwargs)
    
    for s in range(4):
        ax.plot(x_grid, f_samples[:, s], color=colors[s], alpha=0.85, label=f'Sampel #{s+1}')
    
    ax.set_title(title, fontsize=10.5, pad=6)
    ax.set_xlim([-3.0, 3.0])
    ax.set_ylim([-3.5, 3.5])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=7.5, ncol=2, framealpha=0.85)
    if i >= 3:
        ax.set_xlabel(r'Domain Input $x$')
    if i % 3 == 0:
        ax.set_ylabel(r'Nilai Fungsi $f(x)$')
        
fig.suptitle(r'Sampel Fungsi Prior $\mathbf{f} \sim \mathcal{GP}(0, K)$ dari Berbagai Kelas Kernel', fontsize=12.5)
plt.tight_layout()

# Simpan ke PDF & PNG
fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_sample_paths_kernels.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_sample_paths_kernels.png'), bbox_inches='tight')
plt.show()
