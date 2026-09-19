"""
Skrip Generator Visualisasi Matematika & Grafis Dokumen Kredit Bimbingan #3
Topik: Karakteristik Kernel Buku Teks Rasmussen & Williams, Eksperimen Hyperparameter, & Sampling GP via Cholesky
Target: Draf Landasan Teori Skripsi / Laporan Tugas Akhir
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Konfigurasi gaya visual publikasi akademik
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.titlesize': 13,
    'lines.linewidth': 1.6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'mathtext.fontset': 'cm',
    'figure.autolayout': True
})

# Direktori penyimpanan gambar
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set random seed untuk reproduktibilitas
np.random.seed(42)

# =========================================================================
# DEFINISI FUNGSI KERNEL BAKU (RASMUSSEN & WILLIAMS, BAB 4)
# =========================================================================

def kernel_squared_exponential(x1, x2, l=1.0, sigma_f=1.0):
    """Squared Exponential (SE / RBF / Gaussian) Kernel"""
    r_sq = (np.subtract.outer(x1, x2)) ** 2
    return (sigma_f ** 2) * np.exp(-0.5 * r_sq / (l ** 2))

def kernel_exponential(x1, x2, l=1.0, sigma_f=1.0):
    """Exponential Kernel (Matern nu=1/2)"""
    r = np.abs(np.subtract.outer(x1, x2))
    return (sigma_f ** 2) * np.exp(-r / l)

def kernel_matern32(x1, x2, l=1.0, sigma_f=1.0):
    """Matern 3/2 Kernel (nu=3/2)"""
    r = np.abs(np.subtract.outer(x1, x2))
    factor = np.sqrt(3.0) * r / l
    return (sigma_f ** 2) * (1.0 + factor) * np.exp(-factor)

def kernel_matern52(x1, x2, l=1.0, sigma_f=1.0):
    """Matern 5/2 Kernel (nu=5/2)"""
    r = np.abs(np.subtract.outer(x1, x2))
    factor = np.sqrt(5.0) * r / l
    return (sigma_f ** 2) * (1.0 + factor + (5.0 * (r ** 2)) / (3.0 * (l ** 2))) * np.exp(-factor)

def kernel_rational_quadratic(x1, x2, l=1.0, sigma_f=1.0, alpha=1.0):
    """Rational Quadratic (RQ) Kernel (Continuous Scale Mixture of SE)"""
    r_sq = (np.subtract.outer(x1, x2)) ** 2
    return (sigma_f ** 2) * (1.0 + r_sq / (2.0 * alpha * (l ** 2))) ** (-alpha)

def kernel_gamma_exponential(x1, x2, l=1.0, sigma_f=1.0, gamma=1.0):
    """Gamma-Exponential Kernel (0 < gamma <= 2)"""
    r = np.abs(np.subtract.outer(x1, x2))
    return (sigma_f ** 2) * np.exp(- (r / l) ** gamma)

def kernel_periodic(x1, x2, l=1.0, sigma_f=1.0, p=1.0):
    """Periodic (Exp-Sine-Squared) Kernel"""
    r = np.abs(np.subtract.outer(x1, x2))
    sin_term = np.sin(np.pi * r / p)
    return (sigma_f ** 2) * np.exp(-2.0 * (sin_term ** 2) / (l ** 2))

def kernel_cosine(x1, x2, sigma_f=1.0, p=1.0):
    """Cosine Kernel"""
    r = np.abs(np.subtract.outer(x1, x2))
    return (sigma_f ** 2) * np.cos(2.0 * np.pi * r / p)

def kernel_linear(x1, x2, sigma_b=0.0, sigma_v=1.0, c=0.0):
    """Linear (Dot Product) Kernel (Non-stationary)"""
    x1_c = x1 - c
    x2_c = x2 - c
    return (sigma_b ** 2) + (sigma_v ** 2) * np.outer(x1_c, x2_c)

def kernel_polynomial(x1, x2, sigma_0=1.0, sigma_v=1.0, d=2):
    """Polynomial Kernel (Non-stationary)"""
    return ((sigma_0 ** 2) + (sigma_v ** 2) * np.outer(x1, x2)) ** d

def kernel_neural_network(x1, x2, sigma_0=1.0, sigma_v=1.0, sigma_f=1.0):
    """Neural Network (Williams, 1998 / Neal, 1995 Arc-sine) Kernel (Non-stationary)"""
    num = 2.0 * ((sigma_0 ** 2) + (sigma_v ** 2) * np.outer(x1, x2))
    den1 = 1.0 + 2.0 * ((sigma_0 ** 2) + (sigma_v ** 2) * (x1 ** 2))
    den2 = 1.0 + 2.0 * ((sigma_0 ** 2) + (sigma_v ** 2) * (x2 ** 2))
    den = np.sqrt(np.outer(den1, den2))
    ratio = np.clip(num / den, -1.0 + 1e-7, 1.0 - 1e-7)
    return (sigma_f ** 2) * (2.0 / np.pi) * np.arcsin(ratio)

def sample_gp_prior(x_grid, kernel_func, n_samples=4, jitter=1e-6, **kwargs):
    """Fungsi pembantu sampling prior GP via Dekomposisi Cholesky"""
    K = kernel_func(x_grid, x_grid, **kwargs)
    K_stable = K + jitter * np.eye(len(x_grid))
    L = np.linalg.cholesky(K_stable)
    u = np.random.randn(len(x_grid), n_samples)
    f_samples = L @ u
    return f_samples, K, L

# =========================================================================
# GAMBAR 1: PROFIL FUNGSI KOVARIANSI RASMUSSEN (k(r) vs r)
# =========================================================================
def generate_fig1_kernel_profiles():
    print("Menghasilkan Gambar 1: Profil Fungsi Kernel Rasmussen & Williams...")
    r = np.linspace(0, 4.0, 500)
    l_val = 1.0
    sigma_f_val = 1.0
    
    # 1. Stasioner Monotonik
    k_exp = (sigma_f_val ** 2) * np.exp(-r / l_val)
    k_mat32 = (sigma_f_val ** 2) * (1.0 + np.sqrt(3.0) * r / l_val) * np.exp(-np.sqrt(3.0) * r / l_val)
    k_mat52 = (sigma_f_val ** 2) * (1.0 + np.sqrt(5.0) * r / l_val + 5.0 * (r ** 2) / (3.0 * (l_val ** 2))) * np.exp(-np.sqrt(5.0) * r / l_val)
    k_se = (sigma_f_val ** 2) * np.exp(-0.5 * (r ** 2) / (l_val ** 2))
    k_rq = (sigma_f_val ** 2) * (1.0 + (r ** 2) / (2.0 * 0.5 * (l_val ** 2))) ** (-0.5)
    
    # 2. Periodik
    k_per1 = (sigma_f_val ** 2) * np.exp(-2.0 * (np.sin(np.pi * r / 2.0) ** 2) / (1.0 ** 2))
    k_per2 = (sigma_f_val ** 2) * np.exp(-2.0 * (np.sin(np.pi * r / 1.0) ** 2) / (0.5 ** 2))
    k_cos = (sigma_f_val ** 2) * np.cos(2.0 * np.pi * r / 2.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    
    # Panel 1: Stasioner Isotropik Standar
    ax1.plot(r, k_exp, label=r'Exponential / Matérn $\nu=1/2$', color='#d95f02', linestyle='--')
    ax1.plot(r, k_mat32, label=r'Matérn $\nu=3/2$', color='#7570b3', linestyle='-.')
    ax1.plot(r, k_mat52, label=r'Matérn $\nu=5/2$', color='#1b9e77', linestyle=':')
    ax1.plot(r, k_se, label=r'Squared Exponential / RBF', color='#2b83ba', linewidth=2.0)
    ax1.plot(r, k_rq, label=r'Rational Quadratic ($\alpha=0.5$)', color='#e7298a', linestyle='-')
    
    ax1.set_title(r'(a) Profil Kernel Stasioner Isotropik ($l=1.0, \sigma_f=1.0$)', fontsize=11)
    ax1.set_xlabel(r'Jarak Euclidean $r = \|\mathbf{x} - \mathbf{x}^\prime\|$')
    ax1.set_ylabel(r'Nilai Kovariansi $k(r)$')
    ax1.set_xlim([0, 4.0])
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.95)
    
    # Panel 2: Periodik & Cosine
    ax2.plot(r, k_per1, label=r'Periodic ($p=2.0, l=1.0$)', color='#e41a1c', lw=1.8)
    ax2.plot(r, k_per2, label=r'Periodic ($p=1.0, l=0.5$)', color='#377eb8', linestyle='--', lw=1.6)
    ax2.plot(r, k_cos, label=r'Cosine ($p=2.0$)', color='#4daf4a', linestyle='-.', lw=1.6)
    
    ax2.set_title(r'(b) Profil Kernel Periodik & Gelombang', fontsize=11)
    ax2.set_xlabel(r'Jarak Euclidean $r = \|\mathbf{x} - \mathbf{x}^\prime\|$')
    ax2.set_ylabel(r'Nilai Kovariansi $k(r)$')
    ax2.set_xlim([0, 4.0])
    ax2.set_ylim([-1.05, 1.05])
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.95)

    fig.suptitle(r'Perbandingan Profil Fungsi Kovariansi $k(r)$ dari Buku Teks Rasmussen \& Williams (2006)', fontsize=12.5)

    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_kernel_profiles.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_kernel_profiles.png'), bbox_inches='tight')
    plt.close(fig)

# =========================================================================
# GAMBAR 2: GALERI SAMPEL FUNGSI PRIOR UNTUK RAGAM KERNEL RASMUSSEN
# =========================================================================
def generate_fig2_sample_paths_kernels():
    print("Menghasilkan Gambar 2: Galeri Sampel Fungsi Prior untuk 6 Tipe Kernel Rasmussen...")
    x_grid = np.linspace(-3.0, 3.0, 300)
    
    kernel_configs = [
        ("Squared Exponential (RBF)\n" + r"($l=1.0, \sigma_f=1.0$)", 
         kernel_squared_exponential, {'l': 1.0, 'sigma_f': 1.0}),
        ("Matérn 3/2\n" + r"($l=1.0, \sigma_f=1.0$)", 
         kernel_matern32, {'l': 1.0, 'sigma_f': 1.0}),
        ("Rational Quadratic (RQ)\n" + r"($l=1.0, \sigma_f=1.0, \alpha=0.5$)", 
         kernel_rational_quadratic, {'l': 1.0, 'sigma_f': 1.0, 'alpha': 0.5}),
        ("Periodic (Exp-Sine-Squared)\n" + r"($p=2.0, l=1.0, \sigma_f=1.0$)", 
         kernel_periodic, {'l': 1.0, 'sigma_f': 1.0, 'p': 2.0}),
        ("Linear (Non-Stasioner)\n" + r"($c=0.0, \sigma_b=0.2, \sigma_v=1.0$)", 
         kernel_linear, {'sigma_b': 0.2, 'sigma_v': 1.0, 'c': 0.0}),
        ("Neural Network (Non-Stasioner)\n" + r"($\sigma_0=1.0, \sigma_v=5.0, \sigma_f=1.0$)", 
         kernel_neural_network, {'sigma_0': 1.0, 'sigma_v': 5.0, 'sigma_f': 1.0})
    ]
    
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    axs = axs.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (title, k_func, kwargs) in enumerate(kernel_configs):
        ax = axs[i]
        np.random.seed(150 + i)
        f_samples, _, _ = sample_gp_prior(x_grid, k_func, n_samples=4, jitter=1e-5, **kwargs)
        
        for s in range(4):
            ax.plot(x_grid, f_samples[:, s], color=colors[s], alpha=0.85, label=f'Sampel #{s+1}' if i == 0 else None)
        
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlim([-3.0, 3.0])
        ax.set_ylim([-3.5, 3.5])
        ax.grid(True, linestyle=':', alpha=0.6)
        if i >= 3:
            ax.set_xlabel(r'Domain Input $x$')
        if i % 3 == 0:
            ax.set_ylabel(r'Nilai Fungsi $f(x)$')
            
    fig.suptitle(r'Galeri Realisasi Sampel Fungsi Prior $\mathbf{f} \sim \mathcal{GP}(0, K)$ dari Berbagai Kelas Kernel', fontsize=12.5)
    fig.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=True)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_sample_paths_kernels.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_sample_paths_kernels.png'), bbox_inches='tight')
    plt.close(fig)

# =========================================================================
# GAMBAR 3: EKSPERIMEN VARIASI HYPERPARAMETER KOMPREHENSIF
# =========================================================================
def generate_fig3_hyperparameter_effects():
    print("Menghasilkan Gambar 3: Analisis Eksperimen Variasi Nilai Hyperparameter...")
    x_grid = np.linspace(-3.5, 3.5, 300)
    
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
    
    np.random.seed(2026)
    u_fixed = np.random.randn(len(x_grid), 3) # Noise baku tetap untuk perbandingan konsisten
    
    # 1. Variasi Lengthscale l pada SE (sigma_f = 1.0)
    ax1 = axs[0, 0]
    for l_val, col in zip([0.3, 1.0, 2.5], ['#d7191c', '#fdae61', '#2b83ba']):
        K = kernel_squared_exponential(x_grid, x_grid, l=l_val, sigma_f=1.0) + 1e-6 * np.eye(len(x_grid))
        L = np.linalg.cholesky(K)
        f_s = L @ u_fixed[:, 0]
        ax1.plot(x_grid, f_s, label=f'$l = {l_val}$', color=col, lw=1.8)
    ax1.set_title(r'(a) SE: Variasi $l$ ($\sigma_f=1.0$)', fontsize=10.5)
    ax1.set_ylabel(r'$f(x)$')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # 2. Variasi Signal Variance sigma_f^2 pada SE (l = 1.0)
    ax2 = axs[0, 1]
    for sf_sq, col in zip([0.25, 1.0, 4.0], ['#abdda4', '#2b83ba', '#d7191c']):
        sf = np.sqrt(sf_sq)
        K = kernel_squared_exponential(x_grid, x_grid, l=1.0, sigma_f=sf) + 1e-6 * np.eye(len(x_grid))
        L = np.linalg.cholesky(K)
        f_s = L @ u_fixed[:, 0]
        ax2.plot(x_grid, f_s, label=r'$\sigma_f^2 = ' + f'{sf_sq}$', color=col, lw=1.8)
    ax2.set_title(r'(b) SE: Variasi $\sigma_f^2$ ($l=1.0$)', fontsize=10.5)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')
    
    # 3. Variasi Scale Mixture alpha pada Rational Quadratic (l = 1.0, sigma_f = 1.0)
    ax3 = axs[0, 2]
    for a_val, col in zip([0.1, 0.5, 2.0, 20.0], ['#e41a1c', '#984ea3', '#4daf4a', '#377eb8']):
        K = kernel_rational_quadratic(x_grid, x_grid, l=1.0, sigma_f=1.0, alpha=a_val) + 1e-6 * np.eye(len(x_grid))
        L = np.linalg.cholesky(K)
        f_s = L @ u_fixed[:, 1]
        lbl = r'$\alpha = 20 (\approx\mathrm{SE})$' if a_val == 20.0 else f'$\\alpha = {a_val}$'
        ax3.plot(x_grid, f_s, label=lbl, color=col, lw=1.7)
    ax3.set_title(r'(c) RQ: Variasi $\alpha$ (Multiskala)', fontsize=10.5)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right')
    
    # 4. Variasi Periode p pada Periodic Kernel (l = 1.0, sigma_f = 1.0)
    ax4 = axs[1, 0]
    for p_val, col in zip([1.0, 2.0, 4.0], ['#e41a1c', '#377eb8', '#4daf4a']):
        K = kernel_periodic(x_grid, x_grid, l=1.0, sigma_f=1.0, p=p_val) + 1e-5 * np.eye(len(x_grid))
        L = np.linalg.cholesky(K)
        f_s = L @ u_fixed[:, 0]
        ax4.plot(x_grid, f_s, label=f'$p = {p_val}$', color=col, lw=1.8)
    ax4.set_title(r'(d) Periodic: Variasi Periode $p$', fontsize=10.5)
    ax4.set_xlabel(r'Domain Input $x$')
    ax4.set_ylabel(r'$f(x)$')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right')
    
    # 5. Variasi Offset c & Slope pada Linear Kernel (sigma_b = 0.2)
    ax5 = axs[1, 1]
    for c_val, col in zip([-1.5, 0.0, 1.5], ['#e41a1c', '#377eb8', '#4daf4a']):
        K = kernel_linear(x_grid, x_grid, sigma_b=0.2, sigma_v=1.0, c=c_val) + 1e-5 * np.eye(len(x_grid))
        L = np.linalg.cholesky(K)
        f_s = L @ u_fixed[:, 2]
        ax5.plot(x_grid, f_s, label=f'$c = {c_val}$', color=col, lw=1.8)
    ax5.set_title(r'(e) Linear: Variasi Titik Tumpu $c$', fontsize=10.5)
    ax5.set_xlabel(r'Domain Input $x$')
    ax5.grid(True, linestyle=':', alpha=0.6)
    ax5.legend(loc='upper right')
    
    # 6. Variasi Bobot sigma_v pada Neural Network Kernel (sigma_0 = 1.0)
    ax6 = axs[1, 2]
    for sv_val, col in zip([1.0, 5.0, 20.0], ['#4daf4a', '#377eb8', '#e41a1c']):
        K = kernel_neural_network(x_grid, x_grid, sigma_0=1.0, sigma_v=sv_val, sigma_f=1.0) + 1e-5 * np.eye(len(x_grid))
        L = np.linalg.cholesky(K)
        f_s = L @ u_fixed[:, 1]
        ax6.plot(x_grid, f_s, label=f'$\\sigma_v = {sv_val}$', color=col, lw=1.8)
    ax6.set_title(r'(f) Neural Network: Variasi Bobot $\sigma_v$', fontsize=10.5)
    ax6.set_xlabel(r'Domain Input $x$')
    ax6.grid(True, linestyle=':', alpha=0.6)
    ax6.legend(loc='upper right')
    
    fig.suptitle(r'Eksperimen Komputasi Pengaruh Variasi Nilai Hyperparameter terhadap Perilaku Fungsi Prior', fontsize=12.5)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_hyperparameter_effects.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_hyperparameter_effects.png'), bbox_inches='tight')
    plt.close(fig)

# =========================================================================
# GAMBAR 4: ALUR KERJA SAMPLING CHOLESKY (K -> L -> L*u = f)
# =========================================================================
def generate_fig4_cholesky_sampling_workflow():
    print("Menghasilkan Gambar 4: Diagram Alur Komputasi Sampling Cholesky...")
    x_small = np.linspace(-2.0, 2.0, 25)
    K = kernel_squared_exponential(x_small, x_small, l=1.0, sigma_f=1.0) + 1e-5 * np.eye(len(x_small))
    L = np.linalg.cholesky(K)
    
    np.random.seed(99)
    u = np.random.randn(len(x_small), 1)
    f = L @ u
    
    fig = plt.figure(figsize=(12, 3.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.8, 1.2])
    
    # 1. Heatmap K
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(K, cmap='viridis', origin='upper')
    ax1.set_title(r'(a) Matriks Gram $K$', fontsize=10.5)
    ax1.set_xlabel('Indeks $j$')
    ax1.set_ylabel('Indeks $i$')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Heatmap L (Segitiga Bawah)
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(L, cmap='plasma', origin='upper')
    ax2.set_title(r'(b) Cholesky $L$ ($K = LL^T$)', fontsize=10.5)
    ax2.set_xlabel('Indeks $j$')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Barplot u (vektor baku independen)
    ax3 = fig.add_subplot(gs[2])
    ax3.stem(np.arange(len(u)), u.flatten(), basefmt=" ", linefmt='gray', markerfmt='ro')
    ax3.set_title(r'(c) Acak $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, I)$', fontsize=10.5)
    ax3.set_xlabel('Indeks $i$')
    ax3.set_ylabel('Nilai $u_i$')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # 4. Kurva fungsi f = L*u
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(x_small, f, 'b-o', markersize=4, label=r'$\mathbf{f} = L\mathbf{u}$')
    ax4.set_title(r'(d) Realisasi $\mathbf{f} \sim \mathcal{N}(\mathbf{0}, K)$', fontsize=10.5)
    ax4.set_xlabel(r'Input $x$')
    ax4.set_ylabel(r'$f(x)$')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right')
    
    fig.suptitle(r'Mekanisme Komputasi Sampling Fungsi GP: Transformasi Linear Vektor Baku via Dekomposisi Cholesky', fontsize=11.5)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_cholesky_sampling_workflow.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_cholesky_sampling_workflow.png'), bbox_inches='tight')
    plt.close(fig)

# =========================================================================
# GAMBAR 5: SAMPLING PRIOR VS POSTERIOR DENGAN PITA KETIDAKPASTIAN 95%
# =========================================================================
def generate_fig5_prior_vs_posterior_samples():
    print("Menghasilkan Gambar 5: Perbandingan Sampel Fungsi Prior vs Posterior...")
    x_test = np.linspace(-3.5, 3.5, 300)
    
    # Parameter model
    l = 1.0
    sigma_f = 1.0
    sigma_n = 0.15
    
    # 1. Prior GP
    K_ss = kernel_squared_exponential(x_test, x_test, l=l, sigma_f=sigma_f) + 1e-6 * np.eye(len(x_test))
    L_prior = np.linalg.cholesky(K_ss)
    np.random.seed(42)
    prior_samples = L_prior @ np.random.randn(len(x_test), 4)
    prior_std = np.sqrt(np.diag(K_ss))
    
    # 2. Data Observasi Training
    X_train = np.array([-2.5, -1.0, 0.5, 2.0])
    y_train = np.sin(X_train * 1.2) + np.random.normal(0, sigma_n, size=len(X_train))
    
    # 3. Posterior GP Conditioning
    K_train = kernel_squared_exponential(X_train, X_train, l=l, sigma_f=sigma_f) + (sigma_n ** 2) * np.eye(len(X_train))
    K_s = kernel_squared_exponential(X_train, x_test, l=l, sigma_f=sigma_f)
    
    L_train = np.linalg.cholesky(K_train)
    alpha = np.linalg.solve(L_train.T, np.linalg.solve(L_train, y_train))
    
    mu_post = K_s.T @ alpha
    v = np.linalg.solve(L_train, K_s)
    cov_post = K_ss - v.T @ v + 1e-6 * np.eye(len(x_test))
    
    L_post = np.linalg.cholesky(cov_post)
    post_samples = mu_post[:, None] + L_post @ np.random.randn(len(x_test), 4)
    post_std = np.sqrt(np.diag(cov_post))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    
    # Plot Prior
    ax1.fill_between(x_test, -2 * prior_std, 2 * prior_std, color='#9ecae1', alpha=0.4, label=r'Pita Ketidakpastian $95\%$ ($\pm 2\sigma$)')
    ax1.plot(x_test, np.zeros_like(x_test), 'k--', lw=1.2, label=r'Mean Prior $\mu(x)=0$')
    for s in range(4):
        ax1.plot(x_test, prior_samples[:, s], lw=1.4, alpha=0.85)
    ax1.set_title(r'(a) Prior GP: Ketidakpastian Merata di Seluruh Ruang')
    ax1.set_xlabel(r'Input $x$')
    ax1.set_ylabel(r'Nilai Fungsi $f(x)$')
    ax1.set_xlim([-3.5, 3.5])
    ax1.set_ylim([-3.0, 3.0])
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=8.5)
    
    # Plot Posterior
    ax2.fill_between(x_test, mu_post - 2 * post_std, mu_post + 2 * post_std, color='#a1d99b', alpha=0.5, label=r'Pita Ketidakpastian Posterior $95\%$ ($\pm 2\sigma_*$)')
    ax2.plot(x_test, mu_post, 'g-', lw=2.0, label=r'Mean Prediktif $\bar{f}_*(x)$')
    for s in range(4):
        ax2.plot(x_test, post_samples[:, s], lw=1.4, alpha=0.85)
    ax2.plot(X_train, y_train, 'ro', markersize=6, zorder=5, label=r'Data Observasi $(x_i, y_i)$')
    ax2.set_title(r'(b) Posterior GP: Ketidakpastian Menciut di Dekat Data')
    ax2.set_xlabel(r'Input $x$')
    ax2.set_xlim([-3.5, 3.5])
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=8.5)
    
    fig.suptitle(r'Evolusi Distribusi GP: Realisasi Sampel Fungsi Sebelum (Prior) dan Sesudah Observasi Data (Posterior)', fontsize=12.5)
    
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_prior_vs_posterior_samples.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_prior_vs_posterior_samples.png'), bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    print("Memulai pembuatan seluruh visualisasi Dokumen Kredit Bimbingan #3...")
    generate_fig1_kernel_profiles()
    generate_fig2_sample_paths_kernels()
    generate_fig3_hyperparameter_effects()
    generate_fig4_cholesky_sampling_workflow()
    generate_fig5_prior_vs_posterior_samples()
    print(f"Semua gambar berhasil digenerate dan disimpan di direktori: {OUTPUT_DIR}")
