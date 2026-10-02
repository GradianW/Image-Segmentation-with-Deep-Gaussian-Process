import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# ==========================================
# CELL 1: HEADER & INTRO
# ==========================================
cells.append(nbf.v4.new_markdown_cell(r"""# Catatan Tanya-Jawab & Eksplorasi Teoretis: Optimasi Parameter GP
**Rujukan Tunggal**: Rasmussen, C. E., & Williams, C. K. (2006). *Gaussian Processes for Machine Learning*. MIT Press, Cambridge, MA.  
**Fokus**: Chapter 5 (*Model Selection and Adaptation of Hyperparameters*)

---

## ❓ Pertanyaan 1:
> **"Apakah benar untuk mempelajari optimasi parameter GP berdasarkan buku Rasmussen ini, maka saya harus membaca Chapter 5? Jika iya, tolong jelaskan secara singkat atau overviewnya apa yang dibahas pada setiap subbab chapter 5 ini."**

---

## 💡 Jawaban & Konfirmasi:
**YA, BENAR SEKALI.**  
Dalam buku teks *Gaussian Processes for Machine Learning* (Rasmussen & Williams, 2006), seluruh pembahasan mengenai bagaimana cara memilih, mengadaptasi, dan mengoptimasi parameter (yang secara formal disebut **hyperparameter**) dibahas secara mendalam dan tuntas pada **Chapter 5: Model Selection and Adaptation of Hyperparameters** (hlm. 105–128).

Berikut adalah **ringkasan / overview komprehensif mengenai apa yang dibahas pada setiap subbab di Chapter 5**:
"""))

# ==========================================
# CELL 2: OVERVIEW SUBBAB CHAPTER 5
# ==========================================
cells.append(nbf.v4.new_markdown_cell(r"""### 🗺️ Pemetaan Lengkap Subbab Chapter 5 Rasmussen & Williams (2006)

---

### 1. Subbab 5.1: *The Model Selection Problem* (hlm. 105–108)
* **Inti Pembahasan**:
  * Menjelaskan definisi formal bahwa parameter pada GP (seperti *lengthscale* $l$, *signal variance* $\sigma_f^2$, dan *noise variance* $\sigma_n^2$) bukanlah bobot model linier konvensional, melainkan **hyperparameter** dari proses stokastik.
  * Memperkenalkan **Hierarki Inferensi Bayesian Tiga Tingkat (3-Level Bayesian Inference Hierarchy)**:
    1. **Level 1 (Inferensi Fungsi Laten $\mathbf{f}$)**: Menghitung posterior $p(\mathbf{f} \mid \mathbf{y}, X, \boldsymbol{\theta})$ dengan asumsi hyperparameter $\boldsymbol{\theta}$ telah tetap/diketahui.
    2. **Level 2 (Adaptasi Hyperparameter $\boldsymbol{\theta}$)**: Menentukan hyperparameter terbaik $\boldsymbol{\theta}$ di bawah kelas kernel tertentu menggunakan *Marginal Likelihood* $p(\mathbf{y} \mid X, \boldsymbol{\theta})$.
    3. **Level 3 (Perbandingan Struktur Model $\mathcal{H}_i$)**: Membandingkan keluarga kernel yang berbeda (misal: *Squared Exponential* vs *Matérn*).
  * Menjelaskan mengapa optimasi langsung terhadap parameter tanpa marginalisasi berujung pada *severe overfitting*.

---

### 2. Subbab 5.2: *The Bayesian Framework* (hlm. 108–111)
* **Inti Pembahasan**:
  * Menguraikan pendekatan Bayesian penuh (*Full Bayesian Treatment*) di mana distribusi posterior hyperparameter diintegralkan:
    $$p(y_* \mid X, \mathbf{y}, \mathbf{x}_*) = \int p(y_* \mid X, \mathbf{y}, \mathbf{x}_*, \boldsymbol{\theta}) \, p(\boldsymbol{\theta} \mid \mathbf{y}, X) \, d\boldsymbol{\theta}$$
  * Karena integral di atas tidak memiliki solusi analitik tertutup (*intractable*), subbab ini membandingkan 3 strategi praktis:
    1. **Type II Maximum Likelihood (ML-II) / Empirical Bayes**: Mengoptimasi titik puncak Marginal Likelihood $\hat{\boldsymbol{\theta}} = \arg\max \log p(\mathbf{y} \mid X, \boldsymbol{\theta})$ (metode standar yang paling banyak digunakan).
    2. **Maximum A Posteriori (MAP)**: Memasukkan distribusi prior pada hyperparameter $p(\boldsymbol{\theta})$.
    3. **Markov Chain Monte Carlo (MCMC)**: Melakukan sampling numerik (misal: *Hybrid Monte Carlo* / HMC).

---

### 3. Subbab 5.3: *Cross-Validation* (hlm. 111–112)
* **Inti Pembahasan**:
  * Membahas alternatif non-Bayesian (metode validasi silang) untuk pemilihan model, khususnya *Leave-One-Out Cross-Validation* (LOO-CV) dan $k$-fold CV.
  * Mengungkapkan **keistimewaan analitik GP**: distribusi prediksi saat titik data ke-$i$ dikeluarkan ($X_{-i}, \mathbf{y}_{-i}$) dapat dihitung langsung dari invers matriks penuh $K_y^{-1}$ dalam waktu $\mathcal{O}(N^3)$ **tanpa perlu melatih ulang model sebanyak $N$ kali**:
    $$\mu_{-i} = y_i - \frac{[K_y^{-1}\mathbf{y}]_i}{[K_y^{-1}]_{ii}} = y_i - \frac{\alpha_i}{[K_y^{-1}]_{ii}}, \quad \sigma_{-i}^2 = \frac{1}{[K_y^{-1}]_{ii}}$$

---

### 4. Subbab 5.4: *Model Selection for GP Regression* (hlm. 112–124) — **[SUBBAB UTAMA / INTI]**
Subbab ini merupakan **jantung teknis komputasi optimasi GP Regresi**, yang terbagi lagi menjadi 3 bagian:
* **5.4.1 Marginal Likelihood Computation and its Derivatives**:
  * Penurunan analitik fungsi **Log Marginal Likelihood (LML)**:
    $$\mathcal{L}(\boldsymbol{\theta}) \triangleq \log p(\mathbf{y} \mid X, \boldsymbol{\theta}) = \underbrace{-\frac{1}{2} \mathbf{y}^T K_y^{-1} \mathbf{y}}_{\text{Data-fit Term}} \;\; \underbrace{-\frac{1}{2} \log |K_y|}_{\text{Complexity Penalty}} \;\; \underbrace{-\frac{N}{2} \log(2\pi)}_{\text{Normalizing Constant}}$$
  * Pembuktian prinsip **Occam's Razor Otomatis** (keseimbangan data-fit vs kompleksitas).
  * Penurunan **Gradien Analitik** (Persamaan 5.9 Rasmussen):
    $$\frac{\partial \mathcal{L}}{\partial \theta_j} = \frac{1}{2} \operatorname{tr}\left( \left(\boldsymbol{\alpha}\boldsymbol{\alpha}^T - K_y^{-1}\right) \frac{\partial K_y}{\partial \theta_j} \right), \quad \boldsymbol{\alpha} = K_y^{-1}\mathbf{y}$$
  * Formulasi **Algoritma 5.1**: Implementasi numerik efisien berbasis faktorisasi Cholesky $K_y = L L^T$ dengan kompleksitas $\mathcal{O}(N^3)$ satu kali per iterasi dan $\mathcal{O}(N^2)$ per evaluasi gradien hyperparameter.
  * Transformasi parameter ke ruang tak-terkendala $\tilde{\theta}_j = \log \theta_j$ untuk menjamin kepositifan parameter ($l, \sigma_f, \sigma_n > 0$).
* **5.4.2 Cross-validation Derivatives**:
  * Penurunan turunan analitik kriteria LOO-CV terhadap hyperparameter.
* **5.4.3 Examples and Discussion (Lanskap Objektif & Multimodalitas)**:
  * Analisis sifat permukaan LML yang non-konveks (*non-convex landscape*).
  * Fenomena **Multimodalitas** (keberadaan beberapa lokal optima, misal: *mode short lengthscale + low noise* vs *mode long lengthscale + high noise*).
  * Strategi *Multi-restart* / *Random Restarts* untuk menemukan optimum global.

---

### 5. Subbab 5.5: *Model Selection for GP Classification* (hlm. 124–128)
* **Inti Pembahasan**:
  * Membahas pemilihan model pada kasus klasifikasi (label diskret $y \in \{-1, +1\}$).
  * Karena fungsi likelihood bernilai non-Gaussian (sigmoid/probit), integral marginal likelihood tidak dapat diselesaikan secara analitik eksak, sehingga memerlukan metode aproksimasi analitik seperti *Laplace Approximation* atau *Expectation Propagation* (EP) yang dibahas pada Bab 3.
"""))

# ==========================================
# CELL 3: CODE DEMONSTRATION & VERIFICATION
# ==========================================
cells.append(nbf.v4.new_markdown_cell(r"""---
## 🧪 Demonstrasi & Pembuktian Komputasi Interaktif
Di bawah ini kita mengimplementasikan secara langsung algoritma dan persamaan dari **Chapter 5 Rasmussen & Williams (2006)** untuk memverifikasi seluruh teori di atas.
"""))

cells.append(nbf.v4.new_code_cell(r"""import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Konfigurasi plot
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 120

# 1. Pembangkitan Data Observasi Sintetis (Rasmussen Style)
N = 20
X_train = np.sort(np.random.uniform(-4, 4, N))[:, None]
true_func = lambda x: np.sin(1.5 * x) + 0.5 * np.cos(3.0 * x)
sigma_n_true = 0.20
y_train = true_func(X_train).ravel() + np.random.normal(0, sigma_n_true, N)

X_test = np.linspace(-5, 5, 250)[:, None]
y_true_test = true_func(X_test).ravel()

print(f"Data training terbentuk: N = {N} sampel.")
"""))

# ==========================================
# CELL 4: KERNEL, LML, AND GRADIENT IMPLEMENTATION
# ==========================================
cells.append(nbf.v4.new_code_cell('''# 2. Implementasi Algoritma 5.1 Rasmussen & Williams (2006)

def se_kernel(X1, X2, l, sigma_f):
    r2 = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=-1)
    return (sigma_f**2) * np.exp(-0.5 * r2 / (l**2))

def compute_lml_and_grad(log_theta, X, y):
    # Evaluasi Negative Log Marginal Likelihood (NLML) dan Gradien Analitik
    # menggunakan Dekomposisi Cholesky (Algoritma 5.1 Rasmussen).
    # Parameter log_theta = [log(l), log(sigma_f), log(sigma_n)]
    l = np.exp(log_theta[0])
    sigma_f = np.exp(log_theta[1])
    sigma_n = np.exp(log_theta[2])
    N = len(y)
    
    # Matriks kovariansi total Ky = K + sigma_n^2 * I
    K = se_kernel(X, X, l, sigma_f)
    Ky = K + (sigma_n**2 + 1e-7) * np.eye(N)
    
    # Faktorisasi Cholesky: Ky = L L^T
    L = np.linalg.cholesky(Ky)
    
    # Vektor bobot prediktif alpha = L^T \\ (L \\ y)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    
    # Komponen LML
    data_fit = -0.5 * np.dot(y, alpha)
    complexity = -np.sum(np.log(np.diag(L)))
    norm_const = -0.5 * N * np.log(2 * np.pi)
    lml = data_fit + complexity + norm_const
    nlml = -lml
    
    # Invers Ky via Cholesky
    Ky_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(N)))
    W = np.outer(alpha, alpha) - Ky_inv
    
    # Turunan parsial matriks kovariansi
    r2 = np.sum((X[:, None, :] - X[None, :, :])**2, axis=-1)
    dK_dl = K * (r2 / (l**3))
    dK_dsf = 2.0 * sigma_f * np.exp(-0.5 * r2 / (l**2))
    dKy_dsn = 2.0 * sigma_n * np.eye(N)
    
    # Gradien dengan aturan rantai log-transform (d/d log_theta = theta * d/d theta)
    grad_log_l = -0.5 * np.sum(W * dK_dl) * l
    grad_log_sf = -0.5 * np.sum(W * dK_dsf) * sigma_f
    grad_log_sn = -0.5 * np.sum(W * dKy_dsn) * sigma_n
    
    return nlml, np.array([grad_log_l, grad_log_sf, grad_log_sn])

print("Fungsi LML dan Gradien Analitik Algoritma 5.1 siap digunakan.")
'''))

# ==========================================
# CELL 5: EXPERIMENT OCCAM'S RAZOR
# ==========================================
cells.append(nbf.v4.new_code_cell(r"""# 3. Visualisasi Dekomposisi Occam's Razor (Gambar 5.1 Rasmussen)

l_grid = np.logspace(-1.2, 1.3, 150)
fixed_sf = 1.0
fixed_sn = 0.20

lmls, data_fits, complexities = [], [], []
for l in l_grid:
    K = se_kernel(X_train, X_train, l, fixed_sf)
    Ky = K + (fixed_sn**2 + 1e-7) * np.eye(N)
    L = np.linalg.cholesky(Ky)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    df = -0.5 * np.dot(y_train, alpha)
    comp = -np.sum(np.log(np.diag(L)))
    lml = df + comp - 0.5 * N * np.log(2 * np.pi)
    
    lmls.append(lml)
    data_fits.append(df)
    complexities.append(comp)

best_l = l_grid[np.argmax(lmls)]

plt.figure(figsize=(8.5, 4.8))
plt.plot(l_grid, data_fits, '--', color='#2ca02c', lw=2.2, label=r'Data-fit Term $\left(-\frac{1}{2}\mathbf{y}^T K_y^{-1}\mathbf{y}\right)$')
plt.plot(l_grid, complexities, '-.', color='#d62728', lw=2.2, label=r'Complexity Penalty $\left(-\frac{1}{2}\log|K_y|\right)$')
plt.plot(l_grid, lmls, color='#1f77b4', lw=2.8, label=r'Total Log Marginal Likelihood $\mathcal{L}(\theta)$')
plt.axvline(best_l, color='purple', linestyle=':', lw=1.8, label=f'Optimal $l^* = {best_l:.2f}$')
plt.plot(best_l, np.max(lmls), 'o', color='purple', markersize=8)

plt.xscale('log')
plt.xlabel('Lengthscale $l$ (skala log)')
plt.ylabel('Nilai Suku LML')
plt.title("Prinsip Occam's Razor Otomatis pada Marginal Likelihood (Rasmussen Bab 5.4.1)", pad=10)
plt.ylim(-65, 10)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
"""))

# ==========================================
# CELL 6: RUN OPTIMIZER
# ==========================================
cells.append(nbf.v4.new_code_cell(r"""# 4. Menjalankan Optimasi Gradien L-BFGS-B

init_params = np.array([1.0, 1.0, 0.5])
res = minimize(compute_lml_and_grad, np.log(init_params), args=(X_train, y_train), jac=True, method='L-BFGS-B')

opt_l, opt_sf, opt_sn = np.exp(res.x)
print("=" * 45)
print("HASIL OPTIMASI HYPERPARAMETER (RASMUSSEN BAB 5)")
print("=" * 45)
print(f"Lengthscale optimal (l*)       : {opt_l:.4f}")
print(f"Signal variance (sigma_f^2)    : {opt_sf**2:.4f} (sigma_f = {opt_sf:.4f})")
print(f"Noise standard deviation (sn)  : {opt_sn:.4f} (sigma_n^2 = {opt_sn**2:.4f})")
print(f"Log Marginal Likelihood Maks   : {-res.fun:.4f}")
print("=" * 45)
"""))

# ==========================================
# CELL 7: POSTERIOR COMPARISON
# ==========================================
cells.append(nbf.v4.new_code_cell(r"""# 5. Komparasi Prediksi Posterior (Underfitting vs Optimal vs Overfitting)

def gp_predict(X_tr, y_tr, X_te, l, sf, sn):
    N_tr = len(y_tr)
    K_tr = se_kernel(X_tr, X_tr, l, sf) + (sn**2 + 1e-7) * np.eye(N_tr)
    K_s = se_kernel(X_tr, X_te, l, sf)
    K_ss = se_kernel(X_te, X_te, l, sf) + 1e-7 * np.eye(len(X_te))
    
    L = np.linalg.cholesky(K_tr)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
    mu = np.dot(K_s.T, alpha)
    
    v = np.linalg.solve(L, K_s)
    cov = K_ss - np.dot(v.T, v)
    std = np.sqrt(np.maximum(np.diag(cov), 1e-8))
    return mu, std

cases = [
    {'title': 'Underfitting (l=4.50, sn=0.20)', 'l': 4.5, 'sf': 1.0, 'sn': 0.20, 'c': '#2ca02c'},
    {'title': f'Optimal LML (l={opt_l:.2f}, sn={opt_sn:.2f})', 'l': opt_l, 'sf': opt_sf, 'sn': opt_sn, 'c': '#1f77b4'},
    {'title': 'Overfitting (l=0.18, sn=0.03)', 'l': 0.18, 'sf': 1.0, 'sn': 0.03, 'c': '#d62728'}
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, case in zip(axes, cases):
    mu, std = gp_predict(X_train, y_train, X_test, case['l'], case['sf'], case['sn'])
    nlml, _ = compute_lml_and_grad(np.log([case['l'], case['sf'], case['sn']]), X_train, y_train)
    
    ax.plot(X_test, y_true_test, 'k--', alpha=0.5, label='Fungsi Sejati')
    ax.scatter(X_train, y_train, c='black', s=25, zorder=5, label='Data Observasi')
    ax.plot(X_test, mu, color=case['c'], lw=2.2, label='Mean Prediktif $\\bar{f}_*$')
    ax.fill_between(X_test.ravel(), mu - 2*std, mu + 2*std, color=case['c'], alpha=0.22, label='Pita $\pm 2\sigma$ ($95\%$)')
    
    ax.set_title(f"{case['title']}\nTotal LML = {-nlml:.2f}", fontsize=10.5)
    ax.set_xlabel('$x$')
    if ax == axes[0]:
        ax.set_ylabel('$y = f(x)$')
        ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(-2.5, 2.5)

plt.suptitle("Perbandingan Hasil Model Fit di Bawah Nilai Hyperparameter Berbeda (Rasmussen Bab 5.4)", y=1.02, fontsize=12)
plt.tight_layout()
plt.show()
"""))

# ==========================================
# CELL 8: LOO-CV ANALYTIC
# ==========================================
cells.append(nbf.v4.new_code_cell(r"""# 6. Verifikasi Rumus Analitik Leave-One-Out CV (Rasmussen Bab 5.3 & 5.4.2)

# Evaluasi pada parameter optimal
K_opt = se_kernel(X_train, X_train, opt_l, opt_sf) + (opt_sn**2 + 1e-7) * np.eye(N)
L_opt = np.linalg.cholesky(K_opt)
Ky_inv_opt = np.linalg.solve(L_opt.T, np.linalg.solve(L_opt, np.eye(N)))
alpha_opt = np.linalg.solve(L_opt.T, np.linalg.solve(L_opt, y_train))

# Formulasi analitik LOO tertutup Rasmussen (Persamaan 5.12 & 5.13):
mu_loo = y_train - (alpha_opt / np.diag(Ky_inv_opt))
sigma2_loo = 1.0 / np.diag(Ky_inv_opt)

# Menghitung Log Pseudo-Likelihood (LPL)
lpl = -0.5 * np.sum(np.log(2 * np.pi * sigma2_loo) + ((y_train - mu_loo)**2 / sigma2_loo))
print(f"Log Pseudo-Likelihood (LOO-CV) pada parameter optimal: {lpl:.4f}")

plt.figure(figsize=(6.5, 4.2))
plt.errorbar(y_train, mu_loo, yerr=2*np.sqrt(sigma2_loo), fmt='o', color='purple', ecolor='gray', elinewidth=1.5, capsize=3, label='Prediksi LOO $\mu_{-i} \pm 2\sigma_{-i}$')
plt.plot([-2, 2], [-2, 2], 'k--', label='Garis Ideal $y_i = \mu_{-i}$')
plt.xlabel('Target Sejati $y_i$')
plt.ylabel('Prediksi LOO $\mu_{-i}$')
plt.title('Validasi Model: Prediksi Analitik LOO-CV (Rasmussen Bab 5.3)')
plt.legend()
plt.tight_layout()
plt.show()
"""))

nb['cells'] = cells

target_file = r"04_dokumen_ta/Dokumen bimbingan/Bimbingan 04/coret.ipynb"
with open(target_file, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"File {target_file} successfully generated!")
