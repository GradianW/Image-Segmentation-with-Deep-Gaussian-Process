import nbformat
from nbformat.v4 import new_markdown_cell

nb_path = r"c:\Users\MyBook Z Series\Desktop\TA\04_dokumen_ta\Dokumen bimbingan\Bimbingan 05\coret.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Remove any existing Pertanyaan 11
cells = []
for cell in nb.cells:
    if cell.cell_type == "markdown" and ("Pertanyaan 11" in cell.source or "Jawaban 11" in cell.source):
        continue
    cells.append(cell)

q11_content = """# ❓ Pertanyaan 11: Daftar Package Python dan Fungsi-Fungsinya untuk Gaussian Process (GP) & Deep Gaussian Process (DGP)

> **Saya ingin lanjut mengerjakan section 3. Saya ingin menambahkan package2 beserta fungsi2nya yang mungkin akan untuk GP. Tolong list dulu di coret.ipynb. Nanti saya akan memilih mana yang akan dimasukkan ke section 3.**

---

## 💡 Jawaban 11: Katalog Package Python untuk Gaussian Processes

Berikut adalah kurasi dan klasifikasi lengkap mengenai pustaka (*packages/libraries*) Python yang umum dan standar digunakan untuk komputasi **Gaussian Process Regression/Classification**, **Sparse GP**, hingga **Deep Gaussian Process (DGP)**, beserta modul, kelas, dan fungsi-fungsi utamanya.

---

### 1. Ringkasan & Taksonomi Package GP

| No | Package | *Backend Framework* | Dukungan GPU | Skalabilitas & Fitur Utama | Tingkat Relevansi untuk TA (DGP / Image) |
|:---:|:---|:---|:---:|:---|:---:|
| **1** | **GPyTorch** | PyTorch | ✅ Penuh (CUDA) | Exact GP, Sparse/Variational GP, Deep GP, Kernel Interfacing, Linear Operator | ⭐⭐⭐⭐⭐ (Sangat Tinggi / Utama) |
| **2** | **GPflow** | TensorFlow 2 / TFP | ✅ Penuh (GPU/TPU) | Modern Variational GP (SVGP), MCMC, Interoperabilitas TensorFlow | ⭐⭐⭐⭐ (Tinggi) |
| **3** | **GPy** | NumPy / SciPy / Cython | ❌ (CPU Only) | Standar klasik Sheffield ML, implementasi algoritma murni, fondasi DeepGP | ⭐⭐⭐ (Menengah / Referensi) |
| **4** | **Scikit-Learn** | NumPy / SciPy | ❌ (CPU Only) | Implementasi standar sederhana, prototyping cepat, baseline GPR/GPC | ⭐⭐⭐ (Dasar / Baseline) |
| **5** | **Pyro (`pyro.contrib.gp`)** | PyTorch | ✅ Penuh (CUDA) | Probabilistic Programming, Bayesian Deep Learning, Variational Inference | ⭐⭐⭐⭐ (Tinggi) |
| **6** | **GPJax** | JAX | ✅ Penuh (GPU/TPU) | Auto-differentiation (`jax.grad`), JIT Compilation (`jax.jit`), Vmap | ⭐⭐⭐ (Modern / Riset Alternatif) |

---

### 2. Rincian Modul dan Fungsi Utama Tiap Package

---

#### 📦 A. GPyTorch (`gpytorch`) — *Rekomendasi Utama untuk Deep GP & PyTorch*
Pustaka mutakhir berbasis PyTorch yang dirancang untuk komputasi GP berkecepatan tinggi dengan akselerasi GPU, dekomposisi matriks Krylov (BBMM / KeOps), dan mendukung arsitektur Deep Gaussian Process berlapis (*stacked layers*).

1. **Modul Model (`gpytorch.models`)**:
   - `gpytorch.models.ExactGP`: Kelas dasar untuk regresi GP standar dengan observasi Gaussian (menggunakan dekomposisi Cholesky / Lanczos).
   - `gpytorch.models.ApproximateGP`: Kelas dasar untuk *Variational Gaussian Process* (VGP/SVGP) pada data skala besar atau data non-Gaussian.
   - `gpytorch.models.deep_gps.DeepGP`: Kelas khusus untuk menyusun arsitektur *Deep Gaussian Process* multi-layer.

2. **Modul Mean Functions (`gpytorch.means`)**:
   - `ConstantMean()`: $\\mu(x) = c$ (prior mean konstan).
   - `LinearMean(input_size)`: $\\mu(x) = \\mathbf{w}^T \\mathbf{x} + b$ (prior mean linear).
   - `ZeroMean()`: $\\mu(x) = 0$.

3. **Modul Kernel / Covariance Functions (`gpytorch.kernels`)**:
   - `RBFKernel(ard_num_dims)`: Kernel Radial Basis Function / Gaussian.
   - `PeriodicKernel(period_length)`: Kernel periodik untuk data bersiklus.
   - `LinearKernel()`: Kernel linear untuk pemodelan tren garis lurus.
   - `MaternKernel(nu=1.5 / 2.5)`: Kernel Matérn untuk fungsi yang kurang halus (*rougher paths*).
   - `SpectralMixtureKernel(num_mixtures)`: Kernel kombinasi spektral untuk pola frekuensi kompleks.
   - `ScaleKernel(base_kernel)`: Menambahkan faktor skala output $\\sigma_f^2$ di depan kernel dasar: $\\sigma_f^2 \\cdot k(\\mathbf{x}, \\mathbf{x}')$.
   - `AdditiveKernel(k1, k2)` / Operator `k1 + k2`: Operasi penjumlahan kernel.
   - `ProductKernel(k1, k2)` / Operator `k1 * k2`: Operasi perkalian kernel.
   - `GridKernel` / `GridInterpolationKernel` (KISS-GP): Akselerasi GP terstruktur untuk citra/grid 2D.

4. **Modul Likelihood (`gpytorch.likelihoods`)**:
   - `GaussianLikelihood()`: Likelihood Gaussian standar $y = f(x) + \\varepsilon, \\; \\varepsilon \\sim \\mathcal{N}(0, \\sigma_n^2)$.
   - `BernoulliLikelihood()`: Likelihood untuk klasifikasi biner.
   - `SoftmaxLikelihood(num_classes)`: Likelihood untuk klasifikasi multi-kelas (relevan untuk segmentasi citra).

5. **Modul Loss & Marginal Likelihood (`gpytorch.mlls`)**:
   - `ExactMarginalLogLikelihood(likelihood, model)`: Menghitung $\\log p(\\mathbf{y}|X, \\boldsymbol{\\theta})$ eksak.
   - `VariationalELBO(likelihood, model, num_data)`: Menghitung *Evidence Lower Bound* (ELBO) pada Sparse/Variational GP.
   - `DeepApproximateMLL(mll)`: Menghitung marginal log likelihood objektif untuk Deep GP.

6. **Modul Variational Distribution & Strategy (`gpytorch.variational`)**:
   - `CholeskyVariationalDistribution(num_inducing_points)`: Distribusi aproksimasi variasional $q(\\mathbf{u})$.
   - `VariationalStrategy(model, inducing_points, ...)`: Strategi penempatan *inducing points* $\\mathbf{u}$.

---

#### 📦 B. GPflow (`gpflow`) — *Standar Industri Berbasis TensorFlow*
Pustaka open-source buatan Cambridge/Imperial College berbasis TensorFlow yang mengimplementasikan metode aproksimasi variasional modern (Hensman et al.).

1. **Modul Model (`gpflow.models`)**:
   - `gpflow.models.GPR(data=(X, y), kernel=k, mean_function=m)`: Gaussian Process Regression eksak.
   - `gpflow.models.SGPR(data, kernel, inducing_variable)`: Sparse Gaussian Process Regression (menggunakan subset *inducing points*).
   - `gpflow.models.SVGP(kernel, likelihood, inducing_variable)`: *Sparse Variational GP* untuk big data dan klasifikasi non-Gaussian dengan optimasi mini-batch.
   - `gpflow.models.VGP(data, kernel, likelihood)`: Variational GP tanpa *inducing points*.

2. **Modul Kernel (`gpflow.kernels`)**:
   - `gpflow.kernels.SquaredExponential(variance, lengthscales)`: Kernel RBF.
   - `gpflow.kernels.Periodic(base_kernel, period)`: Kernel Periodik.
   - `gpflow.kernels.Linear(variance)`: Kernel Linear.
   - `gpflow.kernels.Matern32()`, `gpflow.kernels.Matern52()`.
   - `gpflow.kernels.White(variance)`: Kernel noise $\\sigma_n^2 I$.
   - Operator `k1 + k2` (`Sum`) dan `k1 * k2` (`Product`).

3. **Modul Likelihood (`gpflow.likelihoods`)**:
   - `gpflow.likelihoods.Gaussian(variance)`
   - `gpflow.likelihoods.Bernoulli()`, `gpflow.likelihoods.MultiClass(num_classes)`
   - `gpflow.likelihoods.Robustmax(num_classes)`

4. **Modul Optimasi (`gpflow.optimizers`)**:
   - `gpflow.optimizers.Scipy()`: Membungkus L-BFGS-B dari SciPy untuk optimasi parameter deterministik.
   - `tf.optimizers.Adam(learning_rate)`: Penggunaan optimizer Adam bawaan TensorFlow untuk mini-batch SVGP.

---

#### 📦 C. GPy (`GPy`) — *Pustaka Fondasi & Referensi Teoretis Sheffield ML*
Dikembangkan oleh Sheffield Machine Learning Group (Neil D. Lawrence), merupakan salah satu pustaka GP paling lengkap secara teori statistik, namun berbasis CPU (NumPy).

1. **Model Utama (`GPy.models`)**:
   - `GPy.models.GPRegression(X, Y, kernel)`: Regresi GP standar.
   - `GPy.models.SparseGPRegression(X, Y, Z, kernel)`: Regresi GP sparse dengan inducing inputs $Z$.
   - `GPy.models.GPClassification(X, Y, kernel)`: Klasifikasi dengan Laplace / EP.
   - `GPy.models.BayesianGPLVM(Y, input_dim, ...)`: *Gaussian Process Latent Variable Model*.

2. **Kernel (`GPy.kern`)**:
   - `GPy.kern.RBF(input_dim, variance, lengthscale)`
   - `GPy.kern.StdPeriodic(input_dim, variance, period, lengthscale)`
   - `GPy.kern.Linear(input_dim, variances)`
   - `GPy.kern.Add([k1, k2])`, `GPy.kern.Prod([k1, k2])`

3. **Metode Optimasi**:
   - `model.optimize(optimizer='lbfgs' / 'scg', max_iters=1000)`: Optimasi parameter marginal likelihood bawaan.
   - `model.plot()`: Visualisasi kurva mean, interval kepercayaan, dan titik data secara instan.

---

#### 📦 D. Scikit-Learn (`sklearn.gaussian_process`) — *Baseline & Pustaka Ringan*
Modul GP bawaan dari library machine learning terpopuler di Python, cocok untuk eksperimen perbandingan (*baseline comparison*).

1. **Kelas Estimator**:
   - `GaussianProcessRegressor(kernel=..., alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5)`:
     - Fungsi `.fit(X, y)`: Melakukan pelatihan / optimasi hiperparameter kernel via L-BFGS.
     - Fungsi `.predict(X_test, return_std=True / return_cov=True)`: Menghasilkan prediksi posterior mean $\\mu_*$ dan standar deviasi $\\sigma_*$.
     - Fungsi `.log_marginal_likelihood(theta)`: Menghitung nilai objektif LML.
   - `GaussianProcessClassifier(kernel=..., optimizer='fmin_l_bfgs_b')`: Klasifikasi biner/multikelas berbasis aproksimasi Laplace.

2. **Kernel (`sklearn.gaussian_process.kernels`)**:
   - `RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))`
   - `ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds=...)`: Kernel Periodik.
   - `DotProduct(sigma_0=1.0)`: Kernel Linear homogen / non-homogen.
   - `Matern(length_scale=1.0, nu=1.5)`
   - `WhiteKernel(noise_level=1.0)`: Estimasi noise $\\sigma_n^2$.
   - `ConstantKernel(constant_value=1.0)`: Faktor skala variansi output $\\sigma_f^2$.
   - Operator `+` (Sum) dan `*` (Product).

---

#### 📦 E. Pyro (`pyro.contrib.gp`) — *Probabilistic Programming & Bayesian Deep GP*
Ekstensi Gaussian Process pada bahasa pemrograman probabilistik Pyro (berbasis PyTorch), sangat cocok jika ingin menggabungkan GP dengan *Variational Autoencoders* (VAE) atau arsitektur Bayesian Deep Learning untuk citra.

1. **Model (`pyro.contrib.gp.models`)**:
   - `GPRegression(X, y, kernel, jitter=1e-6)`
   - `VariationalGP(X, y, kernel, likelihood, ...)`
   - `VariationalSparseGP(X, y, kernel, Xu, likelihood, ...)`

2. **Likelihood & Inferensi**:
   - Likelihood: `Gaussian`, `Binary`, `MultiClass`, `Poisson`
   - Inferensi: `pyro.infer.SVI` (*Stochastic Variational Inference*) dengan `pyro.infer.Trace_ELBO()`.

---

#### 📦 F. GPJax (`gpjax`) — *Framework Generasi Baru Berbasis JAX*
Pustaka fungsional modern yang memanfaatkan ekosistem Google JAX untuk kompilasi XLA berkecepatan tinggi dan diferensiasi otomatis tingkat lanjut.

1. **Objek & Fungsi Utama**:
   - `gpjax.gps.Prior(mean_function, kernel)`: Mendefinisikan prior Gaussian Process.
   - `prior * likelihood`: Mengkonstruksi posterior GP.
   - `gpjax.objectives.ConjugateMLL` / `NonConjugateMLL`: Menghitung marginal log-likelihood.
   - `gpjax.fit(model, objective, train_data, optax_optimizer)`: Pelatihan dengan library optimasi `optax` (Adam, SGD, dll.).
   - `gpjax.kernels`: `RBF`, `Periodic`, `Linear`, `Matern32`, `Matern52`.

---

### 3. Rekomendasi Pemilihan Package untuk Section 3 Dokumen Bimbingan

Untuk kebutuhan **Section 3** pada dokumen [kredit_bimbingan_05.tex](file:///c:/Users/MyBook%20Z%20Series/Desktop/TA/04_dokumen_ta/Dokumen%20bimbingan/Bimbingan%2005/kredit_bimbingan_05.tex), berikut adalah beberapa opsi penyusunan yang dapat Anda pilih:

1. **Opsi 1 (Fokus Komparasi Ekosistem & Pustaka Utama — Paling Seimbang)**:
   - Memilih **3 Package Utama**:
     1. **GPyTorch** (Pustaka Modern PyTorch / GPU / Deep GP).
     2. **GPflow** (Pustaka Modern TensorFlow / Variational GP).
     3. **Scikit-Learn** (Pustaka Standar Baseline CPU).
   - *Keunggulan*: Memberikan gambaran komparatif yang sangat representatif bagi dosen pembimbing mengenai ekosistem GP di Python (PyTorch vs TensorFlow vs Scikit-Learn).

2. **Opsi 2 (Fokus Khusus Arsitektur Tugas Akhir — Deep GP & PyTorch Stack)**:
   - Memilih **GPyTorch** dan **Pyro GP**, disertai rincian modul-modul spesifik untuk Deep Gaussian Process (`gpytorch.models.deep_gps`, `VariationalStrategy`, `GridInterpolationKernel`).
   - *Keunggulan*: Langsung tertuju dan terarah pada implementasi Tugas Akhir (*Image Segmentation with Deep Gaussian Process*).

3. **Opsi 3 (Katalog Lengkap Komprehensif)**:
   - Memasukkan tabel perbandingan semua package (GPyTorch, GPflow, GPy, Scikit-Learn, Pyro, GPJax), kemudian merinci hierarki modul dan fungsi-fungsi penting pada package yang dipilih untuk implementasi TA.
"""

cells.append(new_markdown_cell(q11_content))
nb.cells = cells

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Successfully appended Pertanyaan 11 to coret.ipynb")
