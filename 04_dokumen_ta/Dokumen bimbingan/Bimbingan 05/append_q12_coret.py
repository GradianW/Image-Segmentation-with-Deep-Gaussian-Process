import nbformat
from nbformat.v4 import new_markdown_cell

nb_path = r"c:\Users\MyBook Z Series\Desktop\TA\04_dokumen_ta\Dokumen bimbingan\Bimbingan 05\coret.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Remove any existing Pertanyaan 12
cells = []
for cell in nb.cells:
    if cell.cell_type == "markdown" and ("Pertanyaan 12" in cell.source or "Jawaban 12" in cell.source):
        continue
    cells.append(cell)

q12_content = """# ❓ Pertanyaan 12: Perbedaan `scipy.optimize.minimize` vs Package Khusus Gaussian Process dalam Optimasi Hiperparameter

> **Kemaren saya memakai scipy optimize, minimize, untuk mengoptimasi hyperparameter. Apakah tidak termasuk di list yang kamu buat? Apakah package-package pada list yang kamu buat sudah termasuk melakukan optimasi hyperparameter? Apa bedanya?**

---

## 💡 Jawaban 12: Klasifikasi Peran `scipy.optimize` vs Dedicated GP Frameworks

Pertanyaan ini sangat fundamental dalam memahami hierarki arsitektur perangkat lunak untuk Gaussian Process. Secara ringkas:

1. **`scipy.optimize.minimize` adalah General-Purpose Numerical Optimizer (Pengoptimal Numerik Umum)**, bukan pustaka pemodelan Gaussian Process.
2. **Package pada daftar sebelumnya (GPyTorch, GPflow, Scikit-Learn, GPy) SUDAH MENCAKUP optimasi hiperparameter secara otomatis (*end-to-end*)**.
3. **Bahkan di balik layar, beberapa package GP (seperti Scikit-Learn dan GPflow) sebenarnya menggunakan algoritma `scipy.optimize.minimize` (L-BFGS-B)** yang telah dibungkus (*wrapper*) secara otomatis.

---

### 1. Hubungan Hirarki dan Peran

```
+-------------------------------------------------------------------------------+
|                       DEDICATED GP FRAMEWORKS                                 |
|            (GPyTorch, GPflow, Scikit-Learn GPR, GPy, GPJax)                   |
+-------------------------------------------------------------------------------+
| 1. Manajemen Struktur Kernel (RBF, Periodic, Linear, Komposisi +, *)          |
| 2. Perhitungan Matriks Kovariansi K_y dan Invers / Cholesky L L^T             |
| 3. Formulasi Objektif Marginal Log Likelihood (MLL / ELBO)                   |
| 4. Otomasi Batasan Parameter (Constraint sigma_f > 0, l > 0 via Softplus/Exp)|
| 5. Perhitungan Gradien Otomatis (Autograd PyTorch/TF/JAX atau Rumus Trace)   |
| 6. Modul Prediksi Posterior: Mean mu_* dan Standar Deviasi sigma_*           |
+-------------------------------------------------------------------------------+
                                      |
                                      v  (Memanfaatkan Engine Pengoptimal)
+-------------------------------------------------------------------------------+
|                       NUMERICAL OPTIMIZATION ENGINES                          |
|         - scipy.optimize.minimize (L-BFGS-B, CG, SLSQP, Nelder-Mead)          |
|         - torch.optim (Adam, SGD, L-BFGS di GPU)                              |
|         - tf.optimizers (Adam, SGD di TPU/GPU)                                |
|         - optax (Adam di JAX/XLA)                                             |
+-------------------------------------------------------------------------------+
```

---

### 2. Tabel Komparasi: `scipy.optimize.minimize` (Manual) vs Package Khusus GP

| Aspek | Menggunakan `scipy.optimize.minimize` (Manual / Scratch) | Menggunakan Dedicated GP Package (GPyTorch / GPflow / Sklearn) |
| :--- | :--- | :--- |
| **Kategori Pustaka** | Solver optimasi matematika umum (*General Numerical Solver*). | Kerangka kerja pemodelan probabilistik terpadu (*Dedicated GP Framework*). |
| **Penyusunan Fungsi Objektif** | **Manual:** Pengguna harus merumuskan sendiri fungsi NLL $\\mathcal{L}(\\boldsymbol{\\theta}) = \\frac{1}{2}\\mathbf{y}^T K_y^{-1}\\mathbf{y} + \\frac{1}{2}\\log\\|K_y\\| + \\frac{n}{2}\\log 2\\pi$. | **Otomatis:** Disediakan langsung oleh kelas `ExactMarginalLogLikelihood` / `log_marginal_likelihood()`. |
| **Perhitungan Gradien** | **Manual:** Pengguna harus menurunkan dan mengodekan rumus trace $(\\boldsymbol{\\alpha}\\boldsymbol{\\alpha}^T - K_y^{-1})\\frac{\\partial K_y}{\\partial \\boldsymbol{\\theta}}$, atau mengandalkan estimasi numerik (*finite difference*) yang lambat. | **Otomatis:** Dihitung otomatis melalui mesin *Automatic Differentiation* (PyTorch Autograd / TensorFlow GradientTape) yang sangat presisi dan cepat. |
| **Batasan Positivitas Parameter ($\\boldsymbol{\\theta} > 0$)** | Harus diatur manual via argumen `bounds=((1e-5, None), ...)` atau melakukan re-parameterisasi $\\theta = \\exp(\\tilde{\\theta})$. | Dikelola otomatis melalui modul constraint bawaan (misal transformasi `Positive()`, `Softplus`, atau `Interval`). |
| **Eksekusi Komputasi Hardware** | Terbatas pada **CPU** (berbasis array NumPy/C). | Mendukung akselerasi penuh pada **GPU / CUDA** dan multi-GPU (GPyTorch, GPflow, Pyro, GPJax). |
| **Prediksi Posterior ($\\mu_*, \\sigma_*$)** | Harus ditulis manual (menghitung $K_*$, $K_{**}$, solve linear system untuk mean & variansi). | Cukup memanggil fungsi bawaan `model(X_test)` atau `model.predict(X_test)`. |
| **Skalabilitas Data ($N$)** | Sangat terbatas ($N < 2.000$) karena beban memori dan komputasi Cholesky $\\mathcal{O}(N^3)$ di CPU. | Mampu menangani hingga puluhan ribu/jutaan data melalui *Sparse Gaussian Process* (SVGP) dan akselerasi GPU Krylov. |
| **Ekstensi ke Deep GP** | Sangat sulit dan tidak realistis dibangun dari nol (memerlukan layer-by-layer variational sampling). | Sudah tersedia arsitektur siap pakai (`gpytorch.models.deep_gps.DeepGP`). |

---

### 3. Di Balik Layar: Keterkaitan Scikit-Learn & GPflow dengan `scipy.optimize`

Faktanya, pustaka GP tingkat tinggi tertentu menggunakan `scipy.optimize.minimize` sebagai mesin internalnya:

1. **Scikit-Learn (`GaussianProcessRegressor`)**:
   ```python
   # Di balik layar fungsi model.fit(X, y), scikit-learn mengeksekusi:
   from scipy.optimize import minimize
   res = minimize(self.log_marginal_likelihood, initial_theta, method="L-BFGS-B", jac=True)
   ```
2. **GPflow (`gpflow.optimizers.Scipy`)**:
   ```python
   # GPflow membungkus scipy minimize agar kompatibel dengan variabel TensorFlow:
   opt = gpflow.optimizers.Scipy()
   opt.minimize(model.training_loss, model.trainable_variables, method="L-BFGS-B")
   ```
3. **GPyTorch (`torch.optim`)**:
   Berbeda dari Scikit-Learn, GPyTorch tidak memakai SciPy karena SciPy berjalan di CPU. GPyTorch menggunakan optimizer berbasis PyTorch (seperti `torch.optim.Adam` atau `torch.optim.LBFGS`) agar seluruh perhitungan matriks dan gradien tetap berada di dalam VRAM GPU tanpa *overhead* transfer data ke RAM CPU.

---

### 4. Kesimpulan

* Menulis optimasi dengan **`scipy.optimize.minimize` dari nol** (seperti yang kita lakukan pada Section 1 dan script mandiri) sangat berguna untuk **pemahaman teoritis dan validasi numerik bimbingan** (memahami cara kerja gradien log marginal likelihood secara matematis).
* Untuk **implementasi nyata pada Tugas Akhir (terutama Deep Gaussian Process untuk segmentasi citra)**, kita beralih menggunakan **package khusus GP berbasis PyTorch (GPyTorch)** karena memerlukan akselerasi GPU, diferensiasi otomatis multi-layer, dan skalabilitas data piksel yang besar.
"""

cells.append(new_markdown_cell(q12_content))
nb.cells = cells

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Successfully appended Pertanyaan 12 to coret.ipynb")
