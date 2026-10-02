# Rencana Penyusunan Dokumen Kredit Bimbingan #4
**Topik Utama**: Optimasi Parameter (Hyperparameter Adaptation) pada Gaussian Process Berdasarkan Rasmussen & Williams (2006)  
**Target Pelaksanaan Bimbingan**: Sesi Bimbingan #4  
**Peruntukan Dokumen**: Draf Bab Landasan Teori Laporan Tugas Akhir / Skripsi  
**Referensi Utama**: Rasmussen, C. E., \& Williams, C. K. (2006). *Gaussian Processes for Machine Learning*. MIT Press, Cambridge, MA (Khususnya Bab 5: *Model Selection and Adaptation of Hyperparameters*).

---

## 🎯 Fokus & Ruang Lingkup Dokumen Kredit #4

Dokumen ini disusun secara terarah, mendalam, dan terfokus secara eksklusif pada **metodologi dan algoritma optimasi hyperparameter Gaussian Process** sesuai Bab 5 Rasmussen & Williams (2006):

1. **Hierarki Inferensi dan Masalah Pemilihan Model (*Model Selection*)**:
   - Tiga level hierarki inferensi Bayesian: Level 1 (Inferensi fungsi laten $\mathbf{f}$), Level 2 (Adaptasi hyperparameter $\boldsymbol{\theta}$), dan Level 3 (Perbandingan struktur kernel $\mathcal{H}_i$).
   - Mengapa optimasi langsung terhadap posterior fungsi laten tanpa marginalisasi berujung pada *severe overfitting*.

2. **Marginal Likelihood (Evidence)**:
   - Penurunan analitik perumusan marginal likelihood melalui pengintegralan fungsi laten $\mathbf{f}$:
     $$p(\mathbf{y} \mid X, \boldsymbol{\theta}) = \int p(\mathbf{y} \mid \mathbf{f}) p(\mathbf{f} \mid X, \boldsymbol{\theta}) d\mathbf{f}$$
   - Formulasi log marginal likelihood (LML):
     $$\log p(\mathbf{y} \mid X, \boldsymbol{\theta}) = -\frac{1}{2}\mathbf{y}^T K_y^{-1}\mathbf{y} - \frac{1}{2}\log|K_y| - \frac{N}{2}\log(2\pi)$$

3. **Dekomposisi Suku-Suku LML & Prinsip Occam's Razor Otomatis**:
   - **Data-fit term** ($-\frac{1}{2}\mathbf{y}^T K_y^{-1}\mathbf{y}$): Mengukur kesesuaian model dengan data.
   - **Complexity penalty term** ($-\frac{1}{2}\log|K_y|$): Mengukur dan membatasi volume/kelenturan ruang fungsi yang dicakup prior.
   - **Konstanta Normalisasi** ($-\frac{N}{2}\log(2\pi)$).
   - Analisis trade-off otomatis yang mencegah *overfitting* tanpa membutuhkan dataset validasi terpisah.

4. **Penurunan Gradien Analitik Log Marginal Likelihood**:
   - Penurunan parsial terhadap masing-masing hyperparameter $\theta_j$:
     $$\frac{\partial \log p(\mathbf{y} \mid X, \boldsymbol{\theta})}{\partial \theta_j} = \frac{1}{2}\operatorname{tr}\left( \left(\boldsymbol{\alpha}\boldsymbol{\alpha}^T - K_y^{-1}\right) \frac{\partial K_y}{\partial \theta_j} \right)$$
     dengan $\boldsymbol{\alpha} = K_y^{-1}\mathbf{y}$.
   - Evaluasi turunan parsial kernel baku (Squared Exponential terhadap lengthscale $l$, signal variance $\sigma_f^2$, dan noise variance $\sigma_n^2$).

5. **Algoritma Komputasi Numerik Efisien via Dekomposisi Cholesky (Algoritma 5.1)**:
   - Struktur komputasi $\mathcal{O}(N^3)$ untuk faktorisasi Cholesky $K_y = L L^T$ dan $\mathcal{O}(N^2)$ per evaluasi gradien hyperparameter.
   - Transformasi tak-terkendala (*unconstrained log-parameterization*) $\tilde{\theta}_j = \log \theta_j$ untuk menjamin kepositifan hyperparameter ($l, \sigma_f^2, \sigma_n^2 > 0$).
   - Implementasi optimasi gradien (L-BFGS-B / Conjugate Gradient).

6. **Fenomena Non-Konveksitas, Multimodalitas, dan Strategi Multistart**:
   - Karakteristik permukaan objektif LML yang non-konveks dengan kemungkinan banyak lokal optima (misal: mode noise rendah fluktuasi cepat vs mode noise tinggi fluktuasi lambat).
   - Solusi *random restart* / inisialisasi ganda (*multistart optimization*).

7. **Metode Alternatif: Leave-One-Out Cross-Validation (LOO-CV)**:
   - Formulasi analitik LOO prediktif GP tanpa melatih ulang model sebanyak $N$ kali ($\mathcal{O}(N^3)$ total).
   - Komparasi LOO-CV vs Marginal Likelihood.

---

## 📐 Konvensi Notasi Aljabar Linier

| Notasi | Ruang Dimensi | Tipe Entitas | Keterangan Matematis |
|---|---|---|---|
| $N$ | $\mathbb{N}$ | Skalar | Jumlah sampel observasi training |
| $D$ | $\mathbb{N}$ | Skalar | Dimensi domain input |
| $X$ | $\mathbb{R}^{N \times D}$ | Matriks | Matriks input observasi training |
| $\mathbf{y}$ | $\mathbb{R}^{N \times 1}$ | Vektor Kolom | Vektor target observasi training |
| $\boldsymbol{\theta}$ | $\mathbb{R}^{P \times 1}$ | Vektor Kolom | Vektor hyperparameter kernel dan noise ($l, \sigma_f^2, \sigma_n^2$) |
| $\tilde{\boldsymbol{\theta}}$ | $\mathbb{R}^{P \times 1}$ | Vektor Kolom | Vektor hyperparameter dalam skala logaritmik, $\tilde{\theta}_j = \log \theta_j$ |
| $K$ | $\mathbb{R}^{N \times N}$ | Matriks Simetris PSD | Matriks kovariansi kernel murni, $[K]_{pq} = k(\mathbf{x}_p, \mathbf{x}_q)$ |
| $K_y$ | $\mathbb{R}^{N \times N}$ | Matriks Simetris PD | Matriks kovariansi total observasi, $K_y = K + \sigma_n^2 I_N$ |
| $L$ | $\mathbb{R}^{N \times N}$ | Matriks Segitiga Bawah | Faktor dekomposisi Cholesky, $K_y = L L^T$ |
| $\boldsymbol{\alpha}$ | $\mathbb{R}^{N \times 1}$ | Vektor Kolom | Vektor bobot prediktif, $\boldsymbol{\alpha} = K_y^{-1} \mathbf{y} = L^{-T}(L^{-1}\mathbf{y})$ |
| $W$ | $\mathbb{R}^{N \times N}$ | Matriks Simetris | Matriks bobot gradien, $W = \boldsymbol{\alpha}\boldsymbol{\alpha}^T - K_y^{-1}$ |
| $\mathcal{L}(\boldsymbol{\theta})$ | $\mathbb{R}$ | Skalar | Log Marginal Likelihood, $\log p(\mathbf{y} \mid X, \boldsymbol{\theta})$ |
| $\operatorname{NLML}(\boldsymbol{\theta})$ | $\mathbb{R}$ | Skalar | Negative Log Marginal Likelihood, $-\mathcal{L}(\boldsymbol{\theta})$ |
