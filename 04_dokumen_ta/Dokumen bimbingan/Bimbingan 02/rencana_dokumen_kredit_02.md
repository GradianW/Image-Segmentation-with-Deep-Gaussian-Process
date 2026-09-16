# Rencana Penyusunan Dokumen Kredit Bimbingan #2 (Format LaTeX)
**Topik Utama**: Fondasi Matematis, Penurunan Lengkap, Ilustrasi Visual, dan Contoh Perhitungan Step-by-Step *Gaussian Process Regression* (GPR)  
**Target Pelaksanaan Bimbingan**: Rabu, 16 September 2026  

---

## 🎯 Tujuan Dokumen Kredit
1. **Menjawab Evaluasi Pertemuan #1**: Memperkuat dan menguasai definisi formal, konsep probabilistik, dan penurunan analitik *Gaussian Process* (GP) secara presisi tanpa kerancuan.
2. **Kesiapan Diskusi Teknis**: Menyediakan acuan matematis yang siap ditest/diturunkan di papan tulis saat bimbingan berlangsung bersama dosen pembimbing dan tim.
3. **Rigorous & Eksplisit**: Mendefinisikan setiap entitas variabel (skalar, vektor, matriks) lengkap dengan ukuran dimensi matematisnya agar terhindar dari ambiguitas aljabar linier.
4. **Visualisasi Konseptual (Diagram & Plot)**: Menyertakan ilustrasi grafis/plot di setiap bagian penting untuk memperkuat intuisi geometris (prior vs posterior, profil kernel, trade-off Occam's razor, dan hasil plotting worked example).
5. **Worked Homework Example (Detail Hitungan Manual Step-by-Step)**: Menyajikan contoh perhitungan konkret bergaya *homework problem* dari prior ke posterior untuk $N = 3$ titik training data dan $N_* = 2$ titik test data menggunakan dekomposisi Cholesky.
6. **Dokumen Teknis Berkualitas**: Disusun dalam format laporan teknis ilmiah (*technical report*) menggunakan LaTeX yang rapi, elegan, dan profesional.

---

## 🖼️ Rencana Gambar & Ilustrasi Visual per Bagian
Agar dokumen LaTeX sangat intuitif dan mudah dipahami oleh dosen, kita rencanakan gambar pada bagian-bagian berikut:

| No | Lokasi Bagian | Deskripsi Visual & Tujuan | Metode Pembuatan Gambar di LaTeX |
|---|---|---|---|
| **Gbr 1** | **Bab 1: Pendahuluan & Filosofi** | **Prior vs Posterior Gaussian Process**: Menampilkan sampel fungsi acak dari prior (pita ketidakpastian seragam) berdampingan dengan fungsi posterior setelah mengamati beberapa titik data (pita ketidakpastian menciut di dekat data). | Plot grafis Python (`pgf`/`pdf`/`png`) atau TikZ/PGFPlots langsung di LaTeX. |
| **Gbr 2** | **Bab 2: Karakteristik Kernel** | **Profil Fungsi Kernel & Efek Hyperparameter**: Plot kurva kovariansi $k(r)$ terhadap jarak $r = \|x - x'\|$ membandingkan RBF vs Matérn 3/2 vs Matérn 5/2, serta efek variasi *lengthscale* $l$ terhadap kehalusan sampel fungsi. | Gambar plot komparasi grafis. |
| **Gbr 3** | **Bab 4: Geometri Conditioning** | **Diagram Blok Kovariansi Bersama (*Joint Covariance Matrix*)**: Ilustrasi skematis matriks blok partisi $\begin{bmatrix} K_y & \mathbf{k}_* \\ \mathbf{k}_*^T & k_{**} \end{bmatrix}$ yang menunjukkan bagaimana training data dan test point terhubung secara geometris. | Diagram blok elegan via TikZ. |
| **Gbr 4** | **Bab 5: Marginal Likelihood** | **Visualisasi Prinsip Occam's Razor**: Grafik trade-off antara kompleksitas model vs marginal likelihood (kurva *data-fit* menurun, kurva *complexity penalty* menanjak, menghasilkan puncak optimal di tengah). | Grafik ilustrasi konseptual TikZ / Plot. |
| **Gbr 5** | **Bab 7: Worked Homework Example** | **Plot Hasil Perhitungan Manual ($N=3, N_*=2$)**: Menampilkan grafik koordinat 2D yang memplot 3 titik training data $(x_i, y_i)$, kurva mean prediktif $\bar{f}(x)$, pita ketidakpastian $\pm 2\sigma$ (shaded region), serta posisi khusus 2 test point ($x_{*1} = 0.5$ dengan variansi sempit $0.0189$ vs $x_{*2} = 2.5$ dengan variansi lebar $0.1535$). | Plot kurva koordinat eksak berbasis angka perhitungan manual. |

---

## 📐 Konvensi Notasi & Dimensi Matematis Standar
Sebelum masuk ke penurunan formula, dokumen akan memuat tabel konvensi notasi konsisten:
* $N \in \mathbb{N}$: Jumlah sampel training data (*training instances*), pada contoh kasus $N = 3$.
* $N_* \in \mathbb{N}$: Jumlah sampel test data/prediksi (*test instances*), pada contoh kasus $N_* = 2$.
* $D \in \mathbb{N}$: Dimensi fitur input (pada contoh kasus $D = 1$).
* **Skalar**: Ditulis dengan huruf kecil biasa (misal: $y_i, \sigma_n^2, l, \nu \in \mathbb{R}$).
* **Vektor**: Ditulis dengan huruf kecil tebal, diasumsikan sebagai vektor kolom (misal: $\mathbf{x} \in \mathbb{R}^{D \times 1}$, $\mathbf{y} \in \mathbb{R}^{N \times 1}$, $\mathbf{f}_* \in \mathbb{R}^{N_* \times 1}$).
* **Matriks**: Ditulis dengan huruf kapital tebal (misal: $X \in \mathbb{R}^{N \times D}$, $K \in \mathbb{R}^{N \times N}$, $K_* \in \mathbb{R}^{N \times N_*}$).

---

## 📑 Struktur & Rincian Isi Dokumen (Outline LaTeX)

### 1. Pendahuluan & Filosofi Non-Parametrik Bayesian
* **Transisi Paradigma**:
  * Regresi parametrik: $y = \mathbf{w}^T \boldsymbol{\phi}(\mathbf{x}) + \epsilon$, dengan vektor bobot $\mathbf{w} \in \mathbb{R}^{H \times 1}$ dan fungsi basis $\boldsymbol{\phi}(\mathbf{x}) \in \mathbb{R}^{H \times 1}$.
  * Pendekatan *function-space view*: Menempatkan distribusi probabilitas secara langsung di ruang fungsi tak hingga $f(\mathbf{x})$.
* **Definisi Formal Gaussian Process**:
  * Sifat stokastik: Koleksi variabel acak $\{f(\mathbf{x}) \mid \mathbf{x} \in \mathcal{X}\}$ di mana setiap subset berhingga indeks $\{\mathbf{x}_1, \dots, \mathbf{x}_N\}$ memiliki distribusi bersama multivariat Gaussian.
  * *Consistency Property* / Teorema Ekstensi Kolmogorov:
    $$
    p(\mathbf{f}_1) = \int p(\mathbf{f}_1, \mathbf{f}_2) \, d\mathbf{f}_2
    $$
    di mana $\mathbf{f}_1 \in \mathbb{R}^{N_1 \times 1}$ dan $\mathbf{f}_2 \in \mathbb{R}^{N_2 \times 1}$.
  * Notasi matematis:
    $$
    f(\mathbf{x}) \sim \mathcal{GP}\big(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')\big)
    $$
    * $m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})] \in \mathbb{R}$ (skalar): Fungsi mean.
    * $k(\mathbf{x}, \mathbf{x}') = \mathbb{E}\big[(f(\mathbf{x}) - m(\mathbf{x}))(f(\mathbf{x}') - m(\mathbf{x}'))\big] \in \mathbb{R}$ (skalar): Fungsi kovariansi/kernel.
  * Asumsi $m(\mathbf{x}) \equiv 0$ tanpa kehilangan sifat generalitas.
* **🖼️ [Gambar 1: Ilustrasi Sampel Prior vs Posterior GP]**

### 2. Kovariansi & Karakteristik Fungsi Kernel
* **Syarat Teoretis Kernel**:
  * Fungsi kernel $k: \mathbb{R}^D \times \mathbb{R}^D \to \mathbb{R}$.
  * Syarat *Positive Semi-Definite* (PSD) matriks Gram $K \in \mathbb{R}^{N \times N}$: Untuk setiap vektor tak-nol $\mathbf{v} \in \mathbb{R}^{N \times 1}$, berlaku $\mathbf{v}^T K \mathbf{v} \ge 0$.
  * Teorema Mercer dan Teorema Bochner (representasi spektral).
* **Kernel Squared Exponential (RBF/Gaussian)**:
  * Formulasi:
    $$
    k_{\mathrm{SE}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2 l^2}\right)
    $$
    * $\mathbf{x}, \mathbf{x}' \in \mathbb{R}^{D \times 1}$ (vektor input).
    * $\|\mathbf{x} - \mathbf{x}'\|^2 \in \mathbb{R}$ (skalar kuadrat jarak Euclidean).
    * $\sigma_f^2 \in \mathbb{R}^+$ (skalar): Amplitudo variansi fungsi.
    * $l \in \mathbb{R}^+$ (skalar): *Characteristic lengthscale*.
  * Sifat: *Infinitely mean-square differentiable* ($C^\infty$, fungsi sangat mulus).
* **Keluarga Kernel Matérn**:
  * Formulasi umum dengan fungsi Bessel termodifikasi $K_\nu$ dan parameter keterdiferensialan $\nu$.
  * Kasus khusus $\nu = 3/2$ dan $\nu = 5/2$, serta justifikasi mengapa Matérn lebih realistis untuk fenomena fisik/citra dibanding RBF.
* **Automatic Relevance Determination (ARD)**:
  * Penggunaan matriks panjang karakteristik diagonal $M = \operatorname{diag}(l_1^{-2}, \dots, l_D^{-2}) \in \mathbb{R}^{D \times D}$ untuk mengevaluasi relevansi tiap dimensi input.
* **🖼️ [Gambar 2: Perbandingan Profil Kovariansi RBF vs Matérn dan Efek Lengthscale $l$]**

### 3. Formulasi Model Regresi dengan Noise Observasi
* **Model Observasi Titik per Titik**:
  $$
  y_i = f(\mathbf{x}_i) + \epsilon_i, \quad i = 1, \dots, N
  $$
  * $\mathbf{x}_i \in \mathbb{R}^{D \times 1}$: Vektor input ke-$i$.
  * $y_i \in \mathbb{R}$: Skalar target observasi ke-$i$.
  * $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$: Skalar *i.i.d. Gaussian noise*, dengan variansi noise $\sigma_n^2 \in \mathbb{R}^+$.
* **Representasi Vektor & Matriks**:
  * Matriks desain training input: $X \in \mathbb{R}^{N \times D}$.
  * Vektor target observasi training: $\mathbf{y} \in \mathbb{R}^{N \times 1}$.
  * Vektor laten: $\mathbf{f} \in \mathbb{R}^{N \times 1}$.
  * Vektor derau: $\boldsymbol{\epsilon} \in \mathbb{R}^{N \times 1} \sim \mathcal{N}(\mathbf{0}, \sigma_n^2 I_N)$.
* **Kovariansi Observasi Ber-Noise**:
  $$
  K_y = K(X, X) + \sigma_n^2 I_N \in \mathbb{R}^{N \times N}
  $$

### 4. Penurunan Lengkap Distribusi Bersyarat Posterior
* **Spesifikasi Test Data Multi-Titik ($N_* \ge 1$)**:
  * Matriks test point: $X_* \in \mathbb{R}^{N_* \times D}$.
  * Vektor fungsi laten target test: $\mathbf{f}_* \in \mathbb{R}^{N_* \times 1}$.
  * Matriks kovariansi silang training data & test: $K(X, X_*) \in \mathbb{R}^{N \times N_*}$.
  * Matriks kovariansi silang test data & training: $K(X_*, X) = K(X, X_*)^T \in \mathbb{R}^{N_* \times N}$.
  * Matriks kovariansi internal test point: $K(X_*, X_*) \in \mathbb{R}^{N_* \times N_*}$.
* **Distribusi Gabungan Prior (*Joint Prior Distribution*)**:
  $$
  \begin{bmatrix} \mathbf{y} \\ \mathbf{f}_* \end{bmatrix} \sim \mathcal{N}\left(
  \begin{bmatrix} \mathbf{0}_{N \times 1} \\ \mathbf{0}_{N_* \times 1} \end{bmatrix},
  \begin{bmatrix}
  K(X, X) + \sigma_n^2 I_N & K(X, X_*) \\
  K(X_*, X) & K(X_*, X_*)
  \end{bmatrix}
  \right) \in \mathbb{R}^{(N + N_*) \times 1}
  $$
* **🖼️ [Gambar 3: Skema Partisi Blok Matriks Kovariansi Bersama]**
* **Penurunan Analitik Step-by-Step (*Schur Complement*)**:
  * Partisi matriks presisi multivariat Gaussian.
  * Formulasi umum posterior bersyarat: $\mathbf{f}_* \mid X, \mathbf{y}, X_* \sim \mathcal{N}(\bar{\mathbf{f}}_*, \operatorname{cov}(\mathbf{f}_*))$:
    * **Vektor Mean Prediktif**:
      $$
      \bar{\mathbf{f}}_* = K(X_*, X) \left[ K(X, X) + \sigma_n^2 I_N \right]^{-1} \mathbf{y} \in \mathbb{R}^{N_* \times 1}
      $$
    * **Matriks Kovariansi Prediktif**:
      $$
      \operatorname{cov}(\mathbf{f}_*) = K(X_*, X_*) - K(X_*, X) \left[ K(X, X) + \sigma_n^2 I_N \right]^{-1} K(X, X_*) \in \mathbb{R}^{N_* \times N_*}
      $$
    * Variansi individual tiap test point ke-$j$ diambil dari elemen diagonal: $\mathbb{V}[f_{*j}] = [\operatorname{cov}(\mathbf{f}_*)]_{jj} \in \mathbb{R}$.
* **Analisis Epistemic Uncertainty**:
  * Variansi mengecil di dekat sebaran training points $X$.
  * Variansi membesar menuju variansi prior di daerah ekstrapolasi.
  * Kovariansi non-diagonal $[\operatorname{cov}(\mathbf{f}_*)]_{12}$ mencerminkan korelasi bersama antar titik prediksi test.

### 5. Optimasi Hyperparameter via Marginal Likelihood
* **Evidence / Marginal Likelihood**: $p(\mathbf{y} \mid X, \boldsymbol{\theta}) = \int_{\mathbb{R}^N} p(\mathbf{y} \mid \mathbf{f}) p(\mathbf{f} \mid X, \boldsymbol{\theta}) d\mathbf{f}$.
* **Formulasi Log Marginal Likelihood (LML)**:
  $$
  \log p(\mathbf{y} \mid X, \boldsymbol{\theta}) = -\frac{1}{2} \mathbf{y}^T K_y^{-1} \mathbf{y} - \frac{1}{2} \log |K_y| - \frac{N}{2} \log(2\pi) \in \mathbb{R}
  $$
* **Prinsip Occam's Razor**:
  * Suku 1: $-\frac{1}{2} \mathbf{y}^T K_y^{-1} \mathbf{y}$ (Data-fit term).
  * Suku 2: $-\frac{1}{2} \log |K_y|$ (Complexity penalty term).
  * Suku 3: $-\frac{N}{2} \log(2\pi)$ (Konstanta normalisasi).
* **🖼️ [Gambar 4: Visualisasi Kompromi Data-Fit vs Penalti Kompleksitas (Occam's Razor)]**
* **Gradien Analitik**:
  $$
  \frac{\partial \log p(\mathbf{y} \mid X, \boldsymbol{\theta})}{\partial \theta_j} = \frac{1}{2} \operatorname{tr}\left( \left( \boldsymbol{\alpha}\boldsymbol{\alpha}^T - K_y^{-1} \right) \frac{\partial K_y}{\partial \theta_j} \right) \in \mathbb{R}
  $$
  di mana $\boldsymbol{\alpha} = K_y^{-1}\mathbf{y} \in \mathbb{R}^{N \times 1}$.

### 6. Analisis Komputasi & Solusi Numerik Stabil (Cholesky Decomposition)
* **Kelemahan Inversi Langsung**: Ketidakstabilan numerik (*ill-conditioned matrix*).
* **Faktorisasi Cholesky**: $K_y = L L^T$, dengan $L \in \mathbb{R}^{N \times N}$ matriks segitiga bawah.
* **Prosedur Solusi**:
  1. Forward substitution: $L \mathbf{v} = \mathbf{y} \implies \mathbf{v} \in \mathbb{R}^{N \times 1}$.
  2. Backward substitution: $L^T \boldsymbol{\alpha} = \mathbf{v} \implies \boldsymbol{\alpha} \in \mathbb{R}^{N \times 1}$.
* **Determinan Stabil**: $\log |K_y| = 2 \sum_{i=1}^N \log L_{ii}$.
* **Kompleksitas Komputasi**: $\mathcal{O}(N^3)$ waktu komputasi & $\mathcal{O}(N^2)$ memori.

---

### 7. Worked Homework Example: Perhitungan Manual Step-by-Step ($N = 3, N_* = 2$)
*Bagian khusus bergaya pengerjaan tugas manual (*homework problem*) dengan detail aritmatika, matriks angka nyata, dan dekomposisi langkah demi langkah.*

#### 7.1 Definisi Masalah & Spesifikasi Data Dummy
* **Dimensi Input**: $D = 1$.
* **Hyperparameter Kernel RBF**:
  * $\sigma_f^2 = 1.0$, $l = 1.0$, $\sigma_n^2 = 0.01$.
  * Fungsi kernel: $k(x, x') = \exp\left(-\frac{(x - x')^2}{2}\right)$.
* **Dataset Training ($N = 3$)**:
  * $x_1 = 0.0 \implies y_1 = 0.00$
  * $x_2 = 1.0 \implies y_2 = 0.84$
  * $x_3 = 2.0 \implies y_3 = 0.91$
* **Dataset Test ($N_* = 2$)**:
  * $x_{*1} = 0.5$ (kasus interpolasi, di antara $x_1$ dan $x_2$)
  * $x_{*2} = 2.5$ (kasus ekstrapolasi, di luar jangkauan training data)

#### 7.2 Langkah 1: Konstruksi Matriks Kovariansi Training Data $K(X, X)$ dan $K_y$
* Menghitung nilai kernel pasangan antar training data.
* Menyusun matriks Gram $K(X, X) \in \mathbb{R}^{3 \times 3}$ dan $K_y \in \mathbb{R}^{3 \times 3}$.

#### 7.3 Langkah 2: Faktorisasi Cholesky $K_y = L L^T$
* Menghitung elemen matriks segitiga bawah $L \in \mathbb{R}^{3 \times 3}$ kolom demi kolom secara eksplisit ($L_{11}, L_{21}, L_{31}, L_{22}, L_{32}, L_{33}$).

#### 7.4 Langkah 3: Penyelesaian Sistem Linier untuk Mendapatkan Vektor Bobot $\boldsymbol{\alpha}$
* **Forward Substitution**: Menyelesaikan $L \mathbf{v} = \mathbf{y}$ untuk $\mathbf{v} \in \mathbb{R}^{3 \times 1}$.
* **Backward Substitution**: Menyelesaikan $L^T \boldsymbol{\alpha} = \mathbf{v}$ untuk $\boldsymbol{\alpha} \in \mathbb{R}^{3 \times 1}$.

#### 7.5 Langkah 4: Evaluasi Kovariansi Titik Test ($K(X_*, X)$ dan $K(X_*, X_*)$)
* Menghitung elemen matriks silang $K(X_*, X) \in \mathbb{R}^{2 \times 3}$ dan matriks prior test point $K(X_*, X_*) \in \mathbb{R}^{2 \times 2}$.

#### 7.6 Langkah 5: Kalkulasi Mean Prediktif Posterior $\bar{\mathbf{f}}_*$
* Mengalikan $\bar{\mathbf{f}}_* = K(X_*, X) \boldsymbol{\alpha} \in \mathbb{R}^{2 \times 1}$.
  * $x_{*1} = 0.5 \implies \bar{f}_{*1} \approx \mathbf{0.4263}$.
  * $x_{*2} = 2.5 \implies \bar{f}_{*2} \approx \mathbf{0.6426}$.

#### 7.7 Langkah 6: Kalkulasi Matriks Kovariansi & Variansi Prediktif $\operatorname{cov}(\mathbf{f}_*)$
* Menggunakan forward substitution $L W = K(X, X_*)$ dan menghitung $\operatorname{cov}(\mathbf{f}_*) = K(X_*, X_*) - W^T W \in \mathbb{R}^{2 \times 2}$.
  * Variansi titik interpolasi: $\mathbb{V}[f_{*1}] = \mathbf{0.0189} \implies \sigma_{*1} \approx \mathbf{0.1375}$.
  * Variansi titik ekstrapolasi: $\mathbb{V}[f_{*2}] = \mathbf{0.1535} \implies \sigma_{*2} \approx \mathbf{0.3918}$.
  * Kovariansi silang posterior: $[\operatorname{cov}(\mathbf{f}_*)]_{12} = 0.0461$.

#### 7.8 Langkah 7: Visualisasi Plot & Interpretasi Fisis Hasil Pengerjaan
* **🖼️ [Gambar 5: Plot Eksak Hasil Pengerjaan Soal GP Regression]**
  * Kurva mean prediktif $\bar{f}(x)$ melewati ketiga titik data observasi $(0, 0)$, $(1, 0.84)$, $(2, 0.91)$.
  * Pita ketidakpastian $\pm 2\sigma$ berwarna abu-abu/biru transparan yang menciut di sekitar training data dan melebar di area luar.
  * Penandaan khusus lokasi $x_{*1} = 0.5$ dan $x_{*2} = 2.5$ beserta *error bar*-nya.

---

## 🛠️ Format Dokumen LaTeX yang Akan Dibuat
* **File Target**: [`04_dokumen_ta/kredit_bimbingan_02.tex`](file:///c:/Users/MyBook%20Z%20Series/Desktop/TA/04_dokumen_ta/kredit_bimbingan_02.tex)
* **Folder Aset Gambar**: `04_dokumen_ta/figures/` (menyimpan file-file grafik `.pdf` / `.png` beresolusi tinggi agar di-include rapi lewat `\includegraphics`).
* **Document Class**: `article` (format technical report/notes formal, 1 kolom, font 11pt, margin rapi).
* **Packages**: `amsmath`, `amssymb`, `amsfonts`, `bm`, `booktabs`, `graphicx`, `tikz`, `microtype`, `hyperref`.
