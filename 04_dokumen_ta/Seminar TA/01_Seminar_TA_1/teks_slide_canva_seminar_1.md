# Teks Siap Salin ke Canva — Seminar Tugas Akhir I (Bab 1 s.d. Bab 3)

Dokumen ini berisi **teks isi per slide** yang sudah diringkas agar tidak terlalu padat di layar, lengkap dengan **rumus (format teks biasa & LaTeX untuk fitur Equation di Canva)**, **rekomendasi gambar dari folder `figures/`**, serta **poin bicara (*speaker notes*)**.

> **Tips Menyalin Rumus ke Canva**:
> Di Canva, buka menu **Apps (Aplikasi)** di panel kiri $\rightarrow$ cari **"Equations"** (atau tekan `/` lalu ketik `Equation`), lalu tempelkan kode LaTeX yang disediakan di setiap slide.

---

## SLIDE 1: Cover / Judul Presentasi

### 📋 Teks untuk Disalin ke Slide:
**SEMINAR TUGAS AKHIR I**
**Segmentasi Citra Menggunakan Deep Gaussian Process melalui Pendekatan Ruang Bobot (*Weight-Space View*)**
*(Catatan: Sesuaikan redaksi judul dengan judul final yang Anda sepakati)*

**Disusun Oleh:**
[Nama Lengkap Anda]
NIM: [NIM Anda]

**Dosen Pembimbing:**
1. [Nama Dosen Pembimbing Utama] (Pembimbing Utama)
2. Ibu Maharani (Ko-Pembimbing)

**Program Studi [Nama Prodi]**
**[Nama Fakultas] — [Nama Universitas]**
**2026**

---

## SLIDE 2: Outline Presentasi

### 📋 Teks untuk Disalin ke Slide:
**AGENDA PRESENTASI**

1. **BAB I — PENDAHULUAN**
   * Latar Belakang & Motivasi Penelitian
   * Spesifikasi Fokus Penelitian (*Weight-Space View*)
   * Rumusan Masalah, Tujuan, & Batasan Masalah

2. **BAB II — LANDASAN TEORI**
   * Model Observasi: Bobot Konstan vs Bobot Berdistribusi (*MGF*)
   * *Gaussian Process* (GP), Kernel, & Optimasi Hyperparameter
   * Formulasi Matriks *Deep Neural Network* (DNN) & *Backpropagation*
   * Arsitektur *Deep Gaussian Process* (DGP)

3. **BAB III — METODOLOGI PENELITIAN**
   * Diagram Alir & Tahapan Penelitian
   * Desain Arsitektur Model & Hasil Eksperimen Pendahuluan
   * Rencana Eksperimen Segmentasi Citra (Menuju Seminar II)

---

## SLIDE 3: Bab I — Latar Belakang & Motivasi

### 📋 Teks untuk Disalin ke Slide:
**LATAR BELAKANG MASALAH**

* **Pentingnya Segmentasi Citra (*Image Segmentation*)**
  * Mempartisi citra digital ke dalam wilayah bermakna melalui klasifikasi/regresi pada tingkat piksel.
  * Krusial untuk analisis medis, penginderaan jauh, dan sistem otonom.

* **Keterbatasan *Deep Neural Network* (DNN) Konvensional**
  * Parameter bobot bersifat **deterministik / titik skalar** ($W$ bernilai tetap).
  * Menghasilkan prediksi titik (*point estimate*) tanpa ukuran **ketidakpastian (*uncertainty estimation*)**.
  * Rentan *overfitting* dan terlalu percaya diri (*overconfident*) pada area citra yang ambigu atau berderau (*noisy*).

* **Solusi: *Gaussian Process* (GP) $\rightarrow$ *Deep Gaussian Process* (DGP)**
  * **Pendekatan Probabilistik**: Memperlakukan fungsi/bobot sebagai variabel acak berdistribusi Gaussian $\rightarrow$ menghasilkan prediksi beserta **variansi ketidakpastiannya**.
  * **Komposisi Berlapis (*Deep*)**: Mengatasi keterbatasan kernel stasioner pada GP standar (1-layer) dengan mengekstraksi fitur secara hierarkis (*multi-layer*).

### 🎤 Catatan Bicara:
> "Pada segmentasi citra, model Deep Learning biasa hanya memberikan tebakan kelas per piksel tanpa memberi tahu seberapa yakin model terhadap tebakan tersebut. Dengan pendekatan Gaussian Process yang diperluas menjadi berlapis (Deep GP), kita menggabungkan kekuatan representasi multi-layer milik DNN dengan kemampuan kuantifikasi ketidakpastian probabilistik."

---

## SLIDE 4: Bab I — Spesifikasi & Posisi Penelitian

### 📋 Teks untuk Disalin ke Slide:
**SPESIFIKASI FOKUS PENELITIAN**

Ditinjau model observasi berderau:
$$y = f(\mathbf{x}) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma_n^2)$$

**Fokus Spesifik Penelitian (*Weight-Space View*):**
* Meneliti kasus di mana fungsi laten $f$ berbentuk kombinasi linier terhadap fitur dan matriks bobot:
  $$f(\mathbf{x}) = \mathbf{x}^T \mathbf{w} \quad \text{atau dalam bentuk matriks} \quad F = X W$$
* **Dua Perspektif Pemodelan yang Dijembatani:**
  1. **Kasus Deterministik (DNN)**: Bobot $W$ bernilai konstan/skalar $\rightarrow$ dilatih via *forward* & *backward propagation* berbasis gradien.
  2. **Kasus Probabilistik (*Weight-Space* GP & DGP)**: Bobot $W$ berdistribusi probabilitas Gaussian $\rightarrow$ menginduksi distribusi atas fungsi dan matriks kovariansi antar-data.

### 🧮 Kode LaTeX Canva Equation:
* `y = f(\mathbf{x}) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma_n^2)`
* `f(\mathbf{x}) = \mathbf{x}^T \mathbf{w} \quad \Longleftrightarrow \quad F = X W`

---

## SLIDE 5: Bab I — Rumusan Masalah & Tujuan Penelitian

### 📋 Teks untuk Disalin ke Slide:
**RUMUSAN MASALAH**
1. Bagaimana memformulasikan secara matematis perambatan maju (*forward pass*) dan perambatan galat mundur (*backward propagation*) pada arsitektur berlapis untuk kasus bobot deterministik (DNN) dan bobot berdistribusi (*Weight-Space* GP/DGP)?
2. Bagaimana merancang dan mengoptimasi arsitektur *Deep Gaussian Process* untuk memodelkan representasi fitur berlapis pada data citra?
3. Bagaimana performa akurasi dan karakteristik estimasi ketidakpastian model pada tugas segmentasi citra?

**TUJUAN PENELITIAN**
1. Menurunkan formulasi analitis aljabar matriks dan aturan rantai (*backpropagation*) dari model *weight-space* deterministik menuju probabilistik.
2. Mengimplementasikan arsitektur model komposisi berlapis (*Deep GP*) beserta algoritma optimasi hyperparameter-nya.
3. Mengevaluasi kinerja segmentasi citra secara kuantitatif dan visual (termasuk peta ketidakpastian prediksi).

---

## SLIDE 6: Bab I — Batasan Masalah

### 📋 Teks untuk Disalin ke Slide:
**BATASAN MASALAH**

1. **Pendekatan Pemodelan**:
   * Berfokus pada representasi ruang bobot (*weight-space view*) pada persamaan observasi $y = f + \varepsilon$ dengan transformasi linier bobot $f = XW$ yang diperluas ke struktur multi-layer.
2. **Asumsi Derau (*Noise*)**:
   * Galat observasi $\varepsilon$ diasumsikan berdistribusi Normal (Gaussian) aditif, independen, dan identik (*i.i.d.*) dengan rata-rata nol dan variansi $\sigma_n^2$.
3. **Metode Optimasi**:
   * Optimasi parameter dan hyperparameter dilakukan secara numerik berbasis gradien (*Gradient Descent* dan *Adam*).
4. **Cakupan Eksperimen**:
   * Seminar I mencakup penurunan teori matematis dan eksperimen validasi awal (*proof-of-concept*), sedangkan pengujian penuh pada dataset segmentasi citra dilakukan pada tahap menuju Seminar II.

---

## SLIDE 7: Bab II — Landasan Teori: Dua Kasus Bobot pada $y = f(x) + \varepsilon$

### 📋 Teks untuk Disalin ke Slide:
**PEMBUKTIAN DISTRIBUSI OBSERVASI VIA *MOMENT GENERATING FUNCTION* (MGF)**

Model dasar: $\mathbf{y} = X^T \mathbf{w} + \boldsymbol{\varepsilon}$ dengan derau $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_n^2 I_n)$

**Kasus 1: Bobot $\mathbf{w}$ Bernilai Konstan (Deterministik)**
* Fungsi laten $f(\mathbf{x}) = \mathbf{x}^T \mathbf{w}$ bersifat konstan/pasti.
* Berdasarkan pembuktian MGF:
  $$M_{\mathbf{y} \mid X, \mathbf{w}}(\mathbf{t}) = \exp\left( \mathbf{t}^T (X^T \mathbf{w}) + \frac{1}{2}\mathbf{t}^T (\sigma_n^2 I_n) \mathbf{t} \right)$$
* Distribusi observasi saling bebas bersyarat:
  $$\mathbf{y} \mid X, \mathbf{w} \sim \mathcal{N}\left( X^T \mathbf{w}, \; \sigma_n^2 I_n \right)$$

**Kasus 2: Bobot $\mathbf{w}$ Berdistribusi Normal ($\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \Sigma_p)$)**
* Fungsi laten menjadi variabel acak Gaussian: $\mathbf{f} = X^T \mathbf{w} \sim \mathcal{N}(\mathbf{0}, X^T \Sigma_p X)$.
* Berdasarkan pembuktian MGF gabungan ($\mathbf{w}$ dan $\boldsymbol{\varepsilon}$ saling bebas):
  $$\mathbf{y} \mid X \sim \mathcal{N}\left( \mathbf{0}, \; X^T \Sigma_p X + \sigma_n^2 I_n \right)$$
* Memunculkan **Matriks Kovariansi** $K = X^T \Sigma_p X$ yang menangkap korelasi antar-titik data!

### 🧮 Kode LaTeX Canva Equation:
* `\mathbf{y} \mid X, \mathbf{w} \sim \mathcal{N}\left( X^T \mathbf{w}, \; \sigma_n^2 I_n \right)`
* `\mathbf{y} \mid X \sim \mathcal{N}\left( \mathbf{0}, \; X^T \Sigma_p X + \sigma_n^2 I_n \right)`

---

## SLIDE 8: Bab II — Transisi ke Ruang Fitur & *Gaussian Process* (GP)

### 📋 Teks untuk Disalin ke Slide:
**DARI *WEIGHT-SPACE* MENUJU *FUNCTION-SPACE* (GAUSSIAN PROCESS)**

* **Pemetaan ke Ruang Fitur ($\boldsymbol{\phi}(\mathbf{x})$)**:
  * Untuk menangkap pola non-linier, input $\mathbf{x} \in \mathbb{R}^D$ dipetakan ke ruang fitur $\Phi = [\boldsymbol{\phi}(\mathbf{x}_1), \dots, \boldsymbol{\phi}(\mathbf{x}_N)]^T$.
  * Struktur kovariansi $X \Sigma_p X^T$ bertransformasi menjadi:
    $$K = \Phi \Sigma_p \Phi^T \quad \text{dengan elemen} \quad K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) = \boldsymbol{\phi}(\mathbf{x}_i)^T \Sigma_p \boldsymbol{\phi}(\mathbf{x}_j)$$

* **Distribusi Marginal pada Gaussian Process**:
  $$\mathbf{y} \mid X \sim \mathcal{N}\left( \mathbf{0}, \; K + \sigma_n^2 I_N \right)$$

* **Membangkitkan Sampel Fungsi (*Cholesky Sampling*)**:
  * Dekomposisi matriks kovariansi: $K + \epsilon I = L L^T$
  * Sampel fungsi kontinu: $\mathbf{f} = \mathbf{m} + L \mathbf{u}$, dengan $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, I)$

### 🖼️ Gambar Pendukung (Upload dari `01_Seminar_TA_1/figures/`):
* `fig1_prior_posterior.png` atau `fig5_prior_vs_posterior_samples.png`
* `fig4_cholesky_sampling_workflow.png`

---

## SLIDE 9: Bab II — Karakteristik Kernel & Optimasi Hyperparameter

### 📋 Teks untuk Disalin ke Slide:
**FUNGSI KERNEL & OPTIMASI HYPERPARAMETER NUMERIK**

* **Eksplorasi & Komposisi Fungsi Kernel**:
  * Karakteristik kehalusan fungsi ditentukan oleh struktur jarak kernel (*Squared Exponential/RBF*, *Periodic*, *Linear*) dan parameter ($\ell, \sigma_f^2$).
  * Komposisi kernel ($k = k_1 + k_2$ dan $k = k_1 \times k_2$) mampu memodelkan tren gabungan (misal: tren linier + osilasi musiman).

* **Fungsi Objektif: *Log Marginal Likelihood***
  $$\log p(\mathbf{y} \mid X, \boldsymbol{\theta}) = -\frac{1}{2}\mathbf{y}^T K_y^{-1}\mathbf{y} - \frac{1}{2}\log|K_y| - \frac{N}{2}\log(2\pi)$$
  *(Suku 1: Data Fit — Suku 2: Penalti Kompleksitas Model — Suku 3: Konstanta Normalisasi)*

* **Mengapa Optimasi Numerik (Gradient Descent & Adam)?**
  * Turunan $\frac{\partial \log p(\mathbf{y}\mid X, \boldsymbol{\theta})}{\partial \theta_j} = \frac{1}{2}\text{tr}\left( (\boldsymbol{\alpha}\boldsymbol{\alpha}^T - K_y^{-1}) \frac{\partial K_y}{\partial \theta_j} \right) = 0$ bersifat **non-linier terhadap $\boldsymbol{\theta}$** dan tidak memiliki solusi analitik tertutup (*closed-form*).
  * Diselesaikan secara iteratif dengan **Gradient Descent** atau **Adam** (*Adaptive Moment Estimation*).

### 🖼️ Gambar Pendukung (Upload dari `01_Seminar_TA_1/figures/`):
* `fig1_marginal_likelihood_tradeoff.png`
* `kernel_periodik_linear.png` atau `kernel_rbf_periodik.png`

---

## SLIDE 10: Bab II — Formulasi Matriks *Deep Neural Network* (DNN)

### 📋 Teks untuk Disalin ke Slide:
**REPRESENTASI ALZABAR MATRIKS PADA DEEP NEURAL NETWORK (DNN)**

Diberikan matriks input $X^{(0)} = X \in \mathbb{R}^{N \times D}$ ($N$ sampel data, $D$ fitur) pada arsitektur $L$ layer:

* **1. Tingkat Satu Neuron (Data ke-$i$, Neuron ke-$j$, Layer $\ell$):**
  $$z^{(\ell)}_{i, j} = \sum_{k=1}^{n_{\ell-1}} x^{(\ell-1)}_{i, k} w^{(\ell)}_{kj} + b^{(\ell)}_j, \quad x^{(\ell)}_{i, j} = \sigma^{(\ell)}\left(z^{(\ell)}_{i, j}\right)$$

* **2. Tingkat Satu Sampel Data (Vektor Baris $\mathbf{x}_i^{(\ell)} \in \mathbb{R}^{1 \times n_\ell}$):**
  $$\mathbf{z}^{(\ell)}_i = \mathbf{x}^{(\ell-1)}_i W^{(\ell)} + \mathbf{b}^{(\ell)}, \quad \mathbf{x}^{(\ell)}_i = \sigma^{(\ell)}\left(\mathbf{z}^{(\ell)}_i\right)$$

* **3. Tingkat *Batch* $N$ Data (Matriks $X^{(\ell)} \in \mathbb{R}^{N \times n_\ell}$):**
  $$\underbrace{Z^{(\ell)}}_{(N \times n_\ell)} = \underbrace{X^{(\ell-1)}}_{(N \times n_{\ell-1})} \underbrace{W^{(\ell)}}_{(n_{\ell-1} \times n_\ell)} + \underbrace{\mathbf{1}_N}_{(N \times 1)} \underbrace{\mathbf{b}^{(\ell)}}_{(1 \times n_\ell)}, \quad X^{(\ell)} = \sigma^{(\ell)}\left(Z^{(\ell)}\right)$$

* **Prediksi Output ($\ell = L$) & Fungsi Objektif (MSE):**
  $$\hat{Y} = X^{(L-1)} W^{(L)} + \mathbf{1}_N \mathbf{b}^{(L)}, \quad \mathcal{L} = \frac{1}{2N} \|Y - \hat{Y}\|_F^2$$

### 🧮 Kode LaTeX Canva Equation:
* `Z^{(\ell)} = X^{(\ell-1)} W^{(\ell)} + \mathbf{1}_N \mathbf{b}^{(\ell)}, \quad X^{(\ell)} = \sigma^{(\ell)}\left(Z^{(\ell)}\right)`
* `\hat{Y} = X^{(L-1)} W^{(L)} + \mathbf{1}_N \mathbf{b}^{(L)}, \quad \mathcal{L} = \frac{1}{2N} \|Y - \hat{Y}\|_F^2`

---

## SLIDE 11: Bab II — Penurunan *Backward Propagation*

### 📋 Teks untuk Disalin ke Slide:
**MEKANISME *BACKWARD PROPAGATION* (ATURAN RANTAI MATRIKS)**

Pembaruan parameter iteratif dengan *learning rate* $\eta > 0$:
$$W^{(\ell)}_{n+1} = W^{(\ell)}_n - \eta \frac{\partial \mathcal{L}}{\partial W^{(\ell)}_n}, \quad \mathbf{b}^{(\ell)}_{n+1} = \mathbf{b}^{(\ell)}_n - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}_n}$$

* **Gradien pada Layer Output ($\ell = L$):**
  $$\frac{\partial \mathcal{L}}{\partial \hat{Y}} = \frac{1}{N}(\hat{Y} - Y) \implies \frac{\partial \mathcal{L}}{\partial W^{(L)}} = \frac{1}{N}\left(X^{(L-1)}\right)^T (\hat{Y} - Y), \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(L)}} = \frac{1}{N}\mathbf{1}_N^T (\hat{Y} - Y)$$

* **Perambatan Galat ke Hidden Layer ($\ell = L-1, \dots, 1$):**
  $$\frac{\partial \mathcal{L}}{\partial Z^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial X^{(\ell)}} \odot {\sigma^{(\ell)}}'\left(Z^{(\ell)}\right)$$
  $$\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \left(X^{(\ell-1)}\right)^T \frac{\partial \mathcal{L}}{\partial Z^{(\ell)}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}} = \mathbf{1}_N^T \frac{\partial \mathcal{L}}{\partial Z^{(\ell)}}, \quad \frac{\partial \mathcal{L}}{\partial X^{(\ell-1)}} = \frac{\partial \mathcal{L}}{\partial Z^{(\ell)}} \left(W^{(\ell)}\right)^T$$

* **Perluasan ke Kasus Bobot Berdistribusi (Probabilistik)**:
  * Ketika $W^{(\ell)}$ bukan lagi skalar konstan melainkan variabel acak berdistribusi Gaussian, perambatan maju dan balik turut memperhitungkan **parameter rata-rata ($\mu_W$) dan variansi/kovariansi ($\Sigma_W$)**.

---

## SLIDE 12: Bab II — Arsitektur *Deep Gaussian Process* (DGP)

### 📋 Teks untuk Disalin ke Slide:
**ARSITEKTUR HIERARKIS *DEEP GAUSSIAN PROCESS* (DGP)**

* **Komposisi Fungsi Probabilistik Berlapis ($L$ Layer):**
  $$\mathbf{h}^{(0)} = X \xrightarrow{\mathbf{f}^{(1)} \sim \mathcal{GP}} \mathbf{h}^{(1)} \xrightarrow{\mathbf{f}^{(2)} \sim \mathcal{GP}} \dots \xrightarrow{\mathbf{f}^{(L)} \sim \mathcal{GP}} \mathbf{y}$$
  * Hidden layer: $\mathbf{h}^{(\ell)} = \mathbf{f}^{(\ell)}(\mathbf{h}^{(\ell-1)}) + \boldsymbol{\varepsilon}^{(\ell)}$
  * Output layer: $\mathbf{y} = \mathbf{f}^{(L)}(\mathbf{h}^{(L-1)}) + \boldsymbol{\varepsilon}^{(L)}$

* **Tantangan Inferensi pada DGP (*Intractability*)**:
  * Distribusi marginal $p(\mathbf{y} \mid X)$ mengharuskan integrasi terhadap seluruh variabel laten tersembunyi:
    $$p(\mathbf{y} \mid X) = \int p(\mathbf{y} \mid \mathbf{h}^{(L-1)}) \prod_{\ell=1}^{L-1} p(\mathbf{h}^{(\ell)} \mid \mathbf{h}^{(\ell-1)}) \, d\mathbf{h}^{(1)} \dots d\mathbf{h}^{(L-1)}$$
  * Karena variabel laten $\mathbf{h}^{(\ell-1)}$ berada di dalam fungsi matriks kovariansi non-linier $K^{(\ell)}$, integral ini tidak lagi tertutup dan memerlukan pendekatan inferensi aproksimasi / propagasi momen.

### 🖼️ Gambar Pendukung (Upload dari `01_Seminar_TA_1/figures/`):
* `dgp_latent_warping.png` atau `dgp_manifold_2d.png`

---

## SLIDE 13: Bab III — Metodologi: Diagram Alir Penelitian

### 📋 Teks untuk Disalin ke Slide:
**TAHAPAN & ALUR METODOLOGI PENELITIAN**

1. **Tahap 1 — Formulasi & Landasan Matematis *(Selesai — Semester Ini)***
   * Penurunan MGF *weight-space view* ($y = f + \varepsilon$, $f = X^T W$), komposisi kernel GP, matriks DNN, dan *backpropagation*.
2. **Tahap 2 — Eksperimen Validasi Awal / *Proof of Concept* *(Selesai — Semester Ini)***
   * Simulasi numerik optimasi hyperparameter (GD vs Adam) dan komparasi GP 1-Layer vs Deep GP (2-Layer & 3-Layer) pada data deret waktu.
3. **Tahap 3 — Persiapan & *Preprocessing* Dataset Citra *(Menuju Seminar II)***
   * Ekstraksi representasi fitur piksel/spasial citra menjadi matriks input $X \in \mathbb{R}^{N \times D}$ dan label segmen $Y$.
4. **Tahap 4 — Implementasi & Pelatihan Model DGP Segmentasi Citra *(Menuju Seminar II)***
   * Pelatihan *forward* & *backward pass* untuk bobot berdistribusi pada data citra.
5. **Tahap 5 — Evaluasi Performa & Analisis Ketidakpastian *(Menuju Seminar II)***
   * Pengukuran metrik akurasi segmen dan visualisasi peta variansi (*uncertainty map*).

---

## SLIDE 14: Bab III — Hasil Eksperimen Pendahuluan (*Preliminary Results*)

### 📋 Teks untuk Disalin ke Slide:
**HASIL EKSPERIMEN PENDAHULUAN (VALIDASI AWAL)**

* **1. Eksplorasi & Fitting Komposisi Kernel GP**
  * Menguji kombinasi penjumlahan & perkalian kernel (RBF, Periodik, Linier) dengan optimasi *Gradient Descent* & *Adam*.
  * Komposisi kernel berhasil menangkap tren global sekaligus fluktuasi periodik lokal dengan interval kepercayaan (*confidence region* $\pm 2\sigma$) yang terkalibrasi.

* **2. Komparasi Standard GP (1-Layer) vs Deep GP (2-Layer & 3-Layer)**
  * Diujikan pada dataset deret waktu emisi $\text{CO}_2$ sektor *Residential* US ($N = 1.612$ data; 80% *train*, 20% *test*).
  * Arsitektur berlapis (*Deep GP* 2-Layer & 3-Layer) mampu memetakan representasi laten bertingkat $x \to h^{(1)} \to h^{(2)} \to y$ secara lebih adaptif dibandingkan GP 1-Layer.

### 🖼️ Gambar Pendukung (Upload dari `01_Seminar_TA_1/figures/`):
* `us_residential_gp_dgp_comparison.png`
* `kernel_combination_brazil_co2.png` atau `kernel_fitting_co2.png`

---

## SLIDE 15: Bab III — Rencana Penerapan pada Segmentasi Citra (Menuju Seminar II)

### 📋 Teks untuk Disalin ke Slide:
**RENCANA IMPLEMENTASI SEGMENTASI CITRA & EVALUASI (SEMINAR II)**

* **Skema Representasi Data Citra ke Matriks Model**:
  * Setiap piksel/patch citra ke-$i$ direpresentasikan sebagai vektor fitur baris $\mathbf{x}_i \in \mathbb{R}^{1 \times D}$ (intensitas warna/spasial), membentuk matriks $X \in \mathbb{R}^{N \times D}$.
  * Target $Y \in \mathbb{R}^{N \times C}$ merepresentasikan label kelas segmen setiap piksel.

* **Metrik Evaluasi yang Akan Digunakan**:
  1. **Metrik Kualitas Segmentasi**: *Pixel Accuracy*, *Intersection over Union* (IoU / *Jaccard Index*), dan *Dice Coefficient*.
  2. **Metrik Evaluasi Probabilistik & Ketidakpastian**: *Negative Log-Likelihood* (NLL), *Root Mean Squared Error* (RMSE), serta visualisasi *Uncertainty Map* (piksel dengan variansi prediksi tinggi).

* **Target Jadwal Pengerjaan (Semester Depan)**:
  * **Bulan 1–2**: Finalisasi penurunan *backpropagation* bobot berdistribusi & persiapan *pipeline* dataset citra.
  * **Bulan 3**: Eksperimen pelatihan model & *hyperparameter tuning*.
  * **Bulan 4**: Analisis hasil, penulisan Bab IV–V, dan pelaksanaan **Seminar TA II**.

---

## SLIDE 16: Penutup

### 📋 Teks untuk Disalin ke Slide:
**KESIMPULAN SEMENTARA & PENUTUP**

* Fondasi teoretis dari pendekatan *weight-space* ($y = XW + \varepsilon$), pembuktian distribusi via MGF, optimasi kernel GP, hingga formulasi matriks DNN dan *backpropagation* telah berhasil dirumuskan.
* Eksperimen awal menunjukkan kesiapan kerangka komposisi berlapis (*Deep GP*) dalam memodelkan pola data non-linier beserta estimasi ketidakpastiannya.
* Tahap selanjutnya berfokus pada ekstensi *backpropagation* bobot berdistribusi dan penerapannya secara penuh pada tugas *Image Segmentation*.

**TERIMA KASIH**
*Mohon Masukan dan Saran dari Bapak/Ibu Dosen Penguji & Pembimbing*

