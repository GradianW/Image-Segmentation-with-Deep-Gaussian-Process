# Rencana Penyusunan Dokumen Kredit Bimbingan #3 (Draf Landasan Teori Skripsi)
**Topik Utama**: Karakteristik Fungsi Kernel Rasmussen & Williams, Analisis Efek Hyperparameter, dan Mekanisme Sampling Fungsi GP via Dekomposisi Cholesky  
**Target Pelaksanaan Bimbingan**: Rabu, 23 September 2026  
**Peruntukan Dokumen**: Draf Bab Landasan Teori / Metodologi Laporan Tugas Akhir (Skripsi)

---

## 🎯 Tujuan Dokumen Kredit #3
1. **Draf Bab Skripsi Siap Pakai**: Disusun dengan standar akademik formal yang ketat (bahasa baku, penurunan matematis analitik, dimensi aljabar linier eksplisit, dan notasi konsisten) sehingga dapat langsung diadaptasi menjadi Bab Landasan Teori Laporan Tugas Akhir.
2. **Katalog Komprehensif Kernel Rasmussen & Williams**: Mengulas seluruh fungsi kovariansi standar dari buku teks *Gaussian Processes for Machine Learning* (Bab 4 & Tabel 4.1), baik kernel stasioner (Squared Exponential, Matérn, Rational Quadratic, $\gamma$-Exponential, Periodic, Cosine, White Noise) maupun non-stasioner (Linear, Polynomial, Neural Network).
3. **Eksplorasi Efek Fisis Hyperparameter**: Menjelaskan interpretasi geometris dari masing-masing hyperparameter ($l, \sigma_f^2, \alpha, \gamma, p, c, \sigma_v^2, \Sigma$) dan membuktikannya melalui eksperimen komputasi numerik.
4. **Mekanisme Generatif & Sampling Fungsi GP**: Memberikan penurunan formal dan algoritma komputasi pengambilan sampel fungsi (*function sampling*) dari distribusi GP menggunakan faktorisasi Cholesky ($K = LL^T$), baik untuk kondisi *prior* maupun *posterior*, lengkap dengan analisis penanganan kestabilan numerik (*jitter/nugget effect*).
5. **Jembatan Teoretis Menuju Deep GP & Segmentasi Citra**: Menjelaskan mengapa sifat stasioneritas kernel standar pada GP tunggal memerlukan generalisasi ke arsitektur hirarkis (*Deep Gaussian Process*) untuk menangani diskontinuitas dan variasi spasial tajam pada segmentasi citra.
6. **Visualisasi Berkualitas Publikasi**: Menyajikan grafik komparasi beresolusi tinggi yang memvisualisasikan profil kernel, galeri sampel fungsi 6 kelas kernel, eksperimen variasi hyperparameter 6 panel, alur komputasi Cholesky, dan evolusi prior-posterior.

---

## 📐 Konvensi Notasi & Dimensi Aljabar Linier

| Notasi | Ruang Dimensi | Tipe Entitas | Keterangan Matematis |
|---|---|---|---|
| $N$ | $\mathbb{N}$ | Skalar | Jumlah sampel observasi training |
| $N_*$ | $\mathbb{N}$ | Skalar | Jumlah titik lokasi evaluasi / uji (*test points*) |
| $D$ | $\mathbb{N}$ | Skalar | Dimensi domain input (misal: $D=1$ untuk sinyal, $D=2$ untuk koordinat piksel citra) |
| $\mathbf{x}, \mathbf{x}'$ | $\mathbb{R}^{D \times 1}$ | Vektor Kolom | Pasangan vektor titik input dalam domain $\mathcal{X} \subseteq \mathbb{R}^D$ |
| $r$ | $\mathbb{R}^+$ | Skalar | Jarak Euclidean antar titik input, $r = \|\mathbf{x} - \mathbf{x}'\|_2 = \sqrt{\sum_{d=1}^D (x_d - x'_d)^2}$ |
| $X_*$ | $\mathbb{R}^{N_* \times D}$ | Matriks | Matriks koordinat titik uji bertumpuk, $X_* = [\mathbf{x}_{*1}, \dots, \mathbf{x}_{*N_*}]^T$ |
| $k(\mathbf{x}, \mathbf{x}')$ | $\mathbb{R}$ | Skalar | Nilai evaluasi fungsi kernel kovariansi pada pasangan titik $(\mathbf{x}, \mathbf{x}')$ |
| $K(X_*, X_*)$ | $\mathbb{R}^{N_* \times N_*}$ | Matriks Simetris PSD | Matriks kovariansi Gram antar titik evaluasi uji |
| $L$ | $\mathbb{R}^{N_* \times N_*}$ | Matriks Segitiga Bawah | Faktor dekomposisi Cholesky dari matriks kovariansi, $K = L L^T$ |
| $\mathbf{u}$ | $\mathbb{R}^{N_* \times 1}$ | Vektor Kolom | Vektor variabel acak Gaussian baku independen, $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, I_{N_*})$ |
| $\mathbf{f}_*$ | $\mathbb{R}^{N_* \times 1}$ | Vektor Kolom | Vektor sampel fungsi laten tergenerasi, $\mathbf{f}_* = \mathbf{m}(X_*) + L \mathbf{u}$ |
| $l$ | $\mathbb{R}^+$ | Skalar | Hyperparameter *characteristic lengthscale* |
| $\sigma_f^2$ | $\mathbb{R}^+$ | Skalar | Hyperparameter variansi sinyal (*signal/output variance*) |
| $\sigma_n^2$ | $\mathbb{R}^+$ | Skalar | Hyperparameter variansi derau observasi (*noise variance*) |
| $\alpha$ | $\mathbb{R}^+$ | Skalar | Hyperparameter *scale mixture* pada Rational Quadratic |
| $p$ | $\mathbb{R}^+$ | Skalar | Hyperparameter periode pada Periodic / Cosine Kernel |
| $\epsilon_{\text{jitter}}$ | $\mathbb{R}^+$ | Skalar | Konstanta regularisasi kestabilan numerik diagonal (umumnya $10^{-6}$ s.d. $10^{-4}$) |

---

## 📑 Struktur Bab Dokumen Kredit (LaTeX Outline)

### BAB 1: Karakteristik Fungsi Kovariansi (Kernel) dan Eksplorasi Hyperparameter
* **1.1 Definisi Formal dan Klasifikasi Sifat Kernel**
  * Definisi fungsi kernel $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$.
  * Syarat keabsahan kernel: Matriks Gram $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ harus *Positive Semi-Definite* ($\mathbf{v}^T K \mathbf{v} \ge 0, \forall \mathbf{v} \in \mathbb{R}^N \setminus \{\mathbf{0}\}$).
  * Klasifikasi: Stasioner (invarian translasi), Isotropik (fungsi dari jarak $r$), dan Non-Stasioner (tergantung lokasi absolut).
* **1.2 Katalog Fungsi Kovariansi Buku Teks (Rasmussen & Williams)**
  * Tabel komprehensif formula & hyperparameter.
  * Formulasi analitik: Squared Exponential, Matérn ($\nu=1/2, 3/2, 5/2$), Rational Quadratic, $\gamma$-Exponential, Periodic, Cosine, Linear, Neural Network, dan White Noise.
* **1.3 Analisis Efek Fisis dan Geometri Hyperparameter**
  * Efek *lengthscale* ($l$): Frekuensi fluktuasi spasial horizontal.
  * Efek *signal variance* ($\sigma_f^2$): Skala amplitudo vertikal.
  * Efek *scale mixture* ($\alpha$): Superposisi multiskala pada Rational Quadratic.
  * Efek periode ($p$): Jarak perulangan siklus pada Periodic & Cosine.
  * Efek titik tumpu ($c$) & bobot ($\sigma_v^2, \Sigma$): Corong variansi non-stasioner pada Linear & Neural Network Kernel.
* **1.4 Hasil Eksperimen Komputasi dan Visualisasi Ruang Fungsi**
  * Evaluasi profil kovariansi $k(r)$ (`fig1_kernel_profiles.pdf`).
  * Galeri 6 tipe sampel fungsi prior (`fig2_sample_paths_kernels.pdf`).
  * Analisis 6 panel eksperimen variasi nilai hyperparameter (`fig3_hyperparameter_effects.pdf`).

---

### BAB 2: Mekanisme Generatif dan Sampling Fungsi via Dekomposisi Cholesky
* **2.1 Fondasi Teoretis Sampling dari Multivariat Normal**
  * Proposisi & Pembuktian Formal Teorema Transformasi Linear: Jika $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, I)$, maka $\mathbf{f} = \mathbf{m} + L\mathbf{u} \sim \mathcal{N}(\mathbf{m}, LL^T = K)$.
* **2.2 Prosedur Komputasi 5 Langkah Algoritma Sampling GP**
  1. Diskritisasi domain input $X_*$.
  2. Evaluasi matriks kovariansi $K = k(X_*, X_*)$.
  3. Stabilisasi numerik ($\tilde{K} = K + \epsilon I$) dan Dekomposisi Cholesky ($L = \operatorname{cholesky}(\tilde{K})$).
  4. Pembangkitan vektor keacakan murni $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, I)$.
  5. Transformasi linear $\mathbf{f} = \mathbf{m} + L\mathbf{u}$.
* **2.3 Sampling Prior versus Sampling Posterior (Conditioned on Observed Data)**
  * Formulasi analitik posterior conditioning dan visualisasi penciutan ketidakpastian di sekitar titik data training (`fig5_prior_vs_posterior_samples.pdf`).
* **2.4 Analisis Kestabilan Numerik dan Penanganan Jitter**
  * Penyebab kondisi matriks buruk (*ill-conditioned*) dan peran matematis $\epsilon_{\text{jitter}} I$ dalam menggeser spektrum nilai eigen.

---

### BAB 3: Keterbatasan Kernel Stasioner dan Motivasi Deep Gaussian Process
* **3.1 Keterbatasan Single-Layer GP Stasioner untuk Segmentasi Citra**
  * Ketidakseragaman spasial (*spatial non-stationarity*) dan bahaya *over-smoothing* pada batas tepi objek (*sharp edges*).
* **3.2 Konsep Deep Gaussian Process (DGP) sebagai Solusi Representasi Komposit**
  * Arsitektur hirarki: $\mathbf{y} = f_L(f_{L-1}(\dots f_1(\mathbf{x})))$.
  * Lapisan laten melakukan pembengkokan ruang (*spatial warping*).
  * Propagasi ketidakpastian berbasis sampling via Cholesky & Variational Inference.

---

## 🖼️ Daftar Gambar & Visualisasi (Dihasilkan via `generate_figures.py`)

1. **`fig1_kernel_profiles.pdf / .png`**:  
   Kurva profil fungsi kovariansi $k(r)$ terhadap jarak Euclidean $r$ membandingkan kernel stasioner dasar dan kernel periodik.
2. **`fig2_sample_paths_kernels.pdf / .png`**:  
   Galeri 6 panel sampel fungsi prior acak 1D (SE, Matérn 3/2, RQ, Periodic, Linear, Neural Network).
3. **`fig3_hyperparameter_effects.pdf / .png`**:  
   Eksperimen komputasi variasi nilai hyperparameter: $l$ (SE), $\sigma_f^2$ (SE), $\alpha$ (RQ), $p$ (Periodic), $c$ (Linear), dan $\sigma_v$ (NN).
4. **`fig4_cholesky_sampling_workflow.pdf / .png`**:  
   Diagram alur komputasi sampling fungsi: $K \to L \to \mathbf{u} \to \mathbf{f} = L\mathbf{u}$.
5. **`fig5_prior_vs_posterior_samples.pdf / .png`**:  
   Komparasi visual sampel fungsi Prior GP vs Posterior GP dengan pita ketidakpastian 95\%.
