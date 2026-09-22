# Rencana Penyusunan Dokumen Kredit Bimbingan #3
**Topik Utama**: Karakteristik Fungsi Kernel Rasmussen & Williams, Analisis Efek Hyperparameter, dan Mekanisme Sampling Fungsi GP via Dekomposisi Cholesky  
**Target Pelaksanaan Bimbingan**: Rabu, 23 September 2026  
**Peruntukan Dokumen**: Draf Bab Landasan Teori Laporan Tugas Akhir / Skripsi  
**Referensi Utama**: Rasmussen, C. E., \& Williams, C. K. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

---

## 🎯 Fokus & Tujuan Dokumen Kredit #3

Dokumen ini disusun secara sederhana, elegan, dan fokus pada **dua topik inti** dari buku teks rujukan utama Rasmussen & Williams (2006):

1. **Eksplorasi Fungsi Kovariansi (Kernel)**:
   - Definisi formal fungsi kernel $k(\mathbf{x}, \mathbf{x}')$ dan syarat keabsahan *Positive Semi-Definite* (PSD).
   - Klasifikasi sifat spasial: Stasioner (invarian translasi), Isotropik (fungsi dari jarak Euclidean $r = \|\mathbf{x}-\mathbf{x}'\|$), dan Non-stasioner.
   - Katalog formulasi analitik fungsi kovariansi standar (Bab 4 & Tabel 4.1): Squared Exponential (SE), Matérn class ($\nu=1/2, 3/2, 5/2$), Rational Quadratic (RQ), dan Linear / Dot Product.
   - Analisis efek geometris dan fisis dari masing-masing hyperparameter ($l, \sigma_f^2, \nu, \alpha, c, \sigma_v$).
   - Evaluasi komparatif visual dengan grafik beresolusi tinggi yang membandingkan setiap variasi kernel dan pengaruh nilai hyperparameter.

2. **Bagaimana Mencari Sample dari Distribusi Gaussian Process**:
   - Fondasi analitik representasi multivariat normal $\mathbf{f}_* \sim \mathcal{N}(\mathbf{m}_*, K(X_*, X_*))$ dari generator bilangan acak Gaussian baku independen $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, I_{N_*})$.
   - Teorema Transformasi Linear dan Dekomposisi Cholesky ($K = LL^T$): penurunan formal pembuktian mean $\mathbb{E}[\mathbf{f}] = \mathbf{m}$ dan kovariansi $\operatorname{cov}(\mathbf{f}) = K$.
   - Prosedur komputasi 5 langkah algoritma sampling GP (Diskritisasi domain, evaluasi matriks Gram, stabilisasi jitter, pembangkitan derau acak, dan transformasi linear).
   - Analisis kestabilan numerik dan peran matematis penambahan jitter $\epsilon_{\text{jitter}} I$ dalam menggeser spektrum nilai eigen positif.
   - Perbandingan sampling pada kondisi sebelum observasi data (**Prior**) dan sesudah observasi data training (**Posterior Conditioning**, Bab 2).

---

## 📐 Konvensi Notasi & Dimensi Aljabar Linier

| Notasi | Ruang Dimensi | Tipe Entitas | Keterangan Matematis |
|---|---|---|---|
| $N$ | $\mathbb{N}$ | Skalar | Jumlah sampel observasi training |
| $N_*$ | $\mathbb{N}$ | Skalar | Jumlah titik lokasi evaluasi / uji (*test points*) |
| $D$ | $\mathbb{N}$ | Skalar | Dimensi domain input ($D=1$ untuk sinyal, $D=2$ untuk koordinat piksel) |
| $\mathbf{x}, \mathbf{x}'$ | $\mathbb{R}^{D \times 1}$ | Vektor Kolom | Pasangan vektor titik input dalam domain $\mathcal{X} \subseteq \mathbb{R}^D$ |
| $r$ | $\mathbb{R}^+$ | Skalar | Jarak Euclidean antar titik input, $r = \|\mathbf{x} - \mathbf{x}'\|_2$ |
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
| $\epsilon_{\text{jitter}}$ | $\mathbb{R}^+$ | Skalar | Konstanta regularisasi kestabilan numerik diagonal ($10^{-6}$ s.d. $10^{-4}$) |

---

## 📑 Struktur Bab Dokumen Kredit (LaTeX Outline)

### BAB 1: Eksplorasi Fungsi Kovariansi (Kernel)
* **1.1 Katalog Fungsi Kovariansi Baku Rasmussen & Williams**:
  * Tabel komprehensif formula analitik & hyperparameter (SE, Matérn 1/2, 3/2, 5/2, RQ, Linear).
* **1.2 Analisis Efek Fisis dan Geometris Hyperparameter**:
  * Interpretasi geometris $l, \sigma_f^2, \nu, \alpha, c, \sigma_v$.
* **1.3 Hasil Eksperimen Komparasi Visual Variasi Kernel**:
  * Gambar 1: Profil kovariansi $k(r)$ vs $r$ (`fig1_kernel_profiles.pdf`).
  * Gambar 2: Galeri sampel fungsi prior 6 kelas kernel (`fig2_sample_paths_kernels.pdf`).
  * Gambar 3: Analisis 4 panel pengaruh variasi nilai hyperparameter (`fig3_hyperparameter_effects.pdf`).

---

### BAB 2: Bagaimana Mencari Sample dari Distribusi Gaussian Process
* **2.1 Konsep Dasar dan Masalah Sampling Multivariat Normal**:
  * Keterbatasan pembangkit acak diskret dan kebutuhan mengikat korelasi spasial.
* **2.2 Dekomposisi Cholesky dan Teorema Transformasi Linear**:
  * Pembuktian analitik sifat mean dan kovariansi $\mathbf{f} = \mathbf{m} + L\mathbf{u} \sim \mathcal{N}(\mathbf{m}, K)$.
* **2.3 Prosedur Komputasi 5 Langkah Algoritma Sampling GP**:
  * Alur komputasi lengkap: $K \to \tilde{K} \to L \to \mathbf{u} \to \mathbf{f} = L\mathbf{u}$.
  * Gambar 4: Diagram alur kerja sampling Cholesky (`fig4_cholesky_sampling_workflow.pdf`).
* **2.4 Analisis Kestabilan Numerik dan Peran Jitter**:
  * Penanganan kondisi matriks buruk (*ill-conditioned*) via $\epsilon_{\text{jitter}} I$.
* **2.5 Sampling Prior versus Sampling Posterior**:
  * Formulasi posterior conditioning dan visualisasi penciutan pita ketidakpastian 95% di sekitar data training (`fig5_prior_vs_posterior_samples.pdf`).

---

## 📓 Sumber Visualisasi: Jupyter Notebook (`eksplorasi_kernel_dan_sampling.ipynb`)

Seluruh visualisasi komparatif dihasilkan melalui sebuah berkas Jupyter Notebook:
- **`eksplorasi_kernel_dan_sampling.ipynb`**: Berisi implementasi fungsi kernel, helper sampling Cholesky, visualisasi profil, galeri sampel 6 kernel, eksperimen variasi hyperparameter, workflow Cholesky, dan sampling prior vs posterior.
- Seluruh gambar diekspor ke direktori `figures/` dalam format `.pdf` (vektor publikasi) dan `.png` (raster resolusi tinggi 300 DPI).
