# Rencana Penyusunan Dokumen Kredit Bimbingan #3 (Draf Landasan Teori Skripsi)
**Topik Utama**: Karakteristik Fungsi Kernel, Eksplorasi Ruang Fungsi, Efek Hyperparameter, dan Mekanisme Sampling Fungsi GP via Dekomposisi Cholesky  
**Target Pelaksanaan Bimbingan**: Rabu, 23 September 2026  
**Peruntukan Dokumen**: Draf Bab Landasan Teori / Metodologi Laporan Tugas Akhir (Skripsi)

---

## 🎯 Tujuan Dokumen Kredit #3
1. **Draf Bab Skripsi Siap Pakai**: Disusun dengan standar akademik formal yang ketat (bahasa baku, penurunan matematis analitik, dimensi aljabar linier eksplisit, dan notasi konsisten) sehingga dapat langsung diadaptasi menjadi Bab Landasan Teori Laporan Tugas Akhir.
2. **Eksplorasi Mendalam Fungsi Kernel**: Mengupas tuntas perbedaan matematis dan implikasi spasial antara jarak linier $r = \|\mathbf{x}-\mathbf{x}'\|$ vs jarak kuadratik $r^2 = \|\mathbf{x}-\mathbf{x}'\|^2$, kelas keterdiferensialan fungsi (*mean-square differentiability*), keluarga Matérn vs *Squared Exponential*, serta peran hyperparameter ($l, \sigma_f^2, \sigma_n^2$).
3. **Mekanisme Generatif & Sampling Fungsi GP**: Memberikan penurunan formal dan algoritma komputasi pengambilan sampel fungsi (*function sampling*) dari distribusi GP menggunakan faktorisasi Cholesky ($K = LL^T$), baik untuk kondisi *prior* maupun *posterior*, lengkap dengan analisis penanganan kestabilan numerik (*jitter/nugget effect*).
4. **Jembatan Teoretis Menuju Deep GP & Segmentasi Citra**: Menjelaskan mengapa sifat stasioneritas dan kehalusan kernel standar pada GP tunggal memerlukan generalisasi ke arsitektur hirarkis (*Deep Gaussian Process*) untuk menangani diskontinuitas dan variasi spasial tajam pada segmentasi citra.
5. **Visualisasi Berkualitas Publikasi**: Menyajikan grafik komparasi beresolusi tinggi yang memvisualisasikan profil kernel, sampel fungsi berbagai kelas keterdiferensialan, efek *lengthscale*, dan alur komputasi Cholesky.

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
| $\epsilon_{\text{jitter}}$ | $\mathbb{R}^+$ | Skalar | Konstanta regularisasi kestabilan numerik diagonal (umumnya $10^{-6}$ s.d. $10^{-4}$) |

---

## 📑 Struktur Bab & Rincian Dokumen Kredit (LaTeX Outline)

### BAB 1: Karakteristik Fungsi Kovariansi (Kernel) dan Sifat Ruang Fungsi
* **1.1 Definisi Formal dan Syarat Positif Semidefinit (PSD)**
  * Definisi fungsi kernel $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$.
  * Syarat keabsahan kernel: Matriks Gram $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ harus *Positive Semi-Definite* ($\mathbf{v}^T K \mathbf{v} \ge 0, \forall \mathbf{v} \in \mathbb{R}^N \setminus \{\mathbf{0}\}$).
  * Teorema Mercer (representasi ekspansi fungsi eigen) dan Teorema Bochner (representasi transformasi Fourier spektral densitas positif).
* **1.2 Kernel Stasioner vs Isotropik**
  * Sifat stasioneritas: $k(\mathbf{x}, \mathbf{x}') = k(\mathbf{x} - \mathbf{x}')$ (invarian terhadap translasi ruang input).
  * Sifat isotropis: $k(\mathbf{x}, \mathbf{x}') = k(\|\mathbf{x} - \mathbf{x}'\|) = k(r)$ (hanya bergantung pada jarak skalar Euclidean $r$).
* **1.3 Eksplorasi Modifikasi Rumus Jarak: Jarak Linier $r$ vs Jarak Kuadratik $r^2$**
  * **Squared Exponential (RBF / Gaussian) Kernel** ($r^2$):
    $$k_{\text{SE}}(r) = \sigma_f^2 \exp\left(-\frac{r^2}{2l^2}\right)$$
    * Karakteristik: Sangat halus (*infinitely mean-square differentiable*, $C^\infty$).
    * Konsekuensi fisis: Asumsi kehalusan yang seringkali terlalu restriktif untuk data non-halus.
  * **Exponential / Ornstein-Uhlenbeck Kernel** ($r$):
    $$k_{\text{Exp}}(r) = \sigma_f^2 \exp\left(-\frac{r}{l}\right)$$
    * Karakteristik: Kontinu tetapi *tidak dapat didiferensiasi secara mean-square* ($C^0$).
    * Bentuk sampel fungsi: Kasar (*jagged*), setara dengan proses Wiener / Gerak Brown 1D.
* **1.4 Keluarga Kernel Matérn sebagai Jembatan Fleksibilitas Kehalusan**
  * Formulasi umum Matérn dengan fungsi Bessel termodifikasi $K_\nu$:
    $$k_{\text{Matérn}}(r) = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\,r}{l}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}\,r}{l}\right)$$
  * Bentuk eksplisit untuk $\nu = p + 1/2$:
    * $\nu = 1/2$: Mereduksi tepat ke Kernel Exponential ($C^0$): $k_{\nu=1/2}(r) = \sigma_f^2 \exp(-r/l)$.
    * $\nu = 3/2$: Sekali terdiferensiasi *mean-square* ($C^1$): $k_{\nu=3/2}(r) = \sigma_f^2 \left(1 + \frac{\sqrt{3}r}{l}\right) \exp\left(-\frac{\sqrt{3}r}{l}\right)$.
    * $\nu = 5/2$: Dua kali terdiferensiasi *mean-square* ($C^2$): $k_{\nu=5/2}(r) = \sigma_f^2 \left(1 + \frac{\sqrt{5}r}{l} + \frac{5r^2}{3l^2}\right) \exp\left(-\frac{\sqrt{5}r}{l}\right)$.
    * $\nu \to \infty$: Konvergen ke Kernel *Squared Exponential* ($C^\infty$).
  * Justifikasi ilmiah pemilihan Matérn 3/2 & 5/2 untuk pemodelan data visual/citra.
* **1.5 Analisis Pengaruh Hyperparameter**
  * Pengaruh *Lengthscale* ($l$): Mengatur rentang korelasi spasial; nilai $l$ kecil menghasilkan osilasi cepat/lokal, nilai $l$ besar menghasilkan tren global yang kaku.
  * Pengaruh *Signal Variance* ($\sigma_f^2$): Mengatur amplitudo vertikal fluktuasi fungsi dari mean.
  * *Automatic Relevance Determination* (ARD) untuk input multi-dimensi ($D > 1$).

---

### BAB 2: Mekanisme Generatif dan Sampling Fungsi dari GP via Dekomposisi Cholesky
* **2.1 Fondasi Teoretis Sampling dari Multivariat Normal**
  * Proposisi: Jika $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, I_{N_*})$, maka kombinasi afinitas $\mathbf{f}_* = \mathbf{m}_* + L \mathbf{u}$ berdistribusi $\mathcal{N}(\mathbf{m}_*, L L^T)$.
  * **Pembuktian Formal**:
    1. Nilai Ekspektasi:
       $$\mathbb{E}[\mathbf{f}_*] = \mathbb{E}[\mathbf{m}_* + L \mathbf{u}] = \mathbf{m}_* + L \mathbb{E}[\mathbf{u}] = \mathbf{m}_* + L \mathbf{0} = \mathbf{m}_*$$
    2. Kovariansi:
       $$\operatorname{cov}(\mathbf{f}_*) = \mathbb{E}\left[(\mathbf{f}_* - \mathbf{m}_*)(\mathbf{f}_* - \mathbf{m}_*)^T\right] = \mathbb{E}\left[(L\mathbf{u})(L\mathbf{u})^T\right] = L \mathbb{E}[\mathbf{u}\mathbf{u}^T] L^T = L I_{N_*} L^T = L L^T = K$$
* **2.2 Faktorisasi Cholesky dan Efisiensi Komputasi**
  * Karakteristik $L$: Matriks segitiga bawah riil unik dengan elemen diagonal positif $L_{ii} > 0$.
  * Kompleksitas komputasi $\mathcal{O}(N_*^3)$ dan perbandingannya dengan inversi matriks langsung serta dekomposisi nilai eigen (Eigendecomposition).
* **2.3 Prosedur Algoritma Sampling Prior Fungsi GP**
  1. Tentukan grid diskritisasi $X_* = [\mathbf{x}_{*1}, \dots, \mathbf{x}_{*N_*}]^T \in \mathbb{R}^{N_* \times D}$.
  2. Hitung matriks Gram prior $K(X_*, X_*) \in \mathbb{R}^{N_* \times N_*}$ via fungsi kernel terpilih.
  3. Lakukan stabilisasi numerik (penambahan *jitter*): $\tilde{K} = K(X_*, X_*) + \epsilon_{\text{jitter}} I_{N_*}$.
  4. Lakukan dekomposisi Cholesky: $L = \operatorname{cholesky}(\tilde{K})$.
  5. Bangkitkan $S$ vektor acak baku $\mathbf{u}^{(s)} \sim \mathcal{N}(\mathbf{0}, I_{N_*}), \, s = 1, \dots, S$.
  6. Hitung sampel fungsi: $\mathbf{f}_*^{(s)} = \mathbf{m}(X_*) + L \mathbf{u}^{(s)}$.
* **2.4 Prosedur Algoritma Sampling Posterior Fungsi GP (Conditioned on Observed Data)**
  1. Evaluasi mean prediktif $\bar{\mathbf{f}}_* = K(X_*, X)[K(X,X) + \sigma_n^2 I_N]^{-1} \mathbf{y}$.
  2. Evaluasi kovariansi prediktif $\operatorname{cov}(\mathbf{f}_*) = K(X_*, X_*) - K(X_*, X)[K(X,X) + \sigma_n^2 I_N]^{-1} K(X, X_*)$.
  3. Lakukan dekomposisi Cholesky pada kovariansi prediktif: $L_{\text{post}} = \operatorname{cholesky}(\operatorname{cov}(\mathbf{f}_*) + \epsilon I)$.
  4. Bangkitkan realisasi fungsi posterior: $\mathbf{f}_{*,\text{post}}^{(s)} = \bar{\mathbf{f}}_* + L_{\text{post}} \mathbf{u}^{(s)}$.
* **2.5 Analisis Masalah Kestabilan Numerik & Penanganan Jitter**
  * Mengapa matriks Gram pada kernel yang sangat halus (seperti RBF dengan $l$ besar) rentan kehilangan sifat *positive definiteness* numerik (kondisi matriks buruk / *ill-conditioned*, nilai eigen terkecil mendekati nol / negatif akibat *round-off error* floating-point IEEE 754).
  * Peran matematis dan implementasi praktis *jitter* $\epsilon_{\text{jitter}} I$ sebagai pengangkat spektrum nilai eigen tanpa merusak integritas korelasi spasial.

---

### BAB 3: Relevansi Teoretis Menuju Deep Gaussian Process & Segmentasi Citra
* **3.1 Keterbatasan Single-Layer GP Stasioner untuk Segmentasi Citra**
  * Citra alamiah mengandung batas objek (*edges/boundaries*) yang tajam, tekstur lokal yang heterogen, dan dependensi spasial non-stasioner.
  * Kernel stasioner memaksakan asumsi korelasi spasial yang seragam di seluruh citra, menyebabkan *over-smoothing* pada batas tepi objek segmentasi.
* **3.2 Konsep Deep Gaussian Process (DGP) sebagai Solusi Representasi Komposit**
  * Arsitektur hirarki: $\mathbf{y} = f_L(f_{L-1}(\dots f_1(\mathbf{x})))$.
  * Lapisan-lapisan GP laten melakukan *warping* ruang input non-linear, memungkinkan pemodelan fungsi non-stasioner dan estimasi ketidakpastian (*uncertainty*) tingkat tinggi pada batas segmentasi piksel.

---

## 🖼️ Rencana Gambar & Visualisasi (Dihasilkan via `generate_figures.py`)

1. **`fig1_kernel_profiles.pdf / .png`**:  
   Kurva profil fungsi kovariansi $k(r)$ terhadap jarak Euclidean $r = \|x - x'\|$ membandingkan Exponential ($r$), Matérn 3/2, Matérn 5/2, dan Squared Exponential ($r^2$).
2. **`fig2_sample_paths_kernels.pdf / .png`**:  
   Perbandingan 4 panel sampel fungsi prior acak 1D untuk masing-masing kernel di atas, memperlihatkan spektrum kehalusan dari $C^0$ (kasar/bergerigi), $C^1$, $C^2$, hingga $C^\infty$ (sangat mulus).
3. **`fig3_hyperparameter_effects.pdf / .png`**:  
   Dampak variasi *lengthscale* ($l = 0.2, 1.0, 3.0$) dan variasi *signal variance* ($\sigma_f^2 = 0.5, 1.0, 2.0$) terhadap karakteristik sampel fungsi GP.
4. **`fig4_cholesky_sampling_workflow.pdf / .png`**:  
   Diagram visual alur komputasi sampling fungsi: Matriks Kovariansi $K \to$ Faktorisasi Segitiga Bawah Cholesky $L \to$ Transformasi Linear $L\mathbf{u} \to$ Realisasi Sampel Kontinu $\mathbf{f}$.
5. **`fig5_prior_vs_posterior_samples.pdf / .png`**:  
   Komparasi visual antara sampel fungsi dari *Prior GP* (ketidakpastian seragam tanpa data) vs *Posterior GP* (sampel fungsi mengunci titik observasi training dengan pita ketidakpastian $95\%$ / $\pm 2\sigma$).
