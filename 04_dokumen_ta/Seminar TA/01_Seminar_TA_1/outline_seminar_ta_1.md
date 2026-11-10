# Kerangka Struktur Slide (PPT) Seminar Tugas Akhir 1 (Bab 1 – Bab 3)

Dokumen ini berisi rekomendasi struktur slide per halaman untuk **Seminar TA 1 (Proposal / Kemajuan Bab 1–3)** yang disesuaikan dengan topik penelitian dan riwayat bimbingan Anda (*Weight-Space View*, *Gaussian Process*, *Deep Neural Network* $\rightarrow$ *Deep Gaussian Process* untuk *Image Segmentation*).

Estimasi durasi presentasi standar: **15–20 menit** (**14–16 slide**).

---

## Bagian Pembuka (Slide 1–2)

### Slide 1: Judul Presentasi (Cover)
* **Isi Slide**:
  * Judul Tugas Akhir (misal: *Segmentasi Citra Menggunakan Deep Gaussian Process melalui Pendekatan Weight-Space* — sesuaikan dengan rumusan judul final Anda).
  * Subjudul: *Seminar Tugas Akhir I*.
  * Nama Mahasiswa & NIM.
  * Nama Dosen Pembimbing Utama & Dosen Ko-Pembimbing (Ibu Maharani).
  * Nama Program Studi, Fakultas, Universitas, dan Tahun (2026).

### Slide 2: Outline Presentasi
* **Isi Slide**:
  1. **Bab I — Pendahuluan**: Latar Belakang, Rumusan Masalah, Tujuan, & Batasan Masalah.
  2. **Bab II — Landasan Teori**: *Weight-Space View*, *Gaussian Process* (GP), *Deep Neural Network* (DNN), hingga *Deep Gaussian Process* (DGP).
  3. **Bab III — Metodologi Penelitian**: Alur Penelitian, Formulasi Model ($y = f + \varepsilon$ dengan $f = X^T W$), Arsitektur & Pelatihan, serta Rencana Evaluasi.

---

## Bagian 1: Bab I — Pendahuluan (Slide 3–6)

### Slide 3: Latar Belakang Masalah (Motivasi)
* **Poin Utama**:
  * Pentingnya tugas **Segmentasi Citra (*Image Segmentation*)** dalam visi komputer (klasifikasi pada tingkat piksel).
  * **Keterbatasan *Deep Neural Network* (DNN) Konvensional**:
    * Parameter bobot bersifat deterministik/titik (*point estimate* $W$ skalar).
    * Cenderung *overconfident* pada prediksi dan tidak secara alami memodelkan **ketidakpastian (*uncertainty estimation*)**, serta rentan *overfitting* pada data terbatas/berderau.
  * **Mengapa Pendekatan Probabilistik (*Gaussian Process* & *Deep GP*)?**:
    * Memperlakukan parameter/fungsi sebagai variabel acak berdistribusi probabilitas, menghasilkan prediksi beserta ukuran ketidakpastiannya (variansi).
    * **Kebutuhan Struktur *Deep* (Multi-Layer)**: GP standar (1-layer) terbatas pada kernel stasioner; komposisi berlapis (*Deep GP*) memungkinkan pembelajaran representasi fitur hierarkis seperti DNN namun tetap probabilistik.
* **Saran Visual di Canva**: Ilustrasi perbandingan prediksi deterministik (hanya *output* kelas) vs prediksi probabilistik (*output* + peta ketidakpastian/*uncertainty map*).

### Slide 4: Spesifikasi & Fokus Penelitian (Posisi Penelitian)
* **Poin Utama**:
  * Model observasi umum: $y = f(\mathbf{x}) + \varepsilon$, dengan $\varepsilon \sim \mathcal{N}(0, \sigma_n^2)$.
  * **Fokus Spesifik Penelitian Ini (*Weight-Space View*)**:
    * Meninjau fungsi laten $f$ yang berbentuk kombinasi linier terhadap fitur/bobot: $f(\mathbf{x}) = \mathbf{x}^T \mathbf{w}$ (atau dalam bentuk matriks $F = X W$) dengan bobot berdistribusi Gaussian ($\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \Sigma_p)$), lalu diperluas ke struktur berlapis (*multi-layer*).
  * Jembatan antara arsitektur matriks *Deep Neural Network* (DNN) dengan inferensi Bayesian / *Gaussian Process*.

### Slide 5: Rumusan Masalah & Tujuan Penelitian
* **Rumusan Masalah**:
  1. Bagaimana memformulasikan pemodelan matematis dari bobot deterministik (DNN) menuju bobot berdistribusi probabilitas (*Weight-Space View* GP/DGP) beserta mekanisme perambatan maju (*forward*) dan balik (*backward propagation*)?
  2. Bagaimana membangun arsitektur model komposisi berlapis berbasis *Gaussian Process* untuk tugas segmentasi citra?
  3. Bagaimana kinerja dan kualitas estimasi ketidakpastian model yang diusulkan pada segmentasi citra?
* **Tujuan Penelitian**:
  * Menjawab ketiga poin rumusan masalah secara terukur (formulasi analitis, implementasi model, dan evaluasi metrik).

### Slide 6: Batasan Masalah
* **Poin Utama**:
  * Bentuk fungsi laten berfokus pada pendekatan ruang bobot (*weight-space view*: $f = X W$ dengan prior bobot Gaussian).
  * Asumsi derau (*noise*) observasi bersifat Gaussian aditif independen ($\varepsilon \sim \mathcal{N}(0, \sigma_n^2 I)$).
  * Ruang lingkup dataset citra dan resolusi yang digunakan pada eksperimen.

---

## Bagian 2: Bab II — Landasan Teori (Slide 7–11)
*(Bagian ini adalah kekuatan utama dari kredit bimbingan #2 s.d. #8 yang sudah Anda susun)*

### Slide 7: Regresi Linier Deterministik vs Bayesian (*Weight-Space View*)
* **Poin Utama**:
  * Model observasi: $y = \mathbf{x}^T \mathbf{w} + \varepsilon$.
  * Perbandingan dua kasus (hasil pembuktian *Moment Generating Function* / MGF pada Bimbingan #6):
    1. **Bobot $w$ Konstan (Deterministik)**: $f(\mathbf{x})$ deterministik; keacakan $y$ hanya berasal dari derau $\varepsilon$.
    2. **Bobot $w$ Berdistribusi Normal ($\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \Sigma_p)$)**: $f(\mathbf{x}) = \mathbf{x}^T \mathbf{w}$ menjadi peubah acak Gaussian multivariat dengan matriks kovariansi $K = X \Sigma_p X^T$.
* **Saran Visual di Canva**: Tabel 2 kolom yang membandingkan distribusi prior, nilai ekspektasi $\mathbb{E}[y]$, variansi $\text{Var}(y)$, dan pembuktian MGF.

### Slide 8: Gaussian Process (GP) & Fungsi Kernel
* **Poin Utama**:
  * Definisi formal GP: Koleksi variabel acak yang setiap subkumpulan berhingganya berdistribusi Gaussian gabungan, ditentukan oleh fungsi rata-rata $m(\mathbf{x})$ dan fungsi kovariansi/kernel $k(\mathbf{x}, \mathbf{x}')$.
  * Karakteristik Kernel (RBF/Squared Exponential, Periodik, Linier, Komposisi Kernel) & *Sampling* fungsi via Dekomposisi Cholesky ($K = L L^T$).
  * Optimasi *Hyperparameter*: Maksimisasi *Log Marginal Likelihood* secara numerik (Gradient Descent & Adam) karena tidak memiliki solusi analitik tertutup (*closed-form*).
* **Saran Visual di Canva**: Grafik visualisasi *prior/posterior* GP atau hasil fitting komposisi kernel yang sudah Anda buat di bimbingan sebelumnya.

### Slide 9: Formulasi Matematis *Deep Neural Network* (DNN)
* **Poin Utama** (dari Bimbingan #8):
  * Representasi berjenjang dari tingkat **1 Neuron** $\rightarrow$ **1 Data (Vektor)** $\rightarrow$ **$N$ Data (Matriks Batch)**:
    $$Z^{(\ell)} = X^{(\ell-1)} W^{(\ell)} + \mathbf{1}_N \mathbf{b}^{(\ell)}, \quad X^{(\ell)} = \sigma^{(\ell)}(Z^{(\ell)})$$
  * Fungsi Objektif (*Loss Function*) & mekanisme *Backward Propagation* (aturan rantai gradien matriks $\frac{\partial \mathcal{L}}{\partial W^{(\ell)}}$ dan $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}}$).
* **Saran Visual di Canva**: Diagram arsitektur layer $\ell-1 \to \ell \to L$ lengkap dengan dimensi matriks $(N \times n_{\ell-1}) \times (n_{\ell-1} \times n_\ell)$.

### Slide 10: Transisi Menuju *Deep Gaussian Process* (DGP)
* **Poin Utama**:
  * Dari DNN (bobot skalar deterministik $W^{(\ell)}$) menuju pemetaan berlapis probabilistik:
    $$\mathbf{x}^{(0)} \xrightarrow{f^{(1)}} \mathbf{x}^{(1)} \xrightarrow{f^{(2)}} \dots \xrightarrow{f^{(L)}} \mathbf{y}$$
  * Bagaimana setiap *layer* merambatkan distribusi (rata-rata dan variansi/kovariansi), serta perbandingan *backpropagation* ketika bobot bernilai skalar vs bobot berdistribusi.
* **Saran Visual di Canva**: Diagram perbandingan berdampingan antara *Node/Layer* DNN vs *Layer* DGP, atau grafik eksperimen awal 1-Layer vs 2-Layer vs 3-Layer (data $\text{CO}_2$).

### Slide 11: Konsep Segmentasi Citra (*Image Segmentation*)
* **Poin Utama**:
  * Representasi citra digital sebagai matriks/tensor fitur piksel $\mathbf{x}_i \in \mathbb{R}^{1 \times D}$ dan label target kelas segmen $y_i$.
  * Bagaimana model memetakan fitur piksel/spasial ke probabilitas kelas segmen beserta estimasi ketidakpastiannya.

---

## Bagian 3: Bab III — Metodologi Penelitian (Slide 12–15)

### Slide 12: Diagram Alir Penelitian (*Research Flowchart*)
* **Poin Utama**:
  1. Studi Literatur & Penurunan Formulasi Matematis (*Weight-Space* GP, DNN, & DGP).
  2. Eksperimen Validasi Awal pada Data 1D / Deret Waktu (*Proof of Concept*).
  3. Persiapan & *Preprocessing* Dataset Citra.
  4. Perancangan Arsitektur & Pelatihan Model untuk Segmentasi Citra.
  5. Evaluasi Performa & Analisis Ketidakpastian (*Uncertainty Analysis*).
* **Saran Visual di Canva**: *Flowchart* horizontal atau vertikal 5 tahap (tandai tahap 1–2 yang **sudah terlaksana** di semester ini dan tahap 3–5 untuk penyelesaian menuju Seminar 2).

### Slide 13: Desain Model & Algoritma Pelatihan
* **Poin Utama**:
  * Spesifikasi arsitektur model yang diusulkan:
    * Input fitur citra $X^{(0)} \in \mathbb{R}^{N \times D}$.
    * Transformasi antar-layer dengan parameter bobot berdistribusi.
    * Fungsi objektif yang dioptimasi dan metode optimasi gradien (GD / Adam).
  * Skema *Forward Pass* (prediksi & perambatan variansi) dan *Backward Pass* (pembaruan parameter distribusi).

### Slide 14: Hasil Eksperimen Pendahuluan (*Preliminary Results*)
*(Opsional tapi **sangat direkomendasikan** untuk Seminar 1 agar penguji melihat bukti progres nyata Anda)*
* **Poin Utama**:
  * Menampilkan ringkasan eksperimen yang sudah Anda kerjakan selama Bimbingan #2–#8:
    * Simulasi *komposisi kernel* & optimasi hyperparameter (GD vs Adam).
    * Perbandingan GP 1-Layer vs Deep GP 2-Layer & 3-Layer pada dataset uji awal.
* **Saran Visual di Canva**: Plot hasil fitting & tabel metrik error (MSE/RMSE) dari eksperimen yang sudah ada di folder `figures/`.

### Slide 15: Rencana Eksperimen Segmentasi Citra & Metrik Evaluasi (Menuju Seminar 2)
* **Poin Utama**:
  * **Dataset Citra**: Deskripsi singkat dataset citra yang akan digunakan.
  * **Metrik Evaluasi**:
    * Akurasi/Error prediksi (*Pixel Accuracy*, *IoU / Jaccard Index*, *Dice Score*, atau *MSE/NLL*).
    * Evaluasi kualitas ketidakpastian (*Negative Log-Likelihood* / kalibrasi variansi).
  * **Timeline Rencana Kerja Semester Depan**: Target penyelesaian eksperimen utama dan penulisan Bab 4–5.

---

## Bagian Penutup (Slide 16)

### Slide 16: Penutup / Terima Kasih
* **Isi Slide**:
  * Ringkasan 3 poin kontribusi/progres sejauh ini.
  * Ucapan terima kasih & sesi tanya jawab (*Q&A*).

