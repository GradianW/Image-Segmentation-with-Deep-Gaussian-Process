# 🔬 Eksperimen 01: Implementasi Deep Gaussian Process dengan GPyTorch

Eksperimen ini berfokus pada implementasi mandiri **Deep Gaussian Process (DGP)** multi-layer menggunakan library **GPyTorch** dengan pendekatan *Doubly Stochastic Variational Inference (DSVI)* (Salimbeni & Deisenroth, 2017) berbasis **Jupyter Notebook (`.ipynb`)**.

---

## 🎯 Tujuan Eksperimen
1. Membangun model DGP 2-layer ($x \to h \to y$) dengan *inducing points* terlatih pada setiap layer secara *self-contained* di dalam notebook.
2. Membandingkan performa representasi non-linear antara **Single-layer Sparse GP** dan **Deep GP (2 Layers)** pada fungsi sintetis non-stasioner (*step function*).
3. Mengevaluasi metrik akurasi (RMSE, MAE) dan kalibrasi ketidakpastian (Negative Log Predictive Density / NLPD).
4. Memvisualisasikan fenomena *latent space warping* yang meregangkan ruang input pada daerah diskontinuitas.

---

## 📓 Notebook Utama

Seluruh arsitektur, pelatihan, evaluasi metrik, dan visualisasi telah terangkum secara lengkap di dalam file notebook:

👉 [**`01_dgp_gpytorch_implementation.ipynb`**](file:///c:/Users/MyBook%20Z%20Series/Desktop/TA/02_eksperimen/01_dgp_gpytorch/01_dgp_gpytorch_implementation.ipynb)

---

## 🏗️ Struktur Folder Eksperimen

```text
01_dgp_gpytorch/
├── 01_dgp_gpytorch_implementation.ipynb   # [UTAMA] Notebook self-contained implementasi & eksperimen
├── README.md                              # Ringkasan tujuan, hipotesis, dan catatan eksperimen
├── data/                                  # Folder dataset (jika diperlukan)
└── figures/                               # Folder penyimpanan grafik ekspor hasil visualisasi
```

---

## 📊 Format Ringkasan Hasil Eksperimen

| Model | Arsitektur | Inducing Points | Test RMSE | Test MAE | Test NLPD | Catatan |
|:---|:---|:---|:---|:---|:---|:---|
| **Single-Layer GP** | 1 Layer (RBF Kernel) | 32 | *Evaluated* | *Evaluated* | *Evaluated* | Baseline (mengalami *oversmoothing* pada transisi tajam) |
| **2-Layer Deep GP** | 1D $\to$ 2D Latent $\to$ 1D Output | 32, 32 | *Evaluated* | *Evaluated* | *Evaluated* | Transisi tajam terpetakan dengan baik via *latent warping* |
