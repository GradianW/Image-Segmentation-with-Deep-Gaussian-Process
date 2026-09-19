# 🧪 Direktori Eksperimen Tugas Akhir

Direktori ini dirancang khusus untuk mengelola, melacak, dan mendokumentasikan seluruh tahapan eksperimen model, arsitektur, dan *benchmarking* berbasis **Jupyter Notebook (`.ipynb`)**.

---

## 📋 Daftar Eksperimen & Log Perkembangan

| No | Folder Eksperimen | Notebook Utama | Fokus & Topik Eksperimen | Status |
|:---|:---|:---|:---|:---|
| **01** | [`01_dgp_gpytorch/`](file:///c:/Users/MyBook%20Z%20Series/Desktop/TA/02_eksperimen/01_dgp_gpytorch) | [`01_dgp_gpytorch_implementation.ipynb`](file:///c:/Users/MyBook%20Z%20Series/Desktop/TA/02_eksperimen/01_dgp_gpytorch/01_dgp_gpytorch_implementation.ipynb) | Implementasi Deep Gaussian Process (DGP) dengan GPyTorch & DSVI pada fungsi non-stasioner | 🚀 Ready |
| **02** | `02_dgp_vs_single_gp/` | `02_dgp_vs_single_gp_benchmark.ipynb` | Komparasi mendalam Single GP vs Deep GP pada variasi frekuensi spasial & ketidakpastian | ⏳ Planned |
| **03** | `03_dgp_patch_classification/` | `03_dgp_patch_classification.ipynb` | Eksplorasi representasi hierarkis DGP pada patch citra mini | ⏳ Planned |
| **04** | `04_dgp_image_segmentation_toy/` | `04_dgp_image_segmentation_toy.ipynb` | Prototipe awal integrasi DGP untuk segmentasi citra 2D (toy masks & boundary uncertainty) | ⏳ Planned |

---

## 🏗️ Standar Struktur Setiap Folder Eksperimen

Setiap subfolder eksperimen dirancang *self-contained* berbasis notebook:

```text
02_eksperimen/XX_nama_eksperimen/
├── XX_nama_notebook.ipynb        # [UTAMA] Implementasi kode, visualisasi, dan analisis dalam 1 notebook
├── README.md                     # Catatan ringkas tujuan & temuan eksperimen
├── data/                         # Dataset lokal (opsional)
└── figures/                      # Grafik atau plot yang diekspor dari notebook (opsional)
```

---

## 💡 Panduan Penggunaan
1. Buka folder eksperimen yang dituju (misal: [`01_dgp_gpytorch/`](file:///c:/Users/MyBook%20Z%20Series/Desktop/TA/02_eksperimen/01_dgp_gpytorch)).
2. Jalankan notebook secara interaktif untuk melihat kurva loss, evaluasi metrik, dan plot ketidakpastian.
3. Setiap notebook telah dilengkapi penjelasan matematis, kode model, visualisasi grafis, serta interpretasi hasil.
