# Teks Siap Salin ke Canva: Tujuan, Pendekatan, & Referensi

---

## 1. TUJUAN PENELITIAN

### 📋 Opsi Singkat & Langsung (1 Kalimat Inti):
```text
TUJUAN PENELITIAN

Menggunakan model Deep Gaussian Process (DGP) dengan fungsi laten berbentuk linier terhadap bobot f(x) = xw (pada model observasi y = f(x) + ε) untuk melakukan segmentasi citra (image segmentation).
```

---

## 2. PENDEKATAN / METODE YANG DIGUNAKAN

### 📋 Opsi A — Format Poin Ringkas:
```text
PENDEKATAN & METODE

1. Formulasi Model Dasar (Weight-Space View)
   Mengkaji model observasi y = f(x) + ε dengan fungsi laten f(x) = xw (atau F = XW), serta membandingkan sifat matematis ketika bobot w bernilai skalar konstan (deterministik) terhadap ketika bobot w berdistribusi Normal (probabilistik).

2. Perluasan ke Arsitektur Berlapis (Deep Gaussian Process)
   Menurunkan operasi matriks perambatan maju (forward pass) dan perambatan balik (backward propagation) dari jaringan berlapis deterministik (DNN) menuju komposisi fungsi berlapis probabilistik (Deep Gaussian Process).

3. Optimasi Parameter & Penerapan pada Segmentasi Citra
   Mengoptimasi parameter/hyperparameter model secara numerik berbasis gradien (Gradient Descent / Adam) dan menerapkannya pada representasi fitur piksel citra untuk mengelompokkan segmen objek beserta estimasi ketidakpastiannya.
```

### 📋 Opsi B — Format Paragraf Ringkas (Jika ingin senada dengan Latar Belakang):
```text
PENDEKATAN & METODE

Pendekatan yang digunakan diawali dengan mengkaji model observasi y = f(x) + ε menggunakan fungsi laten linier f(x) = xw (atau F = XW), baik ketika bobot w dipandang sebagai nilai skalar konstan maupun ketika bobot w berdistribusi Normal. Selanjutnya, dirumuskan mekanisme perambatan maju (forward pass) dan perambatan gradien mundur (backward propagation) secara berlapis dari arsitektur Deep Neural Network (DNN) menuju komposisi berlapis Deep Gaussian Process (DGP) dengan optimasi numerik berbasis gradien (Gradient Descent / Adam), untuk kemudian diterapkan pada ekstraksi fitur piksel guna menyelesaikan tugas segmentasi citra.
```

---

## 3. REFERENSI (DAFTAR PUSTAKA)

### 📋 Teks untuk Disalin ke Slide Referensi:
```text
REFERENSI

[1] C. E. Rasmussen and C. K. I. Williams, "Gaussian Processes for Machine Learning," Cambridge, MA: The MIT Press, 2006.

[2] A. Damianou and N. D. Lawrence, "Deep Gaussian Processes," in Proceedings of the 16th International Conference on Artificial Intelligence and Statistics (AISTATS), vol. 31, pp. 207–215, PMLR, 2013.

[3] R. Salinas-Gutiérrez, A. Hernández-Aguirre, M. J. J. Rivera-Meraz, and E. R. Villa-Diharce, "Using Gaussian Copulas in Supervised Probabilistic Classification," in Soft Computing for Intelligent Control and Mobile Robotics (Studies in Computational Intelligence, vol. 318), Berlin, Heidelberg: Springer, pp. 355–372, 2011.
```
*(Catatan: Jika ingin format APA Style, tersedia juga di bawah)*

### 📋 Alternatif Format APA Style:
```text
REFERENSI

• Damianou, A., & Lawrence, N. D. (2013). Deep Gaussian processes. In Proceedings of the Sixteenth International Conference on Artificial Intelligence and Statistics (AISTATS) (Vol. 31, pp. 207–215). PMLR.

• Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes for machine learning. The MIT Press.

• Salinas-Gutiérrez, R., Hernández-Aguirre, A., Rivera-Meraz, M. J. J., & Villa-Diharce, E. R. (2011). Using Gaussian copulas in supervised probabilistic classification. In Soft Computing for Intelligent Control and Mobile Robotics (pp. 355–372). Springer.
```
