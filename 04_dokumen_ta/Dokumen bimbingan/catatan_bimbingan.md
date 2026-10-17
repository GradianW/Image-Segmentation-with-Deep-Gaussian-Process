# Catatan & Log Bimbingan Tugas Akhir

Dokumen ini digunakan untuk mencatat hasil diskusi, saran, revisi, dan tindak lanjut dari pertemuan bimbingan bersama Dosen Pembimbing.

---

## ℹ️ Informasi Rutin Bimbingan
* **Jadwal Tetap**: Setiap hari Rabu, pukul 09:00 – 11:00 WIB
* **Kewajiban Tiap Sesi**: Membawa "Kredit" (resume konsep, tulisan, atau draft progres berkala)

---

## 📝 Format Template Catatan Bimbingan
*(Salin template ini setiap kali mencatat bimbingan baru di bagian Log Riwayat Bimbingan)*

```markdown
### Pertemuan #[Nomor] - [Hari, Tanggal Bulan Tahun] ([Offline/Online])
* **Agenda**: [Agenda utama pertemuan]
* **Poin Pembahasan**:
  - [Topik / progres yang dipaparkan]
* **Saran & Masukan Dosen Pembimbing**:
  1. [Masukan / koreksi dari dosen]
* **Tindak Lanjut (To-Do)**:
  - [ ] [Tugas / revisi untuk pertemuan berikutnya]
```

---

## 📅 Log Riwayat Bimbingan

### Pertemuan #1 - Rabu, 9 September 2026 (Offline)
* **Agenda**: Diskusi Awal Pemahaman Topik & Arah Tugas Akhir
* **Poin Pembahasan**:
  - Menyampaikan progres awal bahwa sudah mulai masuk mempelajari *Deep Gaussian Process* (DGP) karena merasa telah memahami *Gaussian Process* (GP Regresi).
  - Evaluasi pemahaman: Diminta menjelaskan konsep dasar Gaussian Process secara langsung di hadapan anggota kelompok bimbingan.
  - Berhasil memaparkan konsep secara umum, namun terdapat beberapa catatan/kesalahan pada ketepatan definisi formal (perlu disegarkan kembali karena jeda belajar sekitar 10 hari).
  - Pembahasan judul definitif: Saat ini fokus masih berupa topik umum, paper rujukan, dan buku acuan.
* **Saran & Masukan Dosen Pembimbing**:
  1. **Penguatan Fondasi GP**: Jangan terburu-buru melompat ke DGP sebelum fondasi teoretis, konsep matematis, dan definisi formal dari GP benar-benar matang dan melekat.
  2. **Jadwal Rutin Bimbingan**: Disepakati jadwal asistensi/bimbingan rutin setiap **Rabu, pukul 09:00 – 11:00 WIB**.
  3. **Tugas Kredit / Progress Draft**: Setiap sesi bimbingan diwajibkan membawa "kredit" berupa tulisan, resume materi, atau draft progres berkala.
  4. **Window Shopping Judul**: Terkait judul tugas akhir, dipersilakan melakukan eksplorasi referensi (*window shopping* paper/studi kasus) terlebih dahulu sebelum mendiskusikan judul final yang lebih spesifik.
* **Tindak Lanjut (To-Do)**:
  - [ ] Mereview dan memperdalam kembali definisi formal serta teori dasar Gaussian Process (matematika kernel, prior, posterior, marginal likelihood).
  - [ ] Melakukan *window shopping* membaca paper/literatur terkait penerapan GP/DGP untuk segmentasi citra guna merumuskan kandidat judul.
  - [ ] Menyiapkan dokumen "kredit" (resume konsep/draft progres) untuk dibawa pada bimbingan Rabu pekan depan.

---

### Pertemuan #2 - Rabu, 16 September 2026 (Offline)
* **Anggota Kelompok Topik**: Saya & Nabila Azizah Andien (*Topik: Image Segmentation with Deep Gaussian Process*)
* **Agenda**: Diskusi Kredit Kelompok, Klarifikasi Fokus Tulisan Kredit, & Arahan Eksplorasi Teori Kernel serta Sampling GP
* **Poin Pembahasan**:
  - Bimbingan dilaksanakan secara berkelompok. Pada sesi ini, dokumen kredit bimbingan yang dipresentasikan dan dibahas adalah tulisan milik **Nabila Azizah Andien**, sehingga draf kredit bimbingan saya tidak terbahas mendalam secara individual.
  - Sempat dibahas konsep **Regresi Linear (Bayesian Linear Regression) sebagai salah satu bentuk GP** karena Nabila memasukkannya ke dalam dokumen kreditnya:
    - *Catatan/Klarifikasi*: Dosen menilai materi regresi linear ini berstatus *good to know knowledge*, namun **bukan merupakan fokus utama yang diinginkan**.
    - *Fungsi Hakiki Dokumen Kredit*: Tulisan kredit bimbingan sejatinya disiapkan sebagai **draf bab/landasan teori untuk penulisan laporan Tugas Akhir (skripsi)**. Oleh karena itu, isi kredit harus berfokus langsung pada konsep-konsep inti Gaussian Process, kernel, dan mekanisme inferensi yang menjadi fondasi langsung menuju *Deep Gaussian Process* dan *Image Segmentation*.
* **Saran & Masukan Dosen Pembimbing**:
  1. **Eksplorasi Perubahan Rumus & Parameter Kernel**:
     - Mempelajari dan menganalisis perilaku fungsi sampel jika parameter kernel diubah (misalnya variasi *lengthscale* $l$ dan *signal variance* $\sigma_f^2$).
     - Mengamati efek matematis pada rumus kernel, khususnya perbandingan jarak linier $r = \|\mathbf{x} - \mathbf{x}'\|$ vs jarak kuadratik $r^2 = \|\mathbf{x} - \mathbf{x}'\|^2$ (*Squared Exponential* vs *Exponential/Ornstein-Uhlenbeck* serta kaitannya dengan kelas keterdiferensialan fungsi / keluarga Matérn).
  2. **Pemahaman Mendalam Sampling Fungsi dari GP (Sampling Function Prior/Posterior)**:
     - Memahami secara konseptual, matematis, dan komputasional bagaimana cara membangkitkan (*generate*) sampel fungsi kontinu dari distribusi Gaussian Process menggunakan dekomposisi Cholesky ($K = L L^T \implies \mathbf{f} = \mathbf{m} + L \mathbf{u}$ dengan $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, I)$).
     - Memahami peran *jitter* / regularisasi diagonal ($\epsilon I$) untuk menjaga kestabilan numerik komputasi.
* **Tindak Lanjut (To-Do)**:
  - [ ] Menyusun dokumen kredit bimbingan #3 yang berfokus langsung pada teori inti GP (karakteristik kernel, variasi rumus/parameter, dan sampling fungsi Cholesky) yang dirancang sebagai draf siap pakai untuk Bab Landasan Teori Skripsi.
  - [ ] Membuat skrip Python visualisasi dan eksperimen komparasi fungsi sampel dari berbagai kernel dan variasi hyperparameter.
  - [ ] Menyiapkan dokumen teknis dalam format LaTeX beserta visualisasi pendukung untuk bimbingan berikutnya.

---

### Pertemuan #3 - Rabu, 23 September 2026 (Dibatalkan)
* **Agenda**: Bimbingan Rutin Mingguan (Tidak Terlaksana)
* **Keterangan Kejadian**:
  Pada tanggal 23 September 2026, seharusnya diadakan bimbingan rutin. Tapi, saat saya sudah sampai di kampus, Bapak pembimbing tidak kunjung datang. Teman bimbingan saya, Nabila Azizah Andien, akhirnya mengirimkan pesan ke dosen pembimbing untuk melakukan konfirmasi apakah pada hari itu (23 September 2026) ada bimbingan atau tidak. Beliau pun menjawab dan mengonfirmasi bahwa tidak jadi ada bimbingan karena sedang pergi untuk menghadiri konferensi, dan jadwal bimbingan di-*reschedule* menjadi hari Senin minggu selanjutnya, yaitu 28 September 2026. Sayangnya, karena jadwal saya bentrok, saya tidak dapat menghadiri bimbingan pengganti pada hari Senin tersebut. Selain itu, jadwal bimbingan yang biasanya dilakukan pada hari Rabu, untuk beberapa minggu ke depan akan dilaksanakan pada hari Jumat.

---

### Pertemuan #4 - Jumat, 02 Oktober 2026 (Offline)
* **Agenda**: Pembahasan Dokumen Kredit #4: Optimasi Hyperparameter GP via Marginal Likelihood & Cross-Validation serta Turunannya
* **Poin Pembahasan**:
  - Mempresentasikan dokumen kredit bimbingan #4 mengenai fungsi objektif *Marginal Likelihood*, *Leave-One-Out Cross-Validation* (LOO-CV), serta turunan matematis dari kedua fungsi objektif tersebut berdasarkan Rasmussen & Williams (2006) Bab 5.
  - Memaparkan secara garis besar konsep minimisasi atau maksimisasi fungsi objektif untuk mencari hyperparameter optimal.
  - Diskusi metode penyelesaian analitik vs numerik: Sempat mengira bahwa penentuan parameter optimal mungkin dapat diselesaikan secara analitik dengan menurunkan fungsi objektif dan menyamakannya dengan nol ($\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} = 0$). Namun, Bapak dosen pembimbing meluruskan bahwa fungsi tersebut tidak bisa diselesaikan secara analitik tertutup (*closed-form*) dan harus dikerjakan secara numerik menggunakan *Gradient Descent*.
  - Bapak dosen pembimbing kemudian memberikan penjelasan dan penjabaran konsep *Gradient Descent* secara langsung di papan tulis.
  - Evaluasi pemahaman: Ketika masuk ke pembahasan bagian *Cross-Validation* pada dokumen, Bapak dosen pembimbing bertanya mengenai apa itu konsep dasar *Cross-Validation*, namun saya belum dapat menjawabnya karena kurang membaca dan belum mempelajari materi tersebut secara mendalam.
  - Menunjukkan implementasi kode optimasi hyperparameter dengan memanfaatkan pustaka/package Python (`scipy.optimize.minimize`), namun diakui masih terlalu bergantung pada *package* (*heavily lean on package*) tanpa melakukan eksperimen mendalam pada mekanisme internalnya.
* **Saran & Masukan Dosen Pembimbing**:
  1. **Eksplorasi Penurunan Turunan Fungsi Objektif**:
     - Melakukan eksplorasi lebih lanjut pada turunan fungsi objektif dengan menjabarkannya secara aljabar langkah demi langkah.
     - Menyertakan contoh simulasi perhitungan turunan menggunakan angka konkret agar konsep matematisnya lebih terbayang dan teruji.
  2. **Bedah Mekanisme dan Parameter Fungsi pada Package Optimasi**:
     - Mengulik secara detail fungsi-fungsi optimasi pada *package* yang digunakan (seperti parameter pada fungsi `scipy.optimize.minimize`).
     - Memahami kondisi apa yang membuat iterasi fungsi tersebut berhenti (*stopping criteria*), apakah karena sudah konvergen (selisih nilai/gradien lebih kecil dari batas toleransi `tol`) atau karena telah mencapai batas maksimum iterasi (`maxiter`).
  3. **Penguatan Fondasi Konseptual**:
     - Memperdalam kembali pemahaman konsep dasar mengenai *Cross-Validation* dan metode optimasi berbasis gradien (*Gradient Descent*).
  4. **Eksplorasi Fitting Grafik dengan Komposisi / Gabungan Beberapa Kernel**:
     - Melakukan eksplorasi *curve fitting* pada grafik/data menggunakan kombinasi atau gabungan beberapa fungsi kernel (misalnya penjumlahan $k = k_1 + k_2$ atau perkalian $k = k_1 \times k_2$, seperti kombinasi RBF/SE dengan Periodic kernel, Linear kernel, atau White Noise kernel) untuk memodelkan struktur data yang lebih kompleks (misalnya perpaduan tren linier/halus dengan osilasi periodik).
* **Tindak Lanjut (To-Do)**:
  - [x] Membuat eksplorasi dan penjabaran penurunan turunan fungsi objektif secara aljabar serta dilengkapi contoh perhitungan dengan angka konkret (dimasukkan ke Kredit #5).
  - [x] Membedah dokumentasi dan parameter fungsi pada *package* optimasi (`scipy.optimize.minimize` dan `gpytorch`).
  - [x] Mempelajari kembali konsep dasar algoritma *Gradient Descent* dan *Adam*.
  - [x] Melakukan eksperimen dan visualisasi *fitting* grafik GP dengan komposisi/gabungan beberapa kernel (penjumlahan dan perkalian kernel).

---

### Pertemuan #5 - Jumat, 09 Oktober 2026 (Offline)
* **Anggota Kelompok Topik**: Saya & Nabila Azizah Andien (*Topik: Image Segmentation with Deep Gaussian Process*)
* **Agenda**: Presentasi Dokumen Kredit #5, Diskusi Optimasi (Gradient Descent vs Adam & Metaheuristik), Evaluasi Pemahaman Deep Neural Network (DNN), serta Transisi Menuju *Weight-Space View* & *Deep Gaussian Process* (DGP)
* **Poin Pembahasan**:
  - **Presentasi Dokumen Kredit Bimbingan #5**:
    - Dokumen kredit bimbingan yang dibawa memuat contoh perhitungan numerik turunan *log marginal likelihood* ukuran $2 \times 2$, eksperimen *fitting* fungsi dengan kombinasi kernel (penjumlahan dan perkalian RBF, Periodik, Linier), serta eksplorasi parameter library (`scipy.optimize.minimize` dan `gpytorch`).
    - Konten kredit antara saya dan Nabila Azizah Andien secara umum serupa, dengan pembeda utama pada algoritma optimasi numerik: saya menambahkan implementasi simulasi langkah algoritma **Adam** (*Adaptive Moment Estimation*), sedangkan Nabila hanya menyajikan **Gradient Descent** (GD) konvensional.
  - **Evaluasi Pemahaman Arsitektur Deep Neural Network (DNN)**:
    - Karena pembahasan teknis kredit telah dipaparkan pada sesi Nabila, dosen pembimbing meminta saya secara spontan (tanpa persiapan khusus) untuk menjelaskan konsep dasar arsitektur *Deep Neural Network* (DNN).
    - Berhasil memaparkan konsep dasar DNN (arsitektur multi-layer, fungsi aktivasi, propagasi maju/balik) secara intuitif dan konseptual dengan baik.
    - Terdapat beberapa ketidaktepatan minor saat menyebutkan keselarasan dimensi matriks bobot/bias dan operasi aljabar linier antar-layer (menjadi catatan evaluasi untuk lebih presisi dalam notasi aljabar linier dan representasi tensor).
  - **Transisi Menuju Deep Gaussian Process (DGP)**:
    - Sesi bimbingan 09 Oktober dan sesi lanjutan (21 Oktober) diarahkan sebagai jembatan transisi teoretis dari *Gaussian Process* (GP) standar menuju *Deep Gaussian Process* (DGP).
    - Dosen pembimbing memberikan intermezzo konseptual mendasar: Pada regresi linier klasik (*deterministic linear regression*), parameter bobot bersifat pasti (*fixed/deterministic values*). Sebaliknya, pada kerangka Bayesian dan GP/DGP, bobot serta pemetaan fungsi diperlakukan sebagai variabel acak yang memiliki distribusi probabilitas (*stochastic/probabilistic distributions*).
  - **Diskusi Pendalaman Metode Optimasi (GD vs Adam vs Metaheuristik)**:
    - Dosen pembimbing menyoroti rasionalisasi pemilihan algoritma optimasi: Mengapa menggunakan Adam dibandingkan GD biasa? Apa keunggulan adaptifnya?
    - Ditegaskan bahwa algoritma seperti GD dan Adam bergantung mutlak pada ketersediaan turunan/gradien fungsi objektif (*gradient-based methods*). Apabila fungsi objektif tidak memiliki turunan (*non-differentiable*), berupa *black-box*, atau terputus-putus (*discontinuous*), maka pendekatan yang harus digunakan adalah algoritma **Metaheuristik** (*derivative-free optimization* seperti Genetic Algorithm, PSO, Simulated Annealing, atau Nelder-Mead).
* **Saran & Masukan Dosen Pembimbing**:
  1. **Kajian Teoretis Subbab 2.1 Rasmussen & Williams (Weight-Space View)**:
     - Mempelajari secara mendalam Subbab 2.1 dari buku *Gaussian Processes for Machine Learning* (Rasmussen & Williams, 2006) mengenai sudut pandang ruang bobot (*weight-space view*) pada regresi linier Bayesian.
  2. **Pembuktian Distribusi Menggunakan Moment Generating Function (MGF)**:
     - Menurunkan pembuktian analitis menggunakan *Moment Generating Function* (MGF) untuk membuktikan distribusi dari fungsi linier $f(\mathbf{x}) = \mathbf{x}^T \mathbf{w}$ dan nilai observasi berderau $y = f(\mathbf{x}) + \varepsilon$ ketika prior bobot berdistribusi Gaussian ($\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \Sigma_p)$) dan noise Gaussian ($\varepsilon \sim \mathcal{N}(0, \sigma_n^2)$), serta melihat bagaimana matriks kovariansi $K = X \Sigma_p X^T$ muncul secara natural sebagai jembatan menuju *function-space view*.
  3. **Rasionalisasi Komparasi Adam vs Gradient Descent & Konsep Metaheuristik**:
     - Memasukkan penjelasan komparasi yang komprehensif ke dalam dokumen kredit mengenai alasan pemilihan Adam dibanding GD (mekanisme momentum momen pertama, adaptasi *learning rate* momen kedua/RMSProp, peredaman osilasi, serta penanganan *sparse gradient* & *saddle points*).
     - Menjelaskan batasan metode berbasis gradien dan kapan metode metaheuristik diperlukan.
  4. **Eksperimen Pemodelan GP pada Dataset $\text{CO}_2$ (Mauna Loa)**:
     - Mencoba mengimplementasikan Gaussian Process Regression pada dataset konsentrasi $\text{CO}_2$ atmosfer Mauna Loa (*Keeling Curve*) sebagaimana dibahas pada buku Rasmussen & Williams Bab 5 Subbab 5.4.3 untuk membuktikan keandalan kombinasi berbagai kernel (tren jangka panjang, variasi musiman periodik, ketidakteraturan jangka menengah, dan noise putih).
* **Tindak Lanjut (To-Do)**:
  - [ ] Membaca dan membedah Subbab 2.1 (*Weight-space View*) dari buku Rasmussen & Williams (2006).
  - [ ] Menyusun penurunan analitis lengkap pembuktian distribusi Gaussian untuk $f(\mathbf{x}) = \mathbf{x}^T \mathbf{w}$ dan $y = f(\mathbf{x}) + \varepsilon$ menggunakan MGF (*Moment Generating Function*).
  - [ ] Menuliskan tinjauan perbandingan teoretis antara Gradient Descent dan Adam, serta penjelasan kapan optimasi metaheuristik digunakan dalam konteks machine learning/GP.
  - [ ] Menyiapkan skrip Python eksperimen GPR pada dataset deret waktu Mauna Loa $\text{CO}_2$ dengan rancangan komposisi kernel sesuai Bab 5.4.3 Rasmussen.
  - [ ] Menyusun dokumen Kredit Bimbingan #6 (format LaTeX) untuk dibawa pada bimbingan berikutnya (Rabu, 21 Oktober 2026).

