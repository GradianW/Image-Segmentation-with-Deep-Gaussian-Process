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
  - [x] Membaca dan membedah Subbab 2.1 (*Weight-space View*) dari buku Rasmussen & Williams (2006).
  - [x] Menyusun penurunan analitis lengkap pembuktian distribusi Gaussian untuk $f(\mathbf{x}) = \mathbf{x}^T \mathbf{w}$ dan $y = f(\mathbf{x}) + \varepsilon$ menggunakan MGF (*Moment Generating Function*).
  - [ ] Menuliskan tinjauan perbandingan teoretis antara Gradient Descent dan Adam, serta penjelasan kapan optimasi metaheuristik digunakan dalam konteks machine learning/GP.
  - [x] Menyiapkan skrip Python eksperimen GPR pada dataset deret waktu $\text{CO}_2$ (diuji komparasi model GP 1-layer, Deep GP 2-layer, dan 3-layer pada emisi $\text{CO}_2$ Residential US).
  - [x] Menyusun draf dokumen Kredit Bimbingan #6 (format LaTeX) untuk dibawa pada bimbingan berikutnya.

---

### Catatan Jeda — Pekan Ujian (12 – 16 Oktober 2026)
* **Keterangan**:
  Sesi bimbingan mingguan ditiadakan sementara karena berlangsungnya pekan ujian bagi mahasiswa. Kegiatan asistensi disepakati untuk dilanjutkan kembali pada pekan setelah ujian selesai.

---

### Pertemuan #6 - Rabu, 21 Oktober 2026 (Dibatalkan - Miskomunikasi Jadwal)
* **Agenda**: Pembahasan Dokumen Kredit #6 bersama Dosen Pembimbing Utama (Tidak Terlaksana)
* **Keterangan Kejadian**:
  - Pada hari Rabu, 21 Oktober 2026, saya dan rekan satu topik (Nabila Azizah Andien) datang ke ruangan bimbingan sesuai perkiraan jadwal asistensi rutin dengan membawa draf dokumen Kredit Bimbingan #6.
  - Setelah menunggu beberapa saat di ruangan, Bapak Pembimbing tidak kunjung hadir. Salah satu dari kami kemudian menanyakan konfirmasi kehadiran beliau melalui *group chat*.
  - Dari tanggapan yang diberikan, terkonfirmasi bahwa terjadi **miskomunikasi penentuan jadwal bimbingan**: kami berkeyakinan bahwa jadwal bimbingan dilaksanakan pada hari **Rabu (21 Oktober 2026)**, sedangkan Bapak Dosen Pembimbing meyakini bahwa jadwal yang disepakati adalah hari **Selasa (20 Oktober 2026)**.
  - Akibat perbedaan persepsi hari tersebut, sesi bimbingan bersama Pembimbing Utama pada pekan ini tidak terlaksana.

---

### Pertemuan #7 - Jumat, 23 Oktober 2026 (Konsultasi Ko-Pembimbing - Ibu Maharani)
* **Anggota Kelompok Topik**: Saya & Nabila Azizah Andien (*Topik: Image Segmentation with Deep Gaussian Process*)
* **Agenda**: Konsultasi Progres Tugas Akhir bersama Dosen Ko-Pembimbing (Ibu Maharani)
* **Poin Pembahasan**:
  - Menindaklanjuti tidak terlaksananya bimbingan bersama Pembimbing Utama pada hari Rabu sebelumnya, rekan satu topik (Nabila) berinisiatif menghubungi Ko-Pembimbing, Ibu Maharani, untuk mengajukan jadwal konsultasi bimbingan pada hari Jumat, 23 Oktober 2026.
  - Melaporkan dan mendiskusikan status progres pengerjaan topik tugas akhir secara umum.
  - *Catatan Evaluasi*: Berdasarkan tinjauan pribadi terhadap jalannya diskusi, substansi pembahasan lebih bersifat koordinasi umum progres studi dan tidak menghasilkan masukan teknis baru, koreksi spesifik, maupun poin krusial yang berdampak langsung terhadap pengerjaan materi teoretis GP/DGP yang sedang saya susun.
* **Saran & Masukan Dosen Ko-Pembimbing**:
  1. Tetap menjaga keberlanjutan progres dan segera mengonsolidasikan materi bimbingan teknis bersama Dosen Pembimbing Utama.
  2. Mematangkan kesiapan bahan/kredit yang telah dikerjakan untuk dipaparkan secara komprehensif pada jadwal bimbingan berikutnya.
* **Tindak Lanjut (To-Do)**:
  - [x] Menyesuaikan dan mematangkan draf dokumen Kredit Bimbingan #6 yang sebelumnya telah disiapkan (memperbarui tanggal dan melengkapi materi pendukung).
  - [ ] Melengkapi dokumen kredit dengan ringkasan komparasi algoritma optimasi (Gradient Descent vs Adam & batasan gradien menuju Metaheuristik) yang menjadi to-do dari Pertemuan #5.
  - [ ] Mengintegrasikan konsep teoretis "Neuron" pada DGP serta contoh simulasi perambatan variansi manual (dari suplemen teknis) ke dalam dokumen kredit.
  - [x] Menghadiri bimbingan bersama Dosen Pembimbing Utama pada jadwal yang telah disepakati: **Selasa, 27 Oktober 2026**.

---

### Pertemuan #8 - Selasa, 27 Oktober 2026 (Offline)
* **Anggota Kelompok Topik**: Saya & Nabila Azizah Andien (*Topik: Image Segmentation with Deep Gaussian Process*)
* **Kehadiran**: Bapak Pembimbing (Dosen Pembimbing Utama) & Ibu Maharani (Dosen Ko-Pembimbing)
* **Agenda**: Presentasi Dokumen Kredit Bimbingan #6, Pembagian Spesifikasi Fokus Topik Penelitian, serta Arahan Penjabaran Matematis DNN & Penurunan Backpropagation
* **Poin Pembahasan**:
  - **Presentasi Dokumen Kredit Bimbingan #6**:
    - Sesi bimbingan tatap muka dihadiri lengkap oleh Bapak Pembimbing Utama dan Ibu Ko-Pembimbing (Ibu Maharani).
    - Pada pertemuan kali ini, sesi presentasi berfokus pada pemaparan dokumen `kredit_bimbingan_06.pdf` yang saya bawa (mencakup hasil eksperimen pemodelan deret waktu emisi $\text{CO}_2$ sektor Residential US dengan perbandingan Standard GP 1-Layer, Deep GP 2-Layer, dan 3-Layer; pembuktian MGF distribusi normal univariat dan multivariat pada model $y = f(x) + \varepsilon$ untuk bobot $w$ konstan vs $w$ berdistribusi normal; serta komparasi formulasi arsitektur DNN vs GP vs DGP).
  - **Spesifikasi & Diferensiasi Topik Penelitian (Saya vs Nabila Azizah Andien)**:
    - Terjadi penegasan pembagian fokus dan spesifikasi penelitian tugas akhir antara saya dan rekan satu topik (Nabila Azizah Andien) pada model observasi $y = f + \varepsilon$:
      - **Fokus Penelitian Saya**: Khusus meneliti kasus di mana fungsi laten $f$ berbentuk linier terhadap bobot/fitur, yaitu $f = X^T W$ (atau $f = \mathbf{x}^T \mathbf{w}$ dalam *weight-space view*).
      - **Fokus Penelitian Nabila Azizah Andien**: Meneliti kasus di mana $f$ merupakan fungsi dalam bentuk secara general (*general function-space view* / non-parametrik langsung dengan kernel).
  - **Arahan Pendalaman Matematis DNN & Penurunan Backpropagation**:
    - Dari presentasi yang disampaikan, Bapak Pembimbing memberikan arahan agar untuk pertemuan berikutnya saya menjabarkan bagaimana Deep Neural Network (DNN) bekerja secara matematis formal.
    - Beliau menghendaki agar operasi matriks dari setiap *layer* dan *neuron* dijabarkan secara rinci dan terstruktur (mulai dari dimensi matriks bobot, penambahan bias, perambatan maju/*forward pass*, hingga fungsi aktivasi).
    - Selain itu, beliau juga meminta saya untuk mencoba menuliskan dan menurunkan formulasi algoritma *backpropagation* secara matematis dalam dua kasus:
      1. Penurunan *backpropagation* jika bobot $w$ bernilai konstan/**skalar** (deterministik).
      2. Penurunan *backpropagation* jika bobot $w$ **berdistribusi** (probabilistik / stokastik).
* **Saran & Masukan Dosen Pembimbing**:
  1. **Penjabaran Matematis Operasi Matriks DNN (Layer & Neuron)**:
     - Menguraikan operasi aljabar linier/matriks pada setiap lapisan (*layer*) dan setiap unit neuron secara formal, presisi, dan runut ke dalam dokumen kredit berikutnya.
  2. **Penurunan Analitis Backpropagation (Kasus Bobot Skalar vs Bobot Berdistribusi)**:
     - Menuliskan turunan berantai (*chain rule*) perambatan galat mundur (*backpropagation*) konvensional dengan parameter bobot skalar.
     - Mengeksplorasi dan mencoba menurunkan formulasi pembaruan gradien apabila bobot $w$ diperlakukan sebagai peubah acak yang memiliki distribusi probabilitas.
  3. **Penajaman Landasan Teori Sesuai Ruang Lingkup Spesifik**:
     - Mengarahkan pembahasan teoretis yang disusun agar selaras dengan spesifikasi penelitian tugas akhir saya ($y = f + \varepsilon$ dengan $f = X^T W$).
* **Tindak Lanjut (To-Do)**:
  - [x] Menyusun draf dokumen Kredit Bimbingan #8 yang memuat penjabaran matematis DNN secara mendalam (operasi matriks per layer dan neuron).
  - [x] Menuliskan dan menurunkan rumus analitis *backpropagation* untuk kasus bobot deterministik ($w$ skalar).
  - [ ] Menuliskan penurunan dan eksplorasi *backpropagation* untuk kasus bobot yang berdistribusi probabilitas ($w$ stokastik/Gaussian).
  - [ ] Mematangkan formulasi model spesifik $y = f + \varepsilon$ dengan $f = X^T W$ sebagai fondasi menuju pembentukan kovariansi dan Deep GP.

---

### Pertemuan #9 - Selasa, 03 November 2026 (Offline)
* **Anggota Kelompok Topik**: Saya & Nabila Azizah Andien (*Topik: Image Segmentation with Deep Gaussian Process*)
* **Agenda**: Pemaparan Progres Masing-Masing (Dokumen Kredit #8), Arahan *Reparameterization Trick* untuk Backpropagation Bobot Berdistribusi, Simulasi Arsitektur 2-Layer, & Eksplorasi Fitur Citra
* **Poin Pembahasan**:
  - **Pemaparan Progres Masing-Masing Anggota**:
    - Pada sesi bimbingan ini, saya dan Nabila Azizah Andien masing-masing mempresentasikan progres materi yang telah dipelajari sesuai fokus pembagian topik.
  - **Presentasi Dokumen Kredit Bimbingan #8 (`kredit_bimbingan_08.pdf`)**:
    - Memaparkan formulasi matematis arsitektur *Deep Neural Network* (DNN) secara berjenjang dan terstruktur: mulai dari komputasi tingkat satu neuron (skalar), tingkat satu sampel data (vektor baris $\mathbf{x}_i^{(\ell)} \in \mathbb{R}^{1 \times n_\ell}$), hingga tingkat *batch* untuk $N$ sampel data ($X^{(\ell)} \in \mathbb{R}^{N \times n_\ell}$), serta perumusan fungsi objektif *Mean Squared Error* (MSE) berbasis norma Frobenius.
    - Memaparkan penurunan analitis *Backward Propagation* untuk kasus bobot deterministik ($w$ skalar), mencakup pembuktian turunan fungsi objektif terhadap matriks prediksi $\hat{Y}$, gradien bobot $W^{(L)}$ dan bias $\mathbf{b}^{(L)}$ pada *output layer*, hingga formulasi rekursif gradien pada *hidden layer* ($\ell = L-1, \dots, 2, 1$).
* **Saran & Masukan Dosen Pembimbing**:
  1. **Eksplorasi *Reparameterization* dalam *Deep Gaussian Learning* untuk Backpropagation Bobot Berdistribusi**:
     - Membaca literatur mengenai teknik *reparameterization* (*reparameterization trick*) pada *Deep Gaussian Learning* sebagai solusi/ide untuk melakukan *backpropagation* apabila bobot $w$ merupakan variabel acak yang berdistribusi (misalnya berdistribusi Gaussian $w \sim \mathcal{N}(\mu, \sigma^2)$).
     - Memanfaatkan hubungan transformasi variabel acak baku $z = \frac{w - \mu}{\sigma}$ (atau $w = \mu + \sigma z$ dengan $z \sim \mathcal{N}(0, 1)$) agar aliran gradien pada proses *backpropagation* dapat diturunkan secara terdiferensialkan terhadap parameter distribusi ($\mu$ dan $\sigma$).
  2. **Simulasi Perhitungan Arsitektur 2-Layer Menggunakan Data Dummy**:
     - Menyusun contoh perhitungan konkret (simulasi numerik *forward* dan *backward pass*) menggunakan arsitektur jaringan 2 *layer* dengan data *dummy* pada dua skenario:
       1. Ketika bobot $w$ bernilai **skalar** (deterministik).
       2. Ketika bobot $w$ **berdistribusi** (probabilistik/stokastik).
  3. **Eksplorasi Representasi Fitur pada *Training* Citra (*Image*) dan Alternatifnya**:
     - Mencari tahu dan merumuskan apa yang menjadi fitur input ($\mathbf{x}$) ketika melakukan *training* pada data gambar/citra, serta mengeksplorasi berbagai alternatif representasi fitur yang dapat digunakan (seperti intensitas warna piksel RGB/Grayscale/Lab, koordinat spasial piksel, fitur berbasis *patch*/ketetanggaan lokal, *superpixel*, maupun ekstraksi fitur tekstur/filter).
* **Tindak Lanjut (To-Do)**:
  - [ ] Membaca literatur dan merangkum konsep *reparameterization* dalam *Deep Gaussian Learning* ($z = \frac{w - \mu}{\sigma} \iff w = \mu + \sigma z$) untuk penurunan *backpropagation* pada bobot berdistribusi.
  - [ ] Menurunkan persamaan *backpropagation* terhadap parameter distribusi bobot ($\mu$ dan $\sigma$) menggunakan teknik reparameterisasi.
  - [ ] Membuat contoh perhitungan lengkap dengan arsitektur 2 *layer* menggunakan data *dummy* untuk kasus bobot $w$ skalar dan kasus bobot $w$ berdistribusi.
  - [ ] Mengkaji dan menyusun penjelasan mengenai fitur-fitur yang digunakan saat melakukan *training* gambar (*image*) beserta berbagai alternatifnya.

