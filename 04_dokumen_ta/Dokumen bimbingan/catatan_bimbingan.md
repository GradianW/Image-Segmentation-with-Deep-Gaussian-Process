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

### Pertemuan #3 - Rabu, 23 September 2026 (Offline)
* **Agenda**: Pembahasan Dokumen Kredit #3: Karakteristik Fungsi Kernel, Efek Hyperparameter, dan Mekanisme Sampling Fungsi GP via Dekomposisi Cholesky
* **Poin Pembahasan**:
  - [Menyampaikan draf landasan teori skripsi mengenai eksplorasi kernel dan sampling fungsi GP]
  - [Mendiskusikan hasil visualisasi komparasi kernel r vs r^2, Matérn, serta kestabilan numerik dekomposisi Cholesky]
* **Saran & Masukan Dosen Pembimbing**:
  1. [Masukan / koreksi dari dosen]
* **Tindak Lanjut (To-Do)**:
  - [ ] [Tugas / revisi untuk pertemuan berikutnya]



