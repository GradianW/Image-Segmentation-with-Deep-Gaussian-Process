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
* **Agenda**: Pembahasan Regresi Linear sebagai Kasus Khusus GP, Evaluasi Kredit Kelompok, & Arahan Eksplorasi Kernel
* **Poin Pembahasan**:
  - Bimbingan dilaksanakan secara berkelompok. Pada sesi ini, dokumen kredit bimbingan yang dipresentasikan dan dibahas adalah tulisan milik **Nabila Azizah Andien**, sehingga draf kredit bimbingan saya tidak terbahas mendalam secara individual.
  - Pembahasan konsep **Regresi Linear (Bayesian Linear Regression) sebagai salah satu bentuk Gaussian Process**:
    - Pembuktian dan penurunan matematika untuk nilai ekspektasi (*mean function*) dan kovariansi (*covariance function*) dari model regresi linear dalam sudut pandang *function-space* GP:
      - Model: $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x}$ dengan prior bobot $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \Sigma_p)$.
      - Mean: $\mathbb{E}[f(\mathbf{x})] = \mathbb{E}[\mathbf{w}^T \mathbf{x}] = \mathbf{0}^T \mathbf{x} = 0$.
      - Kovariansi: $\operatorname{cov}(f(\mathbf{x}), f(\mathbf{x}')) = \mathbb{E}[f(\mathbf{x}) f(\mathbf{x}')] = \mathbf{x}^T \mathbb{E}[\mathbf{w}\mathbf{w}^T] \mathbf{x}' = \mathbf{x}^T \Sigma_p \mathbf{x}'$ (membuktikan bahwa model linear mendefinisikan GP dengan kernel linear non-stasioner $k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \Sigma_p \mathbf{x}'$).
* **Saran & Masukan Dosen Pembimbing**:
  1. **Eksplorasi Perubahan Rumus & Parameter Kernel**:
     - Melakukan simulasi/eksperimen terhadap perilaku fungsi sampel jika parameter kernel diubah (misalnya variasi *lengthscale* $l$ dan *signal variance* $\sigma_f^2$).
     - Mengamati efek perubahan matematis pada rumus kernel, contohnya jika jarak (*distance*) $r = \|\mathbf{x} - \mathbf{x}'\|$ pada kernel *Squared Exponential* tidak dikuadratkan ($r$ vs $r^2$).
  2. **Pemahaman Sampling Fungsi dari GP (Sampling Function Prior/Posterior)**:
     - Memahami secara konseptual dan komputasional bagaimana cara *generate* / mengambil sampel fungsi dari distribusi Gaussian Process menggunakan matriks kovariansi kernel (misal via dekomposisi Cholesky $K = L L^T$ dan sampling $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, I) \implies \mathbf{f} = L \mathbf{u}$).
* **Tindak Lanjut (To-Do)**:
  - [ ] Memperdalam kembali pembuktian formal penurunan mean dan kovariansi dari Bayesian Linear Regression ke bentuk GP.
  - [ ] Membuat kode eksperimen/visualisasi (Python/Jupyter Notebook) untuk membandingkan sampel fungsi GP dari berbagai konfigurasi parameter kernel dan modifikasi rumus (misal $r$ vs $r^2$).
  - [ ] Memperkuat pemahaman mengenai langkah-langkah *generate* sampel fungsi dari distribusi GP menggunakan dekomposisi Cholesky.
  - [ ] Menyiapkan dokumen kredit bimbingan dan visualisasi untuk pertemuan berikutnya.

---

### Pertemuan #3 - [Hari, Tanggal Bulan Tahun] ([Offline/Online])
* **Agenda**: [Agenda utama pertemuan]
* **Poin Pembahasan**:
  - [Topik / progres yang dipaparkan]
* **Saran & Masukan Dosen Pembimbing**:
  1. [Masukan / koreksi dari dosen]
* **Tindak Lanjut (To-Do)**:
  - [ ] [Tugas / revisi untuk pertemuan berikutnya]


