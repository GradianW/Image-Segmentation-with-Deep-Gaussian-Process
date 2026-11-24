# Folder Penulisan Buku Skripsi (Laporan Tugas Akhir)

Folder ini digunakan khusus untuk menyimpan naskah utama buku skripsi beserta aset gambar, daftar pustaka, dan lampiran.

## Struktur Folder (Rekomendasi LaTeX)

- **`main.tex`** *(tambahkan saat mulai menulis)* — Dokumen utama LaTeX (preamble, template/format kampus, serta pemanggilan `\include` untuk setiap bab).
- **`references.bib`** *(tambahkan saat mulai menulis)* — File daftar pustaka format BibTeX.
- **`bab/`** — Menyimpan file `.tex` terpisah per bab (misalnya `bab1_pendahuluan.tex`, `bab2_tinjauan_pustaka.tex`, `bab3_metodologi.tex`, `bab4_hasil_dan_pembahasan.tex`, `bab5_kesimpulan.tex`).
- **`figures/`** — Menyimpan gambar, grafik, diagram, dan hasil visualisasi yang digunakan di dalam buku skripsi.
- **`lampiran/`** — Menyimpan materi lampiran tambahan (misalnya penurunan matematis detail, tabel tambahan, atau cuplikan kode).

## Catatan Jika Menggunakan Microsoft Word

Jika buku skripsi ditulis menggunakan Microsoft Word (`.docx`), Anda dapat meletakkan file `.docx` utama langsung di dalam folder `buku_skripsi/` ini dan tetap menggunakan subfolder **`figures/`** serta **`lampiran/`** untuk merapikan aset pendukung.
