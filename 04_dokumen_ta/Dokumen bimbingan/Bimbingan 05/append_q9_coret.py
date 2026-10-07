import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

nb_path = r"c:\Users\MyBook Z Series\Desktop\TA\04_dokumen_ta\Dokumen bimbingan\Bimbingan 05\coret.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Filter out any existing Pertanyaan 9 if present
filtered_cells = []
for cell in nb.cells:
    if cell.cell_type == "markdown" and ("Pertanyaan 9" in cell.source or "Jawaban 9" in cell.source):
        break
    filtered_cells.append(cell)

nb.cells = filtered_cells

md_q9 = r"""# ❓ Pertanyaan 9: Analisis Komparasi Operasi Kernel: Kapan Perkalian ($k_1 \times k_2$) Lebih Bagus dari Penjumlahan ($k_1 + k_2$) dan Sebaliknya?

> **Wah ternyata lebih bagus jika menggunakan operasi perkalian. Kapan hasil perkalian lebih bagus dari penjumlahan dan sebaliknya?**

---

## 💡 Jawaban 9: Analisis Komprehensif Desain & Kombinasi Kernel (*Kernel Engineering*)

Dalam pemodelan Gaussian Process (GP), pemilihan operasi antara **Penjumlahan ($+$)** dan **Perkalian ($\times$)** ditentukan oleh **struktur fisik, hubungan antar-proses laten, dan karakteristik variansi data** (*Rasmussen & Williams, 2006, Bab 4.2.4; Duvenaud, 2014*).

---

### 1. Intuisi Logika Aljabar Kernel: Operasi "OR" vs Operasi "AND"

| Dimensi Komparasi | Operasi Penjumlahan ($k_{\mathrm{sum}} = k_1 + k_2$) | Operasi Perkalian ($k_{\mathrm{prod}} = k_1 \times k_2$) |
| :--- | :--- | :--- |
| **Logika Konseptual** | **Operasi "OR" (Additive Decomposition)** | **Operasi "AND" (Interaction / Modulation)** |
| **Representasi Proses Laten** | Superposisi dua fungsi laten independen:<br>$f(\mathbf{x}) = f_1(\mathbf{x}) + f_2(\mathbf{x})$<br>dengan $f_1 \sim \mathcal{GP}(0, k_1)$ dan $f_2 \sim \mathcal{GP}(0, k_2)$ | Modulasi atau perkalian fungsi laten pada ruang fitur RKHS:<br>$f(\mathbf{x}) \sim \mathcal{GP}(0, k_1 \times k_2)$ |
| **Interpretasi Kovarians** | Dua titik $\mathbf{x}$ dan $\mathbf{x}'$ berkorelasi tinggi jika mereka mirip menurut kernel $k_1$ **ATAU** menurut kernel $k_2$. | Dua titik $\mathbf{x}$ dan $\mathbf{x}'$ hanya berkorelasi tinggi jika mereka mirip menurut kernel $k_1$ **DAN** menurut kernel $k_2$. |
| **Pengaruh Amplitudo** | Amplitudo masing-masing komponen bersifat **konstan/tetap** sepanjang domain. | Komponen pertama bertindak sebagai **amplop (*envelope*)** yang memodifikasi/memodulasi amplitudo komponen kedua. |

---

### 2. Kapan Operasi PERKALIAN ($k_1 \times k_2$) Lebih Bagus?

Operasi perkalian kernel unggul ketika data menunjukkan sifat **modulasi amplitudo**, **peluruhan korelasi jarak jauh (*locally periodic*)**, atau **interaksi silang antar-dimensi fitur**:

#### a. Modulasi Amplitudo Dinamis (*Dynamic Scale / Heteroscedasticity*)
* **Ciri Data**: Fluktuasi osilasi yang amplitudonya membesar atau menyusut seiring tren waktu/posisi (misal: $f(x) = 0{,}2 x \sin(\pi x)$ atau getaran mesin yang amplitudonya membesar saat kecepatan putaran naik).
* **Kernel Terbaik**: $k_{\mathrm{Lin}} \times k_{\mathrm{Per}}$ atau $k_{\mathrm{RBF}} \times k_{\mathrm{Per}}$.
* **Mengapa Menang**: Kernel linier $k_{\mathrm{Lin}}$ mengalikan kovarians periodik, sehingga variansi lokal membesar secara proporsional terhadap jarak dari titik asal ($x$). Penjumlahan ($k_{\mathrm{Lin}} + k_{\mathrm{Per}}$) gagal total karena hanya memiringkan garis tengah tanpa bisa memperbesar ayunan gelombang.

#### b. Osilasi Lokal yang Meredup (*Locally Periodic / Damped Waves*)
* **Ciri Data**: Pola periodik yang bentuk gelombangnya perlahan berubah atau meredup seiring jarak (misal: riak gelombang air, siklus ekonomi makro, fluktuasi cuaca musiman yang bervariasi antar-dekade).
* **Kernel Terbaik**: $k_{\mathrm{RBF}} \times k_{\mathrm{Per}}$.
* **Mengapa Menang**: Karena $k_{\mathrm{RBF}}(\mathbf{x}, \mathbf{x}') \to 0$ saat $|\mathbf{x} - \mathbf{x}'| \to \infty$, perkalian ini memastikan bahwa korelasi antar-titik periodik hanya berlaku secara lokal dan meluruh pada jarak jauh.

#### c. Interaksi Antar-Dimensi Fitur (Fitur Spasio-Temporal)
* **Ciri Data**: Data multi-dimensi di mana pengaruh satu fitur bergantung pada fitur lainnya (misal: suhu udara yang bergantung pada Koordinat Lokasi $\mathbf{x}$ dan Waktu Musim $t$).
* **Kernel Terbaik**: $k_{\mathrm{space}}(\mathbf{x}, \mathbf{x}') \times k_{\mathrm{time}}(t, t')$.
* **Mengapa Menang**: Menerapkan aturan "AND": Titik $A$ dan $B$ hanya memiliki korelasi suhu tinggi jika mereka berada di lokasi berdekatan **DAN** diukur pada waktu musim yang sama.

---

### 3. Kapan Operasi PENJUMLAHAN ($k_1 + k_2$) Lebih Bagus?

Operasi penjumlahan kernel unggul ketika data tersusun dari **superposisi komponen-komponen independen** yang bekerja pada skala atau karakteristik yang berbeda:

#### a. Penguraian Tren Global + Fluktuasi Musiman Stabil (*Additive Time-Series Decomposition*)
* **Ciri Data**: Data deret waktu dengan tren jangka panjang yang bertumpuk dengan pola musiman yang amplitudonya stabil (misal: Konsentrasi gas $\text{CO}_2$ atmosfer di Mauna Loa / *Keeling Curve*).
  * Tren kenaikan $\text{CO}_2$ akibat emisi industri terus meningkat.
  * Fluktuasi musiman tahunan akibat fotosintesis tanaman selalu stabil di kisaran $\pm 3\text{ ppm}$ sepanjang 60 tahun terakhir.
* **Kernel Terbaik**: $k_{\mathrm{RBF}}(\text{tren lambat}) + k_{\mathrm{Per}}(\text{musiman}) + k_{\mathrm{Mat\acute{e}rn}}(\text{anomali menengah})$.
* **Mengapa Menang**: Operasi penjumlahan memisahkan komponen tren dan musiman secara aditif. Jika dipaksakan perkalian ($k_{\mathrm{Lin}} \times k_{\mathrm{Per}}$), model akan keliru memprediksi bahwa variasi fotosintesis tanaman membesar berlipat ganda seiring kenaikan emisi $\text{CO}_2$.

#### b. Struktur Multi-Skala (*Multi-Scale Variations*)
* **Ciri Data**: Sinyal yang memiliki korelasi jarak jauh (tren lambat dengan $\ell_1$ besar) sekaligus riak fluktuasi lokal yang cepat ($\ell_2$ kecil).
* **Kernel Terbaik**: $k_{\mathrm{RBF}}^{(\text{long})} + k_{\mathrm{RBF}}^{(\text{short})}$.
* **Mengapa Menang**: Memungkinkan GP menangkap variasi frekuensi rendah dan frekuensi tinggi secara simultan tanpa meredam salah satunya.

#### c. Ekstrapolasi Periodik Murni Tanpa Peluruhan
* **Ciri Data**: Sistem fisika dengan osilasi abadi (misal: orbit planet atau bandul ideal) yang bertumpuk di atas tren pergeseran koordinat.
* **Mengapa Menang**: $k_{\mathrm{sum}}$ mempertahankan gelombang periodik tak hingga pada daerah ekstrapolasi ($|x| > 10$), sedangkan $k_{\mathrm{prod}}$ dengan RBF akan mematikan gelombang menuju nol.

---

### 4. Diagram Alur Panduan Pemilihan Kernel (*Decision Guide*)

```
                         Struktur Hubungan Antar-Pola pada Data
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
       [ Pola Saling Tergantung / Modulasi ]          [ Pola Independen / Bertumpuk ]
                   │                                             │
         ( OPERASI PERKALIAN: × )                       ( OPERASI PENJUMLAHAN: + )
                   │                                             │
      ┌────────────┴────────────┐                   ┌────────────┴────────────┐
      ▼                         ▼                   ▼                         ▼
Amplitudo membesar       Amplitudo meluruh   Tren Global + Musiman     Variasi Multi-Skala
seiring tren linier      secara lokal        dengan amplitudo tetap    (Panjang + Pendek)
      │                         │                   │                         │
      ▼                         ▼                   ▼                         ▼
[ k_Lin × k_Per ]        [ k_RBF × k_Per ]   [ k_RBF + k_Per ]         [ k_RBF + k_RBF ]
```
"""

nb.cells.append(new_markdown_cell(md_q9))

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Appended Pertanyaan 9 to coret.ipynb.")
