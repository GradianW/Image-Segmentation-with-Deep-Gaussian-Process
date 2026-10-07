import nbformat
from nbformat.v4 import new_markdown_cell

nb_path = r"c:\Users\MyBook Z Series\Desktop\TA\04_dokumen_ta\Dokumen bimbingan\Bimbingan 05\coret.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Check if Pertanyaan 10 already exists, if so remove it
cells = []
for cell in nb.cells:
    if cell.cell_type == "markdown" and ("Pertanyaan 10" in cell.source or "Jawaban 10" in cell.source):
        continue
    cells.append(cell)

q10_content = """# ❓ Pertanyaan 10: Mengapa Rumus Kernel Periodik di Dokumen/PDF Berbeda dengan Buku Rasmussen & Williams Halaman 92 Persamaan (4.31)?

> **Kenapa rumus kernel periodik di pdf berbeda dengan buku rasmussen halaman 92 (4.31)?**

---

## 💡 Jawaban 10: Perbandingan Formulasi Kernel Periodik

Perbedaan antara rumus kernel periodik pada **Buku Rasmussen & Williams (2006) hlm. 92 Persamaan (4.31)** dengan **Dokumen PDF / implementasi praktis** terletak pada **generalisasi periode ($p$)** dan **parameter variansi output ($\sigma_p^2$)**.

---

### 1. Perbandingan Formula Matematis

* **Buku Rasmussen & Williams (2006) hlm. 92, Persamaan (4.31):**
  $$k(x, x') = \\exp\\left( -\\frac{2\\sin^2\\left(\\frac{x - x'}{2}\\right)}{\\ell^2} \\right)$$

* **Dokumen PDF / Implementasi Pustaka GP (GPy, GPflow, Scikit-learn):**
  $$k_{\\mathrm{Per}}(x, x') = \\sigma_p^2 \\exp\\left( -\\frac{2\\sin^2\\left(\\frac{\\pi |x - x'|}{p}\\right)}{\\ell_{\\mathrm{Per}}^2} \\right)$$

---

### 2. Tiga Perbedaan Utama & Penjelasannya

#### A. Periode Arbitrer ($p$) vs Periode Standar ($2\\pi$)
1. **Penurunan di Buku Rasmussen (hlm. 92):**
   Rasmussen mengonstruksi kernel periodik 1D dengan memetakan input $x \\in \\mathbb{R}$ ke ruang 2D pada lingkaran satuan (*unit circle*):
   $$\\mathbf{u}(x) = \\begin{bmatrix} \\cos(x) \\\\ \\sin(x) \\end{bmatrix}$$
   Karena fungsi trigonometri $\\cos(x)$ dan $\\sin(x)$ memiliki periode alami $2\\pi$, pemetaan ini secara implisit **mengasumsikan periode bernilai tetap $p = 2\\pi$**.
   
   Jarak Euclidean kuadrat di ruang lingkaran $\\mathbf{u}$ adalah:
   $$\\begin{aligned}
   \\|\\mathbf{u}(x) - \\mathbf{u}(x')\\|^2 &= (\\cos x - \\cos x')^2 + (\\sin x - \\sin x')^2 \\\\
   &= (\\cos^2 x + \\sin^2 x) + (\\cos^2 x' + \\sin^2 x') - 2(\\cos x \\cos x' + \\sin x \\sin x') \\\\
   &= 1 + 1 - 2\\cos(x - x') \\\\
   &= 2 - 2\\cos(x - x') \\\\
   &= 4 \\sin^2\\left(\\frac{x - x'}{2}\\right)
   \\end{aligned}$$
   
   Ketika jarak kuadrat ini dimasukkan ke dalam fungsi kovariansi *Squared Exponential* (RBF) 2D $k(\\mathbf{u}, \\mathbf{u}') = \\exp\\left(-\\frac{\\|\\mathbf{u} - \\mathbf{u}'\\|^2}{2\\ell^2}\\right)$, diperoleh:
   $$k(x, x') = \\exp\\left( -\\frac{4\\sin^2\\left(\\frac{x - x'}{2}\\right)}{2\\ell^2} \\right) = \\exp\\left( -\\frac{2\\sin^2\\left(\\frac{x - x'}{2}\\right)}{\\ell^2} \\right) \\quad \\text{(Persamaan 4.31)}$$

2. **Generalisasi pada Dokumen PDF:**
   Dalam aplikasi nyata, fenomena berulang memiliki panjang periode sembarang $p > 0$ (misalnya $p = 2{,}0$ pada eksperimen bimbingan, $p = 12$ untuk siklus bulanan, $p = 24$ untuk siklus jam harian).
   Untuk mengakomodasi periode $p$, domain input di-*rescale* menjadi $\\frac{2\\pi x}{p}$, sehingga pemetaannya menjadi:
   $$\\mathbf{u}(x) = \\begin{bmatrix} \\cos\\left(\\frac{2\\pi x}{p}\\right) \\\\ \\sin\\left(\\frac{2\\pi x}{p}\\right) \\end{bmatrix}$$
   Jarak kuadratnya menjadi:
   $$\\|\\mathbf{u}(x) - \\mathbf{u}(x')\\|^2 = 4 \\sin^2\\left( \\frac{\\frac{2\\pi x}{p} - \\frac{2\\pi x'}{p}}{2} \\right) = 4 \\sin^2\\left( \\frac{\\pi (x - x')}{p} \\right)$$
   
   Jika disubstitusikan $p = 2\\pi$:
   $$\\frac{\\pi (x - x')}{2\\pi} = \\frac{x - x'}{2}$$
   **Artinya, Persamaan (4.31) di buku Rasmussen adalah kasus khusus ketika $p = 2\\pi$, sedangkan rumus di PDF adalah bentuk umum (generalisasi) untuk sembarang periode $p$.**

---

#### B. Parameter Variansi Sinyal Output ($\\sigma_p^2$)
* **Buku Rasmussen (Eq. 4.31):** Ditulis dalam bentuk dasar (*normalized/unit variance*), di mana amplitudo kovariansi maksimum diatur bernilai 1.
* **Dokumen PDF:** Menambahkan hiperparameter $\\sigma_p^2 > 0$ sebagai pengali variansi sinyal output (*signal variance*), agar GP dapat memodelkan skala vertikal data yang amplitudonya tidak harus bernilai 1.

---

#### C. Tanda Nilai Mutlak $|x - x'|$ vs $(x - x')$
Karena fungsi sinus memiliki sifat ganjil $\\sin(-\\theta) = -\\sin(\\theta)$, maka saat dikuadratkan:
$$\\sin^2(-\\theta) = [-\\sin(\\theta)]^2 = \\sin^2(\\theta) = \\sin^2(|\\theta|)$$
Oleh karena itu:
$$\\sin^2\\left(\\frac{\\pi |x - x'|}{p}\\right) \\equiv \\sin^2\\left(\\frac{\\pi (x - x')}{p}\\right)$$
Kedua notasi menghasilkan nilai yang **persis sama secara aljabar dan numerik**. Notasi $|x - x'|$ digunakan di PDF untuk menegaskan bahwa kernel periodik ini tergolong kernel berbasis jarak stasioner (*distance-based stationary metric*).

---

### 3. Tabel Ringkasan Komparasi

| Komponen | Buku Rasmussen (2006) Eq. (4.31) | Dokumen PDF / Praktis | Keterangan |
| :--- | :--- | :--- | :--- |
| **Periode ($p$)** | $p = 2\\pi$ (implisit/tetap) | $p > 0$ (eksplisit & fleksibel) | PDF merupakan generalisasi untuk sembarang periode. |
| **Variansi Output ($\sigma^2$)** | Tersirat 1 (dasar) | $\\sigma_p^2$ (hiperparameter) | Mengatur skala amplitudo vertikal data. |
| **Argumen Sinus** | $\\sin\\left(\\frac{x - x'}{2}\\right)$ | $\\sin\\left(\\frac{\\pi \\|x - x'\\|}{p}\\right)$ | Identik ketika $p = 2\\pi$. |
| **Sifat Nilai Mutlak** | $(x - x')$ | $\\|x - x'\\|$ | Nilai kuadrat $\\sin^2(\\cdot)$ selalu identik. |

$$\\underbrace{\\sigma_p^2 \\exp\\left( -\\frac{2\\sin^2\\left(\\frac{\\pi |x - x'|}{p}\\right)}{\\ell_{\\mathrm{Per}}^2} \\right)}_{\\text{Rumus PDF}} \\xrightarrow{p = 2\\pi, \\; \\sigma_p^2 = 1} \\underbrace{\\exp\\left( -\\frac{2\\sin^2\\left(\\frac{x - x'}{2}\\right)}{\\ell^2} \\right)}_{\\text{Rasmussen (4.31)}}$$
"""

cells.append(new_markdown_cell(q10_content))
nb.cells = cells

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Successfully appended Pertanyaan 10 to coret.ipynb")
