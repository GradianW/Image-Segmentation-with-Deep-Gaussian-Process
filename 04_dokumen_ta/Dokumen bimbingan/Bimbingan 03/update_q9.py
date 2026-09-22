import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nb_path = r'c:\Users\MyBook Z Series\Desktop\TA\04_dokumen_ta\Dokumen bimbingan\Bimbingan 03\coret2.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f_in:
    nb = json.load(f_in)

q9_md = """---
## Pertanyaan 9: Bagaimana Cara Mengombinasikan Dua atau Lebih Kernel Berdasarkan Rasmussen & Williams (2006, Section 4.2.4)?

### 1. Landasan Teoretis Operasi Kernel
Menurut **Rasmussen & Williams (2006, Bab 4, hlm. 89–91)**, dalam pemodelan data dunia nyata yang kompleks, satu fungsi kovariansi dasar sering kali tidak memadai untuk menangkap seluruh variasi data (misalnya kombinasi tren jangka panjang, variasi musiman berulang, dan fluktuasi derau lokal).

Kita dapat mengonstruksi keluarga kernel baru yang **valid dan definit positif** melalui operasi aljabar:

1. **Penjumlahan Kernel ($k_{\\text{sum}} = k_1 + k_2$)**:
   - **Makna Fisis**: Superposisi dari dua proses Gaussian acak yang saling independen:
     $$ f(\\mathbf{x}) = f_1(\\mathbf{x}) + f_2(\\mathbf{x}), \\quad \\text{dengan } f_1 \\sim \\mathcal{GP}(0, k_1), \\; f_2 \\sim \\mathcal{GP}(0, k_2) $$
   - **Logika**: Operator **"OR"** / Dekomposisi Multi-skala.
   - **Karakteristik**: Menggabungkan struktur aditif, seperti tren global lambat ($k_{\\text{SE}}$ ber-$l$ besar) ditambah fluktuasi detail cepat ($k_{\\text{Matérn}}$ ber-$l$ kecil), atau tren sekuler linier ditambah komponen musiman ($k_{\\text{Linear}} + k_{\\text{Periodic}}$).

2. **Perkalian Kernel ($k_{\\text{prod}} = k_1 \\times k_2$)**:
   - **Makna Fisis**: Interaksi simultan / Modulasi Selubung Amplitudo (*Envelope Modulation*):
     $$ f(\\mathbf{x}) = f_1(\\mathbf{x}) \\cdot f_2(\\mathbf{x}) $$
   - **Logika**: Operator **"AND"** (korelasi antartitik hanya bernilai tinggi jika *kedua* kernel bernilai tinggi).
   - **Keabsahan Matematis**: Dijamin oleh **Schur Product Theorem** (perkalian elemen-demi-elemen / Hadamard product dari dua matriks kovariansi PSD menghasilkan matriks yang tetap PSD).
   - **Karakteristik**: Menghasilkan fenomena variasi lokal, misalnya *Locally Periodic Kernel* ($k_{\\text{Periodic}} \\times k_{\\text{SE}}$) di mana osilasi periodik yang kaku termodulasi sehingga bentuk gelombang atau amplitudonya bermutasi/meluruh seiring jarak spasial.

3. **Penskalaan Skalar dan Pemetaan Fitur Deterministik**:
   - Penskalaan positif: $a \\cdot k_1(\\mathbf{x}, \\mathbf{x}')$ untuk konstanta $a > 0$.
   - Pemetaan deterministik / non-linier: $k_1(\\phi(\\mathbf{x}), \\phi(\\mathbf{x}'))$ untuk sembarang pemetaan fitur $\\phi: \\mathcal{X} \\to \\mathcal{V}$.

---
### 2. Eksperimen Komputasi & Visualisasi Sampel Kombinasi Kernel
Di bawah ini disimulasikan 4 konfigurasi kombinasi 2 kernel:
1. **Superposisi Aditif 1**: $k_{\\text{SE}}(l=3.0) + k_{\\text{Matérn 3/2}}(l=0.3)$ (Tren Lambat Global + Fluktuasi Cepat Lokal).
2. **Superposisi Aditif 2**: $k_{\\text{Linear}} + k_{\\text{Periodic}}$ (Tren Pertumbuhan Linier + Osilasi Musiman).
3. **Modulasi Multiplikatif 1**: $k_{\\text{Periodic}}(p=1.2, l=1.0) \\times k_{\\text{SE}}(l=3.5)$ (*Locally Periodic* / Pembusukan Periodisitas Jarak Jauh).
4. **Modulasi Multiplikatif 2**: $k_{\\text{Linear}} \\times k_{\\text{Periodic}}$ (Osilasi Periodik dengan Amplitudo yang Membesar Proporsional)."""

q9_code = """# ==============================================================================
# Demonstrasi Komputasi & Visualisasi Kombinasi Kernel (Rasmussen & Williams 2006)
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt

# Definisi Fungsi-Fungsi Kernel Dasar
def k_se(x1, x2, l=1.0, sf=1.0):
    d = np.subtract.outer(x1, x2)
    return (sf**2) * np.exp(-0.5 * (d / l)**2)

def k_matern32(x1, x2, l=1.0, sf=1.0):
    r = np.abs(np.subtract.outer(x1, x2))
    factor = np.sqrt(3.0) * r / l
    return (sf**2) * (1.0 + factor) * np.exp(-factor)

def k_periodic(x1, x2, p=1.0, l=1.0, sf=1.0):
    d = np.abs(np.subtract.outer(x1, x2))
    sin_term = np.sin(np.pi * d / p)
    return (sf**2) * np.exp(-2.0 * (sin_term / l)**2)

def k_linear(x1, x2, c=0.0, sb=0.1, sv=0.5):
    return sb**2 + (sv**2) * np.multiply.outer(x1 - c, x2 - c)

def sample_gp(K, n_samples=3, jitter=1e-7, seed=42):
    np.random.seed(seed)
    N = K.shape[0]
    K_stab = K + jitter * np.eye(N)
    L = np.linalg.cholesky(K_stab)
    u = np.random.randn(N, n_samples)
    return L @ u

# Grid Evaluasi
x_grid = np.linspace(-4, 4, 300)

# 1. Kasus Penjumlahan 1: SE (l=3.0) + Matern3/2 (l=0.3)
K_se_slow = k_se(x_grid, x_grid, l=3.0, sf=1.0)
K_mat_fast = k_matern32(x_grid, x_grid, l=0.3, sf=0.3)
K_sum1 = K_se_slow + K_mat_fast

# 2. Kasus Penjumlahan 2: Linear + Periodic
K_lin = k_linear(x_grid, x_grid, c=-4.0, sb=0.05, sv=0.4)
K_per = k_periodic(x_grid, x_grid, p=1.5, l=1.0, sf=0.6)
K_sum2 = K_lin + K_per

# 3. Kasus Perkalian 1: Periodic * SE (Locally Periodic)
K_per_pure = k_periodic(x_grid, x_grid, p=1.2, l=1.0, sf=1.0)
K_se_env = k_se(x_grid, x_grid, l=3.5, sf=1.0)
K_prod1 = K_per_pure * K_se_env

# 4. Kasus Perkalian 2: Linear * Periodic (Growing Amplitude)
K_lin_grow = k_linear(x_grid, x_grid, c=-4.0, sb=0.1, sv=0.5)
K_prod2 = K_lin_grow * K_per_pure

# Plotting 4 Kasus Kombinasi
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle(r'Eksplorasi Kombinasi Dua Kernel Berdasarkan Rasmussen & Williams (2006, Section 4.2.4)', fontsize=15, fontweight='bold', y=0.98)

# Panel (a): Penjumlahan 1
samples_sum1 = sample_gp(K_sum1, n_samples=3, seed=101)
std_sum1 = np.sqrt(np.diag(K_sum1))
axes[0, 0].fill_between(x_grid, -2*std_sum1, 2*std_sum1, color='tab:blue', alpha=0.15, label=r'Uncertainty Band $\\pm 2\\sigma(x)$')
for i in range(3):
    axes[0, 0].plot(x_grid, samples_sum1[:, i], lw=2.0, label=f'Sampel $f_{{sum}}^{{({i+1})}}(x)$')
axes[0, 0].set_title(r'(a) Penjumlahan: $k_{\\mathrm{SE}}(l=3.0) + k_{\\mathrm{Mat\\acute{e}rn\\,3/2}}(l=0.3)$' + '\\n' + r'[Superposisi: Tren Mulus Lambat + Fluktuasi Cepat Lokal]', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('x', fontsize=10)
axes[0, 0].set_ylabel('f(x)', fontsize=10)
axes[0, 0].grid(True, linestyle=':', alpha=0.6)
axes[0, 0].legend(loc='upper right', fontsize=8.5)

# Panel (b): Penjumlahan 2
samples_sum2 = sample_gp(K_sum2, n_samples=3, seed=202)
std_sum2 = np.sqrt(np.diag(K_sum2))
axes[0, 1].fill_between(x_grid, -2*std_sum2, 2*std_sum2, color='tab:green', alpha=0.15, label=r'Uncertainty Band $\\pm 2\\sigma(x)$')
for i in range(3):
    axes[0, 1].plot(x_grid, samples_sum2[:, i], lw=2.0, label=f'Sampel $f_{{sum}}^{{({i+1})}}(x)$')
axes[0, 1].set_title(r'(b) Penjumlahan: $k_{\\mathrm{Linear}}(c=-4, \\sigma_v=0.4) + k_{\\mathrm{Periodic}}(p=1.5, l=1.0)$' + '\\n' + r'[Superposisi: Tren Pertumbuhan Global + Osilasi Musiman]', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('x', fontsize=10)
axes[0, 1].set_ylabel('f(x)', fontsize=10)
axes[0, 1].grid(True, linestyle=':', alpha=0.6)
axes[0, 1].legend(loc='upper left', fontsize=8.5)

# Panel (c): Perkalian 1
samples_prod1 = sample_gp(K_prod1, n_samples=3, seed=303)
std_prod1 = np.sqrt(np.diag(K_prod1))
axes[1, 0].fill_between(x_grid, -2*std_prod1, 2*std_prod1, color='tab:purple', alpha=0.15, label=r'Uncertainty Band $\\pm 2\\sigma(x)$')
for i in range(3):
    axes[1, 0].plot(x_grid, samples_prod1[:, i], lw=2.0, label=f'Sampel $f_{{prod}}^{{({i+1})}}(x)$')
axes[1, 0].set_title(r'(c) Perkalian: $k_{\\mathrm{Periodic}}(p=1.2) \\times k_{\\mathrm{SE}}(l=3.5)$ (Locally Periodic)' + '\\n' + r'[Modulasi: Osilasi Periodik Bermutasi/Meluruh Seiring Jarak]', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('x', fontsize=10)
axes[1, 0].set_ylabel('f(x)', fontsize=10)
axes[1, 0].grid(True, linestyle=':', alpha=0.6)
axes[1, 0].legend(loc='upper right', fontsize=8.5)

# Panel (d): Perkalian 2
samples_prod2 = sample_gp(K_prod2, n_samples=3, seed=404)
std_prod2 = np.sqrt(np.diag(K_prod2))
axes[1, 1].fill_between(x_grid, -2*std_prod2, 2*std_prod2, color='tab:red', alpha=0.15, label=r'Uncertainty Band $\\pm 2\\sigma(x)$')
for i in range(3):
    axes[1, 1].plot(x_grid, samples_prod2[:, i], lw=2.0, label=f'Sampel $f_{{prod}}^{{({i+1})}}(x)$')
axes[1, 1].set_title(r'(d) Perkalian: $k_{\\mathrm{Linear}}(c=-4, \\sigma_v=0.5) \\times k_{\\mathrm{Periodic}}(p=1.2)$' + '\\n' + r'[Modulasi: Amplitudo Gelombang Membesar Mengikuti Garis Linier]', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('x', fontsize=10)
axes[1, 1].set_ylabel('f(x)', fontsize=10)
axes[1, 1].grid(True, linestyle=':', alpha=0.6)
axes[1, 1].legend(loc='upper left', fontsize=8.5)

plt.tight_layout()
plt.subplots_adjust(top=0.91)
plt.savefig('q9_kernel_operations.png', dpi=300, bbox_inches='tight')
plt.show()
print('Demonstrasi operasi 2 kernel sukses disimulasikan dan divisualisasikan.')
"""

has_q9 = any('Pertanyaan 9' in ''.join(c.get('source', [])) for c in nb['cells'])

if not has_q9:
    nb['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in q9_md.split('\n')]
    })
    nb['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in q9_code.split('\n')]
    })
else:
    for i, c in enumerate(nb['cells']):
        if 'Pertanyaan 9' in ''.join(c.get('source', [])):
            nb['cells'][i]['source'] = [line + '\n' for line in q9_md.split('\n')]
            if i + 1 < len(nb['cells']) and nb['cells'][i+1]['cell_type'] == 'code':
                nb['cells'][i+1]['source'] = [line + '\n' for line in q9_code.split('\n')]
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Updated coret2.ipynb successfully with Pertanyaan 9!')
