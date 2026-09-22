import json
import io
import base64
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nb_path = r'c:\Users\MyBook Z Series\Desktop\TA\04_dokumen_ta\Dokumen bimbingan\Bimbingan 03\eksplorasi_kernel_dan_sampling.ipynb'
OUTPUT_DIR = r'c:\Users\MyBook Z Series\Desktop\TA\04_dokumen_ta\Dokumen bimbingan\Bimbingan 03\figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open('fig1_src.py', 'r', encoding='utf-8') as f:
    fig1_code = f.readlines()

with open('fig2_src.py', 'r', encoding='utf-8') as f:
    fig2_code = f.readlines()

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 4 (Markdown)
nb['cells'][4]['source'] = [
    "## Eksplorasi 1: Karakteristik Nilai Kovariansi dan Realisasi Sampel Fungsi Prior (Rasmussen & Williams, 2006)\n",
    "\n",
    "Berikut adalah visualisasi komparatif antara fungsi kovariansi $k(r)$ terhadap jarak Euclidean $r = \\|\\mathbf{x} - \\mathbf{x}^\\prime\\|$ berdampingan dengan realisasi sampel fungsi prior $f(x)$ yang dibangkitkan dari vektor noise baku $\\mathbf{u}$ yang identik.\n"
]

# Cell 5 (Code for Fig 1)
nb['cells'][5]['source'] = fig1_code

# Cell 6 (Markdown)
nb['cells'][6]['source'] = [
    "## Eksplorasi 2: Galeri Realisasi Sampel Fungsi Prior $\\mathbf{f} \\sim \\mathcal{GP}(0, K)$\n",
    "\n",
    "Di bawah ini disajikan galeri kurva sampel fungsi acak yang dibangkitkan dari 6 kelas kernel baku dari buku teks Rasmussen & Williams:\n",
    "1. **Squared Exponential (SE)**: Kurva sangat mulus.\n",
    "2. **Exponential / Matérn 1/2**: Kurva kasar bergerigi.\n",
    "3. **Matérn 3/2**: Kurva dengan kehalusan teratur untuk fenomena fisik realistis.\n",
    "4. **Matérn 5/2**: Kurva lebih mulus dari Matérn 3/2.\n",
    "5. **Rational Quadratic (RQ)**: Memodelkan variasi multi-skala.\n",
    "6. **Linear**: Menghasilkan garis lurus acak dengan corong variansi non-stasioner yang melebar seiring menjauhi titik tumpu $c$.\n"
]

# Cell 7 (Code for Fig 2)
nb['cells'][7]['source'] = fig2_code

# Save notebook structure
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Updated notebook structure successfully.')

# Execute all cells
global_env = {'__name__': '__main__'}
exec_count = 1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = ''.join(cell['source'])
        print(f"Executing cell {i}...")
        plt.close('all')
        exec(code, global_env)
        
        figs = [plt.figure(n) for n in plt.get_fignums()]
        outputs = []
        if figs:
            for fig in figs:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                outputs.append({
                    'output_type': 'display_data',
                    'data': {
                        'image/png': img_b64,
                        'text/plain': ['<Figure size ... with ... Axes>']
                    },
                    'metadata': {}
                })
            plt.close('all')
        
        cell['execution_count'] = exec_count
        cell['outputs'] = outputs
        exec_count += 1

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("All cells in eksplorasi_kernel_dan_sampling.ipynb executed cleanly and outputs attached!")
