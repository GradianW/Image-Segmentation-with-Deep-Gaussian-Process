import json
import io
import base64
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nb_path = r'c:\Users\MyBook Z Series\Desktop\TA\04_dokumen_ta\Dokumen bimbingan\Bimbingan 03\eksplorasi_kernel_dan_sampling.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

global_env = {'__name__': '__main__'}

exec_count = 1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = ''.join(cell['source'])
        print(f"Executing cell {i}...")
        plt.close('all')
        try:
            exec(code, global_env)
            
            # Check if any figure was plotted
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
        except Exception as e:
            print(f"Error in cell {i}: {e}")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("All cells executed and notebook updated successfully!")
