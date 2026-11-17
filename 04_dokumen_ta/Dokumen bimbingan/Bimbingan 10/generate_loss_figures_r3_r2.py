import numpy as np
import matplotlib.pyplot as plt
import os

fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(fig_dir, exist_ok=True)

# Data Input X
X = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.1, 0.5],
    [0.2, 0.6, 0.1],
    [0.5, 0.3, 0.2],
    [0.3, 0.4, 0.4]
], dtype=np.float64)

# Target R3 -> R2
y1 = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.8 * X[:, 2]
y2 = -0.4 * X[:, 0] + 0.7 * X[:, 1] + 0.2 * X[:, 2]
Y = np.column_stack([y1, y2])
N = 5
n0, n1, n2, n3 = 3, 4, 6, 2
lr = 0.5

def run_simulation(sifat, init_type, act_name):
    np.random.seed(67)
    
    # Activation
    if act_name == 'sigmoid':
        act = lambda z: 1.0 / (1.0 + np.exp(-z))
        dact = lambda z: act(z) * (1.0 - act(z))
    else:
        act = lambda z: np.maximum(0.0, z)
        dact = lambda z: (z > 0.0).astype(np.float64)
        
    # Init
    if sifat == 'skalar':
        if init_type == 'ones':
            W1 = np.ones((n0, n1), dtype=np.float64)
            b1 = np.ones((1, n1), dtype=np.float64)
            W2 = np.ones((n1, n2), dtype=np.float64)
            b2 = np.ones((1, n2), dtype=np.float64)
            W3 = np.ones((n2, n3), dtype=np.float64)
            b3 = np.ones((1, n3), dtype=np.float64)
        else:
            W1 = np.array([[ 0.1, -0.2,  0.3,  0.1], [-0.1,  0.3, -0.1,  0.2], [ 0.2,  0.1, -0.2, -0.3]], dtype=np.float64)
            b1 = np.array([[0.1, -0.1, 0.2, 0.0]], dtype=np.float64)
            W2 = np.array([[ 0.2, -0.1,  0.3,  0.1, -0.2,  0.1], [-0.2,  0.2, -0.1,  0.3,  0.1, -0.1], [ 0.1, -0.3,  0.2, -0.1,  0.2,  0.3], [-0.1,  0.1, -0.2,  0.2, -0.1,  0.2]], dtype=np.float64)
            b2 = np.array([[0.1, 0.0, -0.1, 0.2, -0.2, 0.1]], dtype=np.float64)
            W3 = np.array([[-0.2,  0.3], [ 0.1, -0.1], [-0.3,  0.2], [-0.1,  0.4], [ 0.2, -0.3], [-0.3,  0.1]], dtype=np.float64)
            b3 = np.array([[0.1, -0.1]], dtype=np.float64)
    else:
        if init_type == 'ones':
            M1 = np.ones((n0, n1), dtype=np.float64); Sigma1 = np.ones((n0, n1), dtype=np.float64); b1 = np.ones((1, n1), dtype=np.float64)
            M2 = np.ones((n1, n2), dtype=np.float64); Sigma2 = np.ones((n1, n2), dtype=np.float64); b2 = np.ones((1, n2), dtype=np.float64)
            M3 = np.ones((n2, n3), dtype=np.float64); Sigma3 = np.ones((n2, n3), dtype=np.float64); b3 = np.ones((1, n3), dtype=np.float64)
        else:
            M1 = np.array([[ 0.1, -0.2,  0.3,  0.1], [-0.1,  0.3, -0.1,  0.2], [ 0.2,  0.1, -0.2, -0.3]], dtype=np.float64)
            Sigma1 = np.array([[0.1, 0.2, 0.1, 0.2], [0.2, 0.1, 0.2, 0.1], [0.1, 0.2, 0.1, 0.2]], dtype=np.float64)
            b1 = np.array([[0.1, -0.1, 0.2, 0.0]], dtype=np.float64)
            M2 = np.array([[ 0.2, -0.1,  0.3,  0.1, -0.2,  0.1], [-0.2,  0.2, -0.1,  0.3,  0.1, -0.1], [ 0.1, -0.3,  0.2, -0.1,  0.2,  0.3], [-0.1,  0.1, -0.2,  0.2, -0.1,  0.2]], dtype=np.float64)
            Sigma2 = np.array([[0.1, 0.2, 0.1, 0.2, 0.1, 0.2], [0.2, 0.1, 0.2, 0.1, 0.2, 0.1], [0.1, 0.2, 0.1, 0.2, 0.1, 0.2], [0.2, 0.1, 0.2, 0.1, 0.2, 0.1]], dtype=np.float64)
            b2 = np.array([[0.1, 0.0, -0.1, 0.2, -0.2, 0.1]], dtype=np.float64)
            M3 = np.array([[-0.2,  0.3], [ 0.1, -0.1], [-0.3,  0.2], [-0.1,  0.4], [ 0.2, -0.3], [-0.3,  0.1]], dtype=np.float64)
            Sigma3 = np.array([[0.1, 0.2], [0.2, 0.1], [0.1, 0.2], [0.2, 0.1], [0.1, 0.2], [0.2, 0.1]], dtype=np.float64)
            b3 = np.array([[0.1, -0.1]], dtype=np.float64)

    losses = []
    for epoch in range(1, 501):
        if sifat == 'skalar':
            Z1 = X @ W1 + b1; X1 = act(Z1)
            Z2 = X1 @ W2 + b2; X2 = act(Z2)
            Z3 = X2 @ W3 + b3; Yhat = Z3
            loss = (0.5 / N) * np.sum((Y - Yhat) ** 2)
            losses.append(loss)
            
            dL_dYhat = (1.0 / N) * (Yhat - Y)
            dL_dW3 = X2.T @ dL_dYhat; dL_db3 = np.ones((1, N)) @ dL_dYhat
            dL_dX2 = dL_dYhat @ W3.T; dL_dZ2 = dL_dX2 * dact(Z2)
            dL_dW2 = X1.T @ dL_dZ2; dL_db2 = np.ones((1, N)) @ dL_dZ2
            dL_dX1 = dL_dZ2 @ W2.T; dL_dZ1 = dL_dX1 * dact(Z1)
            dL_dW1 = X.T @ dL_dZ1; dL_db1 = np.ones((1, N)) @ dL_dZ1
            W3 -= lr * dL_dW3; b3 -= lr * dL_db3
            W2 -= lr * dL_dW2; b2 -= lr * dL_db2
            W1 -= lr * dL_dW1; b1 -= lr * dL_db1
        else:
            Zw1 = np.random.randn(n0, n1); W1 = M1 + Sigma1 * Zw1
            Z1 = X @ W1 + b1; X1 = act(Z1)
            Zw2 = np.random.randn(n1, n2); W2 = M2 + Sigma2 * Zw2
            Z2 = X1 @ W2 + b2; X2 = act(Z2)
            Zw3 = np.random.randn(n2, n3); W3 = M3 + Sigma3 * Zw3
            Z3 = X2 @ W3 + b3; Yhat = Z3
            loss = (0.5 / N) * np.sum((Y - Yhat) ** 2)
            losses.append(loss)
            
            dL_dYhat = (1.0 / N) * (Yhat - Y)
            dL_dW3 = X2.T @ dL_dYhat; dL_db3 = np.ones((1, N)) @ dL_dYhat
            dL_dM3 = dL_dW3; dL_dSigma3 = dL_dW3 * Zw3
            dL_dX2 = dL_dYhat @ W3.T; dL_dZ2 = dL_dX2 * dact(Z2)
            dL_dW2 = X1.T @ dL_dZ2; dL_db2 = np.ones((1, N)) @ dL_dZ2
            dL_dM2 = dL_dW2; dL_dSigma2 = dL_dW2 * Zw2
            dL_dX1 = dL_dZ2 @ W2.T; dL_dZ1 = dL_dX1 * dact(Z1)
            dL_dW1 = X.T @ dL_dZ1; dL_db1 = np.ones((1, N)) @ dL_dZ1
            dL_dM1 = dL_dW1; dL_dSigma1 = dL_dW1 * Zw1
            M3 -= lr * dL_dM3; Sigma3 -= lr * dL_dSigma3; b3 -= lr * dL_db3
            M2 -= lr * dL_dM2; Sigma2 -= lr * dL_dSigma2; b2 -= lr * dL_db2
            M1 -= lr * dL_dM1; Sigma1 -= lr * dL_dSigma1; b1 -= lr * dL_db1
    return np.array(losses)

var_list = [
    ('v1', 'ones', 'sigmoid', 'Variasi 1 (Init 1, Sigmoid)', '#1f77b4'),
    ('v2', 'ones', 'relu', 'Variasi 2 (Init 1, ReLU)', '#ff7f0e'),
    ('v3', 'different', 'sigmoid', 'Variasi 3 (Init Beda, Sigmoid)', '#2ca02c'),
    ('v4', 'different', 'relu', 'Variasi 4 (Init Beda, ReLU)', '#d62728')
]

epochs = np.arange(1, 501)
milestones = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

for sifat, label_sifat in [('skalar', 'Bobot Skalar Deterministik'), ('distribusi', 'Bobot Berdistribusi')]:
    all_losses = {}
    for vid, itype, aname, vtitle, color in var_list:
        all_losses[vid] = run_simulation(sifat, itype, aname)
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # 4 Subplots per Variasi
    for idx, (vid, _, _, vtitle, color) in enumerate(var_list):
        ax = axes[idx]
        l_arr = all_losses[vid]
        ax.plot(epochs, l_arr, color=color, linewidth=2, label='MSE Loss')
        m_vals = [l_arr[m - 1] for m in milestones]
        ax.scatter(milestones, m_vals, color='black', s=30, zorder=5, label='Kelipatan 50')
        ax.set_title(vtitle, fontsize=11, fontweight='bold')
        ax.set_xlabel('Iterasi (Epoch)', fontsize=10)
        ax.set_ylabel('MSE Loss', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=8, loc='upper right')
        ax.text(0.96, 0.78, f'Loss Akhir (500): {l_arr[-1]:.6f}', transform=ax.transAxes,
                fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    plt.suptitle(f'Grafik Fungsi Loss (MSE Loss) Pemetaan Multi-Output R^3 -> R^2: {label_sifat}', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_file = os.path.join(fig_dir, f"loss_{sifat}_r3_r2.png")
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Gambar tersimpan: {out_file}")

print("Generasi grafik fungsi loss untuk R3 -> R2 selesai!")

