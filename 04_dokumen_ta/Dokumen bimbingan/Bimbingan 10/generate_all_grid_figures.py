import numpy as np
import matplotlib.pyplot as plt
import os

fig_dir = r"04_dokumen_ta/Dokumen bimbingan/Bimbingan 10/figures"
os.makedirs(fig_dir, exist_ok=True)

# 1. Dataset
X = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.1, 0.5],
    [0.2, 0.6, 0.1],
    [0.5, 0.3, 0.2],
    [0.3, 0.4, 0.4]
], dtype=np.float64)
N = X.shape[0]

y1 = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.8 * X[:, 2]
y2 = -0.4 * X[:, 0] + 0.7 * X[:, 1] + 0.2 * X[:, 2]

Y_r1 = y1.reshape(-1, 1)
Y_r2 = np.column_stack([y1, y2])

def sig(z): return 1.0 / (1.0 + np.exp(-z))
def dsig(z): s = sig(z); return s * (1.0 - s)
def relu(z): return np.maximum(0.0, z)
def drelu(z): return (z > 0.0).astype(np.float64)

# Function to run simulation
def run_sim(sifat, target_dim, init_type, act_name):
    np.random.seed(67)
    Y = Y_r1 if target_dim == 1 else Y_r2
    n0, n1, n2, n3 = 3, 4, 6, target_dim
    act = sig if act_name == 'sigmoid' else relu
    dact = dsig if act_name == 'sigmoid' else drelu
    lr = 0.5
    
    # Init
    if sifat == 'skalar':
        if init_type == 'ones':
            W1 = np.ones((n0, n1)); b1 = np.ones((1, n1))
            W2 = np.ones((n1, n2)); b2 = np.ones((1, n2))
            W3 = np.ones((n2, n3)); b3 = np.ones((1, n3))
        else:
            W1 = np.array([
                [ 0.1, -0.2,  0.3,  0.1],
                [-0.1,  0.3, -0.1,  0.2],
                [ 0.2,  0.1, -0.2, -0.3]
            ], dtype=np.float64)
            b1 = np.array([[0.1, -0.1, 0.2, 0.0]], dtype=np.float64)
            W2 = np.array([
                [ 0.2, -0.1,  0.3,  0.1, -0.2,  0.1],
                [-0.2,  0.2, -0.1,  0.3,  0.1, -0.1],
                [ 0.1, -0.3,  0.2, -0.1,  0.2,  0.3],
                [-0.1,  0.1, -0.2,  0.2, -0.1,  0.2]
            ], dtype=np.float64)
            b2 = np.array([[0.1, 0.0, -0.1, 0.2, -0.2, 0.1]], dtype=np.float64)
            if target_dim == 1:
                W3 = np.array([[ 0.3], [-0.2], [ 0.4], [-0.1], [ 0.2], [-0.3]], dtype=np.float64)
                b3 = np.array([[0.1]], dtype=np.float64)
            else:
                W3 = np.array([
                    [ 0.3, -0.1], [-0.2,  0.3], [ 0.4, -0.2],
                    [-0.1,  0.4], [ 0.2, -0.3], [-0.3,  0.1]
                ], dtype=np.float64)
                b3 = np.array([[0.1, -0.1]], dtype=np.float64)
    else: # distribusi
        if init_type == 'ones':
            M1 = np.ones((n0, n1)); Sigma1 = np.ones((n0, n1)); b1 = np.ones((1, n1))
            M2 = np.ones((n1, n2)); Sigma2 = np.ones((n1, n2)); b2 = np.ones((1, n2))
            M3 = np.ones((n2, n3)); Sigma3 = np.ones((n2, n3)); b3 = np.ones((1, n3))
        else:
            M1 = np.array([
                [ 0.1, -0.2,  0.3,  0.1],
                [-0.1,  0.3, -0.1,  0.2],
                [ 0.2,  0.1, -0.2, -0.3]
            ], dtype=np.float64)
            Sigma1 = np.array([
                [0.1, 0.2, 0.1, 0.2], [0.2, 0.1, 0.2, 0.1], [0.1, 0.2, 0.1, 0.2]
            ], dtype=np.float64)
            b1 = np.array([[0.1, -0.1, 0.2, 0.0]], dtype=np.float64)
            M2 = np.array([
                [ 0.2, -0.1,  0.3,  0.1, -0.2,  0.1],
                [-0.2,  0.2, -0.1,  0.3,  0.1, -0.1],
                [ 0.1, -0.3,  0.2, -0.1,  0.2,  0.3],
                [-0.1,  0.1, -0.2,  0.2, -0.1,  0.2]
            ], dtype=np.float64)
            Sigma2 = np.array([
                [0.1, 0.2, 0.1, 0.2, 0.1, 0.2], [0.2, 0.1, 0.2, 0.1, 0.2, 0.1],
                [0.1, 0.2, 0.1, 0.2, 0.1, 0.2], [0.2, 0.1, 0.2, 0.1, 0.2, 0.1]
            ], dtype=np.float64)
            b2 = np.array([[0.1, 0.0, -0.1, 0.2, -0.2, 0.1]], dtype=np.float64)
            if target_dim == 1:
                M3 = np.array([[ 0.3], [-0.2], [ 0.4], [-0.1], [ 0.2], [-0.3]], dtype=np.float64)
                Sigma3 = np.array([[0.1], [0.2], [0.1], [0.2], [0.1], [0.2]], dtype=np.float64)
                b3 = np.array([[0.1]], dtype=np.float64)
            else:
                M3 = np.array([
                    [ 0.3, -0.1], [-0.2,  0.3], [ 0.4, -0.2],
                    [-0.1,  0.4], [ 0.2, -0.3], [-0.3,  0.1]
                ], dtype=np.float64)
                Sigma3 = np.array([
                    [0.1, 0.2], [0.2, 0.1], [0.1, 0.2], [0.2, 0.1], [0.1, 0.2], [0.2, 0.1]
                ], dtype=np.float64)
                b3 = np.array([[0.1, -0.1]], dtype=np.float64)
                
    history = {}
    for epoch in range(1, 501):
        if sifat == 'skalar':
            Z1 = X @ W1 + b1; X1 = act(Z1)
            Z2 = X1 @ W2 + b2; X2 = act(Z2)
            Z3 = X2 @ W3 + b3; Yhat = Z3
            loss = (0.5 / N) * np.sum((Y - Yhat) ** 2)
            if epoch in [1, 2] or epoch % 50 == 0:
                history[epoch] = (loss, Yhat.copy())
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
            if epoch in [1, 2] or epoch % 50 == 0:
                history[epoch] = (loss, Yhat.copy())
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
    return history

# Run all 16 simulations
cases = [
    ('skalar', 1), ('skalar', 2),
    ('distribusi', 1), ('distribusi', 2)
]
variations = [
    ('v1', 'ones', 'sigmoid', 'Variasi 1 (Init 1, Sigmoid)'),
    ('v2', 'ones', 'relu', 'Variasi 2 (Init 1, ReLU)'),
    ('v3', 'different', 'sigmoid', 'Variasi 3 (Init Beda, Sigmoid)'),
    ('v4', 'different', 'relu', 'Variasi 4 (Init Beda, ReLU)')
]

results = {}
for sifat, tdim in cases:
    results[(sifat, tdim)] = {}
    for vname, itype, aname, title in variations:
        results[(sifat, tdim)][vname] = run_sim(sifat, tdim, itype, aname)

print("Simulasi selesai. Membuat gambar grid...")

# Plotting Function
def create_grid(sifat, tdim, part, epochs_subset):
    fig, axes = plt.subplots(5, 4, figsize=(14, 15))
    samples = np.arange(1, 6)
    Y = Y_r1 if tdim == 1 else Y_r2
    
    for r_idx, ep in enumerate(epochs_subset):
        for c_idx, (vname, _, _, vtitle) in enumerate(variations):
            ax = axes[r_idx, c_idx]
            loss, pred = results[(sifat, tdim)][vname][ep]
            
            if tdim == 1:
                ax.plot(samples, Y.ravel(), 'o-', color='#1f77b4', markersize=5, linewidth=1.5, label='Target Y')
                ax.plot(samples, pred.ravel(), 's--', color='#d62728', markersize=5, linewidth=1.2, label='Prediksi Y_hat')
            else:
                ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7, label='Ideal (0)' if (r_idx == 0 and c_idx == 0) else "")
                ax.plot(samples, pred[:, 0] - Y[:, 0], 'o-', color='#1f77b4', markersize=4, linewidth=1.2, label=r'Selisih $y_1$')
                ax.plot(samples, pred[:, 1] - Y[:, 1], 's-', color='#d62728', markersize=4, linewidth=1.2, label=r'Selisih $y_2$')
                
            if r_idx == 0:
                ax.set_title(vtitle, fontsize=10, fontweight='bold', pad=8)
            if c_idx == 0:
                ax.set_ylabel(f'Iterasi {ep}\nLoss: {loss:.4f}', fontsize=9, fontweight='bold')
            else:
                ax.set_ylabel(f'Loss: {loss:.4f}', fontsize=8)
            ax.set_xticks(samples)
            ax.grid(True, linestyle=':', alpha=0.5)
            if r_idx == 0 and c_idx == 0:
                ax.legend(fontsize=7, loc='best')

    plt.tight_layout()
    filename = f"grid_{sifat}_r3_r{tdim}_part{part}.png"
    filepath = os.path.join(fig_dir, filename)
    plt.savefig(filepath, dpi=200)
    plt.close()
    print(f"Disimpan: {filename}")

# Generate 8 Grid Figures (Part 1: 50..250, Part 2: 300..500)
for sifat, tdim in cases:
    create_grid(sifat, tdim, 1, [50, 100, 150, 200, 250])
    create_grid(sifat, tdim, 2, [300, 350, 400, 450, 500])

print("Semua 8 gambar grid komparatif berhasil dibuat!")

