import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# DATASET DENGAN 5 SAMPEL (R^3 -> R^1)
# ==============================================================================
X = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.1, 0.5],
    [0.2, 0.6, 0.1],
    [0.5, 0.3, 0.2],
    [0.3, 0.4, 0.4]
], dtype=np.float64)

# Target Pemetaan R^3 -> R^1: y = 0.5*x1 - 0.3*x2 + 0.8*x3
Y = (0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.8 * X[:, 2]).reshape(-1, 1)
N = X.shape[0]

# FUNGSI AKTIVASI RELU & TURUNANNYA
def relu(z):
    return np.maximum(0.0, z)

def relu_deriv(z):
    return (z > 0.0).astype(np.float64)

# ==============================================================================
# INISIALISASI PARAMETER: 1 SEMUA (W dan b)
# ==============================================================================
W1 = np.ones((3, 4), dtype=np.float64)
b1 = np.ones((1, 4), dtype=np.float64)

W2 = np.ones((4, 6), dtype=np.float64)
b2 = np.ones((1, 6), dtype=np.float64)

W3 = np.ones((6, 1), dtype=np.float64)
b3 = np.ones((1, 1), dtype=np.float64)

learning_rate = 0.5
num_epochs = 500
plot_interval = 50

sample_indices = np.arange(1, N + 1)
history = []

print("=== MEMULAI SIMULASI RELU DENGAN INISIALISASI 1 SEMUA (500 ITERASI) ===")
for epoch in range(1, num_epochs + 1):
    # Forward Propagation
    Z1 = X @ W1 + np.ones((N, 1)) @ b1
    X1 = relu(Z1)
    
    Z2 = X1 @ W2 + np.ones((N, 1)) @ b2
    X2 = relu(Z2)
    
    Z3 = X2 @ W3 + np.ones((N, 1)) @ b3
    Y_hat = Z3
    
    loss = (0.5 / N) * np.sum((Y - Y_hat) ** 2)
    
    # Backward Propagation
    dL_dYhat = (1.0 / N) * (Y_hat - Y)
    dL_dW3 = X2.T @ dL_dYhat
    dL_db3 = np.ones((1, N)) @ dL_dYhat
    
    dL_dX2 = dL_dYhat @ W3.T
    dL_dZ2 = dL_dX2 * relu_deriv(Z2)
    dL_dW2 = X1.T @ dL_dZ2
    dL_db2 = np.ones((1, N)) @ dL_dZ2
    
    dL_dX1 = dL_dZ2 @ W2.T
    dL_dZ1 = dL_dX1 * relu_deriv(Z1)
    dL_dW1 = X.T @ dL_dZ1
    dL_db1 = np.ones((1, N)) @ dL_dZ1
    
    if epoch in [1, 2] or epoch % plot_interval == 0:
        history.append({
            'epoch': epoch,
            'loss': loss,
            'Y_hat': Y_hat.copy()
        })
        print(f"Iterasi {epoch:3d}: MSE Loss = {loss:.8f}")
    
    # Update Parameter
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2
    W3 -= learning_rate * dL_dW3
    b3 -= learning_rate * dL_db3

print("\n=== TARGET vs HASIL PREDIKSI PADA ITERASI KE-500 ===")
print("Target Asli Y   :", Y.flatten())
print("Prediksi Y_hat  :", np.round(history[-1]['Y_hat'].flatten(), 4))
print(f"MSE Loss Akhir  : {history[-1]['loss']:.8f}")
