import numpy as np
import matplotlib.pyplot as plt

def kernel_rbf(X1, X2, sigma_f=1.0, lengthscale=1.0):
    dist_sq = (X1 - X2.T)**2
    return (sigma_f**2) * np.exp(-0.5 * dist_sq / (lengthscale**2))

def kernel_periodic(X1, X2, sigma_p=1.0, lengthscale=1.0, period=2.0):
    dist = np.abs(X1 - X2.T)
    sin_term = np.sin(np.pi * dist / period)
    return (sigma_p**2) * np.exp(-2.0 * (sin_term**2) / (lengthscale**2))

def kernel_linear(X1, X2, sigma_b=0.2, sigma_v=1.0, c=0.0):
    return (sigma_b**2) + (sigma_v**2) * (X1 - c) * (X2.T - c)

def gp_predict(X_train, y_train, X_test, kernel_fn, sigma_n=0.15):
    K = kernel_fn(X_train, X_train) + (sigma_n**2) * np.eye(len(X_train))
    K_s = kernel_fn(X_train, X_test)
    K_ss = kernel_fn(X_test, X_test) + 1e-7 * np.eye(len(X_test))
    
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    
    mu_s = K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    cov_s = K_ss - v.T @ v
    var_s = np.diag(cov_s)
    std_s = np.sqrt(np.maximum(var_s, 1e-8))
    
    return mu_s.flatten(), std_s.flatten()

# Test 1: f(x) = 0.3 x sin(x) with p = 2*pi
np.random.seed(42)
N = 65
X_tr = np.sort(np.random.uniform(-10, 10, N))[:, None]
f_tr = 0.3 * X_tr * np.sin(X_tr)
y_tr = f_tr + np.random.normal(0, 0.15, size=X_tr.shape)

X_te = np.linspace(-13, 13, 600)[:, None]
f_te = 0.3 * X_te * np.sin(X_te)

p_val = 2 * np.pi

# Models for f(x) = 0.3 x sin(x)
# 1a: RBF + Per
k_1a = lambda a, b: kernel_rbf(a, b, sigma_f=1.0, lengthscale=5.0) + kernel_periodic(a, b, sigma_p=1.8, lengthscale=1.5, period=p_val)
# 1b: RBF * Per
k_1b = lambda a, b: kernel_rbf(a, b, sigma_f=1.2, lengthscale=6.0) * kernel_periodic(a, b, sigma_p=1.8, lengthscale=1.5, period=p_val)
# 2a: Lin + Per
k_2a = lambda a, b: kernel_linear(a, b, sigma_b=0.5, sigma_v=0.1, c=0.0) + kernel_periodic(a, b, sigma_p=1.8, lengthscale=1.5, period=p_val)
# 2b: Lin * Per
k_2b = lambda a, b: kernel_linear(a, b, sigma_b=0.05, sigma_v=0.3, c=0.0) * kernel_periodic(a, b, sigma_p=1.0, lengthscale=1.5, period=p_val)

mu_1a, s_1a = gp_predict(X_tr, y_tr, X_te, k_1a)
mu_1b, s_1b = gp_predict(X_tr, y_tr, X_te, k_1b)
mu_2a, s_2a = gp_predict(X_tr, y_tr, X_te, k_2a)
mu_2b, s_2b = gp_predict(X_tr, y_tr, X_te, k_2b)

fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
axes[0, 0].plot(X_te, f_te, 'r--', label='True'); axes[0, 0].plot(X_te, mu_1a, 'b-', label='GP'); axes[0, 0].scatter(X_tr, y_tr, c='k', s=15); axes[0, 0].set_title('1a: RBF + Per')
axes[0, 1].plot(X_te, f_te, 'r--', label='True'); axes[0, 1].plot(X_te, mu_1b, 'purple', label='GP'); axes[0, 1].scatter(X_tr, y_tr, c='k', s=15); axes[0, 1].set_title('1b: RBF * Per')
axes[1, 0].plot(X_te, f_te, 'r--', label='True'); axes[1, 0].plot(X_te, mu_2a, 'teal', label='GP'); axes[1, 0].scatter(X_tr, y_tr, c='k', s=15); axes[1, 0].set_title('2a: Lin + Per')
axes[1, 1].plot(X_te, f_te, 'r--', label='True'); axes[1, 1].plot(X_te, mu_2b, 'darkorange', label='GP'); axes[1, 1].scatter(X_tr, y_tr, c='k', s=15); axes[1, 1].set_title('2b: Lin * Per')

for ax in axes.flat:
    ax.legend()
    ax.set_xlim(-13, 13)

plt.tight_layout()
plt.savefig("test_sim_lin_sin.png")
plt.close()
print("Simulation done.")
