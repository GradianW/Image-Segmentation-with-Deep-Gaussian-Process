import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Set styling for professional publication quality
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['lines.linewidth'] = 1.8
plt.rcParams['lines.markersize'] = 6

out_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(out_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. GENERATE SYNTHETIC DATASET (Rasmussen Example Style)
# -------------------------------------------------------------
np.random.seed(42)
N = 20
X_train = np.sort(np.random.uniform(-4, 4, N))[:, None]
# True underlying function: non-trivial smooth function
true_func = lambda x: np.sin(1.5 * x) + 0.5 * np.cos(3.0 * x)
sigma_n_true = 0.2
y_train = true_func(X_train).ravel() + np.random.normal(0, sigma_n_true, N)

X_test = np.linspace(-5, 5, 250)[:, None]
y_true_test = true_func(X_test).ravel()

# Kernel implementation: Squared Exponential
def se_kernel(X1, X2, l, sigma_f):
    r2 = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=-1)
    return (sigma_f**2) * np.exp(-0.5 * r2 / (l**2))

def dK_dl(X1, X2, l, sigma_f):
    r2 = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=-1)
    return (sigma_f**2) * np.exp(-0.5 * r2 / (l**2)) * (r2 / (l**3))

def dK_dsigmaf(X1, X2, l, sigma_f):
    r2 = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=-1)
    return 2.0 * sigma_f * np.exp(-0.5 * r2 / (l**2))

def compute_lml_components(X, y, l, sigma_f, sigma_n):
    N = len(y)
    K = se_kernel(X, X, l, sigma_f)
    Ky = K + (sigma_n**2 + 1e-7) * np.eye(N)
    
    try:
        L = np.linalg.cholesky(Ky)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        data_fit = -0.5 * np.dot(y, alpha)
        complexity = -np.sum(np.log(np.diag(L)))
        norm_const = -0.5 * N * np.log(2 * np.pi)
        lml = data_fit + complexity + norm_const
        return lml, data_fit, complexity, norm_const
    except np.linalg.LinAlgError:
        return -np.inf, -np.inf, -np.inf, -np.inf

def gp_posterior(X_train, y_train, X_test, l, sigma_f, sigma_n):
    N = len(y_train)
    K_train = se_kernel(X_train, X_train, l, sigma_f) + (sigma_n**2 + 1e-7) * np.eye(N)
    K_s = se_kernel(X_train, X_test, l, sigma_f)
    K_ss = se_kernel(X_test, X_test, l, sigma_f) + 1e-7 * np.eye(len(X_test))
    
    L = np.linalg.cholesky(K_train)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu = np.dot(K_s.T, alpha)
    
    v = np.linalg.solve(L, K_s)
    cov = K_ss - np.dot(v.T, v)
    var = np.diag(cov)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)
    return mu, std

# -------------------------------------------------------------
# FIGURE 1: OCCAM'S RAZOR TRADE-OFF
# -------------------------------------------------------------
print("Generating Figure 1: Occam's Razor Trade-off...")
l_values = np.logspace(-1.2, 1.3, 150)
fixed_sigma_f = 1.0
fixed_sigma_n = 0.2

lmls, data_fits, complexities = [], [], []
for l in l_values:
    lml, df, comp, _ = compute_lml_components(X_train, y_train, l, fixed_sigma_f, fixed_sigma_n)
    lmls.append(lml)
    data_fits.append(df)
    complexities.append(comp)

lmls = np.array(lmls)
data_fits = np.array(data_fits)
complexities = np.array(complexities)

best_idx = np.argmax(lmls)
best_l = l_values[best_idx]

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
ax.plot(l_values, data_fits, label=r'Data-fit term $\left(-\frac{1}{2}\mathbf{y}^T K_y^{-1}\mathbf{y}\right)$', color='#2ca02c', lw=2.2, linestyle='--')
ax.plot(l_values, complexities, label=r'Complexity penalty $\left(-\frac{1}{2}\log|K_y|\right)$', color='#d62728', lw=2.2, linestyle='-.')
ax.plot(l_values, lmls, label=r'Log Marginal Likelihood $\log p(\mathbf{y}|X,\theta)$', color='#1f77b4', lw=2.8)

ax.axvline(best_l, color='purple', linestyle=':', lw=1.8, label=f'Optimal $l^* = {best_l:.2f}$')
ax.plot(best_l, lmls[best_idx], 'o', color='purple', markersize=8, zorder=5)

ax.set_xscale('log')
ax.set_xlabel(r'Lengthscale $l$ (skala logaritmik)')
ax.set_ylabel('Nilai Suku / Log-Likelihood')
ax.set_title("Dekomposisi Log Marginal Likelihood: Prinsip Occam's Razor Otomatis", pad=12)
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend(loc='lower left', framealpha=0.95)
ax.set_ylim(-65, 10)

# Annotations
ax.annotate('Overfitting\n(Kompleksitas Tinggi)', xy=(0.1, -40), xytext=(0.08, -25),
            arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.2),
            ha='center', fontsize=8.5, color='#d62728', weight='bold')

ax.annotate('Underfitting\n(Data-fit Buruk)', xy=(12, -40), xytext=(10, -25),
            arrowprops=dict(arrowstyle="->", color='#2ca02c', lw=1.2),
            ha='center', fontsize=8.5, color='#2ca02c', weight='bold')

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig1_marginal_likelihood_tradeoff.pdf"))
fig.savefig(os.path.join(out_dir, "fig1_marginal_likelihood_tradeoff.png"))
plt.close(fig)

# -------------------------------------------------------------
# FIGURE 2: 2D OPTIMIZATION SURFACE & GRADIENT TRAJECTORIES
# -------------------------------------------------------------
print("Generating Figure 2: 2D Marginal Likelihood Surface & Optimization Trajectories...")

def nlml_and_grad(log_theta, X, y):
    l = np.exp(log_theta[0])
    sigma_f = np.exp(log_theta[1])
    sigma_n = np.exp(log_theta[2])
    N = len(y)
    
    K = se_kernel(X, X, l, sigma_f)
    Ky = K + (sigma_n**2 + 1e-7) * np.eye(N)
    
    L = np.linalg.cholesky(Ky)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    
    # NLML
    data_fit = -0.5 * np.dot(y, alpha)
    complexity = -np.sum(np.log(np.diag(L)))
    norm_const = -0.5 * N * np.log(2 * np.pi)
    nlml = -(data_fit + complexity + norm_const)
    
    # Invert Ky via Cholesky
    Ky_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(N)))
    W = np.outer(alpha, alpha) - Ky_inv
    
    # Gradients w.r.t theta
    dK_l = dK_dl(X, X, l, sigma_f)
    dK_sf = dK_dsigmaf(X, X, l, sigma_f)
    dKy_sn = 2.0 * sigma_n * np.eye(N)
    
    grad_l = -0.5 * np.sum(W * dK_l) * l        # chain rule for log(l)
    grad_sf = -0.5 * np.sum(W * dK_sf) * sigma_f # chain rule for log(sigma_f)
    grad_sn = -0.5 * np.sum(W * dKy_sn) * sigma_n # chain rule for log(sigma_n)
    
    return nlml, np.array([grad_l, grad_sf, grad_sn])

# Optimize to find true best
opt_res = minimize(nlml_and_grad, np.log([1.0, 1.0, 0.2]), args=(X_train, y_train), jac=True, method='L-BFGS-B')
opt_l, opt_sf, opt_sn = np.exp(opt_res.x)

# Grid for contour (log(l) vs log(sigma_n)) with fixed optimal sigma_f
log_l_grid = np.linspace(np.log(0.15), np.log(5.0), 60)
log_sn_grid = np.linspace(np.log(0.04), np.log(1.5), 60)
L_mesh, SN_mesh = np.meshgrid(log_l_grid, log_sn_grid)
Z_lml = np.zeros_like(L_mesh)

for i in range(L_mesh.shape[0]):
    for j in range(L_mesh.shape[1]):
        l_val = np.exp(L_mesh[i, j])
        sn_val = np.exp(SN_mesh[i, j])
        lml_val, _, _, _ = compute_lml_components(X_train, y_train, l_val, opt_sf, sn_val)
        Z_lml[i, j] = lml_val

# Trace trajectories from 3 different initial points
initial_points = [
    np.array([np.log(0.2), np.log(opt_sf), np.log(1.2)]),   # Start 1: Small l, large noise
    np.array([np.log(4.0), np.log(opt_sf), np.log(0.06)]),  # Start 2: Large l, small noise
    np.array([np.log(0.25), np.log(opt_sf), np.log(0.06)])  # Start 3: Small l, small noise
]

trajectories = []
for p0 in initial_points:
    path = [p0.copy()]
    def callback_fn(xk):
        path.append(xk.copy())
    minimize(nlml_and_grad, p0, args=(X_train, y_train), jac=True, method='L-BFGS-B', callback=callback_fn, options={'maxiter': 30})
    trajectories.append(np.array(path))

fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
# Clip Z_lml for better contrast
Z_clipped = np.clip(Z_lml, -60, np.max(Z_lml))
cs = ax.contourf(np.exp(L_mesh), np.exp(SN_mesh), Z_clipped, levels=30, cmap='viridis_r')
cbar = fig.colorbar(cs, ax=ax)
cbar.set_label(r'Log Marginal Likelihood $\log p(\mathbf{y}|X,\theta)$')

ax.contour(np.exp(L_mesh), np.exp(SN_mesh), Z_clipped, levels=15, colors='white', alpha=0.3, linewidths=0.7)

colors = ['#ff7f0e', '#e377c2', '#17becf']
for idx, (path, col) in enumerate(zip(trajectories, colors)):
    l_path = np.exp(path[:, 0])
    sn_path = np.exp(path[:, 2])
    ax.plot(l_path, sn_path, '-o', color=col, lw=2.2, markersize=4, label=f'Lintasan Optimasi {idx+1}')
    ax.plot(l_path[0], sn_path[0], 's', color=col, markersize=8, markeredgecolor='black', label=f'Inisialisasi {idx+1}' if idx==0 else "")

ax.plot(opt_l, opt_sn, '*', color='yellow', markersize=14, markeredgecolor='black', label=f'Optimum Global ($l={opt_l:.2f}, \\sigma_n={opt_sn:.2f}$)', zorder=10)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Lengthscale $l$')
ax.set_ylabel(r'Noise Standard Deviation $\sigma_n$')
ax.set_title(r'Kontur Log Marginal Likelihood & Lintasan Konvergensi Optimasi Gradien', pad=12)
ax.legend(loc='lower left', framealpha=0.92, fontsize=8.5)
ax.grid(True, which="both", ls=":", alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig2_optimization_surface_trajectories.pdf"))
fig.savefig(os.path.join(out_dir, "fig2_optimization_surface_trajectories.png"))
plt.close(fig)

# -------------------------------------------------------------
# FIGURE 3: MODEL FIT COMPARISON (Underfit vs Optimal vs Overfit)
# -------------------------------------------------------------
print("Generating Figure 3: Model Fit Comparison Under Different Hyperparameters...")

cases = [
    {
        'title': 'Kasus A: Underfitting / Over-smooth',
        'sub': r'($l=4.50, \sigma_n=0.20$)',
        'l': 4.5, 'sf': 1.0, 'sn': 0.20,
        'color': '#2ca02c'
    },
    {
        'title': 'Kasus B: Optimal (Terekstrimkan oleh LML)',
        'sub': f'($l={opt_l:.2f}, \\sigma_n={opt_sn:.2f}$)',
        'l': opt_l, 'sf': opt_sf, 'sn': opt_sn,
        'color': '#1f77b4'
    },
    {
        'title': 'Kasus C: Overfitting / Over-complex',
        'sub': r'($l=0.18, \sigma_n=0.03$)',
        'l': 0.18, 'sf': 1.0, 'sn': 0.03,
        'color': '#d62728'
    }
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300, sharey=True)

for ax, case in zip(axes, cases):
    l, sf, sn = case['l'], case['sf'], case['sn']
    mu, std = gp_posterior(X_train, y_train, X_test, l, sf, sn)
    lml, df, comp, _ = compute_lml_components(X_train, y_train, l, sf, sn)
    
    # Plot true function and data
    ax.plot(X_test, y_true_test, 'k--', alpha=0.5, label='Fungsi Sejati' if ax == axes[0] else "")
    ax.scatter(X_train, y_train, c='black', s=24, zorder=5, label='Data Observasi' if ax == axes[0] else "")
    
    # Plot GP Posterior
    ax.plot(X_test, mu, color=case['color'], lw=2.2, label='Mean Prediktif $\\bar{f}_*$')
    ax.fill_between(X_test.ravel(), mu - 2*std, mu + 2*std, color=case['color'], alpha=0.22, label=r'Pita $\pm 2\sigma$ ($95\%$)')
    
    ax.set_title(f"{case['title']}\n{case['sub']}", fontsize=10.5)
    ax.set_xlabel('$x$')
    if ax == axes[0]:
        ax.set_ylabel('$y = f(x)$')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(True, ls=':', alpha=0.5)
    
    # Metrics textbox
    stats_text = (
        f"Data Fit: {df:+.2f}\n"
        f"Complexity: {comp:+.2f}\n"
        f"Total LML: {lml:+.2f}"
    )
    bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9, lw=0.8)
    ax.text(0.04, 0.06, stats_text, transform=ax.transAxes, fontsize=8.5, family='monospace', bbox=bbox_props)

axes[0].legend(loc='upper right', fontsize=8, framealpha=0.9)
plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig3_hyperparameter_fit_comparison.pdf"))
fig.savefig(os.path.join(out_dir, "fig3_hyperparameter_fit_comparison.png"))
plt.close(fig)

# -------------------------------------------------------------
# FIGURE 4: OPTIMIZATION WORKFLOW SCHEMATIC (Matplotlib diagram)
# -------------------------------------------------------------
print("Generating Figure 4: Numerical Optimization Flowchart...")
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
ax.axis('off')

boxes = [
    {"text": "1. Inisialisasi Parameter\n$\\tilde{\\boldsymbol{\\theta}}_0 = \\log \\boldsymbol{\\theta}_0$\n$(l_0, \\sigma_{f0}, \\sigma_{n0})$", "pos": (0.10, 0.5), "color": "#e0f2fe", "border": "#0284c7"},
    {"text": "2. Konstruksi & Faktorisasi\n$K_y = K(\\boldsymbol{\\theta}) + \\sigma_n^2 I$\n$K_y = L L^T$ (Cholesky, $\\mathcal{O}(N^3)$)", "pos": (0.34, 0.5), "color": "#fef3c7", "border": "#d97706"},
    {"text": "3. Evaluasi LML & Bobot\n$\\boldsymbol{\\alpha} = L^{-T} L^{-1} \\mathbf{y}$\n$\\log p = -\\frac{1}{2}\\mathbf{y}^T\\boldsymbol{\\alpha} - \\sum \\log L_{ii} - \\text{c}$", "pos": (0.58, 0.5), "color": "#dcfce7", "border": "#16a34a"},
    {"text": "4. Evaluasi Gradien Analitik\n$W = \\boldsymbol{\\alpha}\\boldsymbol{\\alpha}^T - K_y^{-1}$\n$\\frac{\\partial \\log p}{\\partial \\theta_j} = \\frac{1}{2}\\operatorname{tr}\\left(W \\frac{\\partial K_y}{\\partial \\theta_j}\\right)$", "pos": (0.82, 0.5), "color": "#fae8ff", "border": "#c026d3"},
]

for b in boxes:
    ax.text(b["pos"][0], b["pos"][1], b["text"], ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.6", fc=b["color"], ec=b["border"], lw=1.5))

# Draw forward arrows
for i in range(len(boxes)-1):
    x1 = boxes[i]["pos"][0] + 0.085
    x2 = boxes[i+1]["pos"][0] - 0.085
    y = 0.5
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#334155", lw=1.8))

# Draw feedback loop (Optimizer update)
ax.annotate('5. Update Optimizer (L-BFGS-B / CG)\n$\\tilde{\\boldsymbol{\\theta}}_{t+1} = \\tilde{\\boldsymbol{\\theta}}_t + \\Delta \\tilde{\\boldsymbol{\\theta}}$',
            xy=(0.34, 0.22), xytext=(0.82, 0.22),
            ha='center', va='center', fontsize=9, color="#991b1b",
            arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#dc2626", lw=1.8, linestyle='--'),
            bbox=dict(boxstyle="round,pad=0.4", fc="#fee2e2", ec="#dc2626", lw=1.2))

# Connect step 4 to feedback
ax.plot([0.82, 0.82], [0.35, 0.22], color="#dc2626", lw=1.8, linestyle='--')
# Connect feedback back to step 2
ax.plot([0.34, 0.34], [0.22, 0.35], color="#dc2626", lw=1.8, linestyle='--')

ax.set_title("Diagram Alur Komputasi Numerik Optimasi Hyperparameter GP (Algoritma 5.1 Rasmussen & Williams)", pad=15, weight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0.08, 0.92)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig4_optimization_algorithm_workflow.pdf"))
fig.savefig(os.path.join(out_dir, "fig4_optimization_algorithm_workflow.png"))
plt.close(fig)

print("All figures successfully generated and saved to figures/ directory.")
