import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import kagglehub
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ExpSineSquared, RationalQuadratic, DotProduct, Matern, ConstantKernel
)
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Style persis seperti di Bimbingan 05 / generate_kernel_figures.py
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.labelsize'] = 10.5
plt.rcParams['axes.titlesize'] = 11.5
plt.rcParams['legend.fontsize'] = 8.5

# 1. Load Data
dataset_dir = kagglehub.dataset_download("saloni1712/co2-emissions")
csv_path = os.path.join(dataset_dir, "dataset.csv")
df = pd.read_csv(csv_path)
# Filter US Residential
df_us_res = df[(df['country'] == 'US') & (df['sector'] == 'Residential')].copy()
df_us_res['date'] = pd.to_datetime(df_us_res['date'], format='%d/%m/%Y')
df_us_res = df_us_res.sort_values('date').reset_index(drop=True)
df_weekly = df_us_res.resample('W', on='date').agg({'value': 'mean'}).dropna().reset_index()

t0 = df_weekly['date'].min()
df_weekly['time_year'] = (df_weekly['date'] - t0).dt.days / 365.25

X = df_weekly['time_year'].values.reshape(-1, 1)
y = df_weekly['value'].values

n_train = int(len(X) * 0.85)
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]

dates_train = df_weekly['date'].iloc[:n_train]
dates_test = df_weekly['date'].iloc[n_train:]
split_date = df_weekly['date'].iloc[n_train]

# 2. Definisi 4 Model Kernel Non-RBF
kernel_configs = [
    {
        'title': r'(a) Penjumlahan: $k_{\mathrm{Per}} + k_{\mathrm{RQ}}$',
        'kernel': (
            ConstantKernel(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds=(0.95, 1.05)) +
            ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(0.1, 10.0))
        ),
        'color': 'blue',
        'color_line': 'b-'
    },
    {
        'title': r'(b) Penjumlahan: $k_{\mathrm{Per}} + k_{\mathrm{Lin}}$',
        'kernel': (
            ConstantKernel(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds=(0.95, 1.05)) +
            ConstantKernel(0.1, (1e-3, 1e2)) * DotProduct(sigma_0=1.0)
        ),
        'color': 'purple',
        'color_line': 'purple'
    },
    {
        'title': r'(c) Penjumlahan: $k_{\mathrm{Mat25}} + k_{\mathrm{Per}}$',
        'kernel': (
            ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(0.1, 10.0)) +
            ConstantKernel(1.0, (1e-3, 1e3)) * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds=(0.95, 1.05))
        ),
        'color': 'teal',
        'color_line': 'teal'
    },
    {
        'title': r'(d) Penjumlahan: $k_{\mathrm{Mat15}} + k_{\mathrm{RQ}}$',
        'kernel': (
            ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(0.1, 10.0)) +
            ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(0.1, 10.0))
        ),
        'color': 'darkorange',
        'color_line': 'darkorange'
    }
]

# Dense grid for predictions
X_dense = np.linspace(X.min(), X.max() + 0.1, 600).reshape(-1, 1)
dense_dates = [t0 + pd.Timedelta(days=float(t * 365.25)) for t in X_dense.flatten()]

fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300)

for idx, cfg in enumerate(kernel_configs):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Train GP
    gp = GaussianProcessRegressor(kernel=cfg['kernel'], normalize_y=True, alpha=0.02, n_restarts_optimizer=2, random_state=42)
    gp.fit(X_train, y_train)
    
    # Predict
    mu_dense, std_dense = gp.predict(X_dense, return_std=True)
    
    # 1. Garis Data Observasi Asli (True trajectory)
    ax.plot(df_weekly['date'], df_weekly['value'], 'r--', lw=1.5, label='Data Observasi Asli')
    
    # 2. Kurva Prediksi GP mu_*
    ax.plot(dense_dates, mu_dense, cfg['color_line'], lw=2, label=r'Prediksi GP $\mu_*$')
    
    # 3. Interval Keyakinan 95% (mu +- 2 sigma)
    ax.fill_between(
        dense_dates, mu_dense - 2 * std_dense, mu_dense + 2 * std_dense,
        color=cfg['color'], alpha=0.18, label=r'Interval $95\%$ ($\pm 2\sigma_*$)'
    )
    
    # 4. Scatter Data Latih (Train)
    ax.scatter(dates_train, y_train, c='black', s=16, zorder=5, label=f'Data Latih ($N={n_train}$)')
    
    # 5. Scatter Data Uji (Test)
    ax.scatter(dates_test, y_test, c='red', marker='x', s=20, zorder=5, label=f'Data Uji ($N={len(y_test)}$)')
    
    # Batas Train-Test vertical line
    ax.axvline(split_date, color='gray', linestyle=':', lw=1.2, alpha=0.7)
    
    # Title & styling
    ax.set_title(cfg['title'], fontweight='bold')
    ax.set_xlabel('Tahun Observasi')
    ax.set_ylabel(r'Emisi $\mathrm{CO}_2$ (Mt)')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(df_weekly['date'].min() - pd.Timedelta(days=30), df_weekly['date'].max() + pd.Timedelta(days=50))

plt.tight_layout()

out_dir = r"c:\Users\MyBook Z Series\Desktop\TA\04_dokumen_ta\Dokumen bimbingan\Bimbingan 06\figures"
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, "kernel_fitting_co2.png")
pdf_path = os.path.join(out_dir, "kernel_fitting_co2.pdf")
plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, bbox_inches='tight')
plt.close()

print(f"Gambar berhasil disimpan di:\n- {png_path}\n- {pdf_path}")
