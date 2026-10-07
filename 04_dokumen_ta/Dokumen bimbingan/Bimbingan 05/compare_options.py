import numpy as np
import matplotlib.pyplot as plt

# Compare the two options
# Option 1: f(x) = 0.3 x sin(x) with p = 2*pi
# Option 2: f(x) = 0.2 x sin(2*pi*x / 2.0) with p = 2.0

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=150)

x = np.linspace(-13, 13, 600)
axes[0].plot(x, 0.3 * x * np.sin(x), 'r-', lw=2)
axes[0].set_title(r"$f(x) = 0.3 x \sin(x)$")
axes[0].grid(True)

axes[1].plot(x, 0.2 * x * np.sin(2 * np.pi * x / 2.0), 'b-', lw=2)
axes[1].set_title(r"$f(x) = 0.2 x \sin\left(\frac{2\pi x}{2.0}\right)$")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("compare_options.png")
plt.close()
print("Comparison saved.")
