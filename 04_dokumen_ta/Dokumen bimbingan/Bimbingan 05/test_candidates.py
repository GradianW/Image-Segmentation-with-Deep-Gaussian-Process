import numpy as np
import matplotlib.pyplot as plt

# Test various candidates of linear * sin(x)
x = np.linspace(-12, 12, 500)

plt.figure(figsize=(10, 6))
plt.plot(x, 0.3 * x * np.sin(x), label=r"$f(x) = 0.3 x \sin(x)$ (period $2\pi \approx 6.28$)")
plt.plot(x, 0.2 * x * np.sin(np.pi * x), label=r"$f(x) = 0.2 x \sin(\pi x)$ (period $2.0$)")
plt.plot(x, 0.25 * x * np.sin(1.5 * x), label=r"$f(x) = 0.25 x \sin(1.5 x)$ (period $4.19$)")
plt.legend()
plt.title("Candidate functions")
plt.savefig("test_candidates.png")
plt.close()
print("Candidate test generated.")
