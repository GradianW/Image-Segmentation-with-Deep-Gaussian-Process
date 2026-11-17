from generate_all_grid_figures import results, variations, cases

eps = [1, 2, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
for (sifat, tdim) in cases:
    print(f"% === TABEL: {sifat.upper()} R3 -> R{tdim} ===")
    for ep in eps:
        losses = [f"{results[(sifat, tdim)][v][ep][0]:.6f}" for v, _, _, _ in variations]
        print(f"        {ep:<4} & " + " & ".join(losses) + r" \\")

