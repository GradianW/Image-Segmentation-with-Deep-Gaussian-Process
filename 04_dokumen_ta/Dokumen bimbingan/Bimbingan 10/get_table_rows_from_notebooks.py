import nbformat, glob, os

cases = [
    ("W skalar", "R3 R1"),
    ("W skalar", "R3 R2"),
    ("W berdistribusi", "R3 R1"),
    ("W berdistribusi", "R3 R2")
]

base = r"04_dokumen_ta/Dokumen bimbingan/Bimbingan 10"

for sifat, sub in cases:
    folder = os.path.join(base, sifat, sub)
    v_files = sorted([f for f in os.listdir(folder) if f.startswith("v") and f.endswith(".ipynb")])
    
    # Read tables from the 4 files
    data = {} # ep -> [loss_v1, loss_v2, loss_v3, loss_v4]
    milestones = [1, 2, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for ep in milestones:
        data[ep] = []
        
    for vf in v_files:
        nb = nbformat.read(os.path.join(folder, vf), as_version=4)
        out_txt = nb.cells[2].outputs[0]["text"]
        lines = [l.strip() for l in out_txt.split("\n") if "|" in l and not l.startswith("Iterasi") and not l.startswith("-")]
        for l in lines:
            parts = [p.strip() for p in l.split("|")]
            if len(parts) >= 2 and parts[0].isdigit():
                ep = int(parts[0])
                loss = float(parts[1])
                data[ep].append(f"{loss:.6f}")
                
    print(f"% === TABEL {sifat.upper()} {sub} ===")
    for ep in milestones:
        print(f"        {ep:<4} & " + " & ".join(data[ep]) + r" \\")

