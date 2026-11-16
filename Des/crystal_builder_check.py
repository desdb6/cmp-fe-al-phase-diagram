"""Check that read_cif() gives cells in which pw.x can detect ALL symmetry operations.
Run in the folder containing crystal_builder_edit.py and utils.py:  python check_symmetry.py"""
import glob, itertools, os, tempfile
import numpy as np, spglib
import Des.crystal_builder as cb

# Rotations that pw.x tests in the Cartesian frame: 24 cubic + 12 hexagonal (each also times inversion)
cands = []
for p in itertools.permutations(range(3)):
    for s in itertools.product([1, -1], repeat=3):
        M = np.zeros((3, 3))
        for i in range(3):
            M[i, p[i]] = s[i]
        if np.linalg.det(M) > 0:
            cands.append(M)
for k in range(6):
    c, s = np.cos(k*np.pi/3), np.sin(k*np.pi/3)
    cands.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
    c, s = np.cos(k*np.pi/3), np.sin(k*np.pi/3)          # 2-fold in-plane axes at k*30 deg
    cands.append(np.array([[c, s, 0], [s, -c, 0], [0, 0, -1]]))

def count(atoms):
    M = np.asarray(atoms.cell, dtype=float)
    cell = (
        M.tolist(),
        np.asarray(atoms.get_scaled_positions(), dtype=float).tolist(),
        np.asarray(atoms.numbers, dtype=int).tolist(),
    )
    ds = spglib.get_symmetry_dataset(cell, symprec=1e-4)
    if ds is None:
        raise ValueError(f"spglib could not determine the symmetry for {atoms}")
    found = 0
    for R in ds.rotations:
        Rc = M.T @ R @ np.linalg.inv(M.T)
        Rp = Rc if np.linalg.det(Rc) > 0 else -Rc
        found += any(np.allclose(Rp, C, atol=1e-4) for C in cands)
    return ds.international, len(ds.rotations), found

out = tempfile.mkdtemp()
structs = cb.gen_sym_ineq_supercells(3, 2, 11.8, cell_type="primitive")   # 27 atoms, 2 Al
cb.write_results_ineq_supercells(structs, out)
print("\n%-34s %-8s %5s %8s %8s" % ("file", "group", "ops", "no align", "aligned"))
for f in sorted(glob.glob(out + "/*.cif")):
    sg, n, a = count(cb.read_cif(f, align=False))
    _, _, b = count(cb.read_cif(f))
    print("%-34s %-8s %5d %8d %8d  %s" % (os.path.basename(f), sg, n, a, b, "OK" if b == n else "PROBLEM"))