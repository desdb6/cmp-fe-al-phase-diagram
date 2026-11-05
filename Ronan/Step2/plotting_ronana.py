import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# ============================================================
# DATA: BCC Fe, 12x12x12, ecutwfc=120 Ry, ecutrho=1000 Ry
# ============================================================

data = {
    "volume_A3": [
        21.31005867,
        21.53676142,
        21.76346417,
        21.99016692,
        22.21686968,
        22.44357243,
        22.67027518,
        22.89697793,
        23.12368068,
        23.35038344,
        23.57708619,
        23.80378894,
        24.03049169,
    ],

    "energy_Ry": [
        -658.53965558,
        -658.54107002,
        -658.54220240,
        -658.54306910,
        -658.54368187,
        -658.54405544,
        -658.54420225,
        -658.54413189,
        -658.54385810,
        -658.54339029,
        -658.54274281,
        -658.54192364,
        -658.54094866,
    ],
}

df = pd.DataFrame(data)

V = df["volume_A3"].to_numpy()
E = df["energy_Ry"].to_numpy()


# ============================================================
# THIRD-ORDER BIRCH-MURNAGHAN EQUATION OF STATE
# ============================================================

def birch_murnaghan(V, E0, V0, B0, B0_prime):

    eta = (V0 / V)**(2 / 3) - 1

    return E0 + (9 * V0 * B0 / 16) * (
        B0_prime * eta**3
        + (6 - 4 * (eta + 1)) * eta**2
    )


# ============================================================
# FIT THE EQUATION OF STATE
# ============================================================

p0 = [E.min(), V[np.argmin(E)], 0.09, 4.0]

popt, pcov = curve_fit(
    birch_murnaghan,
    V,
    E,
    p0=p0,
    maxfev=100000
)

E0, V0, B0, B0_prime = popt

errors = np.sqrt(np.diag(pcov))

B0_GPa = B0 * 2179.872

# Lattice parameter for conventional cubic bcc cell
a0 = V0**(1/3)

print("\n========== BIRCH-MURNAGHAN EOS FIT ==========")
print(f"Equilibrium volume V0 = {V0:.6f} A^3")
print(f"Equilibrium lattice parameter a0 = {a0:.6f} A")
print(f"Equilibrium energy E0 = {E0:.9f} Ry")
print(f"Bulk modulus B0 = {B0_GPa:.2f} GPa")
print(f"Pressure derivative B0' = {B0_prime:.4f}")
print("==============================================\n")


# ============================================================
# PLOT DFT DATA AND FITTED CURVE
# ============================================================

V_fit = np.linspace(V.min(), V.max(), 500)
E_fit = birch_murnaghan(V_fit, *popt)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(V, E, "o", label="DFT calculations")
ax.plot(V_fit, E_fit, "-", label="Birch-Murnaghan fit")

ax.set_xlabel(r"Volume ($\mathrm{\AA}^3$)")
ax.set_ylabel("Total energy (Ry)")
ax.set_title("Equation of state of bcc iron")

ax.grid(alpha=0.5)
ax.legend()

fig.tight_layout()


# ============================================================
# SAVE OUTPUTS NEXT TO THIS PYTHON FILE
# ============================================================

folder = Path(__file__).resolve().parent

df.to_csv(
    folder / "FeBcc_EOS_data(k=12).csv",
    index=False
)

fig.savefig(
    folder / "FeBcc_Birch_Murnaghan(k=12).png",
    dpi=200
)

plt.show()