
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data: volume (A^3) and energy (Ry)
data = {
    "volume_A3": [
    21.30152257,
    21.52814145,
    21.75474551,
    21.98136439,
    22.20798326,
    22.43458733,
    22.66120620,
    22.88781026,
    23.11442914,
    23.34103320,
    23.56765208,
    23.79425614,
    24.02087502,
 ],
"energy_Ry": [
    -658.539040190000,
    -658.540392430000,
    -658.541464400000,
    -658.542275050000,
    -658.542840370000,
    -658.543178550000,
    -658.543301660000,
    -658.543221770000,
    -658.542947840000,
    -658.542488180000,
    -658.541852430000,
    -658.541049310000,
    -658.540088450000,
],
}

df = pd.DataFrame(data)
V = df["volume_A3"].to_numpy()
E = df["energy_Ry"].to_numpy()


def birch_murnaghan(V, E0, V0, B0, B0_prime):
    eta = (V0 / V)**(2 / 3) - 1
    return E0 + (9 * V0 * B0 / 16) * (
        B0_prime * eta**3 + (6 - 4 * (eta + 1)) * eta**2
    )


# Fit the equation of state
p0 = [E.min(), V[np.argmin(E)], 0.09, 4.0]
popt, pcov = curve_fit(
    birch_murnaghan, V, E, p0=p0, maxfev=100000
)

E0, V0, B0, B0_prime = popt
errors = np.sqrt(np.diag(pcov))
B0_GPa = B0 * 2179.872

print(f"Equilibrium volume V0 = {V0:.6f} A^3")
print(f"Equilibrium energy E0 = {E0:.9f} Ry")
print(f"Bulk modulus B0 = {B0_GPa:.2f} GPa")
print(f"Pressure derivative B0' = {B0_prime:.4f}")

# Plot DFT data and fitted curve
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

# Save outputs next to this Python file
from pathlib import Path
folder = Path(__file__).resolve().parent
df.to_csv(folder / "FeBcc_EOS_data(k=8).csv", index=False)
fig.savefig(folder / "FeBcc_Birch_Murnaghan.png", dpi=200)

plt.show()
