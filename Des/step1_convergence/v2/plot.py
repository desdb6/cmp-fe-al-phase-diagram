import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


def plot_convergence(x_label: str, df: pd.DataFrame):
    x = df[x_label]
    wall = df["Wall time"]

    # One shared color scale so both plots are comparable
    norm = mpl.colors.Normalize(vmin=wall.min(), vmax=wall.max())
    cmap = "viridis"

    fig, (ax_e, ax_p) = plt.subplots(
        2, 1, figsize=(7, 8), sharex=True, constrained_layout=True
    )

    # --- Energy ---
    ax_e.plot(x, df["energy (eV)"], color="gray", lw=0.8, zorder=1)
    sc = ax_e.scatter(
        x, df["energy (eV)"], c=wall, cmap=cmap, norm=norm,
        s=50, edgecolor="k", linewidth=0.5, zorder=2,
    )
    ax_e.set_ylabel("energy (eV)")
    ax_e.grid(alpha=0.3)

    # --- Pressure (left axis) and moment (right axis) ---
    ax_m = ax_p.twinx()

    ax_p.plot(x, df["pressure (kbar)"], color="gray", lw=0.8, zorder=1)
    ax_p.scatter(
        x, df["pressure (kbar)"], c=wall, cmap=cmap, norm=norm,
        marker="o", s=50, edgecolor="k", linewidth=0.5, zorder=2,
        label="pressure",
    )
    ax_p.set_ylabel("pressure (kbar)  [circles]")

    ax_m.plot(x, df["moment (muB)"], color="gray", lw=0.8, ls="--", zorder=1)
    ax_m.scatter(
        x, df["moment (muB)"], c=wall, cmap=cmap, norm=norm,
        marker="^", s=60, edgecolor="k", linewidth=0.5, zorder=2,
        label="moment",
    )
    ax_m.set_ylabel("moment (muB)  [triangles]")

    ax_p.set_xlabel(x_label)
    ax_p.grid(alpha=0.3)

    # Single colorbar for wall time shared by both plots
    cbar = fig.colorbar(sc, ax=[ax_e, ax_p], pad=0.12)
    cbar.set_label("Wall time")

    return fig, (ax_e, ax_p, ax_m)


df_kmesh = pd.read_csv("/home/des/cmp-fe-al-phase-diagram/Des/step1_convergence/v2/kmesh/results.csv")
df_ecutwfc = pd.read_csv("/home/des/cmp-fe-al-phase-diagram/Des/step1_convergence/v2/ecutwfc/results.csv")
df_ecutrho = pd.read_csv("/home/des/cmp-fe-al-phase-diagram/Des/step1_convergence/v2/ecutrho/results.csv")

fig_kmesh, _ = plot_convergence("k1", df_kmesh)
plt.savefig("/home/des/cmp-fe-al-phase-diagram/Des/step1_convergence/v2/kmesh/conv_plot.png")

fig_ecutwfc, _ = plot_convergence("ecutwfc", df_ecutwfc)
plt.savefig("/home/des/cmp-fe-al-phase-diagram/Des/step1_convergence/v2/ecutwfc/conv_plot.png")

fig_ecutrho, _ = plot_convergence("ecutrho", df_ecutrho)
plt.savefig("/home/des/cmp-fe-al-phase-diagram/Des/step1_convergence/v2/ecutrho/conv_plot.png")
