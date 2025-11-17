import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import argparse
from scripts import matplotlib_lhcb_style

# Apply LHCb style
matplotlib_lhcb_style(plt)

def _annotate_combined(ax, comb_val, comb_err, xpad=0.02, color='k', label=r'$\epsilon_{\text{tag}}$'):
    """
    Writes 'label = value ± err' at the right edge of the axes near y=comb_val.
    xpad is a fractional padding from the right border.
    """
    # Place text at a small padding from the right border, vertically at comb_val.
    ax.text(
        xpad, comb_val,
        rf'{label} = {comb_val:.2f} $\pm$ {comb_err:.2f}',
        color=color, fontsize=12, va='center', ha='left',
        transform=ax.get_yaxis_transform(),  # x in axes fraction, y in data coords
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, lw=0)
    )


def plot_decay(ax, title, singles_run2, singles_run3, comb_run2, comb_run3):
    taggers = list(singles_run2.keys())
    x = np.arange(len(taggers))

    # --- Run 2 single taggers ---
    vals2 = [singles_run2[t][0] for t in taggers]
    errs2 = [singles_run2[t][1] for t in taggers]
    ax.errorbar(x - 0.1, vals2, yerr=errs2, fmt='o', mec='C0', mew=1.5,mfc='none',capsize=3, markersize=7,)

    # --- Run 3 single taggers ---
    vals3 = [singles_run3[t][0] for t in taggers]
    errs3 = [singles_run3[t][1] for t in taggers]
    ax.errorbar(x + 0.1, vals3, yerr=errs3, fmt='s', mec='C3',mew=1.5,mfc='none', capsize=3, markersize=7,)

    # --- Sums (no uncertainty) ---
    sum2 = float(np.sum(vals2))
    sum3 = float(np.sum(vals3))
    ax.axhline(sum2, color='C0', linestyle='--')
    ax.axhline(sum3, color='C3', linestyle='--')

    # --- Combined tagging powers (line + shaded ±1σ band) ---
    comb2_val, comb2_err = comb_run2
    comb3_val, comb3_err = comb_run3

    ax.set_xlim(-0.5, len(taggers) - 0.5)
    ax.axhspan(comb2_val - comb2_err, comb2_val + comb2_err,
               color='C0', alpha=0.22, zorder=0)
    ax.axhspan(comb3_val - comb3_err, comb3_val + comb3_err,
               color='C3', alpha=0.22, zorder=0)
    ax.axhline(comb2_val, color='C0', linestyle='-', linewidth=1.8)
    ax.axhline(comb3_val, color='C3', linestyle='-', linewidth=1.8)

    # --- Centered annotations for Run 2 and Run 3 combined tagging power ---
    ax.text(
        0.45, 0.55,  # position (x in axes fraction, y in data fraction)
        rf"$\epsilon_{{\text{{tag}}}}^{{\text{{Run3}}}} = {comb3_val:.2f} \pm {comb3_err:.2f}$",
        color='black', fontsize=17, ha='center', va='bottom',
        transform=ax.transAxes
    )
    ax.text(
        0.45, 0.53,
        rf"$\epsilon_{{\text{{tag}}}}^{{\text{{Run2}}}} = {comb2_val:.2f} \pm {comb2_err:.2f}$",
        color='black', fontsize=17, ha='center', va='top',
        transform=ax.transAxes
    )

    # --- Style ---
    ax.set_xticks(x)
    ax.set_xticklabels(taggers, rotation=25, fontsize=17)
    ax.set_ylabel(r"$\epsilon_{\text{eff}}$ [%]", fontsize=17)
    ax.set_title(title, fontsize=17)
    ax.grid(axis='y', linestyle=':', alpha=0.8)

def main(save=False):
    # === Data (MC) ===
    Bu_Run2 = {"OSElectron": (0.24, 0.02), "OSMuon": (0.79, 0.03), "OSKaon": (1.88, 0.05)}
    Bu_Run3 = {"OSElectron": (0.47, 0.03), "OSMuon": (1.08, 0.04), "OSKaon": (2.02, 0.06)}
    Bu_Run2_comb = (2.86, 0.06)
    Bu_Run3_comb = (3.48, 0.07)

    Bs_Run2 = {"OSElectron": (0.28, 0.03), "OSMuon": (0.86, 0.06), "OSKaon": (1.79, 0.08), "SSKaon": (2.97, 0.10)}
    Bs_Run3 = {"OSElectron": (0.47, 0.04), "OSMuon": (1.05, 0.06), "OSKaon": (1.78, 0.08), "SSKaon": (2.81, 0.10)}
    Bs_Run2_comb = (5.69, 0.13)
    Bs_Run3_comb = (5.85, 0.13)



    Bd_Run2 = {"OSElectron": (0.34, 0.04), "OSMuon": (0.78, 0.06), "OSKaon": (1.84, 0.10),
               "SSPion": (1.20, 0.08), "SSProton": (0.12, 0.02)}
    Bd_Run3 = {"OSElectron": (0.52, 0.05), "OSMuon": (1.01, 0.07), "OSKaon": (1.83, 0.10),
               "SSPion": (1.11, 0.08), "SSProton": (0.07, 0.02)}
    Bd_Run2_comb = (4.14, 0.14)
    Bd_Run3_comb = (4.46, 0.15)

    # === Plot ===
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    plt.subplots_adjust(top=0.73, wspace=0.25)  # leave space for shared legend

    plot_decay(axes[1], r"$B_s^0 \to D_s^- \pi^+$", Bs_Run2, Bs_Run3, Bs_Run2_comb, Bs_Run3_comb)
    plot_decay(axes[0], r"$B^+ \to J/\psi K^+$",   Bu_Run2, Bu_Run3, Bu_Run2_comb, Bu_Run3_comb)
    plot_decay(axes[2], r"$B^0 \to J/\psi K^{*0}$", Bd_Run2, Bd_Run3, Bd_Run2_comb, Bd_Run3_comb)

    # --- Shared (figure-level) legend ---
    legend_elements = [
        Line2D([0], [0], marker='o', mec='C0',color='none', mew=1.5, mfc='none',markerfacecolor='C0', markersize=7, label=r'Run2 single taggers'),
        Line2D([0], [0], marker='s', mec='C3',color='none', mew=1.5, mfc='none',markerfacecolor='C3', markersize=7, label=r'Run3 single taggers'),
        Line2D([0], [0], color='C0', linestyle='--', label=rf'$\epsilon_{{\text{{tag}}}}^{{\text{{Run2}}}}$ sum'),
        Line2D([0], [0], color='C3', linestyle='--', label=rf'$\epsilon_{{\text{{tag}}}}^{{\text{{Run3}}}}$ sum'),
        Line2D([0], [0], color='C0', linestyle='-',  label=rf'$\epsilon_{{\text{{tag}}}}^{{\text{{Run2}}}}$ comb.'),
        Line2D([0], [0], color='C3', linestyle='-',  label=rf'$\epsilon_{{\text{{tag}}}}^{{\text{{Run3}}}}$ comb.'),
        Patch(facecolor='C0', alpha=0.22, edgecolor='none', label=r'$\pm 1\sigma$ Run2 comb.'),
        Patch(facecolor='C3', alpha=0.22, edgecolor='none', label=r'$\pm 1\sigma$ Run3 comb.'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.90),
        ncol=4,
        fontsize=13,
        frameon=False
    )

    #fig.suptitle("Calibrated Tagging Powers — Run 2 vs Run 3 (MC)", fontsize=16, y=0.92)

    if save:
        fig.savefig("tagging_powers_Run2_vs_Run3_MC.pdf", bbox_inches='tight')
        print("✅ Figure saved as tagging_powers_Run2_vs_Run3_MC.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save plot instead of showing it")
    args = parser.parse_args()
    main(save=args.save)
