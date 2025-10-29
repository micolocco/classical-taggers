import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import argparse
from scripts import matplotlib_lhcb_style

# Apply LHCb style
matplotlib_lhcb_style(plt)

def plot_decay(ax, title,
               singles_run2, singles_run3,
               comb_run2, comb_run3,
               singles_run2_mc=None, singles_run3_mc=None,
               mc_run2_comb=None, mc_run3_comb=None):
    """Plot tagging powers for Data (Run 2 & 3) and optional MC overlay."""
    taggers = list(singles_run2.keys())
    x = np.arange(len(taggers))

    # === DATA single taggers ===
    vals2 = [singles_run2[t][0] for t in taggers]
    errs2 = [singles_run2[t][1] for t in taggers]
    vals3 = [singles_run3[t][0] for t in taggers]
    errs3 = [singles_run3[t][1] for t in taggers]

    ax.errorbar(x - 0.12, vals2, yerr=errs2, fmt='o', color='C0', capsize=3, markersize=7)
    ax.errorbar(x + 0.12, vals3, yerr=errs3, fmt='s', color='C3', capsize=3, markersize=7)

    # === MC single taggers (empty markers) ===
    if singles_run2_mc:
        mc_vals2 = [singles_run2_mc[t][0] for t in taggers]
        mc_errs2 = [singles_run2_mc[t][1] for t in taggers]
        ax.errorbar(x - 0.12, mc_vals2, yerr=mc_errs2, fmt='o', mfc='none', mec='C0',
                    capsize=3, markersize=7, mew=1.5)

    if singles_run3_mc:
        mc_vals3 = [singles_run3_mc[t][0] for t in taggers]
        mc_errs3 = [singles_run3_mc[t][1] for t in taggers]
        ax.errorbar(x + 0.12, mc_vals3, yerr=mc_errs3, fmt='s', mfc='none', mec='C3',
                    capsize=3, markersize=7, mew=1.5)

    # === Sums (Data) ===
    sum2, sum3 = np.sum(vals2), np.sum(vals3)
    ax.axhline(sum2, color='C0', linestyle='--')
    ax.axhline(sum3, color='C3', linestyle='--')

    # === Combined Tagging Power (Data) ===
    d2, d2e = comb_run2
    d3, d3e = comb_run3
    ax.axhspan(d2 - d2e, d2 + d2e, color='C0', alpha=0.22)
    ax.axhspan(d3 - d3e, d3 + d3e, color='C3', alpha=0.22)
    ax.axhline(d2, color='C0', linestyle='-', linewidth=1.8)
    ax.axhline(d3, color='C3', linestyle='-', linewidth=1.8)

    # === Combined Tagging Power (MC) ===
    if mc_run2_comb:
        m2, m2e = mc_run2_comb
        ax.axhspan(m2 - m2e, m2 + m2e, color='C0', alpha=0.10)
        ax.axhline(m2, color='C0', linestyle=':', linewidth=1.8)
    if mc_run3_comb:
        m3, m3e = mc_run3_comb
        ax.axhspan(m3 - m3e, m3 + m3e, color='C3', alpha=0.10)
        ax.axhline(m3, color='C3', linestyle=':', linewidth=1.8)

    # === Centered text for Data ===
    ax.text(0.45, 0.55, rf"$\epsilon_{{\text{{tag}}}}^{{\text{{Run3}}}} = {d3:.2f} \pm {d3e:.2f}$",
            color='black', fontsize=16, ha='center', va='bottom', transform=ax.transAxes)
    ax.text(0.45, 0.53, rf"$\epsilon_{{\text{{tag}}}}^{{\text{{Run2}}}} = {d2:.2f} \pm {d2e:.2f}$",
            color='black', fontsize=16, ha='center', va='top', transform=ax.transAxes)

    # === Style ===
    ax.set_xticks(x)
    ax.set_xticklabels(taggers, rotation=25, fontsize=15)
    ax.set_ylabel(r"$\epsilon_{\text{eff}}$ [%]", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.grid(axis='y', linestyle=':', alpha=0.8)
    ax.set_xlim(-0.5, len(taggers) - 0.5)


def main(save=False):
    # --- DATA ---
    Bu_Run2 = {"OSElectron": (0.203, 0.015), "OSMuon": (0.72, 0.03), "OSKaon": (1.01, 0.03)}
    Bu_Run3 = {"OSElectron": (0.34, 0.02), "OSMuon": (0.82, 0.03), "OSKaon": (0.88, 0.03)}
    Bu_Run2_comb_data = (1.92, 0.05)
    Bu_Run3_comb_data = (2.03, 0.05)

    Bd_Run2 = {"OSElectron": (0.24, 0.05), "OSMuon": (0.52, 0.07), "OSKaon": (1.03, 0.11),
               "SSPion": (1.08, 0.08), "SSProton": (0.33, 0.06)}
    Bd_Run3 = {"OSElectron": (0.46, 0.08), "OSMuon": (0.62, 0.09), "OSKaon": (0.89, 0.11),
               "SSPion": (0.95, 0.08), "SSProton": (0.26, 0.04)}
    Bd_Run2_comb_data = (3.06, 0.17)
    Bd_Run3_comb_data = (3.16, 0.19)

    # --- MC ---
    Bu_Run2_MC = {"OSElectron": (0.24, 0.02), "OSMuon": (0.79, 0.03), "OSKaon": (1.88, 0.05)}
    Bu_Run3_MC = {"OSElectron": (0.47, 0.03), "OSMuon": (1.08, 0.04), "OSKaon": (2.02, 0.06)}
    Bu_Run2_comb_MC = (2.86, 0.06)
    Bu_Run3_comb_MC = (3.48, 0.07)

    Bd_Run2_MC = {"OSElectron": (0.34, 0.04), "OSMuon": (0.78, 0.06), "OSKaon": (1.84, 0.10),
                  "SSPion": (1.20, 0.08), "SSProton": (0.12, 0.02)}
    Bd_Run3_MC = {"OSElectron": (0.52, 0.05), "OSMuon": (1.01, 0.07), "OSKaon": (1.83, 0.10),
                  "SSPion": (1.11, 0.08), "SSProton": (0.07, 0.02)}
    Bd_Run2_comb_MC = (4.14, 0.14)
    Bd_Run3_comb_MC = (4.46, 0.15)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    plt.subplots_adjust(top=0.73, wspace=0.3)

    # B+ (OS)
    plot_decay(axes[0], r"$B^+ \to J/\psi K^+$",
               Bu_Run2, Bu_Run3,
               Bu_Run2_comb_data, Bu_Run3_comb_data,
               singles_run2_mc=Bu_Run2_MC, singles_run3_mc=Bu_Run3_MC,
               mc_run2_comb=Bu_Run2_comb_MC, mc_run3_comb=Bu_Run3_comb_MC)

    # B0 (OS+SS)
    plot_decay(axes[1], r"$B^0 \to J/\psi K^{*0}$",
               Bd_Run2, Bd_Run3,
               Bd_Run2_comb_data, Bd_Run3_comb_data,
               singles_run2_mc=Bd_Run2_MC, singles_run3_mc=Bd_Run3_MC,
               mc_run2_comb=Bd_Run2_comb_MC, mc_run3_comb=Bd_Run3_comb_MC)

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='C0', linestyle='none', markersize=7, label='Data Run2 single'),
        Line2D([0], [0], marker='s', color='C3', linestyle='none', markersize=7, label='Data Run3 single'),
        Line2D([0], [0], marker='o', color='C0', linestyle='none', markersize=7, mew=1.5, mfc='none', label='MC Run2 single'),
        Line2D([0], [0], marker='s', color='C3', linestyle='none', markersize=7, mew=1.5, mfc='none',label='MC Run3 single'),
        Line2D([0], [0], color='C0', linestyle='--', label='Data Run2 sum'),
        Line2D([0], [0], color='C3', linestyle='--', label='Data Run3 sum'),
        Line2D([0], [0], color='C0', linestyle='-', label='Data Run2 comb.'),
        Line2D([0], [0], color='C3', linestyle='-', label='Data Run3 comb.'),
        Line2D([0], [0], color='C0', linestyle=':', label='MC Run2 comb.'),
        Line2D([0], [0], color='C3', linestyle=':', label='MC Run3 comb.')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.90),
               ncol=4, fontsize=12, frameon=False)

    if save:
        fig.savefig("tagging_powers_DATA_with_MC_openmarkers.pdf", bbox_inches='tight')
        print("✅ Figure saved as tagging_powers_DATA_with_MC_openmarkers.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save plot instead of showing it")
    args = parser.parse_args()
    main(save=args.save)
