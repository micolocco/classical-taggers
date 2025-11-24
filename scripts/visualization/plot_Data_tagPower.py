import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import argparse
from scripts import matplotlib_lhcb_style

# Apply LHCb style
matplotlib_lhcb_style(plt)

def _pair_text_side_by_side(ax, y_data, left_text, right_text, fontsize=15, color_left='black', color_right='black', dy=0.0):
    """
    Place two texts side-by-side around the center (x in axes fraction), at a given data y.
    """
    ax.text(0.47, y_data + dy, left_text, color=color_left, fontsize=fontsize,
            ha='right', va='top', transform=ax.get_yaxis_transform())
    ax.text(0.53, y_data + dy, right_text, color=color_right, fontsize=fontsize,
            ha='left', va='top', transform=ax.get_yaxis_transform())

def plot_decay(ax, title,
               singles_run2, singles_run3,
               comb_run2, comb_run3,
               singles_run2_mc=None, singles_run3_mc=None,
               mc_run2_comb=None, mc_run3_comb=None):
    """
    Plot tagging powers for Data (Run 2 & 3) and optional MC overlay.
    Returns a dict of which plot elements were drawn for legend building.
    """
    taggers = list(singles_run2.keys())
    x = np.arange(len(taggers))

    # Detect if this subplot has any MC overlay
    has_mc = any([
        singles_run2_mc is not None, singles_run3_mc is not None,
        mc_run2_comb is not None, mc_run3_comb is not None
    ])

    drawn = {
        'data_single_run2': True,
        'data_single_run3': True,
        'mc_single_run2': False,
        'mc_single_run3': False,
        'data_sum_run2': False,
        'data_sum_run3': False,
        'data_comb_run2': True,
        'data_comb_run3': True,
        'data_band_run2': True,
        'data_band_run3': True,
        'mc_comb_run2': False,
        'mc_comb_run3': False,
        'mc_band_run2': False,
        'mc_band_run3': False,
    }

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
        drawn['mc_single_run2'] = True

    if singles_run3_mc:
        mc_vals3 = [singles_run3_mc[t][0] for t in taggers]
        mc_errs3 = [singles_run3_mc[t][1] for t in taggers]
        ax.errorbar(x + 0.12, mc_vals3, yerr=mc_errs3, fmt='s', mfc='none', mec='C3',
                    capsize=3, markersize=7, mew=1.5)
        drawn['mc_single_run3'] = True

    # === Optional Sums (Data) — only if NO MC is plotted in this subplot ===
    if not has_mc:
        sum2, sum3 = np.sum(vals2), np.sum(vals3)
        ax.axhline(sum2, color='C0', linestyle='--')
        ax.axhline(sum3, color='C3', linestyle='--')
        drawn['data_sum_run2'] = True
        drawn['data_sum_run3'] = True

    # === Combined Tagging Power (Data) ===
    d2, d2e = comb_run2
    d3, d3e = comb_run3
    ax.axhspan(d2 - d2e, d2 + d2e, color='C0', alpha=0.22)
    ax.axhspan(d3 - d3e, d3 + d3e, color='C3', alpha=0.22)
    ax.axhline(d2, color='C0', linestyle='-', linewidth=1.8)
    ax.axhline(d3, color='C3', linestyle='-', linewidth=1.8)

    # === Combined Tagging Power (MC) ===
    m2 = m2e = m3 = m3e = None
    if mc_run2_comb:
        m2, m2e = mc_run2_comb
        ax.axhspan(m2 - m2e, m2 + m2e, color='C0', alpha=0.10)
        ax.axhline(m2, color='C0', linestyle=':', linewidth=1.8)
        drawn['mc_band_run2'] = True
        drawn['mc_comb_run2'] = True
    if mc_run3_comb:
        m3, m3e = mc_run3_comb
        ax.axhspan(m3 - m3e, m3 + m3e, color='C3', alpha=0.10)
        ax.axhline(m3, color='C3', linestyle=':', linewidth=1.8)
        drawn['mc_band_run3'] = True
        drawn['mc_comb_run3'] = True

    # === Place side-by-side numeric labels under the bands ===
    # Compute a small offset (in data units) for text placement just under the lower edge of the bands.
    ymin, ymax = ax.get_ylim()
    dy = 0.02 * (ymax - ymin)

    # Data labels: side-by-side, just below the lower edge of the DATA bands (use the lower of the two)
    y_data_text = min(d2 - d2e, d3 - d3e) - dy
    _pair_text_side_by_side(
        ax, y_data_text,
        left_text = rf"$\epsilon_{{\text{{tag, data}}}}^{{\text{{Run3}}}} = {d3:.2f} \pm {d3e:.2f}$",
        right_text = rf"$\epsilon_{{\text{{tag, data}}}}^{{\text{{Run2}}}} = {d2:.2f} \pm {d2e:.2f}$",
        fontsize=15, color_left='black', color_right='black', dy=0.0
    )

    # MC labels (if present): side-by-side, just below the lower edge of the MC bands (use the lower of the two)
    if (m2 is not None and m2e is not None) or (m3 is not None and m3e is not None):
        # Use available values; if one is missing, fall back safely to the other
        mc_low_edges = []
        if (m2 is not None and m2e is not None): mc_low_edges.append(m2 - m2e)
        if (m3 is not None and m3e is not None): mc_low_edges.append(m3 - m3e)
        y_mc_text = (min(mc_low_edges) if mc_low_edges else ymin) - dy

        # Build texts depending on availability
        mc_left =  rf"$\epsilon_{{\text{{tag, MC}}}}^{{\text{{Run3}}}} = {m3:.2f} \pm {m3e:.2f}$" if (m3 is not None and m3e is not None) else ""
        mc_right = rf"$\epsilon_{{\text{{tag, MC}}}}^{{\text{{Run2}}}} = {m2:.2f} \pm {m2e:.2f}$" if (m2 is not None and m2e is not None) else ""
        _pair_text_side_by_side(
            ax, y_mc_text,
            left_text = mc_left,
            right_text = mc_right,
            fontsize=14,  dy=0.0
        )

    # === Style ===
    ax.set_xticks(x)
    ax.set_xticklabels(taggers, rotation=25, fontsize=15)
    ax.set_ylabel(r"$\epsilon_{\text{eff}}$ [%]", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.grid(axis='y', linestyle=':', alpha=0.8)
    ax.set_xlim(-0.5, len(taggers) - 0.5)

    return drawn

def plot_summary_figure(Bd_Run2_2018_comb_data,
                        Bd_Run2_comb_data,
                        Bd_Run3_comb_data,
                        Bd_Mixed_comb_data,
                        save=False):
    """
    Summary figure comparing:
      - Run2 taggers on 2018 data  (reference band)
      - Run2 taggers on Run3 data  (point + error bar)
      - Run3 taggers on Run3 data  (point + error bar)
      - Mixed configuration on Run3 data (point + error bar)

    The 2018 Run2 combination is shown as a horizontal band across the plot,
    while the other three configurations are plotted as markers with error bars
    at separate x positions, with different colours.
    """

    # Color-blind friendly Okabe–Ito palette
    CBLUE   = "#0072B2"
    CORANGE = "#E69F00"
    CRED    = "#D55E00"
    CGREEN  = "#009E73"

    # Unpack values
    r2_2018_val, r2_2018_err = Bd_Run2_2018_comb_data
    r2_run3_val, r2_run3_err = Bd_Run2_comb_data
    r3_run3_val, r3_run3_err = Bd_Run3_comb_data
    mix_val,     mix_err     = Bd_Mixed_comb_data

    fig, ax = plt.subplots(figsize=(7, 6))

    # --- Reference band: Run2 on 2018 data ---
    x_min, x_max = -0.5, 2.5
    ax.set_xlim(x_min, x_max)

    ax.axhspan(r2_2018_val - r2_2018_err,
               r2_2018_val + r2_2018_err,
               color=CORANGE, alpha=0.20)
    ax.axhline(r2_2018_val, color=CORANGE, linewidth=2, label=r"Run2 comb. on 2018 data")
    # Place the 2018 label under the band
    ymin_text = (r2_2018_val - r2_2018_err) - 0.05  # adjust spacing as needed
    #ax.text(0.5, ymin_text,
    #        rf"{r2_2018_val:.2f} ± {r2_2018_err:.2f}",
    #        ha='center', va='top',
    #        fontsize=16, transform=ax.get_yaxis_transform())
    ## --- Points with error bars for the three Run3-era configurations ---
    labels = [
        r"Run2 comb. on 2024 data",
        r"Run3 comb. on 2024 data",
        r"Run3 OSMuon, OSElectron + Run2 OSKaon, SSPion, SSProton on 2024 data",
    ]
   # labels = [
   # "Run2 taggers\non 2024 data",
   # "Run3 taggers\non 2024 data",
   # "Run3 OSMu, OSE + \n Run2 OSK, SSPi, SSP \non on 2024 data",
   # ]

    x = np.arange(len(labels))  # 0, 1, 2

    vals = [r2_run3_val, r3_run3_val, mix_val]
    errs = [r2_run3_err, r3_run3_err, mix_err]
    colors = [CBLUE, CRED, CGREEN]
    markers = ["o", "s", "D"]

    for xi, yi, ei, ci, mk, lab in zip(x, vals, errs, colors, markers, labels):
        ax.errorbar(xi, yi, yerr=ei, fmt=mk, color=ci,
                    capsize=4, markersize=8, label=lab)
        ax.text(xi, yi + ei+  0.05, f"{yi:.2f}±{ei:.2f}",
                ha='center', va='bottom', fontsize=16)

    #ax.set_xticks(x)
    ax.tick_params(axis='x', labelbottom=False)
    #ax.set_xticklabels(labels, rotation=0, fontsize=12)
    ax.set_ylabel(r"$\epsilon_{\text{eff}}$ [%]", fontsize=18)
    ax.set_title("OS+SS combination in "+r"$B^0 \to J/\psi K^{*0}$", fontsize=18)
    ax.grid(axis='y', linestyle=':', alpha=0.8)

    # Add a bit of headroom for labels
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.3)

    # Legend: one entry for the reference band + the 3 points
    ax.legend(loc='upper left', fontsize=11, frameon=False)

    if save:
        fig.savefig("tagging_powers_Bd_summary.pdf", bbox_inches='tight')
        print("✅ Summary figure saved as tagging_powers_Bd_summary.pdf")
    else:
        plt.show()



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
    # Run 2 taggers evaluated on 2018 data (Bd)
    Bd_Run2_2018_comb_data = (3.95, 0.08)
     # Mixed configuration: Run3 e, mu + Run2 K, pi, p on Run3 data
    Bd_Mixed_comb_data = (3.43, 0.19)
    

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))
    plt.subplots_adjust(top=0.73, wspace=0.3)

    # Left: B+ (OS) with MC overlay
    drawn_left = plot_decay(
        axes[0], r"$B^+ \to J/\psi K^+$",
        Bu_Run2, Bu_Run3,
        Bu_Run2_comb_data, Bu_Run3_comb_data,
        singles_run2_mc=Bu_Run2_MC, singles_run3_mc=Bu_Run3_MC,
        mc_run2_comb=Bu_Run2_comb_MC, mc_run3_comb=Bu_Run3_comb_MC
    )

    # Right: B0 (OS+SS) with MC overlay
    drawn_right = plot_decay(
        axes[1], r"$B^0 \to J/\psi K^{*0}$",
        Bd_Run2, Bd_Run3,
        Bd_Run2_comb_data, Bd_Run3_comb_data,
        singles_run2_mc=Bd_Run2_MC, singles_run3_mc=Bd_Run3_MC,
        mc_run2_comb=Bd_Run2_comb_MC, mc_run3_comb=Bd_Run3_comb_MC
    )

    # ==== Dynamic legend across all subplots ====
    drawn_all = {k: (drawn_left.get(k, False) or drawn_right.get(k, False))
                 for k in set(drawn_left) | set(drawn_right)}

    legend_elements = []

    if drawn_all['data_single_run2']:
        legend_elements.append(Line2D([0], [0], marker='o', color='C0', linestyle='none', markersize=7, label='Data Run2 single'))
    if drawn_all['data_single_run3']:
        legend_elements.append(Line2D([0], [0], marker='s', color='C3', linestyle='none', markersize=7, label='Data Run3 single'))

    if drawn_all['mc_single_run2']:
        legend_elements.append(Line2D([0], [0], marker='o', color='C0', mfc='none', mew=1.5, linestyle='none', markersize=7, label='MC Run2 single'))
    if drawn_all['mc_single_run3']:
        legend_elements.append(Line2D([0], [0], marker='s', color='C3', mfc='none', mew=1.5, linestyle='none', markersize=7, label='MC Run3 single'))

    if drawn_all['data_sum_run2']:
        legend_elements.append(Line2D([0], [0], color='C0', linestyle='--', label='Data Run2 sum'))
    if drawn_all['data_sum_run3']:
        legend_elements.append(Line2D([0], [0], color='C3', linestyle='--', label='Data Run3 sum'))

    if drawn_all['data_comb_run2']:
        legend_elements.append(Line2D([0], [0], color='C0', linestyle='-', label='Data Run2 comb.'))
    if drawn_all['data_comb_run3']:
        legend_elements.append(Line2D([0], [0], color='C3', linestyle='-', label='Data Run3 comb.'))

    if drawn_all['mc_comb_run2']:
        legend_elements.append(Line2D([0], [0], color='C0', linestyle=':', label='MC Run2 comb.'))
    if drawn_all['mc_comb_run3']:
        legend_elements.append(Line2D([0], [0], color='C3', linestyle=':', label='MC Run3 comb.'))

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.90),
               ncol=4, fontsize=12, frameon=False)
    
    plot_summary_figure(
        Bd_Run2_2018_comb_data,
        Bd_Run2_comb_data,
        Bd_Run3_comb_data,
        Bd_Mixed_comb_data,
        save=save
    )

    if save:
        fig.savefig("tagging_powers_DATA_with_MC_openmarkers_dynamic_labels.pdf", bbox_inches='tight')
        print("✅ Figure saved as tagging_powers_DATA_with_MC_openmarkers_dynamic_labels.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save plot instead of showing it")
    args = parser.parse_args()
    main(save=args.save)
