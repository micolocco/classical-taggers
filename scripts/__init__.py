def matplotlib_lhcb_style(plt):
    """
    In order to add a proper "LHCb Simulation" text box, use:

        plt.text(x, y, 'LHCb Simulation', fontsize=28)
    """
    mplLHCb = {}

    mplLHCb["axes.labelsize"] = 24
    mplLHCb["axes.linewidth"] = 2

    mplLHCb["figure.figsize"] = 8, 6
    mplLHCb["figure.dpi"] = 100

    mplLHCb["font.family"] = "serif"
    mplLHCb["font.serif"] = "Times New Roman"
    mplLHCb["font.size"] = 13
    mplLHCb["font.weight"] = 400

    mplLHCb["text.usetex"] = True
    mplLHCb["text.latex.preamble"] = r"\usepackage{amsmath}"
    mplLHCb["mathtext.fontset"] = "cm"
    mplLHCb["mathtext.rm"] = "Times New Roman"
    mplLHCb["mathtext.it"] = "Times New Roman:italic"
    mplLHCb["mathtext.bf"] = "Times New Roman:bold"
    mplLHCb["mathtext.cal"] = "Times New Roman:caligraphic"
    # mplLHCb["mathtext.fontset"] = "cm"
    # mplLHCb["mathtext.rm"] = "DejaVu Serif:style=Book"
    # mplLHCb["mathtext.it"] = "DejaVu Serif:style=Italic"
    # mplLHCb["mathtext.bf"] = "DejaVu Serif:style=Bold"
    # mplLHCb["mathtext.cal"] = "DejaVu Serif Condensed:style=Condensed,Book"

    mplLHCb["legend.loc"] = "best"
    mplLHCb["legend.frameon"] = "false"
    mplLHCb["legend.handletextpad"] = 0.3
    mplLHCb["legend.numpoints"] = 1
    mplLHCb["legend.labelspacing"] = 0.2
    mplLHCb["legend.fontsize"] = 18
    mplLHCb["legend.title_fontsize"] = 18

    mplLHCb["lines.linewidth"] = 2
    mplLHCb["lines.markeredgewidth"] = 0
    mplLHCb["lines.markersize"] = 8

    mplLHCb["savefig.bbox"] = "tight"
    mplLHCb["savefig.pad_inches"] = 0.1

    mplLHCb["xtick.major.size"] = 13
    mplLHCb["xtick.minor.size"] = 6
    mplLHCb["xtick.major.width"] = 1.5
    mplLHCb["xtick.minor.width"] = 1.5
    mplLHCb["xtick.major.pad"] = 10
    mplLHCb["xtick.minor.pad"] = 10
    mplLHCb["xtick.labelsize"] = 22
    mplLHCb["xtick.top"] = "true"
    mplLHCb["xtick.minor.visible"] = "true"
    mplLHCb["xtick.direction"] = "in"

    mplLHCb["ytick.major.size"] = 13
    mplLHCb["ytick.minor.size"] = 6
    mplLHCb["ytick.major.width"] = 1.5
    mplLHCb["ytick.minor.width"] = 1.5
    mplLHCb["ytick.major.pad"] = 10
    mplLHCb["ytick.minor.pad"] = 10
    mplLHCb["ytick.labelsize"] = 22
    mplLHCb["ytick.direction"] = "in"
    mplLHCb["ytick.minor.visible"] = "true"
    mplLHCb["ytick.right"] = "true"

    mplLHCb["xaxis.labellocation"] = "right"
    mplLHCb["yaxis.labellocation"] = "top"
    mplLHCb["axes.axisbelow"] = True

    plt.rcParams.update(mplLHCb)
    plt.rcParams.update({'axes.unicode_minus' : False})
    return mplLHCb


ranges = {
        "B_nTracks": (0,600),
        "B_Tr_T_P": (0,60000),
        "B_Tr_T_PT": (0,10000),
        'B_nPVs': (0,18),
        'B_PT': (0,60000),
        "B_Tr_T_OWNPVIPSig": (0, 10),
        "B_Tr_T_BPVIP": (0, 5),
        "B_Tr_T_OWNPVIP": (0, 5),
        "B_Tr_T_CHI2DOF": (0, 5),
        'B_Tr_T_PIDK': (-150, 150),
        'B_Tr_T_PIDP': (-150, 150),
        "B_Tr_T_GHOSTPROB": (0, 1),
        "B_Tr_T_absOWNPV_IP": (0,1.5),
        "B_Tr_T_PIDe": (-30, 30),
        "B_Tr_T_PIDmu": (-40, 40),
        "B_Tr_T_ISMUON": (0,1),
        #"B_Tr_T_eoverP": (),
        #"B_Tr_T_DeltaQ_Electron": r'$\Delta\mathrm{Q}_{e}$',
        "B_Tr_T_DeltaQ_Pion": (0, 1000),
        "B_Tr_T_DeltaR": (0, 20),
        "B_Tr_T_Signal_TagPart_PT": (0, 100000),
        "B_Tr_T_PhiDistance": (-3.14, 3.14),
        "B_Tr_T_EtaDistance": (0, 4),
        



    }

nice_names = {
    # B candidate kinematics & PV
    "B_OWNPV_X": r"$x_{\mathrm{PV}}(B)~[\mathrm{mm}]$",
    "B_OWNPV_Y": r"$y_{\mathrm{PV}}(B)~[\mathrm{mm}]$",
    "B_OWNPV_Z": r"$z_{\mathrm{PV}}(B)~[\mathrm{mm}]$",
    "B_ENDV_X": r"$x_{\mathrm{end}}(B)~[\mathrm{mm}]$",
    "B_ENDV_Y": r"$y_{\mathrm{end}}(B)~[\mathrm{mm}]$",
    "B_ENDV_Z": r"$z_{\mathrm{end}}(B)~[\mathrm{mm}]$",
    "B_ENERGY": r"$E(B)~[\mathrm{MeV}]$",
    "B_ETA": r"$\eta(B)$",
    "B_M": r"$m(B)~[\mathrm{MeV}/c^2]$",
    "B_P": r"$p(B)~[\mathrm{MeV}/c]$",
    "B_PHI": r"$\phi(B)~[\mathrm{rad}]$",
    "B_PT": r"$p_{T}(B)~[\mathrm{MeV}/c]$",
    "B_PX": r"$p_{x}(B)~[\mathrm{MeV}/c]$",
    "B_PY": r"$p_{y}(B)~[\mathrm{MeV}/c]$",
    "B_PZ": r"$p_{z}(B)~[\mathrm{MeV}/c]$",
    "B_nPVs": r"$\mathrm{nPVs}$",
    "B_nTracks": r"$\mathrm{nTracks}$",

    # Tag track basic info
    "B_Tr_T_TRACKISLONG": r"$\mathrm{isLong}(\mathrm{tag})$",
    "B_Tr_T_Charge": r"$q(\mathrm{tag})$",
    "B_Tr_T_ISMUON": r"$\mathrm{isMuon}(\mathrm{tag})$",

    # Tag track IP variables
    #"B_Tr_T_OWNPVIP": r"$\mathrm{IP}(\mathrm{tag})_{\mathrm{ownPV}}~[\mathrm{mm}]$",
    #"B_Tr_T_OWNPVIPSig": r"$\sqrt{\chi^{2}_{\mathrm{IP}}(\mathrm{tag})_{\mathrm{ownPV}}}$",
    #"B_Tr_T_OWNPVIPCHI2": r"$\chi^{2}_{\mathrm{IP}}(\mathrm{tag})_{\mathrm{ownPV}}$",
    #"B_Tr_T_absOWNPV_IP": r"$|\mathrm{IP}(\mathrm{tag})|_{\mathrm{ownPV}}~[\mathrm{mm}]$",
    #"B_Tr_T_IPChi2BVTX": r"$\chi^{2}_{\mathrm{IP}}(\mathrm{tag},\mathrm{PV}_{\mathrm{best}}(B))$",
    #"B_Tr_T_IPBVTX": r"$\mathrm{IP}(\mathrm{tag},\mathrm{PV}_{\mathrm{best}}(B))~[\mathrm{mm}]$",
    "B_Tr_T_OWNPVIP": r"$\mathrm{IP}(\mathrm{tag}, \mathrm{PV}_{\mathrm{own}})~[\mathrm{mm}]$",
    "B_Tr_T_OWNPVIPSig": r"$\sqrt{\chi^{2}_{\mathrm{IP}}(\mathrm{tag}, \mathrm{PV}_{\mathrm{own}})}$",
    "B_Tr_T_OWNPVIPCHI2": r"$\chi^{2}_{\mathrm{IP}}(\mathrm{tag}, \mathrm{PV}_{\mathrm{own}})$",
    "B_Tr_T_absOWNPV_IP": r"$|\mathrm{IP}(\mathrm{tag}, \mathrm{PV}_{\mathrm{own}})|~[\mathrm{mm}]$",
    "B_Tr_T_IPChi2BVTX": r"$\chi^{2}_{\mathrm{IP}}(\mathrm{tag}, \mathrm{PV}_{\mathrm{best}}(B))$",
    "B_Tr_T_IPBVTX": r"$\mathrm{IP}(\mathrm{tag}, \mathrm{PV}_{\mathrm{best}}(B))~[\mathrm{mm}]$",

    # Tag track kinematics
    "B_Tr_T_ENERGY": r"$E(\mathrm{tag})~[\mathrm{MeV}]$",
    "B_Tr_T_P": r"$p(\mathrm{tag})~[\mathrm{MeV}/c]$",
    "B_Tr_T_PT": r"$p_{T}(\mathrm{tag})~[\mathrm{MeV}/c]$",
    "B_Tr_T_PX": r"$p_{x}(\mathrm{tag})~[\mathrm{MeV}/c]$",
    "B_Tr_T_PY": r"$p_{y}(\mathrm{tag})~[\mathrm{MeV}/c]$",
    "B_Tr_T_PZ": r"$p_{z}(\mathrm{tag})~[\mathrm{MeV}/c]$",
    "B_Tr_T_X": r"$x(\mathrm{tag})~[\mathrm{mm}]$",
    "B_Tr_T_Y": r"$y(\mathrm{tag})~[\mathrm{mm}]$",
    "B_Tr_T_Z": r"$z(\mathrm{tag})~[\mathrm{mm}]$",
    "B_Tr_T_Eta": r"$\eta(\mathrm{tag})$",

    # Tag track quality
    "B_Tr_T_MINIP": r"$\mathrm{MINIP}(\mathrm{tag})~[\mathrm{mm}]$",
    "B_Tr_T_MINIPChi2": r"$\chi^{2}_{\mathrm{MINIP}}(\mathrm{tag})$",
    "B_Tr_T_CHI2DOF": r"$\chi^2/\mathrm{ndof}$",
    "B_Tr_T_GHOSTPROB": r"$P_{\mathrm{ghost}}(\mathrm{tag})$",

    # PID
    "B_Tr_T_PIDK": r"$\mathrm{PID}_{K}(\mathrm{tag})$",
    "B_Tr_T_PIDe": r"$\mathrm{PID}_{e}(\mathrm{tag})$",
    "B_Tr_T_PIDmu": r"$\mathrm{PID}_{\mu}(\mathrm{tag})$",
    "B_Tr_T_PIDP": r"$\mathrm{PID}_{p}(\mathrm{tag})$",
    "B_Tr_T_PROBNN_GHOST": r"$\mathrm{ProbNN}_{\mathrm{ghost}}(\mathrm{tag})$",
    "B_Tr_T_PROBNN_E": r"$\mathrm{ProbNN}_{e}(\mathrm{tag})$",
    "B_Tr_T_PROBNN_K": r"$\mathrm{ProbNN}_{K}(\mathrm{tag})$",
    "B_Tr_T_PROBNN_P": r"$\mathrm{ProbNN}_{p}(\mathrm{tag})$",
    "B_Tr_T_PROBNN_MU": r"$\mathrm{ProbNN}_{\mu}(\mathrm{tag})$",
    "B_Tr_T_PROBNN_PI": r"$\mathrm{ProbNN}_{\pi}(\mathrm{tag})$",

    # Δ variables & angular distances
    "B_Tr_T_PhiDistance": r"$\Delta\phi(\mathrm{tag},B)~[\mathrm{rad}]$",
    "B_Tr_T_cos_PhiDistance": r"$\cos\Delta\phi(\mathrm{tag},B)$",
    "B_Tr_T_EtaDistance": r"$\Delta\eta(\mathrm{tag},B)$",
    "B_Tr_T_DeltaR": r"$\Delta R(\mathrm{tag},B)$",
    "B_Tr_T_diff_z": r"$\Delta z(\mathrm{tag},B)~[\mathrm{mm}]$",

    # ΔQ mass differences
    "B_Tr_T_DeltaQ_Kaon": r"$\Delta Q_{K}~[\mathrm{MeV}]$",
    "B_Tr_T_DeltaQ_Electron": r"$\Delta Q_{e}~[\mathrm{MeV}]$",
    "B_Tr_T_DeltaQ_Muon": r"$\Delta Q_{\mu}~[\mathrm{MeV}]$",
    "B_Tr_T_DeltaQ_Pion": r"$\Delta Q_{\pi}~[\mathrm{MeV}]$",
    "B_Tr_T_DeltaQ_Proton": r"$\Delta Q_{p}~[\mathrm{MeV}]$",

    # Tag + signal combined
    "B_Tr_T_Signal_TagPart_PT": r"$p_{T}(\mathrm{tag}+\mathrm{B})~[\mathrm{MeV}/c]$",

    # Other physics-motivated features
    "B_Tr_T_EoverP": r"$E/p(\mathrm{tag})$",
    "diff_P": r"$\Delta p~[\mathrm{MeV}/c]$",
    "P_proj": r"$p_{\mathrm{proj}}~[\mathrm{MeV}/c]$",
    "t": r"$t_{\mathrm{POCA}}[mm]$",
    "EVIP": r"$\mathrm{EVIP}[mm]$",
    "logEVIP": r"$\log(\mathrm{EVIP})$",
    "logP_proj": r"$\log(p_{\mathrm{proj}})$",
    "B_Tr_T_atanPT_PZ": r"$\arctan\frac{p_{T}}{p_{z}}(\mathrm{tag})$",
}
