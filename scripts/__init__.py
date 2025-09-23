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

    mplLHCb["text.usetex"] = False
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

'''
ranges = {
        "B_nTracks": (0,600),
        "B_Tr_T_P": (0,60000),
        "B_Tr_T_PT": (0,10000),
        'B_nPVs': (0,18),
        'B_PT': (0,60000),
        "B_Tr_T_BPVIPSig": (0, 10),
        "B_Tr_T_BPVIP": (0, 5),
        "B_Tr_T_OWNPVIP": (0, 5),
        "B_Tr_T_CHI2DOF": (0, 5),
        'B_Tr_T_PIDK': (-150, 150),
        'B_Tr_T_PIDP': (-150, 150),
        "B_Tr_T_GHOSTPROB": (0, 1),
        "B_Tr_T_absIP": (0,1.5),
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
'''

'''
nice_names = {
        "B_nTracks": r'$\mathrm{nTracks}$',
        "B_Tr_T_P": r'$p(tag)$',
        "B_Tr_T_PT": r'$p_{T}(tag)$',
        'B_nPVs': r'$\mathrm{nPVs}$',
        'B_PT': r'$p_{T}(B)$',
        "B_Tr_T_BPVIPSig": r'$\sqrt{\chi^{2}_{\rm{IP}}(tag)~\rm{wrt}~\rm{BestPV}}$',
        "B_Tr_T_BPVIP": r'$\mathrm{IP}~\rm{wrt}~\rm{BestPV}$',
        "B_Tr_T_OWNPVIP": r'$\mathrm{IP}~\rm{wrt}~\rm{ownPV}$',
        "B_Tr_T_CHI2DOF": r'$\chi^2/\mathrm{ndof}~$',
        'B_Tr_T_PIDK': r'$\mathrm{PID}_{K}(tag)$',
        'B_Tr_T_PIDP': r'$\mathrm{PID}_{p}(tag)$',
        'B_Tr_T_PIDe': r'$\mathrm{PID}_{e}(tag)$',
        'B_Tr_T_PIDmu': r'$\mathrm{PID}_{mu}(tag)$',
        "B_Tr_T_GHOSTPROB": r'$P_{\mathrm{ghost}}(tag)$',
        "B_Tr_T_absIP": r'$|\mathrm{IP}|(tag)$',
        "B_Tr_T_PhiDistance": r'$\mathrm{\Delta}\mathrm{\Phi}(tag, signal)$' ,
        "B_Tr_T_EtaDistance": r'$\mathrm{\Delta}\mathrm{\eta}(tag, signal)$' ,
        #"B_Tr_T_eoverP": r'$E/p(tag)$',
        #"B_Tr_T_DeltaQ_Electron": r'$\Delta\mathrm{Q}_{e}$',
        "B_Tr_T_DeltaQ_Pion": r'$\mathrm{\Delta}\mathrm{Q}_{pi}$',
        "B_Tr_T_DeltaR": r'$\mathrm{\Delta}\mathrm{R}$',
        "B_Tr_T_Signal_TagPart_PT": r'$p_{T}(tag+signal)$',
    }
'''
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
    "B_Tr_T_DeltaQ_Kaon": r"$\Delta Q_{K}~[\mathrm{MeV}/c^2]$",
    "B_Tr_T_DeltaQ_Electron": r"$\Delta Q_{e}~[\mathrm{MeV}/c^2]$",
    "B_Tr_T_DeltaQ_Muon": r"$\Delta Q_{\mu}~[\mathrm{MeV}/c^2]$",
    "B_Tr_T_DeltaQ_Pion": r"$\Delta Q_{\pi}~[\mathrm{MeV}/c^2]$",
    "B_Tr_T_DeltaQ_Proton": r"$\Delta Q_{p}~[\mathrm{MeV}/c^2]$",

    # Tag + signal combined
    "B_Tr_T_Signal_TagPart_PT": r"$p_{T}(\mathrm{tag}+\mathrm{B})~[\mathrm{MeV}/c]$",

    # Other physics-motivated features
    "B_Tr_T_eoverP": r"$Q_{e}/p(\mathrm{tag})~[c/\mathrm{MeV}]$",
    "diff_P": r"$\Delta p~[\mathrm{MeV}/c]$",
    "P_proj": r"$p_{\mathrm{proj}}~[\mathrm{MeV}/c]$",
    "t": r"$t_{\mathrm{POCA}}[mm]$",
    "EVIP": r"$\mathrm{EVIP}[mm]$",
    "logEVIP": r"$\log(\mathrm{EVIP})$",
    "logP_proj": r"$\log(p_{\mathrm{proj}})$",
    "B_Tr_T_atanPT_PZ": r"$\arctan\frac{p_{T}}{p_{z}}(\mathrm{tag})$",
}

ranges = {
    # PV info
    "B_OWNPV_X": (0.2, 0.6),      # mm
    "B_OWNPV_Y": (-0.2, 0.2),      # mm
    "B_OWNPV_Z": (-300, 300),      # mm
    "B_ENDV_X": (-7.5, 7.5),         # mm
    "B_ENDV_Y": (-7.5, 7.5),         # mm
    "B_ENDV_Z": (-300, 300),       # mm

    # B kinematics
    "B_ENERGY": (0, 3e5),          # MeV
    "B_ETA": (1.5, 5.0),
    "B_M": (5000, 5600),           # MeV/c²
    "B_P": (0, 5e5),               # MeV/c
    "B_PHI": (-3.14, 3.14),
    "B_PT": (0, 6e4),              # MeV/c
    "B_PX": (-5e4, 5e4),
    "B_PY": (-5e4, 5e4),
    "B_PZ": (0, 5e5),
    "B_nPVs": (0, 17),
    "B_nTracks": (0, 600),

    # Tag track basic info
    "B_Tr_T_TRACKISLONG": (0, 1),
    "B_Tr_T_Charge": (-1, 1),
    "B_Tr_T_ISMUON": (0, 1),

    # IP-related
    "B_Tr_T_OWNPVIP": (0, 2),              # mm
    "B_Tr_T_OWNPVIPSig": (0, 30),
    "B_Tr_T_OWNPVIPCHI2": (0, 700),
    "B_Tr_T_absOWNPV_IP": (0, 2),
    "B_Tr_T_IPChi2BVTX": (0, 1000),
    "B_Tr_T_IPBVTX": (0, 5),
    "B_Tr_T_absIP": (0, 5),

    # Tag kinematics
    "B_Tr_T_ENERGY": (0, 1e5),             # MeV
    "B_Tr_T_P": (0, 8e4),                  # MeV/c
    "B_Tr_T_PT": (0, 8e3),                 # MeV/c
    "B_Tr_T_PX": (-7e3, 7e3),
    "B_Tr_T_PY": (-7e3, 7e3),
    "B_Tr_T_PZ": (0, 1e5),
    "B_Tr_T_X": (-2.5, 2.5),                 # mm
    "B_Tr_T_Y": (-2.5, 2.5),                 # mm
    "B_Tr_T_Z": (-300, 300),               # mm
    "B_Tr_T_Eta": (1.5, 5.0),

    # Track quality
    "B_Tr_T_MINIP": (0, 2),                 # mm
    "B_Tr_T_MINIPChi2": (0, 700),
    "B_Tr_T_CHI2DOF": (0, 5),
    "B_Tr_T_GHOSTPROB": (0, 1),

    # PID
    "B_Tr_T_PIDK": (-100, 100),
    "B_Tr_T_PIDe": (-50, 50),
    "B_Tr_T_PIDmu": (-25, 25),
    "B_Tr_T_PIDP": (-100, 100),
    "B_Tr_T_PROBNN_GHOST": (0, 1),
    "B_Tr_T_PROBNN_E": (0, 1),
    "B_Tr_T_PROBNN_K": (0, 1),
    "B_Tr_T_PROBNN_P": (0, 1),
    "B_Tr_T_PROBNN_MU": (0, 1),
    "B_Tr_T_PROBNN_PI": (0, 1),

    # Δ variables & angular distances
    "B_Tr_T_PhiDistance": (-3.14, 3.14),
    "B_Tr_T_cos_PhiDistance": (-1, 1),
    "B_Tr_T_EtaDistance": (0, 4),
    "B_Tr_T_DeltaR": (0, 10),
    "B_Tr_T_diff_z": (0, 150),           # mm

    # ΔQ
    "B_Tr_T_DeltaQ_Kaon": (0, 3e3),
    "B_Tr_T_DeltaQ_Electron": (0, 3e3),
    "B_Tr_T_DeltaQ_Muon": (0, 3e3),
    "B_Tr_T_DeltaQ_Pion": (0, 3e3),
    "B_Tr_T_DeltaQ_Proton": (0, 3e3),

    # Tag+signal
    "B_Tr_T_Signal_TagPart_PT": (0, 2e4),

    # Other features
    "B_Tr_T_eoverP": (-0.001, 0.001),
    "diff_P": (0, 3e5),
    "P_proj": (0, 1e5),
    "t": (-0.1, 0.1),                          
    "EVIP": (0, 10),
    "logEVIP": (0, 6),
    "logP_proj": (0, 12),
    "B_Tr_T_atanPT_PZ": (0, 0.5)
}
