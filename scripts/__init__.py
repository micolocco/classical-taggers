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
    mplLHCb["legend.fontsize"] = 22
    mplLHCb["legend.title_fontsize"] = 22

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
        "B_Tr_T_P": (0,100000),
        "B_Tr_T_PT": (0,10000),
        'B_nPVs': (0,18),
        'B_PT': (0,60000),
        "B_Tr_T_BVIPSig": (0, 60),
        "B_Tr_T_CHI2DOF": (0, 5),
        'B_Tr_T_PIDK': (-150, 150),
        'B_Tr_T_PIDP': (-150, 150),
        "B_Tr_T_GHOSTPROB": (0, 1),
        "B_Tr_T_absIP": (0,1.5),
        "B_Tr_T_PIDe": (-30, 30),
        "B_Tr_T_PIDmu": (-40, 40),
        "B_Tr_T_ISMUON": (0,1)

    }

nice_names = {
        "B_nTracks": r'$\mathrm{nTracks}$',
        "B_Tr_T_P": r'$p(tag)$',
        "B_Tr_T_PT": r'$p_{T}(tag)$',
        'B_nPVs': r'$\mathrm{nPVs}$',
        'B_PT': r'$p_{T}(B)$',
        "B_Tr_T_BVIPSig": r'$\sqrt{\chi^{2}_{\rm{IP}}(tag)~\rm{wrt}~\rm{PV}(B)}$',
        "B_Tr_T_CHI2DOF": r'$\chi^2_{\rm{vtx}}/\mathrm{ndof}~\left(tag\right)$',
        'B_Tr_T_PIDK': r'$\mathrm{PID}_{K}(tag)$',
        'B_Tr_T_PIDP': r'$\mathrm{PID}_{p}(tag)$',
        "B_Tr_T_GHOSTPROB": r'$P_{\mathrm{ghost}}(tag)$',
        "B_Tr_T_absIP": r'$|\mathrm{IP}|(tag)$'
    }