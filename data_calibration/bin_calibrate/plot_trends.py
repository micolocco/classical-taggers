import json
import numpy as np
import argparse
from matplotlib import pyplot as plt, ticker as mticker
import mplhep as hep
hep.style.use("LHCb2")

class MinorSymLogLocator(mticker.Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        # major tick difference at lower end
        dmlower = majorlocs[1] - majorlocs[0]
        # major tick difference at upper end
        dmupper = majorlocs[-1] - majorlocs[-2]

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or (dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


def plot_deviation(samples, performances, metric, name, normed=False):
    plt.figure()
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (k, d) in enumerate(performances.items()):
        n_all = d[samples["all"]]["selected"][metric][0]*100
        e_all = np.sqrt(
            np.sum(np.array(d[samples["all"]]["selected"][metric][1:]*100)**2))
        n = np.array([d[v]["selected"][metric][0]*100 for k,
                      v in samples.items() if k != "all"])
        e = np.array([np.sqrt(np.sum(np.array(d[v]["selected"][metric][1:]*100)**2))
                      for k, v in samples.items() if k != "all"])
        e = np.sqrt(e**2 + e_all**2 - 2*e*e_all)
        if not normed:
            plt.errorbar(range(len(samples)-1), (n - n_all), yerr=e, label=k,
                         marker="x", markersize=10, linestyle="-", linewidth=0.5, color=color[i])
        else:
            plt.errorbar(range(len(samples)-1), (n - n_all) / e, label=k, marker=".",
                         markersize=10, linestyle="-", linewidth=0.5, color=color[i])

    plt.xticks(range(len(samples)-1),
               [k for k in samples.keys() if k != "all"])
    plt.xlim(left=-0.5, right=len(samples)-1.5)

    if not normed:
        plt.ylim(np.array([-1, 1]) *
                 np.ceil(np.max(np.abs(plt.gca().get_ylim()))*1.05))
        plt.ylabel(f"Deviation from average {metric.replace('_', ' ')} [%]")
    else:
        plt.ylim(np.array(
            [-1, 1])*10**np.ceil(np.log10(np.max(np.abs(plt.gca().get_ylim()))*1.5)))
        plt.yscale("symlog")
        plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.gca().yaxis.set_minor_locator(MinorSymLogLocator(0.1, 10))
        plt.ylabel(f"Deviation from average {metric.replace('_', ' ')}" + r" [$\sigma$]")
    plt.yticks(minor=True)
    plt.grid(axis="y")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(name)


def plot_trend(samples, performances, metric, name):
    plt.figure()
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (k, d) in enumerate(performances.items()):
        plt.axhline(d[samples["all"]]["selected"][metric][0]*100,
                    linestyle="-", alpha=0.4, color=color[i], linewidth=0.5)
        plt.axhspan(d[samples["all"]]["selected"][metric][0]*100 - np.sqrt(np.sum(np.array(d[samples["all"]]["selected"][metric][1:]*100)**2)), d[samples["all"]]
                    ["selected"][metric][0]*100 + np.sqrt(np.sum(np.array(d[samples["all"]]["selected"][metric][1:]*100)**2)), alpha=0.1, color=color[i])
        plt.errorbar([0], [d[samples["all"]]["selected"][metric][0]*100], yerr=[np.sqrt(np.sum(np.array(d[samples["all"]]
                                                                                                        ["selected"][metric][1:]*100)**2))], marker="x", markersize=7, linestyle=":", linewidth=0.5, color=color[i])
        plt.errorbar(range(1, len(samples)), [d[v]["selected"][metric][0]*100 for k, v in samples.items() if k != "all"], yerr=[np.sqrt(np.sum(np.array(
            d[v]["selected"][metric][1:]*100)**2)) for k, v in samples.items() if k != "all"], label=k, marker="x", markersize=7, linestyle=":", linewidth=0.5, color=color[i])
        # print(k, [(k, d[v]["selected"][metric][0]*100, np.sqrt(np.sum(np.array(d[v]
        #    ["selected"][metric][1:]*100)**2))) for k, v in samples.items()])
    plt.axvline(0.5, color="k", alpha=0.5, linestyle="-", linewidth=1)

    plt.xlim(left=-0.5, right=len(samples)-0.5)
    plt.xticks(range(len(samples)), samples.keys())

    if metric == "tagging_power":
        plt.ylim(bottom=0)
        plt.yticks(np.arange(np.ceil(np.max(plt.gca().get_ylim())) * 2) / 2)

    plt.ylabel(f"{metric.replace('_', ' ')} [%]")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, nargs="+", default=["tagging_power"], choices=[
                        "tagging_power", "tag_efficiency", "mistag_rate", "effective_mistag"])
    parser.add_argument(
        "--outdir", type=str, default="/ceph-kernel/users/qfuehring/run3_bd2dpi/block1/")
    parser.add_argument("--reference", type=str,
                        default="/ceph-kernel/users/qfuehring/run3_bd2dpi/block1/")
    parser.add_argument("--bins", type=str, nargs="+", default=["/ceph-kernel/users/qfuehring/run3_bd2dpi/block1/PT-bin0/", "/ceph-kernel/users/qfuehring/run3_bd2dpi/block1/PT-bin1/",
                        "/ceph-kernel/users/qfuehring/run3_bd2dpi/block1/PT-bin2/", "/ceph-kernel/users/qfuehring/run3_bd2dpi/block1/PT-bin3/"])
    parser.add_argument("--var", type=str, default="PT")
    args = parser.parse_args()

    samples = {"all": args.reference}
    samples.update({f"{args.var} bin{i}": bin for i,
                   bin in enumerate(args.bins)})

    performances = {"Combination": {}}
    for sample in samples.values():
        with open(f"{sample}calibration_combination.json", "r") as f:
            performances["Combination"].update(
                {sample: json.load(f)["Combination"]["calibrated"]})
        with open(f"{sample}calibration.json", "r") as f:
            for tagger, performance in json.load(f).items():
                if not tagger in performances.keys():
                    performances.update({tagger: {}})
                performances[tagger].update(
                    {sample: performance["calibrated"]})

    outdir = args.outdir
    if not outdir.endswith("/"):
        outdir += "/"

    for metric in args.metrics:
        plot_trend(samples, performances, metric,
                   name=f"{outdir}trend_{args.var}_{metric}.pdf")
        plot_deviation(samples, performances, metric,
                       name=f"{outdir}trend_{args.var}_{metric}_deviations.pdf")
        plot_deviation(samples, performances, metric,
                       name=f"{outdir}trend_{args.var}_{metric}_deviations_normed.pdf", normed=True)


if __name__ == "__main__":
    main()
