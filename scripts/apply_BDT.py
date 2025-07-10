from BDTSelection_HelperFunctions import BDTLoop, BDTCheck
import argparse

def BDT(infile, treename, output):
    Decorator(
    r"""
    ___      __    _____
    |  |     | \     |
    |__|_    |  |    |
    |    |   |  |    |
    |____|   |_/     |
    """
    )

    Decorator('Adding BDT response to the files!')
    nMax = -1
    ROOT.ROOT.EnableImplicitMT()
    file = TFile.Open(infile)
    inTree = file.Get(treename)
    print("Output file name (BDT): {}".format(output))
    outFile = TFile.Open(output, "recreate")
    outFile.cd()
    outTree = inTree.CloneTree(0)
    outTree.SetMaxTreeSize(1000*(2000000000)) #2TB
    outTree.SetAutoSave((100000000)) #autosave every 100MB
    BDTLoop(file, inTree, outFile, outTree, nMax)
    BDTCheck(infile, output)
    return output



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply a preselection for the tagging particles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--selected_input', help='Raw file where cuts need to be applied', type=str)
    parser.add_argument('--output', help='Name of the output file', type=str)
    parser.add_argument('--treename', help='Tree name of the raw ntuples', type=str, default='BdToDsmPi_DsmToKpKmPim/DecayTree')

    cfg = parser.parse_args()
    from pprint import pprint
    pprint(cfg)

    # Define label map to adapt to different naming convention DsPi analysis production VS phis Analysis production
    label_map = {
        'lab0': 'B',
        'lab1': 'piplus',
        'lab2': 'Ds',
        'lab3': 'hplus',
        'lab4': 'hminus',
        'lab5': 'piminus',
    }