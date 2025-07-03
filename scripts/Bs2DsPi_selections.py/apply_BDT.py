def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=str, help="input root file")
    parser.add_argument('--output',type=str, help="the output directory in str format",default=".")
    parser.add_argument('--treename',help="type in the name of the input tuple TTree (including the TDirectory it's in)", default="BdToDsmPi_DsmToKpKmPim/DecayTree")
    args = parser.parse_args()
    return args


def BDT(mini_file):
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
    infile = mini_file
    outName = FileOutNameBDT(mini_file)
    print("Input file name for BDT processing: {}".format(infile))
    ROOT.ROOT.EnableImplicitMT()
    file = TFile.Open(infile)
    inTree = file.Get('Tuple/DecayTree')
    print("Output file name (BDT): {}".format(outName))
    outFile = TFile.Open(outName, "recreate")
    outFile.mkdir("Tuple")
    outFile.cd("Tuple")
    outTree = inTree.CloneTree(0)
    outTree.SetMaxTreeSize(1000*(2000000000)) #2TB
    outTree.SetAutoSave((100000000)) #autosave every 100MB
    BDTLoop(file, inTree, outFile, outTree, nMax)
    BDTCheck(infile, outName)

    return outName
