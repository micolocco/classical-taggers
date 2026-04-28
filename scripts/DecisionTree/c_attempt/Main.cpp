#include <iostream>
// #include <memory>
#include <vector>
// #include "TDirectory.h"
// #include "TTree.h"
// #include "TLeaf.h"
// #include "TH1.h"
// #include "TFile.h"
#include "DecisionTree.hpp"
#include "Dataframe.hpp"
#include <chrono>
#include <numeric>


double gini(const std::vector<int>& y, const std::vector<double>& weights, unsigned int n_classes, unsigned int predictions) {
    if (y.size() == 0) {
        return 0.0; // No samples, no impurity
    }
    // Static thread-local buffer to avoid repeated allocations
    std::vector<double> class_weights(n_classes, 0.0);

    double total_weight = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        class_weights[y[i]] += weights[i];
        total_weight += weights[i];
    }
    
    double gini = 0.0;
    for (size_t i = 0; i < n_classes; ++i) {
        class_weights[i] /= total_weight;
        gini += class_weights[i] * (1 - class_weights[i]);
    }
    return gini;
}


int main() {
    std::string path = "/ceph/users/togasa/FlavourTagging/MC/DT_outputs/testing/balanced/DT_trainingset_10k.root";

    
    Dataframe tree(path);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::vector<std::string> features = {"B_Tr_T_cos_PhiDistance", "B_Tr_T_PhiDistance", "B_Tr_T_diff_z", "B_Tr_T_DeltaR", "diff_P", "P_proj", "t", "EVIP", "B_Tr_T_absOWNPV_IP", "B_Tr_T_EtaDistance", "B_Tr_T_DeltaQ_Pion", "B_Tr_T_DeltaQ_Muon", "B_Tr_T_DeltaQ_Electron", "B_Tr_T_DeltaQ_Proton", "B_Tr_T_DeltaQ_Kaon", "B_Tr_T_Signal_TagPart_PT", "B_Tr_T_OWNPVIPSig", "logEVIP", "logP_proj", "B_Tr_T_atanPT_PZ", "B_OWNPV_X", "B_OWNPV_Y", "B_OWNPV_Z", "B_ENDV_X", "B_ENDV_Y", "B_ENDV_Z", "B_ENERGY", "B_ETA", "B_M", "B_P", "B_PHI", "B_PT", "B_PX", "B_PY", "B_PZ", "B_nPVs", "B_nTracks", "B_Tr_T_TRACKISLONG", "B_Tr_T_OWNPVIP", "B_Tr_T_OWNPVIPCHI2", "B_Tr_T_Charge", "B_Tr_T_ISMUON", "B_Tr_T_ENERGY", "B_Tr_T_Eta", "B_Tr_T_MINIP", "B_Tr_T_MINIPChi2", "B_Tr_T_P", "B_Tr_T_PT", "B_Tr_T_PIDK", "B_Tr_T_PIDe", "B_Tr_T_PIDmu", "B_Tr_T_PIDP", "B_Tr_T_PROBNN_GHOST", "B_Tr_T_PROBNN_E", "B_Tr_T_PROBNN_K", "B_Tr_T_PROBNN_P", "B_Tr_T_PROBNN_MU", "B_Tr_T_PROBNN_PI", "B_Tr_T_CHI2DOF", "B_Tr_T_GHOSTPROB", "B_Tr_T_PX", "B_Tr_T_PY", "B_Tr_T_PZ", "B_Tr_T_X", "B_Tr_T_Y", "B_Tr_T_Z", "B_Tr_T_IPChi2BVTX", "B_Tr_T_IPBVTX"};
    std::vector<std::string> prediction_names = {"OSKaon", "OSMuon", "OSElectron", "SSPion", "SSProton", "SSKaon", "otherK", "otherMu", "otherE", "photonOSEl", "otherPi", "otherP", "Others", "notSamePV"};


    tree.load_features(features);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Features loaded successfully in " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;

    begin = std::chrono::steady_clock::now();
    tree.load_targets("particle");
    end = std::chrono::steady_clock::now();
    std::cout << "Targets loaded successfully in " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
    
    DecisionTree dtree(gini, 6, features, prediction_names, 2, 1, 0.0, 0.009);
    std::cout << "Decision tree initialized successfully." << std::endl;
    begin = std::chrono::steady_clock::now();
    std::cout << "Starting to build the tree..." << std::endl;
    dtree.fit(tree.get_data(), tree.get_targets(), tree.get_balancing_weights());


    end = std::chrono::steady_clock::now();
    std::cout << "Tree built successfully in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    
    dtree.save_model("decision_tree_model.txt");

    dtree.plot_tree("decision_tree.pdf");

    return 0;
}