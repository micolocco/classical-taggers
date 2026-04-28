#include "Dataframe.hpp"
#include "TDirectory.h"
#include "TTree.h"
#include <cstddef>
#include <iostream>
#include <memory>

Dataframe::Dataframe(std::string& filename, unsigned int head_) {
    this->tfile = std::unique_ptr<TFile>(TFile::Open(filename.c_str(), "READ"));
    if (!this->tfile || this->tfile->IsZombie()) {
        throw std::runtime_error("Not a root file or broken file");
    }
    std::cout << "Contents of the root file:" << std::endl;
    this->tfile->ls();
    this->head = head_; 
}

void Dataframe::load_features(const std::vector<std::string>& infeatures) {
    this->feature_names = infeatures;
    this->data.resize(infeatures.size());
    TTree* tree = nullptr;
    this->tfile->GetObject("DecayTree", tree);
    if (!tree) {
        throw std::runtime_error("Tree not found in the root file");
    }
    std::vector<double> read(infeatures.size());
    auto N = tree->GetEntriesFast();

    if (this->head > 0 && this->head < N) {
        N = this->head;
    }

    for (size_t i = 0; i < this->data.size(); ++i) {
        this->data[i].resize(N);
    }

    int ifeature = 0;
    for (const auto& feature : infeatures) {
        std::cout << "Loading feature: " << feature << " with " << N << " entries." << std::endl;
        TLeaf* leaf = tree->GetLeaf(feature.c_str());
        if (!leaf) {
            throw std::runtime_error("Feature " + feature + " not found in the tree");
        }
        tree->SetBranchStatus(feature.c_str(), 1);
        tree->SetBranchAddress(feature.c_str(), &read[ifeature++]);
    }

    for (int i = 0; i < N; ++i) {
        tree->GetEntry(i);
        for (size_t j = 0; j < this->data.size(); ++j) {
            this->data[j][i] = read[j];
        }
    }

    tree->ResetBranchAddresses();

    for (size_t i = 0; i < this->data.size(); ++i) {
        double average = 0.0;
        int n = this->data[i].size();



        for (size_t j = 0; j < this->data[i].size(); ++j) {
            average += this->data[i][j]/n;
        }

        double standard_dev = 0.0;
        for (size_t j = 0; j < this->data[i].size(); ++j) {
            standard_dev += (this->data[i][j] - average) * (this->data[i][j] - average);
        }
        standard_dev = std::sqrt(standard_dev/n);
        std::cout << "Average of feature " << infeatures[i] << ": " << average << " ± " << standard_dev << std::endl;
    }
}

void Dataframe::load_targets(const std::string& target_feature) {
    TTree* tree = nullptr;
    this->tfile->GetObject("DecayTree", tree);
    if (!tree) {
        throw std::runtime_error("Tree not found in the root file");
    }
    TLeaf* leaf = tree->GetLeaf(target_feature.c_str());
    if (!leaf) {
        throw std::runtime_error("Target feature " + target_feature + " not found in the tree");
    }
    unsigned int N = tree->GetEntriesFast();
    if (this->head > 0 && this->head < N) {
        N = this->head;
    }
    this->targets.resize(N);
    long long read;

    tree->SetBranchStatus(target_feature.c_str(), 1);
    tree->SetBranchAddress(target_feature.c_str(), &read);

    for (size_t i = 0; i < this->targets.size(); ++i) {
        tree->GetEntry(i);
        this->targets[i] = read;
    }

    tree->ResetBranchAddresses();

    std::cout << "Loaded target feature: " << target_feature << std::endl;

    std::vector<int> target_counts;

    for (size_t i = 0; i < this->targets.size(); ++i) {
        if (this->targets[i] >= static_cast<int>(target_counts.size())) {
            target_counts.resize(this->targets[i] + 1, 0);
        }
        target_counts[this->targets[i]]++;
    }
    std::cout << "Target counts:" << std::endl;
    for (size_t i = 0; i < target_counts.size(); ++i) {
        std::cout << "Target " << i << ": " << target_counts[i] << std::endl;
    }
}

std::vector<std::vector<double>>& Dataframe::get_data() {
    return this->data;
}

std::vector<int>& Dataframe::get_targets() {
    return this->targets;
}

std::vector<double>& Dataframe::get_balancing_weights() {
    this->balancing_weights.resize(this->targets.size(), 1.0);
    std::vector<int> target_counts;

    for (size_t i = 0; i < this->targets.size(); ++i) {
        if (this->targets[i] >= static_cast<int>(target_counts.size())) {
            target_counts.resize(this->targets[i] + 1, 0);
        }
        target_counts[this->targets[i]]++;
    }

    for (size_t i = 0; i < this->targets.size(); ++i) {
        this->balancing_weights[i] =
            static_cast<double>(this->targets.size()) /
            (static_cast<double>(target_counts[this->targets[i]]) * static_cast<double>(this->balancing_weights.size()));
    }

    return this->balancing_weights;
}

const std::vector<std::string>& Dataframe::get_feature_names() const {
    return this->feature_names;
}