#include <vector>
#include <string>
#include "TFile.h"

class Dataframe {
    private:
    std::unique_ptr<TFile> tfile = nullptr;
    std::vector<std::vector<double>> data;
    std::vector<int> targets;
    std::vector<int> labels;
    std::vector<double> balancing_weights;
    std::vector<std::string> feature_names;
    unsigned int head;

    public:
    Dataframe(std::string& filename, unsigned int head_ = 0);
    void load_features(const std::vector<std::string>& infeatures);
    void load_targets(const std::string& target_feature);
    std::vector<std::vector<double>>& get_data();
    std::vector<int>& get_targets();
    std::vector<double>& get_balancing_weights();
    const std::vector<std::string>& get_feature_names() const;
};