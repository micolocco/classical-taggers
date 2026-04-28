#include <algorithm>
#include <unordered_set>
#include <vector>
#include <optional>
#include <functional>
#include "TreeNode.hpp"
// #include <string>

struct cut_info {
    int column;
    double cut_value;
    double impurity_decrease;
    double impurity_if_leaf;
};

class DecisionTree {
    private:
    unsigned int max_depth;
    unsigned int min_samples_split;
    unsigned int min_samples_leaf;
    double min_weight_fraction_leaf;
    double min_impurity_decrease;
    double (*criterion)(const std::vector<int>&, const std::vector<double>&, unsigned int, unsigned int);
    std::vector<std::vector<double>> data;
    // std::unique_ptr<TreeNode> root;
    TreeNode* root;
    unsigned int n_classes;
    std::vector<std::string> feature_names;
    std::vector<std::string> prediction_names;
    bool is_weighted;


    cut_info get_optimal_split_of_column(const std::vector<double>& x,
                                         const std::vector<int>& y,
                                         const std::vector<double>& weights,
                                         const double N,
                                         unsigned int depth,
                                         unsigned int node_index,
                                         unsigned int column);
    cut_info get_optimal_split(const std::vector<std::vector<double>>& X,
                                const std::vector<int>& y,
                                const std::vector<double>& weights,
                                const double N,
                                unsigned int depth,
                                unsigned int node_index);
    std::unique_ptr<TreeNode> build_tree(const std::vector<std::vector<double>>& X,
                    const std::vector<int>& y,
                    const double N,
                    const std::vector<double>& weights,
                    unsigned int depth,
                    unsigned int node_index);
    void prune_tree(TreeNode* node,
                    const std::vector<std::vector<double>>& X,
                    const std::vector<int>& y,
                    const std::vector<double>& weights);

    template <typename T>
    std::vector<T> unique(const std::vector<T>& vec) {
        if (vec.empty()) return vec;

        std::vector<T> sorted_vec = vec;  // Copy once

        if (!std::is_sorted(sorted_vec.begin(), sorted_vec.end())) {
            std::sort(sorted_vec.begin(), sorted_vec.end());
        }

        // std::unique moves duplicates to the end and returns iterator to new end
        auto last = std::unique(sorted_vec.begin(), sorted_vec.end());

        // Only one resize to remove the duplicates
        sorted_vec.erase(last, sorted_vec.end());

        return sorted_vec;
    }
    
    // std::string node_info(TreeNode* node, 
    //                       std::optional<std::reference_wrapper<const std::vector<std::string>>> column_names);

    public:
    DecisionTree(
        double (*criterion)(const std::vector<int>&, const std::vector<double>&, unsigned int, unsigned int), 
        unsigned int max_depth, 
        const std::vector<std::string>& feature_list,
        const std::vector<std::string>& prediction_list,
        unsigned int min_samples_split = 2, 
        unsigned int min_samples_leaf = 1, 
        double min_weight_fraction_leaf = 0.0, 
        double min_impurity_decrease = 0.0
    );
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y,
             std::optional<std::reference_wrapper<const std::vector<double>>> weights = std::nullopt); // weights is optional, if not provided, uniform weights will be used
    
    std::vector<unsigned int> predict(const std::vector<std::vector<double>>& X);
    void save_model(const std::string& filename);

    std::string _node_info(TreeNode* node);
    void plot_tree(const std::string& filename);

};