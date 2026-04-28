// #include <iostream>
#include <memory>
#include <vector>


class TreeNode{
    private: 
    bool leaf;
    unsigned int depth;
    unsigned int node_index;
    unsigned int column;
    unsigned int n_classes;
    unsigned int num_samples_trained_on;
    int prediction;
    double cut_value;
    double weight_trained_on;
    double impurity_if_leaf;
    double (*criterion)(const std::vector<int>&, const std::vector<double>&, unsigned int, unsigned int);
    std::vector<int> num_samples_per_class;
    std::vector<double> weights_per_class;
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;
    std::string feature_name;
    

    public:
    TreeNode(unsigned int depth,
             unsigned int node_index,
             double (*criterion)(const std::vector<int>&, const std::vector<double>&, unsigned int, unsigned int),
             const std::vector<int>& y,
             const std::vector<double>& weights,
             unsigned int n_classes,
             const std::string &feature_name);
    void set_daughters(std::unique_ptr<TreeNode> l, std::unique_ptr<TreeNode> r);
    void set_leaf(bool leaf);
    unsigned int predict_sample(const std::vector<double>& x);
    double calc_impurity(const std::vector<std::vector<double>>& X,
                         const std::vector<int>& y,
                         const std::vector<double>& weights);
    double impurity_decrease(const std::vector<std::vector<double>>& X,
                             const std::vector<int>& y,
                             const double N,
                             const std::vector<double>& weights);
    bool cut_valid(const std::vector<std::vector<double>>& X,
                   const std::vector<double>& weights,
                   unsigned int max_depth,
                   unsigned int min_samples_split,
                   unsigned int min_samples_leaf,
                   double min_weight_fraction_leaf);
    unsigned int count_leaf_nodes();
    unsigned int max_depth();

    void set_column(unsigned int col);
    void set_cut_value(double cut);
    void set_impurity_if_leaf(double impurity);
    bool is_leaf();
    double get_impurity_if_leaf();
    double get_cut_value();
    double get_column();
    TreeNode* get_left();
    TreeNode* get_right();
    unsigned int get_depth();
    unsigned int get_node_index();
    int get_prediction();

    int get_num_samples_trained_on();
    std::vector<int> get_num_samples_per_class();
    std::vector<double> get_weights_per_class();
};