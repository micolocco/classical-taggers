#include "TreeNode.hpp"
#include <limits>
#include <stdexcept>
#include <iostream>

TreeNode::TreeNode(unsigned int depth_,
                   unsigned int node_index_,
                   double (*criterion_)(const std::vector<int>&, const std::vector<double>&, unsigned int, unsigned int),
                   const std::vector<int>& y_,
                   const std::vector<double>& weights_,
                   unsigned int n_classes_,
                   const std::string &feature_name_) {
    this->leaf = false;
    this->left = nullptr;
    this->right = nullptr;
    this->column = -1;
    this->cut_value = std::numeric_limits<double>::signaling_NaN();
    this->weight_trained_on = 0.0;
    for (const double w : weights_) this->weight_trained_on += w;
    
    this->depth = depth_;
    this->node_index = node_index_;
    this->criterion = criterion_;
    this->feature_name = feature_name_;

    this->n_classes = n_classes_;
    this->num_samples_trained_on = y_.size();

    this->prediction = -1;
    if (y_.size() > 0) {
        this->num_samples_per_class.resize(n_classes_, 0);
        this->weights_per_class.resize(n_classes_, 0);

        for (size_t i = 0; i < y_.size(); ++i) {
            num_samples_per_class[y_[i]] += 1;
            weights_per_class[y_[i]] += weights_[i];
        }
        
        for (size_t i = 0; i < weights_per_class.size(); ++i) {
            if (this->prediction == -1 || weights_per_class[i] > weights_per_class[this->prediction]) {
                this->prediction = i;
            }
        }
    } else {
        this->num_samples_per_class.resize(n_classes_, 0);
    }

    this->impurity_if_leaf = std::numeric_limits<double>::signaling_NaN();
}

void TreeNode::set_daughters(std::unique_ptr<TreeNode> l, std::unique_ptr<TreeNode> r) {
    this->left = std::move(l);
    this->right = std::move(r);
}

void TreeNode::set_leaf(bool l) {
    this->leaf = l;

    if (l) {
        //if the node is a leaf, we can free the memory of the child nodes if they exist
        if (this->left) {
            this->left.reset();
        }
        if (this->right) {
            this->right.reset();
        }
        
        this->left = nullptr;
        this->right = nullptr;
        
    }
}

unsigned int TreeNode::predict_sample(const std::vector<double>& x) {
    if (this->leaf) return this->prediction;
    if (x[this->column] <= this->cut_value) return this->left->predict_sample(x);
    else return this->right->predict_sample(x);
}

double TreeNode::calc_impurity(const std::vector<std::vector<double>>& X,
                     const std::vector<int>& y,
                     const std::vector<double>& weights) {
    if (X.empty()) return 0.0;
    if (this->left == nullptr || this->right == nullptr) {
        if (this->left == nullptr && this->right == nullptr) {
            this->impurity_if_leaf = this->criterion(y, weights, this->n_classes, this->prediction);
            return this->impurity_if_leaf;
        }
        else{
            throw std::runtime_error("One of the child nodes is null while the other is not. This should not happen.");
        }
    }

    double Nt_l = 0.0, Nt_r = 0.0;
    std::vector<std::vector<double>> X_left, X_right;
    std::vector<int> y_left, y_right;
    std::vector<double> weights_left, weights_right;
    
    int N_l = 0, N_r = 0;
    for (size_t i = 0; i < X[0].size(); ++i) {
        if (X[this->column][i] <= this->cut_value) {
            Nt_l += weights[i];
            ++N_l;
        }
        else {
            Nt_r += weights[i];
            ++N_r;
        }
    }

    //initialize the left and right child data
    X_left.resize(X.size());
    X_right.resize(X.size());

    for (size_t i = 0; i < X.size(); ++i) {
        X_left[i].resize(N_l);
        X_right[i].resize(N_r);
    }

    y_left.resize(N_l);
    y_right.resize(N_r);
    weights_right.resize(N_r);
    weights_left.resize(N_l);

    unsigned int left_index = 0;
    unsigned int right_index = 0;
    for (unsigned int i = 0; i < X[0].size(); i++) {
        if (X[this->column][i] <= this->cut_value) {
            for(size_t j = 0; j < X.size(); j++){
                X_left[j][left_index] = X[j][i];
            }
            y_left[left_index] = y[i];
            weights_left[left_index] = weights[i];
            left_index++;
        } else {
            for(size_t j = 0; j < X.size(); j++){
                X_right[j][right_index] = X[j][i];
            }
            y_right[right_index] = y[i];
            weights_right[right_index] = weights[i];
            right_index++;
        }
    }
    X_left.shrink_to_fit();
    X_right.shrink_to_fit();
    

    double imp_l = this->left->calc_impurity(X_left, y_left, weights_left);
    double imp_r = this->right->calc_impurity(X_right, y_right, weights_right);

    // std::cout<< "Calculated impurity for left child: " << imp_l << " with weight " << Nt_l << std::endl;
    // std::cout<< "Calculated impurity for right child: " << imp_r << " with weight " << Nt_r << std::endl;
    // std::cout<< "Calculated weighted impurity of children: " << (Nt_l / (Nt_l + Nt_r)) * imp_l + (Nt_r / (Nt_l + Nt_r)) * imp_r << std::endl;
          
    return ((Nt_l / (Nt_l + Nt_r)) * imp_l + (Nt_r / (Nt_l + Nt_r)) * imp_r);
}


double TreeNode::impurity_decrease(const std::vector<std::vector<double>>& X,
                             const std::vector<int>& y,
                             const double N,
                             const std::vector<double>& weights) {
    // std::cout << "Calculating impurity decrease for node at depth " << this->depth << " with node index " << this->node_index << std::endl;
    // std::cout << "Current impurity if leaf: " << this->impurity_if_leaf << std::endl;
    // std::cout << "Current weight trained on: " << this->weight_trained_on << std::endl;
    double imp = this->calc_impurity(X, y, weights);
    // std::cout << "Calculated impurity of current node: " << imp << std::endl;


    return this->weight_trained_on / N * (this->impurity_if_leaf - imp);
}

bool TreeNode::cut_valid(const std::vector<std::vector<double>>& X,
                   const std::vector<double>& weights,
                   unsigned int max_depth,
                   unsigned int min_samples_split,
                   unsigned int min_samples_leaf,
                   double min_weight_fraction_leaf) {
    if (this->depth >= max_depth) return false;
    if (X.size() <= min_samples_split) return false;
    unsigned int count_left = 0, count_right = 0;
    double weight_left = 0.0, weight_right = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][this->column] <= this->cut_value) {
            count_left += 1;
            weight_left += weights[i];
        }
        else {
            count_right += 1;
            weight_right += weights[i];
        }
    }
    if (count_left <= min_samples_leaf) return false;
    if (count_right <= min_samples_leaf) return false;
    if (weight_left <= min_weight_fraction_leaf) return false;
    if (weight_right <= min_weight_fraction_leaf) return false;

    return true;
}

unsigned int TreeNode::count_leaf_nodes() {
    if (this->leaf) {
        return 1;
    }
    return this->left->count_leaf_nodes() + this->right->count_leaf_nodes();
}

unsigned int TreeNode::max_depth() {
    if (this->leaf) {
        return this->depth;
    }
    return std::max(this->left->max_depth(), this->right->max_depth());
}   

void TreeNode::set_column(unsigned int col) {
    this->column = col;
}

void TreeNode::set_cut_value(double cut) {
    this->cut_value = cut;
}

void TreeNode::set_impurity_if_leaf(double impurity) {
    this->impurity_if_leaf = impurity;
}

double TreeNode::get_impurity_if_leaf() {
    return this->impurity_if_leaf;
}

bool TreeNode::is_leaf() {
    return this->leaf;
}

double TreeNode::get_cut_value() {
    return this->cut_value;
}

double TreeNode::get_column() {
    return this->column;
}

TreeNode* TreeNode::get_left() {
    return this->left.get();
}

TreeNode* TreeNode::get_right() {
    return this->right.get();
}

unsigned int TreeNode::get_depth() {
    return this->depth;
}

unsigned int TreeNode::get_node_index() {
    return this->node_index;
}

int TreeNode::get_prediction() {
    return this->prediction;
}

int TreeNode::get_num_samples_trained_on() {
    return this->num_samples_trained_on;
}

std::vector<int> TreeNode::get_num_samples_per_class() {
    return this->num_samples_per_class;
}

std::vector<double> TreeNode::get_weights_per_class() {
    return this->weights_per_class;
}