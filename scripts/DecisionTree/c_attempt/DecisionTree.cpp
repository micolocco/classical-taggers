#include "DecisionTree.hpp"
#include <cstddef>
#include <limits>
#include <numeric>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <functional>
#include <cstdlib>
#include <cstdio>




DecisionTree::DecisionTree(double (*criterion_)(const std::vector<int>&, const std::vector<double>&, unsigned int, unsigned int), 
                           unsigned int max_depth_, 
                           const std::vector<std::string>& feature_list,
                           const std::vector<std::string>& prediction_list,
                           unsigned int min_samples_split_, 
                           unsigned int min_samples_leaf_, 
                           double min_weight_fraction_leaf_, 
                           double min_impurity_decrease_) {
    this->criterion = criterion_;
    this->max_depth = max_depth_;
    if (this->max_depth == 0) {
        this->max_depth = std::numeric_limits<unsigned int>::max();
    }
    this->min_samples_split = min_samples_split_;
    this->min_samples_leaf = min_samples_leaf_;
    this->min_weight_fraction_leaf = min_weight_fraction_leaf_;
    this->min_impurity_decrease = min_impurity_decrease_;
    this->feature_names = feature_list;
    this->prediction_names = prediction_list;
    this->root = nullptr;
    this->n_classes = -1;
}

cut_info DecisionTree::get_optimal_split_of_column(const std::vector<double>& x,
                                         const std::vector<int>& y,
                                         const std::vector<double>& weights,
                                         const double N,
                                         unsigned int depth,
                                         unsigned int node_index,
                                         unsigned int column) {
    cut_info best_cut = {-1, 0.0, 0.0, 0.0};

    struct IndexedValue {
        double value;
        int target;
        double weight;
    };
    std::vector<IndexedValue> indexed_values(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        indexed_values[i] = {x[i], y[i], weights[i]};
    }
    std::sort(indexed_values.begin(), indexed_values.end(), [](const IndexedValue& a, const IndexedValue& b) {
        return a.value < b.value;
    });
    std::vector<double> x_sorted(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x_sorted[i] = indexed_values[i].value;
    }
    std::vector<double> unique_values = this->unique(x_sorted);

    std::vector<double> cuts;
    cuts.resize(unique_values.size() - 1);
    for (size_t i = 0; i < cuts.size(); i++) {
        cuts[i] = unique_values[i] + (unique_values[i + 1] - unique_values[i]) / 2.0;
    }

    unique_values.clear();
    x_sorted.clear();

    TreeNode node(depth, node_index, this->criterion, y, weights, this->n_classes, this->feature_names[column]);
    node.set_impurity_if_leaf(this->criterion(y, weights, this->n_classes, N));

    std::vector<double> x_left, x_right;
    std::vector<int> y_left, y_right;
    std::vector<double> weights_left, weights_right;
    double N_tl = 0.0, N_tr = 0.0;

    x_left.reserve(x.size());
    y_left.reserve(y.size());
    weights_left.reserve(weights.size());
    x_right.reserve(x.size());
    y_right.reserve(y.size());
    weights_right.reserve(weights.size());

    // Get everything in x, y, and weight initially into each right vector in inverse order, 
    // that allows efficient popping from the back when we move elements to the left vector as we iterate through the cuts
    for (size_t i = indexed_values.size(); i > 0 ; --i) {
        x_right.push_back(indexed_values[i-1].value);
        y_right.push_back(indexed_values[i-1].target);
        weights_right.push_back(indexed_values[i-1].weight);
        N_tr += indexed_values[i-1].weight;
    }
    indexed_values.clear();


    for (size_t i = 0; i < cuts.size(); ++i) {
        // if (i == 0 || cuts[i]-cuts[i-1] > 1e-6){
            for (size_t j = x_right.size()-1; x_right[j] <= cuts[i] && j > 0; --j) {
                x_left.push_back(x_right[j]);
                x_right.pop_back();
                y_left.push_back(y_right[j]);
                y_right.pop_back();
                weights_left.push_back(weights_right[j]);
                weights_right.pop_back();
                N_tl += weights_right[j];
                N_tr -= weights_right[j];

                if (x_left.size() + x_right.size() != x.size()) {
                    std::cerr << "Error: size mismatch after moving element at index " << j << " for cut " << cuts[i] << ": x_left size = " << x_left.size() << ", x_right size = " << x_right.size() << ", total = " << x_left.size() + x_right.size() << ", expected = " << x.size() << std::endl;
                    std::exit(1);
                }
            }



            if (depth <= max_depth && x.size() >= min_samples_split && x_left.size() >= min_samples_leaf && x_right.size() >= min_samples_leaf && N_tl / N >= min_weight_fraction_leaf && N_tr / N >= min_weight_fraction_leaf) {
                double imp_l = this->criterion(y_left, weights_left, this->n_classes, N);
                double imp_r = this->criterion(y_right, weights_right, this->n_classes, N);

                

                double imp_decrease = (N_tl+N_tr)/N*(node.get_impurity_if_leaf() - ((N_tl / (N_tl+N_tr)) * imp_l + (N_tr / (N_tl+N_tr)) * imp_r));
                
                
                // if (this->feature_names[column] == "B_Tr_T_PROBNN_PI"){
                //     std::cout << "Evaluating cut at column " << this->feature_names[column] << " with cut value " << cuts[i] << ":" << std::endl;
                //     std::cout << "impurity if leaf " << node.get_impurity_if_leaf() << ": N_l: " << x_left.size() << " N_R: " << x_right.size() << " decrease: " << imp_decrease << std::endl;
                // }


                if (imp_decrease > best_cut.impurity_decrease && imp_decrease >= this->min_impurity_decrease) {
                    best_cut.cut_value = cuts[i];
                    best_cut.impurity_decrease = imp_decrease;
                    best_cut.impurity_if_leaf = node.get_impurity_if_leaf();
                    best_cut.column = column;
                }
            }
        // }
    }

    return best_cut;
}


cut_info DecisionTree::get_optimal_split(const std::vector<std::vector<double>>& X,
                                const std::vector<int>& y,
                                const std::vector<double>& weights,
                                const double N,
                                unsigned int depth,
                                unsigned int node_index) {
    cut_info best_cut = {-1, 0.0, 0.0, 0.0};
    for (unsigned int column = 0; column < X.size(); column++) {
        cut_info cut = this->get_optimal_split_of_column(X[column], y, weights, N, depth, node_index, column);

        
        // std::cout << "Column " << column << " " << this->feature_names[column] << ": best cut value = " << cut.cut_value << ", impurity decrease = " << cut.impurity_decrease << std::endl;
        // std::cout << "Column " << column << " contains " << X[column].size() << " samples." << std::endl;
        
        if (cut.impurity_decrease > best_cut.impurity_decrease && cut.impurity_decrease >= this->min_impurity_decrease) {
            best_cut = cut;
        }

    }
    return best_cut;
}

std::unique_ptr<TreeNode> DecisionTree::build_tree(const std::vector<std::vector<double>>& X,
                              const std::vector<int>& y,
                              const double N,
                              const std::vector<double>& weights,
                              unsigned int depth,
                              unsigned int node_index){
    std::cout << "Building tree at depth " << depth << " with node index " << node_index << " and " << X.size() << " samples." << std::endl;
    
    cut_info best_cut = this->get_optimal_split(X, y, weights, N, depth, node_index);

    std::cout << "Generating node at depth " << depth << " on column " << best_cut.column << " with cut value " << best_cut.cut_value << " and impurity decrease " << best_cut.impurity_decrease << std::endl;
    std::string feature_name = best_cut.column != -1 ? this->feature_names[best_cut.column] : "N/A";
    std::unique_ptr<TreeNode>  node = std::make_unique<TreeNode>(depth, node_index, this->criterion, y, weights, this->n_classes, feature_name);

    
    if (best_cut.column == -1) {
        std::cout << "Creating leaf node at depth " << depth << " with impurity " << node->get_impurity_if_leaf() << std::endl;
        node->set_leaf(true);
        node->set_impurity_if_leaf(this->criterion(y, weights, this->n_classes, N));
        return node;
    }
    
    node->set_column(best_cut.column);
    node->set_cut_value(best_cut.cut_value);
    node->set_leaf(false);
    node->set_impurity_if_leaf(best_cut.impurity_if_leaf);

    unsigned int n_left = 0;


    std::vector<double> sorted_values = X[best_cut.column]; // For threadscheduling reasons, maybe faster this way?
    std::sort(sorted_values.begin(), sorted_values.end());
    

    for (unsigned int i = 0; i < sorted_values.size(); i++) {
        if (sorted_values[i] <= best_cut.cut_value) {
            n_left++;
        }
    }

    std::cout << "Splitting " << sorted_values.size() << " samples into " << n_left << " left and " << sorted_values.size() - n_left << " right." << std::endl;

    
    std::vector<std::vector<double>> X_left(X.size());
    std::vector<std::vector<double>> X_right(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        X_left[i].resize(n_left);
        X_right[i].resize(sorted_values.size() - n_left);
    }
    std::vector<int> y_left(n_left);
    std::vector<int> y_right(sorted_values.size() - n_left);
    std::vector<double> weights_left(n_left);
    std::vector<double> weights_right(sorted_values.size() - n_left);
    
    
    
    
    
    
    unsigned int left_index = 0;
    unsigned int right_index = 0;
    for (unsigned int i = 0; i < sorted_values.size(); i++) {
        if (X[best_cut.column][i] <= best_cut.cut_value) {
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
    sorted_values.clear();
    X_left.shrink_to_fit();
    X_right.shrink_to_fit();


    // std::cout << "Checking shape of X_left:  " << X_left.size() << " features, " << X_left[0].size() << " samples." << std::endl;
    // std::cout << "Checking shape of X_right: " << X_right.size() << " features, " << X_right[0].size() << " samples." << std::endl;


    std::unique_ptr<TreeNode> left = this->build_tree(X_left, y_left, N, weights_left, depth + 1, node_index * 2 + 1);
    std::unique_ptr<TreeNode> right = this->build_tree(X_right, y_right, N, weights_right, depth + 1, node_index * 2 + 2);

    node->set_daughters(std::move(left), std::move(right));
    return node;
}

void DecisionTree::prune_tree(TreeNode* node, 
                              const std::vector<std::vector<double>>& X, 
                              const std::vector<int>& y, 
                              const std::vector<double>& weights) {
    if (node->is_leaf()) {
        return;
    }

    // Notation is terrible but based on the formula for alpha effective in cost complexity pruning from sklearn documentation / The book
    // "Classification and Regression Trees" by Breiman, Friedman, Olshen, and Stone
    unsigned int T = node->count_leaf_nodes();
    double Rt = node->get_impurity_if_leaf();
    double RT_t = node->calc_impurity(X, y, weights);
    double alpha_eff = (Rt - RT_t) / (T - 1);

    std::cout << "Pruning node at depth " << node->get_depth() << " with node index " << node->get_node_index() << ": impurity if leaf = " << Rt << ", impurity of subtree = " << RT_t << ", number of leaf nodes in subtree = " << T << ", effective alpha = " << alpha_eff << std::endl;

    if (alpha_eff < this->min_impurity_decrease) {
        node->set_leaf(true);
    } else {
        unsigned int n_left = 0;


        std::vector<double> sorted_values = X[node->get_column()]; // For threadscheduling reasons, maybe faster this way?
        std::sort(sorted_values.begin(), sorted_values.end());
        

        for (unsigned int i = 0; i < sorted_values.size(); i++) {
            if (sorted_values[i] <= node->get_cut_value()) {
                n_left++;
            }
        }

        std::cout << "Splitting " << sorted_values.size() << " samples into " << n_left << " left and " << sorted_values.size() - n_left << " right." << std::endl;

        
        std::vector<std::vector<double>> X_left(X.size());
        std::vector<std::vector<double>> X_right(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            X_left[i].resize(n_left);
            X_right[i].resize(sorted_values.size() - n_left);
        }
        std::vector<int> y_left(n_left);
        std::vector<int> y_right(sorted_values.size() - n_left);
        std::vector<double> weights_left(n_left);
        std::vector<double> weights_right(sorted_values.size() - n_left);
        
        unsigned int left_index = 0;
        unsigned int right_index = 0;
        for (unsigned int i = 0; i < sorted_values.size(); i++) {
            if (X[node->get_column()][i] <= node->get_cut_value()) {
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
        sorted_values.clear();
        X_left.shrink_to_fit();
        X_right.shrink_to_fit();
    

        this->prune_tree(node->get_left(), X_left, y_left, weights_left);
        this->prune_tree(node->get_right(), X_right, y_right, weights_right);
    }

}

void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<int>& y,
                       std::optional<std::reference_wrapper<const std::vector<double>>> weights) {
    std::vector<double> norm_weights;
    if (weights.has_value()) {
        double sum_weights = std::accumulate(weights->get().begin(), weights->get().end(), 0.0);
        norm_weights.resize(weights->get().size());
        for (size_t i = 0; i < weights->get().size(); ++i) {
            norm_weights[i] = weights->get()[i] / sum_weights;
        }
        this->is_weighted = true;
    } else {
        norm_weights.resize(y.size(), 1.0 / y.size());
        this->is_weighted = false;
    }


    this->n_classes = *std::max_element(y.begin(), y.end()) + 1; // Assuming classes are labeled from 0 to n_classes-1


    const double N = std::accumulate(norm_weights.begin(), norm_weights.end(), 0.0);


    this->root = this->build_tree(X, y, N, norm_weights, 0, 0).release();

    std::cout << "Tree built successfully. Beginning pruning..." << std::endl;

    this->prune_tree(this->root, X, y, norm_weights);

    std::cout << "Tree pruned successfully." << std::endl;

}

std::vector<unsigned int> DecisionTree::predict(const std::vector<std::vector<double>>& X) {
    std::vector<unsigned int> predictions(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        predictions[i] = this->root->predict_sample(X[i]);
    }
    return predictions;
}

void DecisionTree::save_model(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    file << "decision_tree:\n";
    file << "  max_depth: " << this->max_depth << "\n";
    file << "  min_samples_split: " << this->min_samples_split << "\n";
    file << "  min_samples_leaf: " << this->min_samples_leaf << "\n";
    file << "  min_weight_fraction_leaf: " << this->min_weight_fraction_leaf << "\n";
    file << "  min_impurity_decrease: " << this->min_impurity_decrease << "\n";
    file << "  n_classes: " << this->n_classes << "\n";
    file << "  root:\n";

    std::function<void(TreeNode*, int)> save_node = [&](TreeNode* node, int indent) {
        if (!node) return;
        
        std::string spaces(indent, ' ');
        file << spaces << "depth: " << node->get_depth() << "\n";
        file << spaces << "node_index: " << node->get_node_index() << "\n";
        file << spaces << "is_leaf: " << (node->is_leaf() ? "true" : "false") << "\n";
        
        if (!node->is_leaf()) {
            file << spaces << "column: " << node->get_column() << "\n";
            file << spaces << "column_name: " << this->feature_names[node->get_column()] << "\n";
            file << spaces << "cut_value: " << node->get_cut_value() << "\n";
            file << spaces << "impurity_if_leaf: " << node->get_impurity_if_leaf() << "\n";
            file << spaces << "left_child:\n";
            save_node(node->get_left(), indent + 2);
            file << spaces << "right_child:\n";
            save_node(node->get_right(), indent + 2);
        } else {
            file << spaces << "impurity: " << node->get_impurity_if_leaf() << "\n";
            file << spaces << "prediction: " << node->get_prediction() << "\n";
        }
    };

    save_node(this->root, 4);
    file.close();
}

std::string DecisionTree::_node_info(TreeNode* node) {
    std::ostringstream info;

    if (!node->is_leaf()) {
        std::string col_name;
        if (this->feature_names.empty() || node->get_column() >= this->feature_names.size()) {
            col_name = "X[" + std::to_string((int)node->get_column()) + "]";
        } else {
            col_name = this->feature_names[node->get_column()];
        }
        info << col_name << " <= " << std::fixed << std::setprecision(5) << node->get_cut_value();
    }

    info << "\\nCriterion=" << std::fixed << std::setprecision(5) << node->get_impurity_if_leaf();
    info << "\\nSamples=" << node->get_num_samples_trained_on();

    std::string values = "\\nValue=[";
    if (this->is_weighted){
        for (size_t i = 0; i < node->get_weights_per_class().size(); i++) {
            values += std::to_string(node->get_weights_per_class()[i]);
            if (i < node->get_weights_per_class().size() - 1) {
                values += ", ";
            }

            if (values.length() > 50){
                info << values;
                values = "\n";
            }
        }
    }
    else{
        for (size_t i = 0; i < node->get_num_samples_per_class().size(); i++) {
            values += std::to_string(node->get_num_samples_per_class()[i]);
            if (i < node->get_num_samples_per_class().size() - 1) {
                values += ", ";
            }

            if (values.length() > 50){
                info << values;
                values = "\n";
            }
        }
    }
    info << values << "]";
    info << "\\nPred=" << this->prediction_names[node->get_prediction()];

    return info.str();
}

void DecisionTree::plot_tree(const std::string& filename) {
    if (this->root == nullptr) {
        throw std::runtime_error("Tree has not been fitted yet. Call fit() before plot_tree().");
    }

    // Generate DOT file first
    std::string dot_filename = filename + ".dot";
    std::ofstream dot_file(dot_filename);
    dot_file << "digraph DecisionTree {\n";
    dot_file << "  node [shape=box, style=\"rounded,filled\", color=\"#334155\", fontname=\"Arial\"];\n";
    dot_file << "  edge [color=\"#475569\", fontname=\"Arial\"];\n";

    int node_counter = 0;

    std::function<int(TreeNode*)> visit = [&](TreeNode* node) -> int {
        if (node == nullptr) {
            return -1;
        }

        int node_id = node_counter++;
        std::string node_label = this->_node_info(node);
        std::string color = node->is_leaf() ? "#f4f6f8" : "#dbeafe";

        dot_file << "  node_" << node_id << " [label=\"" << node_label << "\", fillcolor=\"" << color << "\"];\n";

        TreeNode* left = node->get_left();
        TreeNode* right = node->get_right();

        if (left != nullptr) {
            int left_id = visit(left);
            dot_file << "  node_" << node_id << " -> node_" << left_id << " [label=\"True\"];\n";
        }

        if (right != nullptr) {
            int right_id = visit(right);
            dot_file << "  node_" << node_id << " -> node_" << right_id << " [label=\"False\"];\n";
        }

        return node_id;
    };

    visit(this->root);

    dot_file << "}\n";
    dot_file.close();

    // Convert DOT to PDF using graphviz
    std::string command = "dot -Tpdf " + dot_filename + " -o " + filename;
    int result = system(command.c_str());

    if (result != 0) {
        std::cerr << "Warning: Failed to convert DOT to PDF. Make sure Graphviz is installed." << std::endl;
        std::cerr << "Command was: " << command << std::endl;
    } else {
        // Clean up the temporary DOT file
        std::remove(dot_filename.c_str());
    }
}
