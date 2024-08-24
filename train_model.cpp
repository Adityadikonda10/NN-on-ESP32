#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using namespace std;
using namespace Eigen;
using json = nlohmann::json;

// Layer class for dense layers
class Layer_Dense {
public:
    MatrixXd weights;
    VectorXd biases;

    Layer_Dense(int n_inputs, int n_neurons) {
        weights = 0.10 * MatrixXd::Random(n_inputs, n_neurons);
        biases = VectorXd::Zero(n_neurons);
    }

    void forward(const MatrixXd& inputs) {
        this->inputs = inputs;
        output = (inputs * weights).rowwise() + biases.transpose();
    }

    void backward(const MatrixXd& dvalues) {
        dweights = inputs.transpose() * dvalues;
        dbiases = dvalues.colwise().sum();
        dinputs = dvalues * weights.transpose();
    }

    MatrixXd dweights;
    VectorXd dbiases;
    MatrixXd dinputs;

private:
    MatrixXd inputs;
    MatrixXd output;
};

// ReLU activation class
class ActivationReLU {
public:
    void forward(const MatrixXd& inputs) {
        this->inputs = inputs;
        output = inputs.cwiseMax(0);
    }

    void backward(const MatrixXd& dvalues) {
        dinputs = dvalues;
        dinputs.array() *= (inputs.array() > 0).cast<double>().array();
    }

    MatrixXd dinputs;
private:
    MatrixXd inputs;
    MatrixXd output;
};

// Softmax activation class
class ActivationSoftMax {
public:
    void forward(const MatrixXd& inputs) {
        MatrixXd exp_values = (inputs.array() - inputs.colwise().maxCoeff().array()).exp();
        probabilities = exp_values.array().rowwise() / exp_values.rowwise().sum().array();
        output = probabilities;
    }

    void backward(const MatrixXd& dvalues) {
        dinputs = MatrixXd(dvalues.rows(), dvalues.cols());
        for (int i = 0; i < dvalues.rows(); ++i) {
            MatrixXd single_output = probabilities.row(i).transpose() * MatrixXd::Identity(dvalues.cols(), dvalues.cols());
            MatrixXd jacobian_matrix = single_output - (probabilities.row(i).transpose() * probabilities.row(i)).array();
            dinputs.row(i) = jacobian_matrix * dvalues.row(i).transpose();
        }
    }

    MatrixXd dinputs;
    MatrixXd output;

private:
    MatrixXd probabilities;
};

// Loss class
class Loss {
public:
    virtual double calculate(const MatrixXd& output, const VectorXi& y) = 0;
    virtual void backward(const MatrixXd& dvalues, const VectorXi& y_true) = 0;
};

// Categorical Cross-Entropy Loss class
class LossCategoricalCrossEntropy : public Loss {
public:
    double calculate(const MatrixXd& output, const VectorXi& y) override {
        int samples = output.rows();
        MatrixXd y_pred_clipped = output.array().max(1e-7).min(1 - 1e-7);

        VectorXd correct_confidences(samples);
        for (int i = 0; i < samples; ++i) {
            correct_confidences[i] = y_pred_clipped(i, y[i]);
        }

        VectorXd negative_log_likelihoods = (-correct_confidences.array().log()).matrix();
        return negative_log_likelihoods.mean();
    }

    void backward(const MatrixXd& dvalues, const VectorXi& y_true) override {
        int samples = dvalues.rows();
        int labels = dvalues.cols();

        MatrixXd y_true_one_hot(samples, labels);
        y_true_one_hot.setZero();
        for (int i = 0; i < samples; ++i) {
            y_true_one_hot(i, y_true[i]) = 1;
        }

        dinputs = -y_true_one_hot.array() / dvalues.array();
        dinputs /= samples;
    }

    MatrixXd dinputs;
};

// Adam optimizer class
class Optimizer_Adam {
public:
    Optimizer_Adam(double learning_rate = 0.001, double decay = 0.0, double epsilon = 1e-7,
                   double beta_1 = 0.9, double beta_2 = 0.999)
        : learning_rate(learning_rate), decay(decay), epsilon(epsilon),
          beta_1(beta_1), beta_2(beta_2), current_learning_rate(learning_rate), iterations(0) {}

    void pre_update_params() {
        if (decay) {
            current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
        }
    }

    void update_params(Layer_Dense& layer) {
        if (!layer.weight_cache.size()) {
            layer.weight_momentums = MatrixXd::Zero(layer.weights.rows(), layer.weights.cols());
            layer.weight_cache = MatrixXd::Zero(layer.weights.rows(), layer.weights.cols());
            layer.bias_momentums = VectorXd::Zero(layer.biases.size());
            layer.bias_cache = VectorXd::Zero(layer.biases.size());
        }

        layer.weight_momentums = beta_1 * layer.weight_momentums + (1 - beta_1) * layer.dweights;
        layer.bias_momentums = beta_1 * layer.bias_momentums + (1 - beta_1) * layer.dbiases;

        MatrixXd weight_momentums_corrected = layer.weight_momentums / (1 - pow(beta_1, iterations + 1));
        VectorXd bias_momentums_corrected = layer.bias_momentums / (1 - pow(beta_1, iterations + 1));

        layer.weight_cache = beta_2 * layer.weight_cache + (1 - beta_2) * layer.dweights.cwiseProduct(layer.dweights);
        layer.bias_cache = beta_2 * layer.bias_cache + (1 - beta_2) * layer.dbiases.cwiseProduct(layer.dbiases);

        MatrixXd weight_cache_corrected = layer.weight_cache / (1 - pow(beta_2, iterations + 1));
        VectorXd bias_cache_corrected = layer.bias_cache / (1 - pow(beta_2, iterations + 1));

        layer.weights -= current_learning_rate * weight_momentums_corrected.cwiseQuotient(weight_cache_corrected.cwiseSqrt() + epsilon);
        layer.biases -= current_learning_rate * bias_momentums_corrected.cwiseQuotient(bias_cache_corrected.cwiseSqrt() + epsilon);
    }

    void post_update_params() {
        ++iterations;
    }

private:
    double learning_rate;
    double decay;
    double epsilon;
    double beta_1;
    double beta_2;
    double current_learning_rate;
    int iterations;
};

// Function to load MNIST data
void load_mnist(MatrixXd& X_train, VectorXi& y_train, MatrixXd& X_test, VectorXi& y_test) {
    ifstream train_images("train-images-idx3-ubyte");
    ifstream train_labels("train-labels-idx1-ubyte");
    ifstream test_images("t10k-images-idx3-ubyte");
    ifstream test_labels("t10k-labels-idx1-ubyte");

    // Skip header bytes
    train_images.ignore(16);
    train_labels.ignore(8);
    test_images.ignore(16);
    test_labels.ignore(8);

    // Load data
    X_train.resize(60000, 784);
    y_train.resize(60000);
    for (int i = 0; i < 60000; ++i) {
        for (int j = 0; j < 784; ++j) {
            X_train(i, j) = train_images.get();
        }
        y_train[i] = train_labels.get();
    }

    X_test.resize(10000, 784);
    y_test.resize(10000);
    for (int i = 0; i < 10000; ++i) {
        for (int j = 0; j < 784; ++j) {
            X_test(i, j) = test_images.get();
        }
        y_test[i] = test_labels.get();
    }
}

int main() {
    MatrixXd X_train, X_test;
    VectorXi y_train, y_test;

    load_mnist(X_train, y_train, X_test, y_test);

    // Normalize data
    X_train /= 255.0;
    X_test /= 255.0;

    Layer_Dense input_layer(784, 784);
    ActivationReLU activation_input;

    Layer_Dense hidden_layer1(784, 16);
    ActivationReLU activation1;

    Layer_Dense hidden_layer2(16, 16);
    ActivationReLU activation2;

    Layer_Dense output_layer(16, 10);
    ActivationSoftMax activation_output;

    LossCategoricalCrossEntropy loss_function;
    Optimizer_Adam optimizer;

    vector<double> losses;
    vector<double> accuracies;
    int epochs = 50;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        input_layer.forward(X_train);
        activation_input.forward(input_layer.output);

        hidden_layer1.forward(activation_input.output);
        activation1.forward(hidden_layer1.output);

        hidden_layer2.forward(activation1.output);
        activation2.forward(hidden_layer2.output);

        output_layer.forward(activation2.output);
        activation_output.forward(output_layer.output);

        double loss = loss_function.calculate(activation_output.output, y_train);
        losses.push_back(loss);

        VectorXi predictions = activation_output.output.rowwise().maxCoeff();
        double accuracy = (predictions.array() == y_train.array()).cast<double>().mean();
        accuracies.push_back(accuracy);

        loss_function.backward(activation_output.output, y_train);
        activation_output.backward(loss_function.dinputs);
        output_layer.backward(activation_output.dinputs);

        activation2.backward(output_layer.dinputs);
        hidden_layer2.backward(activation2.dinputs);

        activation1.backward(hidden_layer2.dinputs);
        hidden_layer1.backward(activation1.dinputs);

        activation_input.backward(hidden_layer1.dinputs);
        input_layer.backward(activation_input.dinputs);

        optimizer.pre_update_params();
        optimizer.update_params(input_layer);
        optimizer.update_params(hidden_layer1);
        optimizer.update_params(hidden_layer2);
        optimizer.update_params(output_layer);
        optimizer.post_update_params();

        if (epoch % 10 == 0) {
            cout << "Epoch: " << epoch << ", Loss: " << fixed << setprecision(3) << loss
                 << " Accuracy: " << fixed << setprecision(3) << accuracy << endl;
        }
    }

    // Save model parameters to JSON file
    json model_parameters;
    model_parameters["input_layer_weights"] = input_layer.weights;
    model_parameters["input_layer_biases"] = input_layer.biases;
    model_parameters["hidden_layer1_weights"] = hidden_layer1.weights;
    model_parameters["hidden_layer1_biases"] = hidden_layer1.biases;
    model_parameters["hidden_layer2_weights"] = hidden_layer2.weights;
    model_parameters["hidden_layer2_biases"] = hidden_layer2.biases;
    model_parameters["output_layer_weights"] = output_layer.weights;
    model_parameters["output_layer_biases"] = output_layer.biases;

    ofstream model_file("OCR_model_784_16_16_10.json");
    model_file << model_parameters.dump(4);
    model_file.close();

    cout << "Model saved successfully in JSON format." << endl;

    // Plotting (you may need to use an external library for this in C++)
    // For simplicity, you can export the data to files and plot using tools like Python/Matplotlib

    return 0;
}
