#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <json/json.h>

using namespace std;

struct Layer {
    vector<vector<double>> weights;
    vector<double> biases;

    Layer(int input_size, int neuron_size) {
        weights.resize(neuron_size, vector<double>(input_size));
        biases.resize(neuron_size, 0.0);
    }

    vector<double> forward(const vector<double>& inputs) {
        vector<double> output(biases.size(), 0.0);
        for (size_t i = 0; i < biases.size(); ++i) {
            output[i] = biases[i];
            for (size_t j = 0; j < inputs.size(); ++j) {
                output[i] += weights[i][j] * inputs[j];
            }
        }
        return output;
    }
};

vector<double> activationReLU(const vector<double>& inputs) {
    vector<double> output(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        output[i] = max(0.0, inputs[i]);
    }
    return output;
}

vector<double> activationSoftMax(const vector<double>& inputs) {
    double max_val = *max_element(inputs.begin(), inputs.end());
    vector<double> exp_values(inputs.size());
    double sum_exp_values = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        exp_values[i] = exp(inputs[i] - max_val);
        sum_exp_values += exp_values[i];
    }

    vector<double> probabilities(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        probabilities[i] = exp_values[i] / sum_exp_values;
    }

    return probabilities;
}

void loadModelParameters(const string& file_path, Layer& input_layer, Layer& hidden_layer1, Layer& hidden_layer2, Layer& output_layer) {
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << file_path << endl;
        exit(1);
    }

    Json::Value root;
    file >> root;

    auto loadLayerParameters = [&](Layer& layer, const string& weights_key, const string& biases_key) {
        const Json::Value& weights_json = root[weights_key];
        const Json::Value& biases_json = root[biases_key];

        for (size_t i = 0; i < layer.weights.size(); ++i) {
            for (size_t j = 0; j < layer.weights[i].size(); ++j) {
                if (weights_json[static_cast<int>(i)][static_cast<int>(j)].isDouble()) {
                    layer.weights[i][j] = weights_json[static_cast<int>(i)][static_cast<int>(j)].asDouble();
                } else {
                    cerr << "Error: Weight value at [" << i << "][" << j << "] is not a double." << endl;
                }
            }
        }

        for (size_t i = 0; i < layer.biases.size(); ++i) {
            if (biases_json[static_cast<int>(i)].isDouble()) {
                layer.biases[i] = biases_json[static_cast<int>(i)].asDouble();
            } else {
                cerr << "Error: Bias value at [" << i << "] is not a double." << endl;
            }
        }
    };

    loadLayerParameters(input_layer, "input_layer_weights", "input_layer_biases");
    loadLayerParameters(hidden_layer1, "hidden_layer1_weights", "hidden_layer1_biases");
    loadLayerParameters(hidden_layer2, "hidden_layer2_weights", "hidden_layer2_biases");
    loadLayerParameters(output_layer, "output_layer_weights", "output_layer_biases");
}

vector<double> loadImage(const string& file_path) {
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << file_path << endl;
        exit(1);
    }

    vector<double> image_data;
    string line;

    // Skip header if present
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        double pixel_value;
        while (ss >> pixel_value) {
            image_data.push_back(pixel_value);
            if (ss.peek() == ',') ss.ignore(); // Handle CSV delimiter
        }
    }
    return image_data;
}

int main() {
    int input_size = 784; // MNIST images are 28x28
    int hidden1_size = 64;
    int hidden2_size = 16;
    int output_size = 10;

    Layer input_layer(input_size, hidden1_size);
    Layer hidden_layer1(hidden1_size, hidden2_size);
    Layer hidden_layer2(hidden2_size, output_size);
    Layer output_layer(hidden2_size, output_size);

    loadModelParameters("/Users/adityadikonda/PycharmProjects/OCR_on_ESP32/model_params.json", input_layer, hidden_layer1, hidden_layer2, output_layer);

    vector<double> input_data = loadImage("/Users/adityadikonda/PycharmProjects/OCR_on_ESP32/MNIST_CSV/mnist_test.csv");

    if (input_data.size() != static_cast<size_t>(input_size)) {
        cerr << "Error: Image data size does not match input layer size." << endl;
        return 1;
    }

    auto layer1_output = activationReLU(input_layer.forward(input_data));
    auto layer2_output = activationReLU(hidden_layer1.forward(layer1_output));
    auto layer3_output = activationReLU(hidden_layer2.forward(layer2_output));
    auto final_output = activationSoftMax(output_layer.forward(layer3_output));

    cout << "Predicted Probabilities:" << endl;
    for (size_t i = 0; i < final_output.size(); ++i) {
        cout << "Class " << i << ": " << fixed << setprecision(4) << final_output[i] << endl;
    }

    return 0;
}
