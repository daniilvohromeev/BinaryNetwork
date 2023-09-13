#ifndef PERCEPTRON_NEURONS_NETWORK_H
#define PERCEPTRON_NEURONS_NETWORK_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include "perceptron.h"

using namespace std;

class neuron_network {
public:
    struct neuron_network_lay {
        vector<Perceptron> neurons_in_lay;
    };

    neuron_network(int input_neurons, int output_neurons) {
        this->input_neurons = input_neurons;
        this->output_neurons = output_neurons;
        addOutputLayer(output_neurons);  // Add an output layer with the specified number of neurons
    }

    void addOutputLayer(int neuron_count) {
        neuron_network_lay layer;
        for (int i = 0; i < neuron_count; ++i) {
            layer.neurons_in_lay.emplace_back(input_neurons + 1); // Ajouter 1 pour le biais
        }
        neuron_network_lays.push_back(layer);
    }

    void addHiddenLayer(int neuron_count) {
        neuron_network_lay layer;
        for (int i = 0; i < neuron_count; ++i) {
            layer.neurons_in_lay.emplace_back(input_neurons);
        }
        neuron_network_lays.push_back(layer);
    }

    int getLayerCount() const {
        return neuron_network_lays.size();
    }

    int getInputNeurons() const {
        return input_neurons;
    }

    int getOutputNeurons() const {
        return output_neurons;
    }

    vector<neuron_network_lay>& getNeuronNetworkLays() {
        return neuron_network_lays;
    }

    void train(const vector<vector<double>>& training_data, const vector<int>& target_labels, int max_epochs) {
        for (int epoch = 0; epoch < max_epochs; ++epoch) {
            bool did_learn = true;
            for (size_t i = 0; i < training_data.size(); ++i) {
                if (!feedForwardAndBackpropagate(training_data[i], target_labels[i])) {
                    did_learn = false;
                }
            }
            if (did_learn) {
                cout << "Converged in epoch " << epoch << endl;
                break;
            }
        }
    }

    vector<double> predict(const vector<double>& input_data) {
        vector<double> output;

        // Perform feedforward propagation
        vector<double> current_input = input_data;
        for (auto& layer : neuron_network_lays) {
            vector<double> next_output;
            for (auto& neuron : layer.neurons_in_lay) {
                next_output.push_back(neuron.predict(current_input));
            }
            current_input = next_output;
        }

        output = current_input;
        return output;
    }

private:
    int input_neurons;
    int output_neurons;
    vector<neuron_network_lay> neuron_network_lays;

    bool feedForwardAndBackpropagate(const vector<double>& input, int target) {
        vector<vector<double>> outputs;
        outputs.emplace_back(input);

        for (auto& layer : neuron_network_lays) {
            vector<double> next_output;
            next_output.push_back(1.0); // Ajouter une entrée constante pour le biais
            for (auto& neuron : layer.neurons_in_lay) {
                next_output.push_back(neuron.predict(outputs.back()));
            }
            outputs.emplace_back(next_output);
        }

        if (outputs.back().size() != output_neurons) {
            cerr << "Output layer size does not match expected output neurons." << endl;
            return false;
        }

        vector<double> errors(output_neurons);
        for (int i = 0; i < output_neurons; ++i) {
            errors[i] = target - outputs.back()[i];
        }

        int layer_index = neuron_network_lays.size() - 1;
        for (auto layer_it = neuron_network_lays.rbegin(); layer_it != neuron_network_lays.rend(); ++layer_it) {
            vector<double> new_errors;
            for (int i = 0; i < layer_it->neurons_in_lay.size(); ++i) {
                double delta = errors[i] * outputs[layer_index + 1][i] * (1.0 - outputs[layer_index + 1][i]);
                layer_it->neurons_in_lay[i].train(outputs[layer_index], delta);
                double error = 0.0;
                for (int j = 0; j < outputs[layer_index].size(); ++j) {
                    error += delta * layer_it->neurons_in_lay[i].getWeights()[j];
                }
                new_errors.push_back(error);
            }
            errors = new_errors;
            --layer_index;
        }

        return true;
    }

};

#endif //PERCEPTRON_NEURONS_NETWORK_H
