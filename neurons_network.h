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
        std::vector<Perceptron> *neurons_in_lay = nullptr;
    };

    // принимает количество слоев, входных нейронов и нейронов для вывода
    neuron_network(int lay_count, int input_neurons, int output_neurons) {
        this->lay_count = lay_count;
        this->input_neurons = input_neurons;
        this->output_neurons = output_neurons;
        this->neuron_network_lays = new vector<neuron_network_lay>(lay_count);
        for (auto &unit: *this->neuron_network_lays) {
            unit.neurons_in_lay = new vector<Perceptron>(input_neurons,2);

        }//TODO: напиши нормальную инициализацию
    };

    int getLayCount() const {
        return lay_count;
    }

    void setLayCount(int layCount) {
        lay_count = layCount;
    }

    int getInputNeurons() const {
        return input_neurons;
    }

    void setInputNeurons(int inputNeurons) {
        input_neurons = inputNeurons;
    }

    int getOutputNeurons() const {
        return output_neurons;
    }

    void setOutputNeurons(int outputNeurons) {
        output_neurons = outputNeurons;
    }

    vector<neuron_network_lay> *getNeuronNetworkLays() const {
        return neuron_network_lays;
    }

    void setNeuronNetworkLays(vector<neuron_network_lay> *neuronNetworkLays) {
        neuron_network_lays = neuronNetworkLays;
    }

    int lay_count;
    int input_neurons;
    int output_neurons;
    std::vector<neuron_network_lay> *neuron_network_lays = nullptr;
};

#endif //PERCEPTRON_NEURONS_NETWORK_H
