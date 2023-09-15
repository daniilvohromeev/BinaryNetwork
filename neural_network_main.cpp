#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>

// Функция сигмоидальной активации
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Производная функции сигмоидальной активации
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

class Neuron {
public:
    Neuron(int num_inputs) {
        weights.resize(num_inputs);
        for (int i = 0; i < num_inputs; ++i) {
            // Инициализация весов случайными значениями
            weights[i] = static_cast<double>(rand()) / RAND_MAX;
        }
        bias = static_cast<double>(rand()) / RAND_MAX;
    }

    double feedforward(const std::vector<double>& inputs) {
        double sum = bias;
        for (int i = 0; i < inputs.size(); ++i) {
            sum += inputs[i] * weights[i];
        }
        output = sigmoid(sum);
        return output;
    }

    double getOutput() const {
        return output;
    }

    double getWeight(int index) const {
        return weights[index];
    }

    void updateWeight(int index, double delta) {
        weights[index] += delta;
    }

    void updateBias(double delta) {
        bias += delta;
    }

private:
    std::vector<double> weights;
    double bias;
    double output;
};

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layer_sizes) {
        if (layer_sizes.size() < 2) {
            std::cerr << "Error: At least 2 layers are required (input and output layers)." << std::endl;
            exit(1);
        }

        for (int i = 1; i < layer_sizes.size(); ++i) {
            std::vector<Neuron> layer;
            for (int j = 0; j < layer_sizes[i]; ++j) {
                int num_inputs = (i == 1) ? layer_sizes[0] : layer_sizes[i - 1];
                layer.push_back(Neuron(num_inputs));
            }
            layers.push_back(layer);
        }
    }

    std::vector<double> predict(const std::vector<double>& inputs) {
        std::vector<double> outputs;

        for (int i = 0; i < layers.size(); ++i) {
            std::vector<double> layer_outputs;
            for (int j = 0; j < layers[i].size(); ++j) {
                if (i == 0) {
                    layer_outputs.push_back(layers[i][j].feedforward(inputs));
                } else {
                    layer_outputs.push_back(layers[i][j].feedforward(outputs));
                }
            }
            outputs = layer_outputs;
        }

        return outputs;
    }

    void train(const std::vector<std::vector<double>>& input_data, const std::vector<std::vector<double>>& target_data, double learning_rate, int num_epochs) {
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            for (int i = 0; i < input_data.size(); ++i) {
                std::vector<std::vector<double>> layer_inputs(layers.size());
                std::vector<std::vector<double>> layer_outputs(layers.size());

                // Прямое распространение
                for (int j = 0; j < layers.size(); ++j) {
                    if (j == 0) {
                        layer_inputs[j] = input_data[i];
                    } else {
                        layer_inputs[j] = layer_outputs[j - 1];
                    }

                    for (int k = 0; k < layers[j].size(); ++k) {
                        layer_outputs[j].push_back(layers[j][k].feedforward(layer_inputs[j]));
                    }
                }

                // Обратное распространение
                std::vector<std::vector<double>> layer_errors(layers.size());
                for (int j = layers.size() - 1; j >= 0; --j) {
                    layer_errors[j].resize(layers[j].size(), 0.0);

                    if (j == layers.size() - 1) {
                        for (int k = 0; k < layers[j].size(); ++k) {
                            double error = target_data[i][k] - layer_outputs[j][k];
                            layer_errors[j][k] = error * sigmoid_derivative(layer_outputs[j][k]);
                        }
                    } else {
                        for (int k = 0; k < layers[j].size(); ++k) {
                            double error = 0.0;
                            for (int n = 0; n < layers[j + 1].size(); ++n) {
                                error += layer_errors[j + 1][n] * layers[j + 1][n].getWeight(k);
                            }
                            layer_errors[j][k] = error * sigmoid_derivative(layer_outputs[j][k]);
                        }
                    }
                }

                // Обновление весов и смещений
                for (int j = 0; j < layers.size(); ++j) {
                    for (int k = 0; k < layers[j].size(); ++k) {
                        for (int n = 0; n < layer_inputs[j].size(); ++n) {
                            double delta = learning_rate * layer_errors[j][k] * layer_inputs[j][n];
                            layers[j][k].updateWeight(n, delta);
                        }
                        double delta = learning_rate * layer_errors[j][k];
                        layers[j][k].updateBias(delta);
                    }
                }
            }
        }
    }

private:
    std::vector<std::vector<Neuron>> layers;
};

int main() {

    std::vector<std::vector<double>> training_data = {
            {0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 1, 1, 0, 0,
                    0, 0, 1, 1, 1, 0, 0,
                    0, 0, 1, 1, 1, 0, 0,
                    0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0},


            {0, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 1, 1, 0, 0,
                    0, 1, 0, 0, 0, 1, 0,
                    0, 1, 0, 0, 0, 1, 0,
                    0, 1, 0, 0, 0, 1, 0,
                    0, 0, 1, 1, 1, 0, 0,
                    0, 0, 0, 0, 0, 0, 0},

            {0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 0, 0,
                    0, 0, 1, 0, 1, 0, 0,
                    0, 1, 1, 1, 1, 1, 0,
                    0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0},

            // здесь другие примеры для обучения сети.
    };
    std::vector<std::vector<double>> test_data = {
            {0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0,
             0, 0, 1, 1, 1, 0, 0,
             0, 1, 1, 1, 1, 1, 0,
             0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0},
    };
    std::vector<std::vector<double>> targets = {{1,0,0}, {0,1,0}, {0,0,1}};
    // Инициализация генератора случайных чисел
    srand(static_cast<unsigned>(time(nullptr)));

    // Пример данных для обучения и тестирования
    std::vector<std::vector<double>> input_data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> target_data = {{0,1}, {1,0}, {1,0}, {0,1}};

    std::vector<int> layer_sizes = {49, 16, 16, 3}; // Количество нейронов в каждом слое

    NeuralNetwork nn(layer_sizes);

    double learning_rate = 0.5;
    int num_epochs = 10000;

    nn.train(training_data, targets, learning_rate, num_epochs);

    // Тестирование
    for (const auto& input : test_data) {
        std::vector<double> predicted = nn.predict(input);
        std::cout << "Input: {" << input[0] << ", " << input[1] << "} ";
        std::cout << "Output: " << predicted[0] <<"/"<< predicted[1] << "/" << predicted[2] << std::endl;
    }

    return 0;
}
