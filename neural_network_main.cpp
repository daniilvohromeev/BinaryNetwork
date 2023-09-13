#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include "stb_image.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
using namespace std;
// Функция активации (сигмоида).
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Производная функции активации.
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, const std::vector<int>& hidden_layer_sizes, int output_size)
            : input_size_(input_size), hidden_layer_sizes_(hidden_layer_sizes), output_size_(output_size) {
        srand(time(0));
        initializeWeights();
    }

    // Инициализация весов нейронной сети.
    void initializeWeights() {
        int prev_layer_size = input_size_;

        // Инициализируем веса для скрытых слоев.
        for (size_t i = 0; i < hidden_layer_sizes_.size(); ++i) {
            int current_layer_size = hidden_layer_sizes_[i];
            std::vector<std::vector<double>> layer_weights(current_layer_size, std::vector<double>(prev_layer_size));
            hidden_weights_.push_back(layer_weights);
            prev_layer_size = current_layer_size;
        }

        // Инициализируем веса для выходного слоя.
        output_weights_.resize(output_size_, std::vector<double>(prev_layer_size));

        // Инициализируем биасы.
        hidden_biases_.resize(hidden_layer_sizes_.size(), 1.0);
        output_bias_ = 1.0;

        // Заполняем веса случайными значениями.
        for (size_t i = 0; i < hidden_weights_.size(); ++i) {
            for (int j = 0; j < hidden_layer_sizes_[i]; ++j) {
                for (int k = 0; k < prev_layer_size; ++k) {
                    hidden_weights_[i][j][k] = (double)rand() / RAND_MAX;
                }
            }
        }

        for (int i = 0; i < output_size_; ++i) {
            for (int j = 0; j < prev_layer_size; ++j) {
                output_weights_[i][j] = (double)rand() / RAND_MAX;
            }
        }
    }

    // Обучение нейронной сети методом обратного распространения ошибки.
    void train(const std::vector<double>& input_data, const std::vector<double>& target) {
        // Прямое распространение (подсчет выхода сети).
        std::vector<std::vector<double>> layer_outputs;
        std::vector<double> layer_input = input_data;

        for (size_t i = 0; i < hidden_weights_.size(); ++i) {
            std::vector<double> layer_output(hidden_layer_sizes_[i]);
            for (int j = 0; j < hidden_layer_sizes_[i]; ++j) {
                double weighted_sum = 0;
                for (int k = 0; k < input_size_; ++k) {
                    weighted_sum += layer_input[k] * hidden_weights_[i][j][k];
                }
                weighted_sum += hidden_biases_[i];
                layer_output[j] = sigmoid(weighted_sum);
            }
            layer_outputs.push_back(layer_output);
            layer_input = layer_output;
        }

        std::vector<double> output(output_size_);
        for (int i = 0; i < output_size_; ++i) {
            double weighted_sum = 0;
            for (int j = 0; j < hidden_layer_sizes_.back(); ++j) {
                weighted_sum += layer_input[j] * output_weights_[i][j];
            }
            weighted_sum += output_bias_;
            output[i] = sigmoid(weighted_sum);
        }

        // Вычисление ошибки.
        std::vector<double> output_errors(output_size_);
        for (int i = 0; i < output_size_; ++i) {
            output_errors[i] = target[i] - output[i];
        }

        // Обратное распространение ошибки и обновление весов.
        std::vector<double> delta(output_size_);
        for (int i = 0; i < output_size_; ++i) {
            delta[i] = output_errors[i] * sigmoid_derivative(output[i]);
        }

        for (int i = hidden_weights_.size() - 1; i >= 0; --i) {
            std::vector<double> next_delta(hidden_layer_sizes_[i]);
            for (int j = 0; j < hidden_layer_sizes_[i]; ++j) {
                double error_sum = 0;
                for (int k = 0; k < output_size_; ++k) {
                    error_sum += delta[k] * output_weights_[k][j];
                }
                next_delta[j] = error_sum * sigmoid_derivative(layer_outputs[i][j]);
            }

            for (int j = 0; j < hidden_layer_sizes_[i]; ++j) {
                for (int k = 0; k < input_size_; ++k) {
                    hidden_weights_[i][j][k] += learning_rate_ * next_delta[j] * input_data[k];
                }
            }

            delta = next_delta;
        }

        for (int i = 0; i < output_size_; ++i) {
            for (int j = 0; j < hidden_layer_sizes_.back(); ++j) {
                output_weights_[i][j] += learning_rate_ * delta[i] * layer_outputs.back()[j];
            }
        }
    }

    // Предсказание класса на основе входных данных.
    std::vector<double> predict(const std::vector<double>& input_data) {
        std::vector<double> layer_input = input_data;
        for (size_t i = 0; i < hidden_weights_.size(); ++i) {
            std::vector<double> layer_output(hidden_layer_sizes_[i]);
            for (int j = 0; j < hidden_layer_sizes_[i]; ++j) {
                double weighted_sum = 0;
                for (int k = 0; k < input_size_; ++k) {
                    weighted_sum += layer_input[k] * hidden_weights_[i][j][k];
                }
                weighted_sum += hidden_biases_[i];
                layer_output[j] = sigmoid(weighted_sum);
            }
            layer_input = layer_output;
        }

        std::vector<double> output(output_size_);
        for (int i = 0; i < output_size_; ++i) {
            double weighted_sum = 0;
            for (int j = 0; j < hidden_layer_sizes_.back(); ++j) {
                weighted_sum += layer_input[j] * output_weights_[i][j];
            }
            weighted_sum += output_bias_;
            output[i] = sigmoid(weighted_sum);
        }

        return output;
    }

private:
    int input_size_;
    std::vector<int> hidden_layer_sizes_;
    int output_size_;

    std::vector<std::vector<std::vector<double>>> hidden_weights_;
    std::vector<std::vector<double>> output_weights_;

    std::vector<double> hidden_biases_;
    double output_bias_;

    const double learning_rate_ = 0.1;
};

// Функция для преобразования черно-белого изображения в матрицу 7x7
vector<vector<double>> imageToMatrix(const unsigned char* image, int width, int height) {
    vector<vector<double>> result(7, vector<double>(7, 0.0)); // Создаем матрицу 7x7, заполненную нулями

    if (image == nullptr || width != 7 || height != 7) {
        cerr << "Error: Image size must be 7x7 pixels." << endl;
        return result;
    }

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            // В данном примере черный цвет (0) преобразуется в 1, белый цвет (255) преобразуется в 0
            result[i][j] = (image[i * width + j] < 200) ? 1.0 : 0.0;
        }
    }

    return result;
}

int main() {
    //пример загрузки и преобразования изображения в матрицу
    const char* filename = R"(C:\Users\LEGION\CLionProjects\BinaryNetwork\dataset\O.jpg)"; // Укажите путь к вашему изображению JPEG

    int width, height, channels;
    unsigned char* image = stbi_load(filename, &width, &height, &channels, 1); // Загрузка изображения в оттенках серого

    if (image == nullptr) {
        cerr << "Error: Unable to open image." << endl;
        return 1; // Ошибка при загрузке изображения
    }

    vector<vector<double>> imageMatrixTrain = imageToMatrix(image, width, height);
    //TODO: нужно доработать выгрузку датасета а то я прикинул мы заебемся тут

    // Создаем динамическую нейронную сеть с конфигурацией.
    NeuralNetwork neural_network(49, {16}, 1);

    // Обучаем нейронную сеть на примерах.
    std::vector<std::vector<double>> training_data = {
            {0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 1, 0, 0, 0,
             0, 0, 1, 1, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0},

            {0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0,
             0, 1, 1, 1, 0, 0, 0,
             0, 1, 1, 1, 0, 0, 0,
             0, 1, 1, 1, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0},


            {0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0,
             0, 0, 1, 1, 1, 0, 0,
             0, 1, 1, 1, 1, 1, 0,
             0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0},

            // здесь другие примеры для обучения сети.
    };

    std::vector<std::vector<double>> targets = {{1}, {1}, {0}}; // соответствующие метки классов.

    for (int epoch = 0; epoch < 10000; ++epoch) { // Обучение на 10000 эпох.
        for (size_t i = 0; i < training_data.size(); ++i) {
            neural_network.train(training_data[i], targets[i]);
        }
    }

    // Пример предсказания класса для новых данных.
    std::vector<double> new_data = {0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 1, 1, 1,
                                    0, 0, 0, 0, 1, 1, 1,
                                    0, 0, 0, 0, 1, 1, 1};

    std::vector<double> prediction = neural_network.predict(new_data);
    for (auto val: prediction) {
        std::cout << "Predict_output: " << val << std::endl;
    }
    return 0;
}
