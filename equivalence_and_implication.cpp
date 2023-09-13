#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "neurons_network.h"

#include <iostream>
#include <vector>

using namespace std;

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
    const char* filename = R"(C:\Users\LEGION\CLionProjects\BinaryNetwork\dataset\O.jpg)"; // Укажите путь к вашему изображению JPEG

    int width, height, channels;
    unsigned char* image = stbi_load(filename, &width, &height, &channels, 1); // Загрузка изображения в оттенках серого

    if (image == nullptr) {
        cerr << "Error: Unable to open image." << endl;
        return 1; // Ошибка при загрузке изображения
    }

    vector<vector<double>> imageMatrixTrain = imageToMatrix(image, width, height);
    vector<vector<double>> circleData = {
            // Circle matrix (as previously defined)
    };

    vector<vector<double>> squareData = {
            // Square matrix
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
    };

    vector<vector<double>> triangleData = {
            // Triangle matrix
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0},
            {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
            {0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0},
            {0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0},
            {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
    };

    // Define unique target labels for each shape
    int circleLabel = 0;
    int squareLabel = 1;
    int triangleLabel = 2;

    // Combine training data and labels
    vector<vector<double>> trainingData;
    trainingData.insert(trainingData.end(), circleData.begin(), circleData.end());
    trainingData.insert(trainingData.end(), squareData.begin(), squareData.end());
    trainingData.insert(trainingData.end(), triangleData.begin(), triangleData.end());

    vector<int> targetLabels;
    targetLabels.push_back(circleLabel);
    targetLabels.push_back(squareLabel);
    targetLabels.push_back(triangleLabel);

    // Create and configure the neural network
    neuron_network my_network(49, 3);  // 49 input neurons (7x7 matrix), 3 output neurons (for three shapes)
    my_network.addHiddenLayer(10);  // Add a hidden layer with 10 neurons

    // Train the neural network with data for all three shapes
    my_network.train(trainingData, targetLabels, 1000);

    // Test the trained neural network with an example input (e.g., square)
    vector<double> input_data = squareData[0]; // Test with a square
    vector<double> output = my_network.predict(input_data);

    // Display the network's output, which should indicate the recognized shape (0 for circle, 1 for square, 2 for triangle)
    for (double i : output) {
        cout << "Recognized Shape: " << i << endl;
    }
    stbi_image_free(image); // Освобождение памяти, занятой изображением

    return 0;
}
