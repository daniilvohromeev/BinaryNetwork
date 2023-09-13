    #ifndef PERCEPTRON_H
    #define PERCEPTRON_H

    #include <iostream>
    #include <utility>
    #include <vector>
    #include <random>
    #include <algorithm>
    #include <functional>

    using namespace std;

    class Perceptron {
    public:
        double lr; // Learning rate
        vector<double> weights;

        Perceptron(int size, double lr = 0.3) {
            this->lr = lr;
            this->init(size);
        }

        Perceptron(double lr, vector<double>& weights) {
            this->lr = lr;
            this->weights.assign(weights.begin(), weights.end());
        }

        void init(int size) {
            random_device rnd_device;
            mt19937 mersenne_engine{rnd_device()};
            normal_distribution<double> dist{0.5, 0.01};
            auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

            vector<double> loc_weights(size + 1, 0);
            generate(begin(loc_weights), end(loc_weights), gen);
            this->weights.assign(loc_weights.begin(), loc_weights.end());
            this->weights[this->weights.size() - 1] = 1;
        }

        double predict(vector<double> inputs) {
            inputs.push_back(1); // Adding bias
            return this->evalutate(inputs);
        }

        double evalutate(vector<double> inputs) {
            return this->activate(this->weightedSum(std::move(inputs)));
        }

        double weightedSum(vector<double> inputs) {
            double sum = 0;
            for (int i = 0; i < inputs.size(); i++) {
                sum += inputs[i] * this->weights[i];
            }
            return sum + weights[weights.size() - 1];
        }

        double activate(double value) {
            // Using a sigmoid activation function
            return 1.0 / (1.0 + exp(-value));
        }

        // Calculate the gradient for a given input, target, and error
        double calculateGradient(double input, double target, double error) {
            return error * input * (1 - input);
        }

        // Update weights based on the calculated gradient
        void updateWeights(double gradient, vector<double>& inputs) {
            for (int i = 0; i < inputs.size(); i++) {
                this->weights[i] += this->lr * gradient * inputs[i];
            }
        }

        vector<double> getWeights() const {
            return weights;
        }

        bool train(vector<double> inputs, double target) {
            inputs.push_back(1); // Adding bias

            double actual = this->evalutate(inputs);
            double error = target - actual;

            if (abs(error) < 0.000001) {
                return true; // Converged
            }

            double gradient = calculateGradient(actual, target, error);
            updateWeights(gradient, inputs);

            return false;
        }
    };

    #endif
