#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>

using namespace std;

class Perceptron {


public:
    double lr; //learning rate
    vector<double> weights;

    Perceptron(int inputsize, double lr = 0.3) {
        this->lr = lr;
        this->init(inputsize);
    }

    Perceptron(double lr, vector<double> &weights) {

        //this->bias = bias;
        this->lr = lr;
        this->weights.assign(weights.begin(), weights.end());
    }

    //Print perceptron weights for debugging.
    void printWeights() {

        for (auto weight: this->weights) {
            cout << weight << ",";
        }

        cout << endl;
    }


    void init(int size) {

        // First create an instance of an engine.
        random_device rnd_device;
        // Specify the engine and distribution.
        mt19937 mersenne_engine{rnd_device()};  // Generates random double
        normal_distribution<double> dist{0.5, 0.01};
        auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };


        vector<double> weights(size + 1, 0);
        generate(begin(weights), end(weights), gen);
        this->weights.assign(weights.begin(), weights.end());
        //adding bias
        this->weights[this->weights.size() - 1] = 1;

        //this->weights = weights;
    }


    double dt(double actualval, int target, double inputval, double lr) {
        double err = target - actualval;
        return err * lr * inputval;
    }


    bool train(vector<double> inputs, int target) {

        inputs.push_back(1);//adding bias

        double actual = this->evalutate(inputs);
        //double d = round(actual * 1000) / 1000;
        if (abs(target-actual) < 0.000001) return true;


        for (int i = 0; i < this->weights.size(); i++) {

            this->weights[i] += this->dt(actual, target, inputs[i], this->lr);
        }

        return false;
    }

    void traintilllearn(vector<pair<vector<double>, int>> traningset) {

        bool didlearn = false;
        while (!didlearn) {

            didlearn = true;

            for (auto unit: traningset) {
                if (!train(unit.first, unit.second)) {
                    didlearn = false;
                }
            }

        }
        return;

    }


    double predict(vector<double> inputs) {

        //push bias
        inputs.push_back(1);
        return this->evalutate(inputs);
    }

    double weightedSum(vector<double> inputs) {
        double sum = 0;
        for (int i = 0; i < inputs.size(); i++) {
            sum += inputs[i] * this->weights[i];
        }
        return sum+weights[weights.size()-1];


    }

    double evalutate(vector<double> inputs) {
        return this->activate(this->weightedSum(inputs));
    }

    double activate(double value) {
        return tanh(value / 1.0);
    }

};

#endif