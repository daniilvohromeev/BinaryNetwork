#include <iostream>
#include <vector>
#include "perceptron.h"
#include "neurons_network.h"
using namespace std;

int main() {

    auto *network = new neuron_network(2,2,2);

//    vector<pair<vector<double>, int>> trainingset{
//            make_pair(vector<double>{1, 1}, 1),
//            make_pair(vector<double>{0, 1}, 0),
//            make_pair(vector<double>{1, 0}, 0)
//    };
//    vector<pair<vector<double>, int>> trainingset2{
//            make_pair(vector<double>{1, 1}, 1),
//            make_pair(vector<double>{0, 1}, 1),
//            make_pair(vector<double>{1, 0}, 0)
//    };
//    vector<vector<pair<vector<double>, int>>> trainingSets{
//            trainingset, trainingset2
//    };
//
//    struct neurons_lay {
//        Perceptron *per = new Perceptron(2);
//    };
//
//
//    auto *neurons_List = new vector<neurons_lay>(2);
//
//    cout << "********************PRIMARY WEIGHTS******************" << endl;
//    for (auto unit: *neurons_List) {
//        unit.per->printWeights();
//    }
//    cout << "********************EQUIVALENCE**********************" << endl;
//    neurons_List->at(0).per->traintilllearn(trainingSets.at(0));
//    cout << "********************REAL RESULT**********************" << endl;
//    for (const auto &unit: trainingset) {
//        cout <<neurons_List->at(0).per->predict(unit.first)<<endl;
//    }
//    cout << "********************ROUND RESULT********************" << endl;
//    for (const auto &unit: trainingset) {
//        cout <<abs(round(neurons_List->at(0).per->predict(unit.first)))<<endl;
//    }
//    cout << "********************IMPLICATION**********************" << endl;
//    neurons_List->at(1).per->traintilllearn(trainingSets.at(1));
//    cout << "********************REAL RESULT**********************" << endl;
//    for (const auto& unit: trainingset2) {
//        cout << neurons_List->at(1).per->predict(unit.first)<<endl;
//    }
//    cout << "********************ROUND RESULT*********************" << endl;
//    for (const auto &unit: trainingset2) {
//        cout <<abs(round(neurons_List->at(1).per->predict(unit.first)))<<endl;
//    }
//
//    cout << "********************WEIGHTS AFTER TRAINING***********" << endl;
//    for (auto unit: *neurons_List) {
//        unit.per->printWeights();
//    }
//    cout<<neurons_List->at(1).per->predict(vector<double>{0, 0});
    return 0;
}