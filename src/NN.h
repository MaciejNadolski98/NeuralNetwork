#pragma once
#include <armadillo>

using namespace arma;

class NeuralNetwork
{
private:
  int layers;
  std::vector<int> structure;
  double step;
  double regularization;

  std::vector<dmat> matrices;
  std::vector<dmat> ignore;
  std::vector<dmat> delta;
  int numberOfTests;

  static void sigmoid(dvec& A);

public:
  NeuralNetwork(const std::vector<int>& structure, double step, double regularization);

  dvec evaluateUnsupervised(const dvec& input);
  double evaluateSupervised(const dvec& input, const dvec& output, bool calculate_cost);

  double checkGradient(const dvec& input, const dvec& output);

  void learn();

  void print();
};
