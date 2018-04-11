#include "NN.h"
#include <armadillo>
#include <vector>
#include <algorithm>

using namespace arma;

void NeuralNetwork::sigmoid(dvec& A)
{
  A.for_each([](double& elem) { elem = 1.f/(1.f + exp(-elem)); });
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& structure, double step, double regularization)
{
  this->regularization = regularization;
  this->step = step;
  this->structure = structure;
  layers = int(structure.size());

  for(int i = 0; i < layers - 1; i++)
  {
    matrices.push_back(dmat(structure[i+1], structure[i], fill::randu));
    matrices.back().for_each([](double& x) { x = 2.0 * (x - 0.5); });
    delta.push_back(dmat(size(matrices.back()), fill::zeros));
    ignore.push_back(dmat(size(matrices.back()), fill::ones));
    for(int j = 0; j < structure[i+1]; j++)
      ignore.back()(j, 0) = 0;
    for(int j = 0; j < structure[i]; j++)
      ignore.back()(0, j) = 0;
  }

  numberOfTests = 0;
}

dvec NeuralNetwork::evaluateUnsupervised(const dvec& input)
{
  dvec ret = input;

  for (const auto& mat: matrices)
  {
    ret[0] = 1;
    ret = mat * ret;
    sigmoid(ret);
  }

  return ret;
}

double NeuralNetwork::evaluateSupervised(const dvec& input, const dvec& output, bool calculate_cost)
{
  std::vector<dvec> net;
  net.reserve(layers);
  net.push_back(input);

  for(const auto& mat: matrices)
  {
    net.back()[0] = 1;
    net.push_back(mat * net.back());
    sigmoid(net.back());
  }

  dvec ones(size(output), fill::ones);
  double cost = 0.0;
  if(calculate_cost)
  {
    cost = - sum(output % log(net.back()) + (ones - output) % log(ones - net.back()));
    for(int i = 0; i < layers - 1; i++)
    {
      cost += (1.0f / 2.0f) * regularization * accu(matrices[i] % matrices[i] % ignore[i]);
    }
  }

  std::vector<dvec> der;
  der.reserve(layers);
  der.push_back(net.back() - output);

  auto it = matrices.end();
  auto it2 = net.end();
  it--;
  it2--;
  it2--;
  while(it != matrices.begin())
  {
    der.push_back((it->t() * der.back()) % (*it2  - (*it2 % *it2)));
    der.back()[0] = 0;
    it--, it2--;
  }
  der.push_back(dvec(1));
  std::reverse(der.begin(), der.end());

  for(int i = 0; i < layers - 1; i++)
  {
    delta[i] += der[i+1] * net[i].t();
    delta[i] += (matrices[i] * regularization) % ignore[i];
  }
  numberOfTests++;

  return cost;
}

double NeuralNetwork::checkGradient(const dvec &input, const dvec &output)
{
  double epsilon = 0.000001;
  for (int i = 0; i < layers-1; i++)
  {
    delta[i] *= 0;
  }
  double cost = evaluateSupervised(input, output, true);

  std::vector<dmat> delta_priv;
  for(int i = 0; i < layers-1; i++)
  {
    delta_priv.push_back(dmat(delta[i]));
  }
  double tmpmaximum = 0.0f;
  int sign_errors = 0;
  int sum = 0;

  for(int i = 0; i < layers-1; i++)
  {
    for(int x = 0; x < int(delta[i].n_rows); x++)
    {
      for (int y = 0; y < int(delta[i].n_cols); y++)
      {
        if(x == 0 || y == 0)
          continue;
        matrices[i](x, y) += epsilon;
        double tmp_cost = evaluateSupervised(input, output, true);
        matrices[i](x, y) -= epsilon;
        double calc = std::abs((tmp_cost - cost)/epsilon - delta_priv[i](x, y));
        std::cout<<calc<<" = "<<(tmp_cost - cost)/epsilon<<" - "<<delta_priv[i](x, y)<<std::endl;
        tmpmaximum = std::max(tmpmaximum, calc);
        if((tmp_cost - cost)/epsilon * delta_priv[i](x, y) < 0)
          sign_errors++;
        sum++;
      }
    }
  }

  std::cout<<"checking Gradient results:\n";
  std::cout<<"maximum error: "<<tmpmaximum<<std::endl;
  std::cout<<"sign errors: "<<sign_errors<<" out of "<<sum<<std::endl;
  return tmpmaximum;
}

void NeuralNetwork::learn()
{
  for(auto& mat: delta)
  {
    mat /= double(numberOfTests);
  }
  numberOfTests = 0;

  for(int i = 0; i < layers - 1; i++)
  {
    matrices[i] -= (delta[i] * step);
    delta[i] *= 0;
  }
}

void NeuralNetwork::print()
{
  std::cout<<"----------------------------\n";
  for(int i = 0; i < layers - 1; i++)
  {
    std::cout<<i<<":\n"<<matrices[i]<<std::endl;
  }
  std::cout<<"----------------------------\n";
}
