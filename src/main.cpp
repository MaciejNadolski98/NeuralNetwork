#include <armadillo>
#include <fstream>
#include <random>
#include "NN.h"

using namespace std;

const int BATCH_SIZE = 500;
const int ITERATIONS = 80000;

const int TRAINING_SIZE = 500;
const int TEST_SIZE = 1000;

const double STEP = 1.0;
const double REGULARIZATION = 0.0;

const int DISPLAY_STEP = 1000;

const vector<int> STRUCTURE = {2, 7, 7, 1};


double f(double x)
{
  return sin(x);
}

double random_double()
{
  double ret = double(rand()%1000000)/1000000.0;
  return ret * 3.0;
}

int main()
{
  srand(time(NULL));
  vector<dvec> training_set;
  vector<dvec> training_labels;
  for (int i = 0; i < TRAINING_SIZE; i++)
  {
    training_set.push_back(dvec({0.0, random_double()}));
    training_labels.push_back(dvec({f(training_set.back()[1])}));
  }
  vector<dvec> test_set;
  vector<dvec> test_labels;
  for (int i = 0; i < TEST_SIZE; i++)
  {
    test_set.push_back(dvec({0.0, random_double()}));
    test_labels.push_back(dvec({f(training_set.back()[1])}));
  }
  NeuralNetwork nn(STRUCTURE, STEP, REGULARIZATION);

  //nn.checkGradient(training_set[0], training_labels[0]);
  //return 0;

  double cost = 0.0;
  int it = 0;
  for(int i=0;i<ITERATIONS;i++)
  {
    bool calc = false;
    if(i%DISPLAY_STEP == 0)
      calc = true;
    cost = 0.0;
    for(int j=0;j<BATCH_SIZE;j++)
    {
      if(it == TRAINING_SIZE)
        it = 0;
      cost += nn.evaluateSupervised(training_set[it], training_labels[it], calc);
      it++;
    }
    nn.learn();
    cost /= BATCH_SIZE;
    if(i%DISPLAY_STEP == 0)
    {
      cout<<i<<endl;
      cout<<"cost: "<<cost<<endl;
    }
  }

  double epsilon = 0.02;
  int correct = 0;
  double max_error = 0.0;
  for(int i=0;i<TEST_SIZE;i++)
  {
    cout<<"test "<<i<<endl;
    dvec out = nn.evaluateUnsupervised(test_set[i]);
    double ans = f(test_set[i][1]);
    cout<<"  answer: "<<ans<<endl;
    cout<<"  network:"<<out[0]<<endl;
    if (abs(ans - out[0]) < epsilon)
      correct++;
    max_error = max(max_error, abs(ans - out[0]));
  }
  cout<<"correct: "<<correct<<"/"<<TEST_SIZE<<endl;
  cout<<"max error: "<<max_error<<endl;
}
