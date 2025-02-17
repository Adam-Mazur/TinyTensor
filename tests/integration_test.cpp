#include "../include/tensor.h"
#include <iomanip>
#include <iostream>

#define TRUE_W 3.0
#define TRUE_B 2.0
#define NUM_SAMPLES 100
#define EPOCHS 500
#define LEARNING_RATE 0.005

int main()
{
    Tensor<float> x = Tensor<float>::randn({NUM_SAMPLES, 1}) * 10.0;
    Tensor<float> y = x * TRUE_W + TRUE_B + Tensor<float>::randn({NUM_SAMPLES, 1}) * 0.1;

    Tensor<float> w = Tensor<float>::randn({1}, true);
    Tensor<float> b = Tensor<float>::randn({1}, true);

    for (int i = 0; i < EPOCHS; i++)
    {
        w.zero_grad();
        b.zero_grad();

        Tensor<float> y_pred = x * w + b;
        Tensor<float> loss = (y_pred - y).pow(2).mean();

        loss.backward();

        w += (*w.grad) * (-LEARNING_RATE);
        b += (*b.grad) * (-LEARNING_RATE);
    }

    std::cout << std::fixed << std::setprecision(2) << "True w: " << TRUE_W << " Learned w: " << w[{0}] << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "True b: " << TRUE_B << " Learned b: " << b[{0}] << std::endl;

    return 0;
}