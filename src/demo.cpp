/**
This demo showcases a simple convolutional neural network implemented using a
custom tensor object for a binarized version of the MNIST dataset.
The network consists of three convolutional layers followed by two fully connected layers.
Training is performed using cross-entropy loss and optimized with gradient descent.

The demo measures the average training time per iteration and evaluates the final model's
accuracy on the test set, which typically achieves around 99% accuracy.

To get started, download the training and test data by running:

    python3 download_data.py

This script will retrieve the necessary binary files and place them in the correct directory.
*/

#include "../include/tensor.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

// ========================================================
// SECTION: Constants
// ========================================================

#define TRAIN_DATA_PATH "../data/train.bin"
#define TEST_DATA_PATH "../data/test.bin"
#define TRAIN_DATA_LEN 11824
#define TEST_DATA_LEN 2956
#define IMG_SIZE 28
#define TEST_SIZE 1000
#define BATCH_SIZE 32
#define LEARNING_RATE 0.0005

// ========================================================
// SECTION: Auxiliary functions
// ========================================================

std::pair<Tensor<float>, Tensor<int>> load_mnist(const std::string &path, int data_length)
{
    std::ifstream file(path, std::ios::binary);

    // The data is stored as a binary file, the beginning of the file contains the labels,
    // and the rest of the file contains the images. Both labels and images are stored as uint8_t.
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    Tensor<float> images = Tensor<float>::zeros({data_length, IMG_SIZE, IMG_SIZE});
    Tensor<int> labels = Tensor<int>::zeros({data_length});

    for (int i = 0; i < data_length; i++)
    {
        labels[{i}] = data[i];
    }

    for (int i = 0; i < data_length; i++)
    {
        int offset = data_length + i * IMG_SIZE * IMG_SIZE;
        for (int j = 0; j < IMG_SIZE; j++)
        {
            for (int k = 0; k < IMG_SIZE; k++)
            {
                images[{i, j, k}] = data[offset + j * IMG_SIZE + k] / 255.0;
            }
        }
    }

    return {images, labels};
}

Tensor<float> conv2d(const Tensor<float> &in, const Tensor<float> &w, int out_channels, int kernel_size, int stride,
                     int padding)
{
    Tensor<float> inp_unf = Tensor<float>::unfold(in, kernel_size, padding, stride);
    Tensor<float> out_unf = Tensor<float>::matmul(w.view({out_channels, -1}), inp_unf);

    int batch_size = in.size()[0];
    int output_height = (int)((in.size()[2] + 2 * padding - kernel_size) / stride) + 1;

    return out_unf.view({batch_size, out_channels, output_height, -1});
}

Tensor<float> forward(const Tensor<float> &x, const Tensor<float> &w1, const Tensor<float> &w2, const Tensor<float> &w3,
                      const Tensor<float> &w4, const Tensor<float> &w5, const Tensor<float> &b1,
                      const Tensor<float> &b2)
{
    Tensor<float> out1 = conv2d(x, w1, 16, 3, 2, 1);
    Tensor<float> out2 = conv2d(Tensor<float>::relu(out1), w2, 32, 3, 2, 1);
    Tensor<float> out3 = conv2d(Tensor<float>::relu(out2), w3, 64, 3, 2, 1);
    Tensor<float> out4 = Tensor<float>::relu(out3).view({x.size()[0], -1}); // Flatten
    Tensor<float> out5 = Tensor<float>::matmul(out4, w4) + b1;
    Tensor<float> out6 = Tensor<float>::relu(out5);
    return Tensor<float>::matmul(out6, w5) + b2;
}

int main()
{
    auto [train_images, train_labels] = load_mnist(TRAIN_DATA_PATH, TRAIN_DATA_LEN);
    auto [test_images, test_labels] = load_mnist(TEST_DATA_PATH, TEST_DATA_LEN);

    Tensor<float> w1 = Tensor<float>::randn({16, 1, 3, 3}, true);
    Tensor<float> w2 = Tensor<float>::randn({32, 16, 3, 3}, true);
    Tensor<float> w3 = Tensor<float>::randn({64, 32, 3, 3}, true);
    Tensor<float> w4 = Tensor<float>::randn({64 * 4 * 4, 128}, true);
    Tensor<float> w5 = Tensor<float>::randn({128, 2}, true);
    Tensor<float> b1 = Tensor<float>::randn({128}, true);
    Tensor<float> b2 = Tensor<float>::randn({2}, true);

    // ========================================================
    // SECTION: Training
    // ========================================================

    auto start_time = std::chrono::high_resolution_clock::now();

    int i = 0;
    for (; i + BATCH_SIZE < TRAIN_DATA_LEN; i += BATCH_SIZE)
    {
        Tensor<float> x = train_images[{{i, i + BATCH_SIZE}}].view({BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE});
        Tensor<int> y = train_labels[{{i, i + BATCH_SIZE}}];

        w1.zero_grad();
        w2.zero_grad();
        w3.zero_grad();
        w4.zero_grad();
        w5.zero_grad();
        b1.zero_grad();
        b2.zero_grad();

        Tensor<float> out = forward(x, w1, w2, w3, w4, w5, b1, b2);
        Tensor<float> loss = Tensor<float>::cross_entropy(out, y);
        
        loss.backward();

        w1 += (*w1.grad) * (-LEARNING_RATE);
        w2 += (*w2.grad) * (-LEARNING_RATE);
        w3 += (*w3.grad) * (-LEARNING_RATE);
        w4 += (*w4.grad) * (-LEARNING_RATE);
        w5 += (*w5.grad) * (-LEARNING_RATE);
        b1 += (*b1.grad) * (-LEARNING_RATE);
        b2 += (*b2.grad) * (-LEARNING_RATE);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << std::fixed << std::setprecision(3) << "Average time per iteration: " << elapsed.count() / i * 1000
              << " miliseconds" << std::endl;

    // ========================================================
    // SECTION: Inference
    // ========================================================

    NoGradGuard no_grad;

    Tensor<float> x = test_images[{{0, TEST_SIZE}}].view({TEST_SIZE, 1, IMG_SIZE, IMG_SIZE});
    Tensor<float> out = forward(x, w1, w2, w3, w4, w5, b1, b2);

    int correct = 0;
    for (int i = 0; i < TEST_SIZE; i++)
    {
        int predicted = out[{{i, i + 1}}].argmax()[{0}];
        int actual = test_labels[{{i, i + 1}}][{0}];

        if (predicted == actual)
        {
            correct++;
        }
    }

    std::cout << std::fixed << std::setprecision(1) << "Accuracy: " << (float)correct / TEST_SIZE * 100 << "%"
              << std::endl;

    return 0;
}