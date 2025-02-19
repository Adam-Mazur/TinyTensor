# TinyTensor
![License](https://img.shields.io/badge/license-MIT-blue)
![C++](https://img.shields.io/badge/C++-17-blue)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)
![Dependencies](https://img.shields.io/badge/dependencies-none-green)
![Made with Love](https://img.shields.io/badge/made%20with-%E2%9D%A4-red)

This project is a from-scratch reimplementation of PyTorch’s Tensor object in C++. It supports fundamental tensor operations, including indexing, broadcasting, automatic differentiation, and various linear algebra functions. The goal is to provide a lightweight yet fully capable Tensor implementation that serves as a learning tool for exploring PyTorch internals.

I started this project as a university assignment for Object Programming classes, but I decided to continue developing it as a personal project. While it is still far from fully replicating PyTorch’s features, it is already capable of training simple neural networks, such as a CNN for MNIST, with reasonable performance.

## Features
Here is a list of features that the project supports:
- Strided array data structure 
- Memory management and reference counting
- Basic arithmetic operations (addition, subtraction, multiplication, division, etc.)
- Indexing and slicing (designed to closely match PyTorch behavior)
- Broadcasting (for element-wise operations and matrix multiplication)
- Linear algebra operations (e.g., matmul, mm)
- Math functions (e.g., exp, log, sum, mean, var, max, min)
- Machine learning operations (e.g., ReLU, softmax, cross-entropy)
- Automatic differentiation (backpropagation)
- Other PyTorch-inspired functions (e.g., stack, unfold, view, transpose) 

## Dependencies
This project uses the following dependencies:
- CMake
- Catch2 (installed automatically with CMake)
- Valgrind

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Adam-Mazur/TinyTensor.git
cd tiny_tensor
```
2. Create a build directory:
```bash
mkdir build
cd build
```
3. Run CMake:
```bash
cmake ..
```
4. Build the project:
```bash
make
```
5. Run the tests:
```bash
ctest
```
## Usage
Generally, the project is designed to match PyTorch’s API as closely as possible. However, some differences exist mainly because some simplifications were made to keep the project manageable. Here is an example of how to use the project to train a simple linear regressor:

```cpp
// ...
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
// ...
```

For the full example, see `tests/integration_test.cpp` file. 

---
Another example is training a simple CNN for the Binary MNIST dataset. The code below shows the forward pass of the network:

```cpp
// ...
Tensor<float> out1 = conv2d(x, w1, 16, 3, 2, 1);
Tensor<float> out2 = conv2d(Tensor<float>::relu(out1), w2, 32, 3, 2, 1);
Tensor<float> out3 = conv2d(Tensor<float>::relu(out2), w3, 64, 3, 2, 1);
Tensor<float> out4 = Tensor<float>::relu(out3).view({x.size()[0], -1}); // Flatten
Tensor<float> out5 = Tensor<float>::matmul(out4, w4) + b1;
Tensor<float> out6 = Tensor<float>::relu(out5);
return Tensor<float>::matmul(out6, w5) + b2;
// ...
```
For the full example, see the `src/demo.cpp` file. To run this code, you need to download the data with the following command:
```bash
python3 download_data.py
```  
And then run the following command (inside the build folder):
```bash
./demo
```

## Contributing
Contributions are welcome! If you'd like to improve this project, here’s how you can help:

### How to Contribute
1. Fork the repository and clone it to your local machine.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure the code is clean and well-documented.
4. Commit and push your changes.
5. Open a pull request on GitHub.

### Reporting Issues
If you find a bug or have a feature request, please open an issue with a detailed description.


## License
This project is licensed under the **MIT License**. 