#pragma once
#include <cstddef>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

template <typename T> struct TensorData
{
    std::vector<T> vec;
    size_t reference_count;

    TensorData(const std::vector<T> &data);
    TensorData(int size, T value);

    size_t size();

    T &operator[](size_t index);

    const T &operator[](size_t index) const;
};

template <typename T> class Tensor
{
  private:
    TensorData<T> *data;
    std::vector<int> shape;
    int offset;
    std::vector<int> strides;

    Tensor<T> *operand1;
    Tensor<T> *operand2;
    std::function<void(Tensor *)> _backward;

    void add_reference();

    void release();

    Tensor(const std::vector<int> &size, T value, bool requires_grad = false);

    T &operator[](const std::vector<int> &indices);

    Tensor operator[](const std::vector<std::pair<int, int>> &indices);

    template <typename Op> static Tensor broadcast(const Tensor<T> &t1, const Tensor<T> &t2, Op op);

    size_t get_hash();

    bool backward_enabled() const;

    void add_backward();

    void sub_backward();

    void minus_backward();

    void mul_backward();

    void div_backward();

    void sum_backward();

    void sum2_backward(const std::vector<int> &dim, bool keep_dim);

    void sqrt_backward();

    void pow_backward(T exponent);

    void exp_backward();

    void log_backward();

    void relu_backward();

    void cross_entropy_backward(int n, std::vector<int> &target);

    void mm_backward();

    void matmul_backward(std::vector<int> &t1_shape, std::vector<int> &t2_shape, std::vector<int> &new_shape);

    void unfold_backward(int kernel_size, int padding, int stride);

  public:
    Tensor<T> *grad;

    Tensor(const std::vector<T> &data, bool requires_grad = false);

    Tensor();

    Tensor(const Tensor &other);

    Tensor &operator=(const Tensor &other);

    Tensor(Tensor &&other) noexcept;

    Tensor &operator=(Tensor &&other) noexcept;

    static Tensor zeros(const std::vector<int> &size, bool requires_grad = false);

    static Tensor ones(const std::vector<int> &size, bool requires_grad = false);

    static Tensor randn(const std::vector<int> &size, bool requires_grad = false);

    static Tensor xavier_normal(const std::vector<int> &size, float gain = 1.0, bool requires_grad = false);

    static Tensor kaiming_normal(const std::vector<int> &size, bool requires_grad = false);

    T &operator[](const std::initializer_list<int> &indices);

    Tensor operator[](const std::initializer_list<std::pair<int, int>> &indices);

    class Iterator
    {
      private:
        Tensor<T> &tensor;
        int last_elem_index;

      public:
        std::vector<int> indices;
        int data_index;

        Iterator(Tensor<T> &tensor, std::vector<int> indices) : tensor(tensor), indices(indices)
        {
            if (tensor.numel() == 0)
            {
                data_index = 0;
                last_elem_index = 0;
            }
            else
            {
                // Calculate the index of the element at (shape[0]-1, shape[1]-1, shape[2]-1, ...).
                last_elem_index = std::inner_product(tensor.strides.begin(), tensor.strides.end(), tensor.shape.begin(),
                                                     tensor.offset) -
                                  std::accumulate(tensor.strides.begin(), tensor.strides.end(), 0);

                data_index =
                    std::inner_product(tensor.strides.begin(), tensor.strides.end(), indices.begin(), tensor.offset);
            }
        };

        T &operator*()
        {
            return (*tensor.data)[data_index];
        }

        Iterator &operator++()
        {
            // If data_index == last_data_index then the for loop below would reset it back
            // to the index of the first element. So, we need to set it manually.
            if (data_index >= last_elem_index)
            {
                data_index = last_elem_index + tensor.strides.back();
                return *this;
            }

            for (int j = indices.size() - 1; j >= 0; j--)
            {
                indices[j]++;
                data_index += tensor.strides[j];

                if (indices[j] >= tensor.shape[j])
                {
                    indices[j] = 0;
                    data_index -= tensor.shape[j] * tensor.strides[j];
                }
                else
                {
                    break;
                }
            }
            return *this;
        }

        bool operator==(const Iterator &other) const
        {
            return data_index == other.data_index;
        }

        bool operator!=(const Iterator &other) const
        {
            return !(*this == other);
        }

        Iterator &operator=(const Iterator &other)
        {
            indices = other.indices;
            data_index = other.data_index;
            return *this;
        }
    };

    Iterator begin()
    {
        return Iterator(*this, std::vector<int>(shape.size(), 0));
    }

    Iterator end()
    {
        if (this->numel() == 0)
        {
            return Iterator(*this, std::vector<int>(shape.size(), 0));
        }

        std::vector<int> end_indecies = shape;
        end_indecies.back() += 1;

        for (int i = 0; i < end_indecies.size(); i++)
        {
            end_indecies[i] -= 1;
        }

        return Iterator(*this, end_indecies);
    }

    Tensor view(const std::vector<int> &size);

    Tensor transpose(int dim0, int dim1);

    Tensor clone();

    bool equal(Tensor &other);

    std::vector<int> size();

    int numel();

    void backward();

    void zero_grad();

    Tensor operator+(Tensor<T> &other);

    Tensor operator+(T other);

    Tensor operator-(Tensor<T> &other);

    Tensor operator-(T other);

    Tensor operator-();

    Tensor operator*(Tensor<T> &other);

    Tensor operator*(T other);

    Tensor operator/(Tensor<T> &other);

    Tensor operator/(T other);

    template <typename U> friend Tensor<U> operator/(U other, Tensor<U> &t);

    Tensor &operator+=(Tensor<T> &other);

    Tensor sum();

    Tensor sum(const std::vector<int> &dim, bool keep_dim);

    Tensor sqrt();

    Tensor pow(T exponent);

    Tensor exp();

    Tensor log();

    Tensor<int> argmax();

    Tensor max();

    Tensor mean(const std::vector<int> &dim, bool keep_dim);

    Tensor mean();

    Tensor var(const std::vector<int> &dim, bool keep_dim);

    Tensor var();

    static Tensor relu(Tensor &t);

    static Tensor sigmoid(Tensor &tm);

    static Tensor softmax(Tensor &t, int dim = 0);

    static Tensor cross_entropy(Tensor &input, Tensor<int> &target);

    static Tensor mm(Tensor &t1, Tensor &t2, Tensor *out = nullptr);

    static Tensor matmul(Tensor &t1, Tensor &t2);

    static Tensor stack(std::vector<Tensor> tensors, int dim = 0);

    static Tensor unfold(Tensor &in, int kernel_size, int padding, int stride);

    ~Tensor();
};

class NoGradGuard
{
  public:
    inline static bool is_enabled = false;
    bool prev_state;
    NoGradGuard() : prev_state(is_enabled)
    {
        is_enabled = true;
    }
    ~NoGradGuard()
    {
        is_enabled = prev_state;
    }
};

// This is to mitigate the template definition errors
#include "../src/tensor.cpp"