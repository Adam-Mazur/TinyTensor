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

    size_t size() const;

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

    T &operator[](const std::vector<int> &indices) const;

    Tensor operator[](const std::vector<std::pair<int, int>> &indices) const;

    template <typename Op> static Tensor broadcast(const Tensor<T> &t1, const Tensor<T> &t2, Op op);

    size_t get_hash() const;

    bool backward_enabled() const;

    void set_backward(std::function<void(Tensor *)> grad_fn, const Tensor<T> *op1 = nullptr,
                      const Tensor<T> *op2 = nullptr);

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

    void max_and_min_backward(const Tensor<T> &indices);

    void relu_backward();

    void cross_entropy_backward(int n, const std::vector<int> &target);

    void mm_backward();

    void matmul_backward(const std::vector<int> &t1_shape, const std::vector<int> &t2_shape,
                         const std::vector<int> &new_shape);

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

    T &operator[](const std::initializer_list<int> &indices) const;

    Tensor operator[](const std::initializer_list<std::pair<int, int>> &indices) const;

    class Iterator
    {
      private:
        Tensor<T> &tensor;
        int last_elem_index;

      public:
        std::vector<int> indices;
        int data_index;

        Iterator(Tensor<T> &tensor, const std::vector<int> indices) : tensor(tensor), indices(indices)
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

        Iterator(const Tensor<T> &tensor, const std::vector<int> indices)
            : Iterator(const_cast<Tensor<T> &>(tensor), indices) {};

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

    // This is to allow const Tensor to use the non-const begin() and end() functions.
    const Iterator begin() const
    {
        return const_cast<Tensor<T> &>(*this).begin();
    }

    const Iterator end() const
    {
        return const_cast<Tensor<T> &>(*this).end();
    }

    Tensor view(const std::vector<int> &size) const;

    Tensor transpose(int dim0, int dim1) const;

    Tensor clone() const;

    bool equal(const Tensor &other) const;

    std::vector<int> size() const;

    int numel() const;

    void backward();

    void zero_grad();

    Tensor operator+(const Tensor<T> &other) const;

    Tensor operator+(T other) const;

    Tensor operator-(const Tensor<T> &other) const;

    Tensor operator-(T other) const;

    Tensor operator-() const;

    Tensor operator*(const Tensor<T> &other) const;

    Tensor operator*(T other) const;

    Tensor operator/(const Tensor<T> &other) const;

    Tensor operator/(T other) const;

    template <typename U> friend Tensor<U> operator/(U other, const Tensor<U> &t);

    Tensor &operator+=(const Tensor<T> &other);

    Tensor sum() const;

    Tensor sum(const std::vector<int> &dim, bool keep_dim) const;

    Tensor sqrt() const;

    Tensor pow(T exponent) const;

    Tensor exp() const;

    Tensor log() const;

    Tensor<int> argmax() const;

    Tensor max() const;

    Tensor max(int dim, bool keep_dim) const;

    Tensor<int> argmin() const;

    Tensor min() const;

    Tensor min(int dim, bool keep_dim) const;

    Tensor mean(const std::vector<int> &dim, bool keep_dim) const;

    Tensor mean() const;

    Tensor var(const std::vector<int> &dim, bool keep_dim) const;

    Tensor var() const;

    static Tensor relu(const Tensor &t);

    static Tensor sigmoid(const Tensor &tm);

    static Tensor softmax(const Tensor &t, int dim = 0);

    static Tensor cross_entropy(const Tensor &input, const Tensor<int> &target);

    static Tensor mm(const Tensor &t1, const Tensor &t2, Tensor *out = nullptr);

    static Tensor matmul(const Tensor &t1, const Tensor &t2);

    static Tensor stack(const std::vector<Tensor> tensors, int dim = 0);

    static Tensor unfold(const Tensor &in, int kernel_size, int padding, int stride);

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