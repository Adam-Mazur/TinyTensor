#include "../include/tensor.h"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <numeric>
#include <random>
#include <stack>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <vector>

// ========================================================
// SECTION: TensorData
// ========================================================

template <typename T> TensorData<T>::TensorData(const std::vector<T> &data) : vec(data), reference_count(1)
{
}

template <typename T> TensorData<T>::TensorData(int size, T value) : vec(size, value), reference_count(1)
{
}

template <typename T> size_t TensorData<T>::size()
{
    return vec.size();
}

template <typename T> T &TensorData<T>::operator[](size_t index)
{
    return vec[index];
}

template <typename T> const T &TensorData<T>::operator[](size_t index) const
{
    return vec[index];
}

// ========================================================
// SECTION: Tensor's Helper Methods and Constructors
// ========================================================

template <typename T> void Tensor<T>::add_reference()
{
    if (data != nullptr)
    {
        ++data->reference_count;
    }
}

template <typename T> void Tensor<T>::release()
{
    if (data != nullptr)
    {
        data->reference_count--;
        if (data->reference_count == 0)
        {
            delete data;
        }
        data = nullptr;
    }
}

// A private constructor, which is equivalent to torch.full()
template <typename T> Tensor<T>::Tensor(const std::vector<size_t> &size, T value, bool requires_grad)
{
    strides = std::vector<int>(size.size());
    shape = size;
    offset = 0;
    int acc = 1;

    for (int i = size.size() - 1; i >= 0; i--)
    {
        strides[i] = acc;
        acc *= size[i];
    }

    int num_of_elements = acc;
    data = new TensorData<T>(num_of_elements, value);

    if (requires_grad)
    {
        if (!std::is_same<T, float>::value && !std::is_same<T, double>::value)
        {
            throw std::invalid_argument("Only float or double tensors support gradients.");
        }
        grad = new Tensor<T>(size, 0.0);
    }
    else
    {
        grad = nullptr;
    }

    _backward = nullptr;
    operand1 = nullptr;
    operand2 = nullptr;
}

template <typename T>
Tensor<T>::Tensor(const std::vector<T> &data, bool requires_grad)
    : data(new TensorData<T>(data)), shape({data.size()}), offset(0), strides({1}), _backward(nullptr),
      operand1(nullptr), operand2(nullptr)
{
    if (requires_grad)
    {
        if (!std::is_same<T, float>::value && !std::is_same<T, double>::value)
        {
            throw std::invalid_argument("Only float or double tensors support gradients.");
        }
        grad = new Tensor<T>(shape, 0.0);
    }
    else
    {
        grad = nullptr;
    }
}

template <typename T>
Tensor<T>::Tensor()
    : data(nullptr), grad(nullptr), offset(0), shape({}), strides({}), operand1(nullptr), operand2(nullptr), _backward()
{
}

// ========================================================
// SECTION: Copy Operations
// ========================================================

template <typename T>
Tensor<T>::Tensor(const Tensor<T> &other)
    : data(other.data), shape(other.shape), offset(other.offset), strides(other.strides), operand1(other.operand1),
      operand2(other.operand2), _backward(other._backward)
{
    add_reference();
    if (other.grad != nullptr && !NoGradGuard::is_enabled)
    {
        grad = new Tensor<T>(*other.grad);
    }
    else
    {
        grad = nullptr;
    }
}

template <typename T> Tensor<T> &Tensor<T>::operator=(const Tensor<T> &other)
{
    if (this != &other)
    {
        release();
        data = other.data;
        shape = other.shape;
        offset = other.offset;
        strides = other.strides;
        operand1 = other.operand1;
        operand2 = other.operand2;
        _backward = other._backward;

        if (grad != nullptr)
        {
            delete grad;
        }

        if (other.grad != nullptr && !NoGradGuard::is_enabled)
        {
            grad = new Tensor<T>(*other.grad);
        }
        else
        {
            grad = nullptr;
        }

        add_reference();
    }
    return *this;
}

// ========================================================
// SECTION: Move Operations
// ========================================================

// noexcept is needed here because of some compilation issues
template <typename T>
Tensor<T>::Tensor(Tensor<T> &&other) noexcept
    : data(other.data), shape(other.shape), offset(other.offset), strides(other.strides), grad(other.grad),
      operand1(other.operand1), operand2(other.operand2), _backward(other._backward)
{
    other.data = nullptr;
    other.grad = nullptr;
}

// noexcept is needed here because of some compilation issues
template <typename T> Tensor<T> &Tensor<T>::operator=(Tensor<T> &&other) noexcept
{
    if (this != &other)
    {
        release();
    
        if (grad != nullptr)
        {
            delete grad;
        }
        
        data = other.data;
        shape = other.shape;
        offset = other.offset;
        strides = other.strides;
        grad = other.grad;
        operand1 = other.operand1;
        operand2 = other.operand2;
        _backward = other._backward;
        other.data = nullptr;
        other.grad = nullptr;
    }
    return *this;
}

// ========================================================
// SECTION: Tensor's Initialization
// ========================================================

template <typename T> Tensor<T> Tensor<T>::zeros(const std::vector<size_t> &size, bool requires_grad)
{
    return Tensor(size, static_cast<T>(0), requires_grad);
}

template <typename T> Tensor<T> Tensor<T>::ones(const std::vector<size_t> &size, bool requires_grad)
{
    return Tensor(size, static_cast<T>(1), requires_grad);
}

template <typename T> Tensor<T> Tensor<T>::randn(const std::vector<size_t> &size, bool requires_grad)
{
    std::default_random_engine generator;
    std::normal_distribution<T> distribution(static_cast<T>(0), static_cast<T>(1));
    Tensor<T> tensor = Tensor<T>(size, static_cast<T>(0), requires_grad);

    for (int i = 0; i < tensor.data->size(); i++)
    {
        (*tensor.data)[i] = distribution(generator);
    }

    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::xavier_normal(const std::vector<size_t> &size, float gain, bool requires_grad)
{
    std::default_random_engine generator;
    float std = gain * std::sqrt(2.0 / (size[0] + size[1]));
    std::normal_distribution<T> distribution(static_cast<T>(0), static_cast<T>(std));
    Tensor<T> tensor = Tensor<T>(size, static_cast<T>(0), requires_grad);

    for (int i = 0; i < tensor.data->size(); i++)
    {
        (*tensor.data)[i] = distribution(generator);
    }

    return tensor;
}

template <typename T> Tensor<T> Tensor<T>::kaiming_normal(const std::vector<size_t> &size, bool requires_grad)
{
    std::default_random_engine generator;
    float std = std::sqrt(2.0 / (size[0]));
    std::normal_distribution<T> distribution(static_cast<T>(0), static_cast<T>(std));
    Tensor<T> tensor = Tensor<T>(size, static_cast<T>(0), requires_grad);

    for (int i = 0; i < tensor.data->size(); i++)
    {
        (*tensor.data)[i] = distribution(generator);
    }

    return tensor;
}

// ========================================================
// SECTION: Indexing Operators
// ========================================================

template <typename T> T &Tensor<T>::operator[](const std::initializer_list<int> &indices)
{
    std::vector<int> vec_indices(indices.begin(), indices.end());
    return (*this)[vec_indices];
}

template <typename T> T &Tensor<T>::operator[](const std::vector<int> &indices)
{
    if (data == nullptr)
    {
        throw std::invalid_argument("The tensor data is null.");
    }
    if (indices.size() != strides.size())
    {
        throw std::invalid_argument("The number of indices doesn't match the shape of the tensor.");
    }
    if (!std::equal(indices.begin(), indices.end(), shape.begin(), [](int a, int b) { return a < b; }))
    {
        throw std::out_of_range("Some index is out of range.");
    }

    std::vector<int> new_indices(indices.begin(), indices.end());
    for (int i = 0; i < new_indices.size(); i++)
    {
        if (-shape[i] <= new_indices[i] && new_indices[i] < 0)
        {
            new_indices[i] = shape[i] + new_indices[i];
        }
        else if (new_indices[i] < 0)
        {
            throw std::out_of_range("Some index is out of range.");
        }
    }
    int index = std::inner_product(strides.begin(), strides.end(), new_indices.begin(), offset);
    return (*data)[index];
}

template <typename T> Tensor<T> Tensor<T>::operator[](const std::initializer_list<std::pair<int, int>> &indices)
{
    std::vector<std::pair<int, int>> vec_indices(indices.begin(), indices.end());
    return (*this)[vec_indices];
}

template <typename T> Tensor<T> Tensor<T>::operator[](const std::vector<std::pair<int, int>> &indices)
{
    if (data == nullptr)
    {
        throw std::invalid_argument("The tensor data is null.");
    }
    if (indices.size() > strides.size())
    {
        throw std::invalid_argument("The number of indices doesn't match the shape of the tensor.");
    }

    std::vector<std::pair<int, int>> new_indices(indices.begin(), indices.end());

    if (indices.size() < strides.size())
    {
        for (int i = indices.size(); i < strides.size(); i++)
        {
            new_indices.push_back({0, shape[i]});
        }
        return (*this)[new_indices];
    }

    Tensor<T> new_tensor = Tensor<T>(*this);
    new_tensor.offset = offset;
    new_tensor.shape = std::vector<size_t>();
    new_tensor.strides = std::vector<int>();
    for (int i = 0; i < new_indices.size(); i++)
    {
        if (new_indices[i].first < 0)
        {
            new_indices[i].first = shape[i] + new_indices[i].first;
        }
        if (new_indices[i].second < 0)
        {
            new_indices[i].second = shape[i] + new_indices[i].second;
        }
        new_indices[i].first = std::max(new_indices[i].first, 0);
        new_indices[i].second = std::min(new_indices[i].second, static_cast<int>(shape[i]));
        new_tensor.offset += strides[i] * new_indices[i].first;
        if (new_indices[i].second - new_indices[i].first == 0)
            continue;
        new_tensor.shape.push_back(new_indices[i].second - new_indices[i].first);
        new_tensor.strides.push_back(strides[i]);
    }

    if (new_tensor.shape.size() == 0)
    {
        throw std::invalid_argument("The resulting vector can't have 0 dimensions.");
    }

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor.grad->shape = new_tensor.shape;
        new_tensor.grad->strides = new_tensor.strides;
        new_tensor.grad->offset = new_tensor.offset;
        new_tensor.operand1 = new Tensor<T>(*this);
        new_tensor.operand2 = nullptr;
        new_tensor._backward = [](Tensor<T> *) {};
    }

    return new_tensor;
};

// ========================================================
// SECTION: Tensor Misc Methods
// ========================================================

template <typename T> size_t Tensor<T>::get_hash()
{
    std::hash<size_t> hash_fn;
    size_t seed = reinterpret_cast<size_t>(data);

    for (size_t i : shape)
    {
        seed ^= hash_fn(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    for (int i : strides)
    {
        seed ^= hash_fn(static_cast<size_t>(i)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    seed ^= hash_fn(static_cast<size_t>(offset)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

template <typename T> Tensor<T> Tensor<T>::view(const std::vector<int> &size)
{
    // The condition for creating a view of the tensor.
    for (int i = 0; i < shape.size() - 1; i++)
    {
        if (strides[i] != strides[i + 1] * shape[i + 1])
        {
            throw std::invalid_argument("Can't call view on this tensor.");
        }
    }

    int num_of_elements = 1;
    int data_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    bool full = true;

    for (int i = 0; i < size.size(); i++)
    {
        if (size[i] == -1 && full)
        {
            full = false;
            continue;
        }
        else if (size[i] == -1)
        {
            throw std::invalid_argument("Can't have two minus ones.");
        }
        num_of_elements *= size[i];
    }

    if (full && (num_of_elements != data_size))
    {
        throw std::invalid_argument("The number of elements doesn't match.");
    }
    if (!full && (num_of_elements > data_size || data_size % num_of_elements != 0))
    {
        throw std::invalid_argument("Can't create a tensor of this shape.");
    }

    std::vector<size_t> new_size(size.begin(), size.end());

    for (int i = 0; i < size.size(); i++)
    {
        if (size[i] == -1)
        {
            new_size[i] = data_size / num_of_elements;
        }
    }

    Tensor<T> new_tensor = Tensor<T>(*this);
    new_tensor.strides = std::vector<int>(new_size.size());
    new_tensor.shape = new_size;
    new_tensor.offset = offset;
    int acc = 1;

    for (int i = new_size.size() - 1; i >= 0; i--)
    {
        new_tensor.strides[i] = acc;
        acc *= new_size[i];
    }

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor.grad->shape = new_tensor.shape;
        new_tensor.grad->strides = new_tensor.strides;
        new_tensor.grad->offset = new_tensor.offset;
        new_tensor.operand1 = new Tensor<T>(*this);
        new_tensor.operand2 = nullptr;
        new_tensor._backward = [](Tensor<T> *) {};
    }

    return new_tensor;
}

template <typename T> Tensor<T> Tensor<T>::transpose(int dim0, int dim1)
{
    if (dim0 < 0 || dim0 >= shape.size() || dim1 < 0 || dim1 >= shape.size())
    {
        throw std::invalid_argument("Dimensions out of range.");
    }
    if (dim0 == dim1)
    {
        throw std::invalid_argument("The dimensions can't be equal.");
    }

    Tensor<T> new_tensor = Tensor<T>(*this);
    new_tensor.strides[dim0] = strides[dim1];
    new_tensor.strides[dim1] = strides[dim0];
    new_tensor.shape[dim0] = shape[dim1];
    new_tensor.shape[dim1] = shape[dim0];

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor.grad->shape = new_tensor.shape;
        new_tensor.grad->strides = new_tensor.strides;
        new_tensor.operand1 = new Tensor<T>(*this);
        new_tensor.operand2 = nullptr;
        new_tensor._backward = [](Tensor<T> *) {};
    }

    return new_tensor;
}

template <typename T> Tensor<T> Tensor<T>::clone()
{
    Tensor<T> new_tensor = Tensor<T>(data->vec);
    new_tensor.shape = shape;
    new_tensor.offset = offset;
    new_tensor.strides = strides;
    new_tensor.operand1 = operand1;
    new_tensor.operand2 = operand2;
    new_tensor._backward = _backward;

    if (grad != nullptr)
    {
        new_tensor.grad = new Tensor<T>(grad->clone());
    }
    else
    {
        new_tensor.grad = nullptr;
    }

    return new_tensor;
}

template <typename T> bool Tensor<T>::equal(Tensor<T> &other)
{
    if (shape != other.shape)
    {
        return false;
    }

    std::vector<int> indices(shape.size(), 0);
    int num_of_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    for (int i = 0; i < num_of_elements; i++)
    {
        if ((*this)[indices] != other[indices])
        {
            return false;
        }
        for (int j = indices.size() - 1; j >= 0; j--)
        {
            indices[j]++;
            if (indices[j] == shape[j])
            {
                indices[j] = 0;
            }
            else
            {
                break;
            }
        }
    }

    return true;
}

template <typename T> std::vector<size_t> Tensor<T>::size()
{
    return shape;
}

// ========================================================
// SECTION: Autograd Methods
// ========================================================

template <typename T> void Tensor<T>::backward()
{
    if (grad == nullptr)
    {
        throw std::invalid_argument("Can't call backward on a tensor without a gradient.");
    }
    if (_backward == nullptr)
    {
        throw std::invalid_argument("Can't call backward on a tensor that is not a result of a computation.");
    }
    if (shape != std::vector<size_t>({1}))
    {
        throw std::invalid_argument("Can't call backward on a tensor with more than one element.");
    }

    std::vector<Tensor<T> *> topo;
    std::unordered_set<size_t> visited;
    std::stack<Tensor<T> *> stack({this});
    std::unordered_set<Tensor<T> *> to_remove;

    while (!stack.empty())
    {
        Tensor<T> *node = stack.top();

        if (visited.count(node->get_hash()) == 0)
        {
            visited.insert(node->get_hash());

            if (node->operand1 != nullptr)
            {
                to_remove.insert(node->operand1);
                if (visited.count(node->operand1->get_hash()) == 0)
                {
                    stack.push(node->operand1);
                }
            }

            if (node->operand2 != nullptr)
            {
                to_remove.insert(node->operand2);
                if (visited.count(node->operand2->get_hash()) == 0)
                {
                    stack.push(node->operand2);
                }
            }
        }
        else
        {
            stack.pop();

            if (std::find_if(topo.begin(), topo.end(),
                             [node](Tensor<T> *x) { return x->get_hash() == node->get_hash(); }) == topo.end())
            {
                topo.push_back(node);
            }
        }
    }

    std::reverse(topo.begin(), topo.end());
    (*this->grad)[{0}] = 1;

    for (Tensor<T> *node : topo)
    {
        // Call _backward (grad_fn)
        if (node->_backward != nullptr)
        {
            (node->_backward)(node);
        }
    }

    for (Tensor<T> *node : to_remove)
    {
        delete node;
    }
}

template <typename T> void Tensor<T>::zero_grad()
{
    if (grad != nullptr)
    {
        grad->data->vec.assign(grad->data->size(), static_cast<T>(0));
    }
}

// ========================================================
// SECTION: Arithmetic Operators
// ========================================================

template <typename T>
template <typename Op>
Tensor<T> Tensor<T>::broadcast(const Tensor<T> &t1, const Tensor<T> &t2, Op op)
{
    // Padding strides and shapes with ones, so that the number of dimensions
    // of both operands matches.
    std::vector<size_t> t1_shape(t1.shape.begin(), t1.shape.end());
    std::vector<size_t> t2_shape(t2.shape.begin(), t2.shape.end());
    std::vector<int> t1_strides(t1.strides.begin(), t1.strides.end());
    std::vector<int> t2_strides(t2.strides.begin(), t2.strides.end());

    if (t1_shape.size() < t2_shape.size())
    {
        std::vector<size_t> padding(t2_shape.size() - t1_shape.size(), 1);
        t1_shape.insert(t1_shape.begin(), padding.begin(), padding.end());
        t1_strides.insert(t1_strides.begin(), padding.begin(), padding.end());
    }
    else if (t2_shape.size() < t1_shape.size())
    {
        std::vector<size_t> padding(t1_shape.size() - t2_shape.size(), 1);
        t2_shape.insert(t2_shape.begin(), padding.begin(), padding.end());
        t2_strides.insert(t2_strides.begin(), padding.begin(), padding.end());
    }

    std::vector<size_t> new_shape(t1_shape.size());

    for (int i = t1_shape.size() - 1; i >= 0; i--)
    {
        if (t1_shape[i] == 1 || t2_shape[i] == 1)
        {
            new_shape[i] = t1_shape[i] * t2_shape[i];
            continue;
        }
        if (t1_shape[i] != t2_shape[i])
        {
            throw std::invalid_argument("Those shapes are not broadcastable.");
        }
        new_shape[i] = t1_shape[i];
    }

    bool new_requires_grad;

    if (t1.grad != nullptr || t2.grad != nullptr)
    {
        new_requires_grad = true;
    }
    else
    {
        new_requires_grad = false;
    }

    Tensor<T> new_tensor = Tensor<T>::zeros(new_shape, new_requires_grad);

    // For all dimensions equal to one, the corresponding stride
    // must be equal to 0, since we want to broadcast them.
    for (int i = 0; i < new_shape.size(); i++)
    {
        if (t1_shape[i] == 1)
        {
            t1_strides[i] = 0;
        }
        if (t2_shape[i] == 1)
        {
            t2_strides[i] = 0;
        }
    }

    int t1_index = t1.offset;
    int t2_index = t2.offset;
    std::vector<int> indices(new_shape.size(), 0);

    for (int i = 0; i < new_tensor.data->size(); i++)
    {
        (*new_tensor.data)[i] = op((*t1.data)[t1_index], (*t2.data)[t2_index]);

        for (int j = indices.size() - 1; j >= 0; j--)
        {
            indices[j]++;
            t1_index += t1_strides[j];
            t2_index += t2_strides[j];
            if (indices[j] == new_tensor.shape[j])
            {
                indices[j] = 0;
                t1_index -= t1_shape[j] * t1_strides[j];
                t2_index -= t2_shape[j] * t2_strides[j];
            }
            else
            {
                break;
            }
        }
    }

    return new_tensor;
}

template <typename T> Tensor<T> Tensor<T>::operator+(Tensor<T> &other)
{
    Tensor<T> new_tensor = Tensor<T>::broadcast(*this, other, std::plus<>());

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::add_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
        new_tensor.operand2 = new Tensor<T>(other);
    }

    return new_tensor;
}

template <typename T> Tensor<T> Tensor<T>::operator+(T other)
{
    Tensor<T> other_tensor = Tensor<T>({other});
    Tensor<T> new_tensor = Tensor<T>::broadcast(*this, other_tensor, std::plus<>());

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::add_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
        new_tensor.operand2 = new Tensor<T>(other_tensor);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::add_backward()
{
    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        std::vector<int> dim_to_reduce;
        int dim_r = this->shape.size() - 1;
        int dim_op = operand1->shape.size() - 1;

        while (dim_r >= 0)
        {
            if (dim_op < 0 || this->shape[dim_r] != operand1->shape[dim_op])
            {
                dim_to_reduce.push_back(dim_r);
            }
            dim_r--;
            dim_op--;
        }

        std::vector<int> new_shape(operand1->shape.begin(), operand1->shape.end());
        Tensor<T> reduced_grad = this->grad->sum(dim_to_reduce, true).view(new_shape);
        (*operand1->grad) += reduced_grad;
    }

    if (operand2->grad != nullptr)
    {
        std::vector<int> dim_to_reduce;
        int dim_r = this->shape.size() - 1;
        int dim_op = operand2->shape.size() - 1;

        while (dim_r >= 0)
        {
            if (dim_op < 0 || this->shape[dim_r] != operand2->shape[dim_op])
            {
                dim_to_reduce.push_back(dim_r);
            }
            dim_r--;
            dim_op--;
        }

        std::vector<int> new_shape(operand2->shape.begin(), operand2->shape.end());
        Tensor<T> reduced_grad = this->grad->sum(dim_to_reduce, true).view(new_shape);
        (*operand2->grad) += reduced_grad;
    }
}

template <typename T> Tensor<T> Tensor<T>::operator-(Tensor<T> &other)
{
    Tensor<T> new_tensor = Tensor<T>::broadcast(*this, other, std::minus<>());

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::sub_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
        new_tensor.operand2 = new Tensor<T>(other);
    }

    return new_tensor;
}

template <typename T> Tensor<T> Tensor<T>::operator-(T other)
{
    Tensor<T> other_tensor = Tensor<T>({other});
    Tensor<T> new_tensor = Tensor<T>::broadcast(*this, other_tensor, std::minus<>());

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::sub_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
        new_tensor.operand2 = new Tensor<T>(other_tensor);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::sub_backward()
{
    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        std::vector<int> dim_to_reduce;
        int dim_r = this->shape.size() - 1;
        int dim_op = operand1->shape.size() - 1;

        while (dim_r >= 0)
        {
            if (dim_op < 0 || this->shape[dim_r] != operand1->shape[dim_op])
            {
                dim_to_reduce.push_back(dim_r);
            }
            dim_r--;
            dim_op--;
        }

        std::vector<int> new_shape(operand1->shape.begin(), operand1->shape.end());
        Tensor<T> reduced_grad = this->grad->sum(dim_to_reduce, true).view(new_shape);
        (*operand1->grad) += reduced_grad;
    }

    if (operand2->grad != nullptr)
    {
        std::vector<int> dim_to_reduce;
        int dim_r = this->shape.size() - 1;
        int dim_op = operand2->shape.size() - 1;

        while (dim_r >= 0)
        {
            if (dim_op < 0 || this->shape[dim_r] != operand2->shape[dim_op])
            {
                dim_to_reduce.push_back(dim_r);
            }
            dim_r--;
            dim_op--;
        }

        std::vector<int> new_shape(operand2->shape.begin(), operand2->shape.end());
        Tensor<T> reduced_grad = this->grad->sum(dim_to_reduce, true).view(new_shape);
        Tensor<T> temp = -reduced_grad;
        (*operand2->grad) += temp;
    }
}

template <typename T> Tensor<T> Tensor<T>::operator-()
{
    Tensor<T> new_tensor = this->clone();

    for (T &value : new_tensor.data->vec)
    {
        value = -value;
    }

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::minus_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::minus_backward()
{
    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        Tensor<T> new_tensor_grad = this->grad->clone();
        Tensor<T> temp = -new_tensor_grad;
        (*operand1->grad) += temp;
    }
}

template <typename T> Tensor<T> Tensor<T>::operator*(Tensor<T> &other)
{
    Tensor<T> new_tensor = Tensor<T>::broadcast(*this, other, std::multiplies<>());

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::mul_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
        new_tensor.operand2 = new Tensor<T>(other);
    }

    return new_tensor;
}

template <typename T> Tensor<T> Tensor<T>::operator*(T other)
{
    Tensor<T> other_tensor = Tensor<T>({other});
    Tensor<T> new_tensor = Tensor<T>::broadcast(*this, other_tensor, std::multiplies<>());

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::mul_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
        new_tensor.operand2 = new Tensor<T>(other_tensor);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::mul_backward()
{
    NoGradGuard no_grad;
    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        std::vector<int> dim_to_reduce;
        int dim_r = this->shape.size() - 1;
        int dim_op = operand1->shape.size() - 1;

        while (dim_r >= 0)
        {
            if (dim_op < 0 || this->shape[dim_r] != operand1->shape[dim_op])
            {
                dim_to_reduce.push_back(dim_r);
            }
            dim_r--;
            dim_op--;
        }

        std::vector<int> new_shape(operand1->shape.begin(), operand1->shape.end());
        Tensor<T> reduced_grad = ((*this->grad) * (*operand2)).sum(dim_to_reduce, true).view(new_shape);
        (*operand1->grad) += reduced_grad;
    }

    if (operand2->grad != nullptr)
    {
        std::vector<int> dim_to_reduce;
        int dim_r = this->shape.size() - 1;
        int dim_op = operand2->shape.size() - 1;

        while (dim_r >= 0)
        {
            if (dim_op < 0 || this->shape[dim_r] != operand2->shape[dim_op])
            {
                dim_to_reduce.push_back(dim_r);
            }
            dim_r--;
            dim_op--;
        }

        std::vector<int> new_shape(operand2->shape.begin(), operand2->shape.end());
        Tensor<T> reduced_grad = ((*this->grad) * (*operand1)).sum(dim_to_reduce, true).view(new_shape);
        (*operand2->grad) += reduced_grad;
    }
}

template <typename T> Tensor<T> Tensor<T>::operator/(Tensor<T> &other)
{
    Tensor<T> new_tensor = Tensor<T>::broadcast(*this, other, std::divides<>());

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::div_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
        new_tensor.operand2 = new Tensor<T>(other);
    }

    return new_tensor;
}

template <typename T> Tensor<T> Tensor<T>::operator/(T other)
{
    Tensor<T> other_tensor = Tensor<T>({other});
    Tensor<T> new_tensor = Tensor<T>::broadcast(*this, other_tensor, std::divides<>());

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::div_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
        new_tensor.operand2 = new Tensor<T>(other_tensor);
    }

    return new_tensor;
}

template <typename U> Tensor<U> operator/(U other, Tensor<U> &t)
{
    Tensor<U> other_tensor = Tensor<U>({other});
    Tensor<U> new_tensor = Tensor<U>::broadcast(other_tensor, t, std::divides<>());

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<U>::div_backward;
        new_tensor.operand1 = new Tensor<U>(other_tensor);
        new_tensor.operand2 = new Tensor<U>(t);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::div_backward()
{
    NoGradGuard no_grad;
    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        std::vector<int> dim_to_reduce;
        int dim_r = this->shape.size() - 1;
        int dim_op = operand1->shape.size() - 1;

        while (dim_r >= 0)
        {
            if (dim_op < 0 || this->shape[dim_r] != operand1->shape[dim_op])
            {
                dim_to_reduce.push_back(dim_r);
            }
            dim_r--;
            dim_op--;
        }

        std::vector<int> new_shape(operand1->shape.begin(), operand1->shape.end());
        Tensor<T> temp1 = static_cast<T>(1) / (*operand2);
        Tensor<T> reduced_grad = ((*this->grad) * temp1).sum(dim_to_reduce, true).view(new_shape);
        (*operand1->grad) += reduced_grad;
    }

    if (operand2->grad != nullptr)
    {
        std::vector<int> dim_to_reduce;
        int dim_r = this->shape.size() - 1;
        int dim_op = operand2->shape.size() - 1;

        while (dim_r >= 0)
        {
            if (dim_op < 0 || this->shape[dim_r] != operand2->shape[dim_op])
            {
                dim_to_reduce.push_back(dim_r);
            }
            dim_r--;
            dim_op--;
        }

        std::vector<int> new_shape(operand2->shape.begin(), operand2->shape.end());
        Tensor<T> temp1 = (*operand2) * (*operand2);
        Tensor<T> temp2 = -(*operand1) / temp1;
        Tensor<T> reduced_grad = ((*this->grad) * temp2).sum(dim_to_reduce, true).view(new_shape);
        (*operand2->grad) += reduced_grad;
    }
}

template <typename T> Tensor<T> &Tensor<T>::operator+=(Tensor<T> &other)
{
    std::vector<size_t> other_shape(other.shape.begin(), other.shape.end());
    std::vector<int> other_strides(other.strides.begin(), other.strides.end());

    if (other_shape.size() < this->shape.size())
    {
        std::vector<size_t> padding(this->shape.size() - other_shape.size(), 1);
        other_shape.insert(other_shape.begin(), padding.begin(), padding.end());
        other_strides.insert(other_strides.begin(), padding.begin(), padding.end());
    }
    else if (other_shape.size() > this->shape.size())
    {
        throw std::invalid_argument("In-place operations are not supported for tensors with those shapes.");
    }

    for (int i = this->shape.size() - 1; i >= 0; i--)
    {
        if (other_shape[i] == 1)
        {
            other_strides[i] = 0;
        }
        if (other_shape[i] != 1 && this->shape[i] != other_shape[i])
        {
            throw std::invalid_argument("Those shapes are not broadcastable.");
        }
    }

    int this_index = this->offset;
    int other_index = other.offset;
    std::vector<int> indices(this->shape.size(), 0);
    bool run = true;

    while (run)
    {
        (*this->data)[this_index] += (*other.data)[other_index];

        for (int j = indices.size() - 1; j >= 0; j--)
        {
            indices[j]++;
            this_index += this->strides[j];
            other_index += other_strides[j];

            if (indices[j] == this->shape[j] && j == 0)
            {
                run = false;
                break;
            }
            else if (indices[j] == this->shape[j])
            {
                indices[j] = 0;
                this_index -= this->shape[j] * this->strides[j];
                other_index -= other_shape[j] * other_strides[j];
            }
            else
            {
                break;
            }
        }
    }

    return *this;
}

// ========================================================
// SECTION: Math Methods
// ========================================================

template <typename T> Tensor<T> Tensor<T>::sum()
{
    Tensor<T> new_tensor = Tensor<T>::zeros({1}, this->grad != nullptr);
    std::vector<int> indices(shape.size(), 0);
    int num_of_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    for (int i = 0; i < num_of_elements; i++)
    {
        new_tensor[{0}] += (*this)[indices];

        for (int j = indices.size() - 1; j >= 0; j--)
        {
            indices[j]++;
            if (indices[j] == shape[j])
            {
                indices[j] = 0;
            }
            else
            {
                break;
            }
        }
    }

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::sum_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::sum_backward()
{
    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        (*operand1->grad) += (*this->grad);
    }
}

template <typename T> Tensor<T> Tensor<T>::sum(const std::vector<int> &dim, bool keep_dim)
{
    for (int i : dim)
    {
        if (i < 0 || i >= shape.size())
        {
            throw std::invalid_argument("Dimension out of range.");
        }
    }

    std::vector<size_t> new_shape(shape.begin(), shape.end());

    for (int i : dim)
    {
        new_shape[i] = 1;
    }

    Tensor<T> new_tensor = Tensor<T>::zeros(new_shape, this->grad != nullptr);
    std::vector<int> strides_new = new_tensor.strides;

    for (int i : dim)
    {
        strides_new[i] = 0;
    }

    int index_old = offset;
    int index_new = 0;
    std::vector<int> indices(shape.size(), 0);
    int num_of_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    for (int i = 0; i < num_of_elements; i++)
    {
        (*new_tensor.data)[index_new] += (*this->data)[index_old];

        for (int j = indices.size() - 1; j >= 0; j--)
        {
            indices[j]++;
            index_old += strides[j];
            index_new += strides_new[j];
            if (indices[j] == shape[j])
            {
                indices[j] = 0;
                index_old -= shape[j] * strides[j];
                index_new -= new_shape[j] * strides_new[j];
            }
            else
            {
                break;
            }
        }
    }

    if (!keep_dim)
    {
        std::vector<int> final_shape;

        for (int i = 0; i < new_shape.size(); i++)
        {
            if (std::find(dim.begin(), dim.end(), i) != dim.end())
                continue;
            final_shape.push_back(new_shape[i]);
        }

        new_tensor = new_tensor.view(final_shape);
    }

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = std::bind(&Tensor<T>::sum2_backward, std::placeholders::_1, dim, keep_dim);
        new_tensor.operand1 = new Tensor<T>(*this);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::sum2_backward(const std::vector<int> &dim, bool keep_dim)
{
    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        Tensor<T> temp = this->grad->clone();

        if (!keep_dim)
        {
            std::vector<int> new_shape(operand1->shape.begin(), operand1->shape.end());
            for (int i : dim)
            {
                new_shape[i] = 1;
            }
            Tensor<T> temp2 = temp.view(new_shape);
            (*operand1->grad) += temp2;
        }
        else
        {
            (*operand1->grad) += temp;
        }
    }
}

template <typename T> Tensor<T> Tensor<T>::sqrt()
{
    Tensor<T> new_tensor = this->clone();

    for (T &value : new_tensor.data->vec)
    {
        value = std::sqrt(value);
    }

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::sqrt_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::sqrt_backward()
{
    NoGradGuard no_grad;

    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        Tensor<T> temp = static_cast<T>(0.5) / (*this);
        Tensor<T> temp2 = (*this->grad) * temp;
        (*operand1->grad) += temp2;
    }
}

template <typename T> Tensor<T> Tensor<T>::pow(T exponent)
{
    Tensor<T> new_tensor = this->clone();

    for (T &value : new_tensor.data->vec)
    {
        value = std::pow(value, exponent);
    }

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = std::bind(&Tensor<T>::pow_backward, std::placeholders::_1, exponent);
        new_tensor.operand1 = new Tensor<T>(*this);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::pow_backward(T exponent)
{
    NoGradGuard no_grad;

    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        Tensor<T> temp = (*this) / (*operand1);
        Tensor<T> temp2 = temp * exponent;
        Tensor<T> temp3 = (*this->grad) * temp2;
        (*operand1->grad) += temp3;
    }
}

template <typename T> Tensor<T> Tensor<T>::exp()
{
    Tensor<T> new_tensor = this->clone();

    for (T &value : new_tensor.data->vec)
    {
        value = std::exp(value);
    }

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::exp_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::exp_backward()
{
    NoGradGuard no_grad;

    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        Tensor<T> temp = (*this->grad) * (*this);
        (*operand1->grad) += temp;
    }
}

template <typename T> Tensor<T> Tensor<T>::log()
{
    Tensor<T> new_tensor = this->clone();

    for (T &value : new_tensor.data->vec)
    {
        value = std::log(value);
    }

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::log_backward;
        new_tensor.operand1 = new Tensor<T>(*this);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::log_backward()
{
    NoGradGuard no_grad;

    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        Tensor<T> temp = static_cast<T>(1) / (*operand1);
        Tensor<T> temp2 = (*this->grad) * temp;
        (*operand1->grad) += temp2;
    }
}

template <typename T> Tensor<int> Tensor<T>::argmax()
{
    int argmax = -1;
    T max_value;
    std::vector<int> indices(shape.size(), 0);
    int num_of_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    for (int i = 0; i < num_of_elements; i++)
    {
        T value = (*this)[indices];

        if (argmax == -1 || value > max_value)
        {
            max_value = value;
            argmax = 0;
            for (int j = 0; j < indices.size(); j++)
            {
                argmax += indices[j] * strides[j];
            }
        }

        for (int j = indices.size() - 1; j >= 0; j--)
        {
            indices[j]++;
            if (indices[j] == shape[j])
            {
                indices[j] = 0;
            }
            else
            {
                break;
            }
        }
    }

    Tensor<int> new_tensor = Tensor<int>({argmax});
    return new_tensor;
}

template <typename T> Tensor<T> Tensor<T>::max()
{
    Tensor<int> argmax = this->argmax();
    Tensor<T> new_tensor = Tensor<T>(*this);
    new_tensor.offset = this->offset + argmax[{0}];
    new_tensor.shape = std::vector<size_t>({1});
    new_tensor.strides = std::vector<int>({0});

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor.grad->shape = new_tensor.shape;
        new_tensor.grad->strides = new_tensor.strides;
        new_tensor.grad->offset = new_tensor.offset;
        new_tensor.operand1 = new Tensor<T>(*this);
        new_tensor.operand2 = nullptr;
        new_tensor._backward = [](Tensor<T> *) {};
    }

    return new_tensor;
}

template <typename T> Tensor<T> Tensor<T>::mean(const std::vector<int> &dim, bool keep_dim)
{
    Tensor<T> temp = this->sum(dim, keep_dim);
    float num_of_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    float num_of_elements_new = std::accumulate(temp.shape.begin(), temp.shape.end(), 1, std::multiplies<size_t>());
    return temp * static_cast<T>(num_of_elements_new / num_of_elements);
}

template <typename T> Tensor<T> Tensor<T>::mean()
{
    Tensor<T> temp = this->sum();
    int num_of_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    return temp / static_cast<T>(num_of_elements);
}

template <typename T> Tensor<T> Tensor<T>::var(const std::vector<int> &dim, bool keep_dim)
{
    Tensor<T> mean = this->mean(dim, true);
    Tensor<T> diff = *this - mean;
    Tensor<T> squared_diff = diff * diff;
    float num_of_elements = 1;

    for (int i : dim)
    {
        num_of_elements *= shape[i];
    }

    Tensor<T> sum = squared_diff.sum(dim, keep_dim);
    return sum / static_cast<T>(num_of_elements - 1);
}

template <typename T> Tensor<T> Tensor<T>::var()
{
    Tensor<T> mean = this->mean();
    Tensor<T> diff = *this - mean;
    Tensor<T> squared_diff = diff * diff;
    float num_of_elements = std::accumulate(this->shape.begin(), this->shape.end(), 1, std::multiplies<size_t>());
    Tensor<T> sum = squared_diff.sum();
    return sum / static_cast<T>(num_of_elements - 1);
}

// ========================================================
// SECTION: Activation Functions
// ========================================================

template <typename T> Tensor<T> Tensor<T>::relu(Tensor<T> &t)
{
    Tensor<T> new_tensor = t.clone();

    for (T &value : new_tensor.data->vec)
    {
        value = std::max(value, static_cast<T>(0));
    }

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::relu_backward;
        new_tensor.operand1 = new Tensor<T>(t);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::relu_backward()
{
    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        Tensor<T> temp = operand1->clone();
        delete temp.grad;
        temp.grad = nullptr;

        for (int i = 0; i < operand1->data->size(); i++)
        {
            if ((*operand1->data)[i] > 0)
            {
                (*temp.data)[i] = 1;
            }
            else
            {
                (*temp.data)[i] = 0;
            }
        }

        Tensor<T> temp2 = (*this->grad) * temp;
        (*operand1->grad) += temp2;
    }
}

template <typename T> Tensor<T> Tensor<T>::sigmoid(Tensor<T> &t)
{
    Tensor<T> temp1 = -t;
    Tensor<T> temp2 = temp1.exp();
    Tensor<T> temp3 = temp2 + static_cast<T>(1);
    return static_cast<T>(1) / temp3;
}

template <typename T> Tensor<T> Tensor<T>::softmax(Tensor<T> &t, int dim)
{
    Tensor<T> temp1 = t.max();
    Tensor<T> temp2 = t - temp1;
    Tensor<T> temp3 = temp2.exp();
    Tensor<T> temp4 = temp3.sum({dim}, true);
    return temp3 / temp4;
}

// ========================================================
// SECTION: Loss Functions
// ========================================================

template <typename T> Tensor<T> Tensor<T>::cross_entropy(Tensor<T> &input, Tensor<int> &target)
{
    Tensor<T> softmax_output = Tensor<T>::softmax(input, 1);
    Tensor<T> log_softmax_output = softmax_output.log();
    Tensor<T> log_probs = Tensor<T>::zeros({input.shape[0]});

    for (int i = 0; i < input.shape[0]; i++)
    {
        log_probs[{i}] = -log_softmax_output[{i, target[{i}]}];
    }

    Tensor<T> output = log_probs.mean();

    if (input.grad != nullptr && !NoGradGuard::is_enabled)
    {
        output.grad = new Tensor<T>({0});
        int n = input.shape[0];
        std::vector<int> target_int(n);
        for (int i = 0; i < n; i++)
        {
            target_int[i] = target[{i}];
        }
        output._backward = std::bind(&Tensor<T>::cross_entropy_backward, std::placeholders::_1, n, target_int);
        output.operand1 = new Tensor<T>(log_softmax_output);
    }

    return output;
}

template <typename T> void Tensor<T>::cross_entropy_backward(int n, std::vector<int> &target)
{
    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    for (int i = 0; i < n; i++)
    {
        Tensor<T> temp = (*this->grad) * static_cast<T>(-1.0 / n);
        (*operand1->grad)[{i, target[i]}] += temp[{0}];
    }
}

// ========================================================
// SECTION: Matrix Multiplication
// ========================================================

template <typename T> Tensor<T> Tensor<T>::mm(Tensor &t1, Tensor &t2, Tensor *out)
{
    if (t1.shape.size() != 2 || t2.shape.size() != 2)
    {
        throw std::invalid_argument("Both tensors must be 2-dimensional.");
    }
    if (t1.shape[1] != t2.shape[0])
    {
        throw std::invalid_argument(
            "The number of columns of the first tensor must be equal to the number of rows of the second tensor.");
    }
    
    Tensor<T> new_tensor;
    
    if (out == nullptr)
    {
        std::vector<size_t> new_shape = {t1.shape[0], t2.shape[1]};
        new_tensor =
            Tensor<T>::zeros(new_shape, (t1.grad != nullptr || t2.grad != nullptr) && !NoGradGuard::is_enabled);
    }
    else
    {
        new_tensor = *out;
    }
    
    for (int i = 0; i < t1.shape[0]; i++)
    {
        for (int j = 0; j < t2.shape[1]; j++)
        {
            for (int k = 0; k < t1.shape[1]; k++)
            {
                int index_new = new_tensor.offset + i * new_tensor.strides[0] + j * new_tensor.strides[1];
                int index1 = t1.offset + i * t1.strides[0] + k * t1.strides[1];
                int index2 = t2.offset + k * t2.strides[0] + j * t2.strides[1];
                (*new_tensor.data)[index_new] += (*t1.data)[index1] * (*t2.data)[index2];
            }
        }
    }
    
    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward = &Tensor<T>::mm_backward;
        new_tensor.operand1 = new Tensor<T>(t1);
        new_tensor.operand2 = new Tensor<T>(t2);
    }
    
    return new_tensor;
}

template <typename T> void Tensor<T>::mm_backward()
{
    NoGradGuard no_grad;
    
    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }
    
    if (operand1->grad != nullptr)
    {
        Tensor<T> temp = operand2->transpose(0, 1);
        Tensor<T> new_tensor_grad = Tensor<T>::mm(*this->grad, temp);
        (*operand1->grad) += new_tensor_grad;
    }
    
    if (operand2->grad != nullptr)
    {
        Tensor<T> temp = operand1->transpose(0, 1);
        Tensor<T> new_tensor_grad = Tensor<T>::mm(temp, *this->grad);
        (*operand2->grad) += new_tensor_grad;
    }
}

template <typename T> Tensor<T> Tensor<T>::matmul(Tensor<T> &t1, Tensor<T> &t2)
{
    NoGradGuard no_grad;

    std::vector<int> t1_shape(t1.shape.begin(), t1.shape.end());
    std::vector<int> t2_shape(t2.shape.begin(), t2.shape.end());
    bool append_one = false;
    bool prepend_one = false;

    if (t1_shape.size() == 1)
    {
        prepend_one = true;
        t1_shape.insert(t1_shape.begin(), 1);
    }
    if (t2_shape.size() == 1)
    {
        append_one = true;
        t2_shape.push_back(1);
    }

    if (t1_shape.size() < t2_shape.size())
    {
        std::vector<size_t> padding(t2_shape.size() - t1_shape.size(), 1);
        t1_shape.insert(t1_shape.begin(), padding.begin(), padding.end());
    }
    else if (t2_shape.size() < t1_shape.size())
    {
        std::vector<size_t> padding(t1_shape.size() - t2_shape.size(), 1);
        t2_shape.insert(t2_shape.begin(), padding.begin(), padding.end());
    }

    std::vector<size_t> new_shape(t1_shape.size());

    for (int i = 0; i < t1_shape.size() - 2; i++)
    {
        if (t1_shape[i] == 1 || t2_shape[i] == 1)
        {
            new_shape[i] = t1_shape[i] * t2_shape[i];
            continue;
        }

        if (t1_shape[i] != t2_shape[i])
        {
            throw std::invalid_argument("Those shapes are not broadcastable.");
        }
        new_shape[i] = t1_shape[i];
    }

    new_shape[new_shape.size() - 2] = t1_shape[t1_shape.size() - 2];
    new_shape[new_shape.size() - 1] = t2_shape[t2_shape.size() - 1];

    Tensor<T> new_tensor = Tensor<T>::zeros(new_shape);

    Tensor<T> op1 = t1;
    Tensor<T> op2 = t2;

    if (t1_shape.size() != t1.shape.size())
    {
        op1 = t1.view(t1_shape);
    }
    if (t2_shape.size() != t2.shape.size())
    {
        op2 = t2.view(t2_shape);
    }

    if (new_shape.size() == 2)
    {
        new_tensor = Tensor<T>::mm(op1, op2);
    }
    else
    {
        int n_dim = t1_shape.size();
        int n = t1_shape[n_dim - 2];
        int m = t1_shape[n_dim - 1];
        int p = t2_shape[n_dim - 1];

        std::vector<int> strides1(op1.strides.begin(), op1.strides.end() - 2);
        std::vector<int> strides2(op2.strides.begin(), op2.strides.end() - 2);
        std::vector<int> strides_new(new_tensor.strides.begin(), new_tensor.strides.end() - 2);

        for (int i = 0; i < n_dim - 2; i++)
        {
            if (op1.shape[i] == 1)
            {
                strides1[i] = 0;
            }

            if (op2.shape[i] == 1)
            {
                strides2[i] = 0;
            }
        }

        std::vector<int> op1_strides = {op1.strides[n_dim - 2], op1.strides[n_dim - 1]};
        std::vector<int> op2_strides = {op2.strides[n_dim - 2], op2.strides[n_dim - 1]};
        std::vector<int> out_strides = {new_tensor.strides[n_dim - 2], new_tensor.strides[n_dim - 1]};

        int offset1 = op1.offset;
        int offset2 = op2.offset;
        int offset_out = new_tensor.offset;

        std::vector<int> indices(n_dim - 2, 0);
        bool run = true;
        while (run)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    for (int k = 0; k < m; k++)
                    {
                        int index_new = offset_out + i * out_strides[0] + j * out_strides[1];
                        int index1 = offset1 + i * op1_strides[0] + k * op1_strides[1];
                        int index2 = offset2 + k * op2_strides[0] + j * op2_strides[1];
                        (*new_tensor.data)[index_new] += (*op1.data)[index1] * (*op2.data)[index2];
                    }
                }
            }

            for (int j = indices.size() - 1; j >= 0; j--)
            {
                indices[j]++;
                offset1 += strides1[j];
                offset2 += strides2[j];
                offset_out += strides_new[j];

                if (indices[j] == new_shape[j] && j == 0)
                {
                    run = false;
                    break;
                }
                else if (indices[j] == new_shape[j])
                {
                    indices[j] = 0;
                    offset1 -= t1_shape[j] * strides1[j];
                    offset2 -= t2_shape[j] * strides2[j];
                    offset_out -= new_shape[j] * strides_new[j];
                }
                else
                {
                    break;
                }
            }
        }
    }

    if (append_one)
    {
        std::vector<int> new_shape2(new_shape.begin(), new_shape.end());
        new_shape2.pop_back();
        new_tensor = new_tensor.view(new_shape2);
    }
    else if (prepend_one)
    {
        std::vector<int> new_shape2(new_shape.begin(), new_shape.end());
        new_shape2.erase(new_shape2.begin() + new_shape2.size() - 2);
        new_tensor = new_tensor.view(new_shape2);
    }

    no_grad.~NoGradGuard();

    if ((t1.grad != nullptr || t2.grad != nullptr) && !NoGradGuard::is_enabled)
    {
        new_tensor.grad = new Tensor<T>(Tensor<T>::zeros(new_tensor.shape));
        std::vector<int> out_shape(new_shape.begin(), new_shape.end());
        new_tensor._backward =
            std::bind(&Tensor<T>::matmul_backward, std::placeholders::_1, t1_shape, t2_shape, out_shape);
        new_tensor.operand1 = new Tensor<T>(t1);
        new_tensor.operand2 = new Tensor<T>(t2);
    }

    return new_tensor;
}

template <typename T>
void Tensor<T>::matmul_backward(std::vector<int> &t1_shape, std::vector<int> &t2_shape, std::vector<int> &new_shape)
{
    NoGradGuard no_grad;

    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    int n_dim = new_shape.size();

    if (operand1->grad != nullptr)
    {
        Tensor<T> op2 = operand2->view(t2_shape);
        Tensor<T> out_grad = this->grad->view(new_shape);

        Tensor<T> temp = op2.transpose(n_dim - 2, n_dim - 1);
        Tensor<T> new_tensor_grad = Tensor<T>::matmul(out_grad, temp);

        std::vector<int> dim_to_reduce;

        for (int i = 0; i < n_dim - 2; i++)
        {
            if (t1_shape[i] != new_shape[i])
            {
                dim_to_reduce.push_back(i);
            }
        }

        std::vector<int> op1_shape(operand1->shape.begin(), operand1->shape.end());
        Tensor<T> reduced_grad = new_tensor_grad.sum(dim_to_reduce, true).view(op1_shape);
        (*operand1->grad) += reduced_grad;
    }

    if (operand2->grad != nullptr)
    {
        Tensor<T> op1 = operand1->view(t1_shape);
        Tensor<T> out_grad = this->grad->view(new_shape);

        Tensor<T> temp = op1.transpose(n_dim - 2, n_dim - 1);
        Tensor<T> new_tensor_grad = Tensor<T>::matmul(temp, out_grad);

        std::vector<int> dim_to_reduce;

        for (int i = 0; i < n_dim - 2; i++)
        {
            if (t2_shape[i] != new_shape[i])
            {
                dim_to_reduce.push_back(i);
            }
        }

        std::vector<int> op2_shape(operand2->shape.begin(), operand2->shape.end());
        Tensor<T> reduced_grad = new_tensor_grad.sum(dim_to_reduce, true).view(op2_shape);
        (*operand2->grad) += reduced_grad;
    }
}

// ========================================================
// SECTION: Mics Matrix Methods
// ========================================================

template <typename T> Tensor<T> Tensor<T>::stack(std::vector<Tensor<T>> tensors, int dim)
{
    std::vector<size_t> shape = tensors[0].shape;

    for (int i = 1; i < tensors.size(); i++)
    {
        if (tensors[i].shape != shape)
        {
            throw std::invalid_argument("All tensors must have the same shape.");
        }
    }

    std::vector<size_t> new_shape = shape;
    new_shape.insert(new_shape.begin() + dim, tensors.size());
    Tensor<T> new_tensor = Tensor<T>::zeros(new_shape, false);

    int num_of_elements = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    std::vector<int> strides_new = new_tensor.strides;
    int offset_new = new_tensor.offset;

    std::vector<int> indices1(new_shape.size(), 0);
    std::vector<int> indices2(shape.size(), 0);

    for (int i = 0; i < num_of_elements; i++)
    {
        indices2 = indices1;
        indices2.erase(indices2.begin() + dim);
        (*new_tensor.data)[offset_new] = tensors[indices1[dim]][indices2];

        for (int j = indices1.size() - 1; j >= 0; j--)
        {
            indices1[j]++;
            offset_new += strides_new[j];
            if (indices1[j] == new_shape[j])
            {
                indices1[j] = 0;
                offset_new -= new_shape[j] * strides_new[j];
            }
            else
            {
                break;
            }
        }
    }

    return new_tensor;
}

template <typename T> Tensor<T> Tensor<T>::unfold(Tensor &in, int kernel_size, int padding, int stride)
{
    int batch_size = in.shape[0];
    int n_channels = in.shape[1];
    int spacial_dim1 = in.shape[2];
    int spacial_dim2 = in.shape[3];

    int n_block_row = ((spacial_dim1 + 2 * padding - kernel_size) / stride + 1);
    int n_block_col = ((spacial_dim2 + 2 * padding - kernel_size) / stride + 1);

    int n_rows = kernel_size * kernel_size * n_channels;
    int n_cols = n_block_row * n_block_col;

    Tensor<T> new_tensor =
        Tensor<T>::zeros({static_cast<size_t>(batch_size), static_cast<size_t>(n_rows), static_cast<size_t>(n_cols)},
                         in.grad != nullptr);

    int new_off = new_tensor.offset;
    int new_st0 = new_tensor.strides[0];
    int new_st1 = new_tensor.strides[1];
    int new_st2 = new_tensor.strides[2];

    int in_off = in.offset;
    int in_st0 = in.strides[0];
    int in_st1 = in.strides[1];
    int in_st2 = in.strides[2];
    int in_st3 = in.strides[3];

    for (int i = 0; i < batch_size; i++)
    {
        for (int j = 0; j < n_rows; j++)
        {
            for (int k = 0; k < n_cols; k++)
            {
                int channel = j / (kernel_size * kernel_size);
                int row = (j % (kernel_size * kernel_size)) / kernel_size;
                int col = (j % (kernel_size * kernel_size)) % kernel_size;

                int row_in = row + stride * (k / n_block_col);
                int col_in = col + stride * (k % n_block_col);

                int index = new_off + i * new_st0 + j * new_st1 + k * new_st2;

                if (row_in < padding || row_in >= spacial_dim1 + padding || col_in < padding ||
                    col_in >= spacial_dim2 + padding)
                {
                    (*new_tensor.data)[index] = 0;
                }
                else
                {
                    int index2 = in_off + i * in_st0 + channel * in_st1 + (row_in - padding) * in_st2 +
                                 (col_in - padding) * in_st3;
                    (*new_tensor.data)[index] = (*in.data)[index2];
                }
            }
        }
    }

    if (new_tensor.grad != nullptr && !NoGradGuard::is_enabled)
    {
        new_tensor._backward =
            std::bind(&Tensor<T>::unfold_backward, std::placeholders::_1, kernel_size, padding, stride);
        new_tensor.operand1 = new Tensor<T>(in);
    }

    return new_tensor;
}

template <typename T> void Tensor<T>::unfold_backward(int kernel_size, int padding, int stride)
{
    if (this->grad == nullptr)
    {
        std::invalid_argument("Can't call backward if the gradient is nullptr.");
    }

    if (operand1->grad != nullptr)
    {
        int batch_size = operand1->shape[0];
        int n_channels = operand1->shape[1];
        int spacial_dim1 = operand1->shape[2];
        int spacial_dim2 = operand1->shape[3];

        int n_block_row = ((spacial_dim1 + 2 * padding - kernel_size) / stride + 1);
        int n_block_col = ((spacial_dim2 + 2 * padding - kernel_size) / stride + 1);

        int n_rows = kernel_size * kernel_size * n_channels;
        int n_cols = n_block_row * n_block_col;

        int out_off = this->grad->offset;
        int out_st0 = this->grad->strides[0];
        int out_st1 = this->grad->strides[1];
        int out_st2 = this->grad->strides[2];

        int op1_off = operand1->offset;
        int op1_st0 = operand1->strides[0];
        int op1_st1 = operand1->strides[1];
        int op1_st2 = operand1->strides[2];
        int op1_st3 = operand1->strides[3];

        for (int i = 0; i < batch_size; i++)
        {
            for (int j = 0; j < n_rows; j++)
            {
                for (int k = 0; k < n_cols; k++)
                {
                    int channel = j / (kernel_size * kernel_size);
                    int row = (j % (kernel_size * kernel_size)) / kernel_size;
                    int col = (j % (kernel_size * kernel_size)) % kernel_size;

                    int row_in = row + stride * (k / n_block_col);
                    int col_in = col + stride * (k % n_block_col);

                    if (row_in < padding || row_in >= spacial_dim1 + padding || col_in < padding ||
                        col_in >= spacial_dim2 + padding)
                    {
                        continue;
                    }

                    int index = out_off + i * out_st0 + j * out_st1 + k * out_st2;
                    int index2 = op1_off + i * op1_st0 + channel * op1_st1 + (row_in - padding) * op1_st2 +
                                 (col_in - padding) * op1_st3;
                    (*operand1->grad->data)[index2] += (*this->grad->data)[index];
                }
            }
        }
    }
}

template <typename T> Tensor<T>::~Tensor()
{
    release();
    if (grad != nullptr)
    {
        delete grad;
    }
}