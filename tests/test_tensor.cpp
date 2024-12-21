#include "../include/tensor.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <utility>

TEST_CASE("The tensor copy and move operations work properly")
{
    Tensor<float> t1 = Tensor<float>({0, 1, 2, 3, 4});
    SECTION("The copy constructor works properly")
    {
        Tensor<float> t2 = Tensor<float>(t1);
        t1[{0}] = 5;
        REQUIRE(t2[{0}] == 5);
    }
    SECTION("The copy assignment operator works properly")
    {
        Tensor<float> t2 = t1;
        t1[{0}] = 5;
        REQUIRE(t2[{0}] == 5);
    }
    SECTION("The move constructor works properly")
    {
        Tensor<float> t2 = std::move(t1);
        CHECK(t2[{0}] == 0);
        CHECK_THROWS(t1[{0}]);
    }
    SECTION("The move assignment operator works properly")
    {
        Tensor<float> t2 = Tensor<float>({0});
        t2 = std::move(t1);
        CHECK(t2[{0}] == 0);
        CHECK_THROWS(t1[{0}]);
    }
}

TEST_CASE("Tensor initialization works properly")
{
    Tensor<float> t1 = Tensor<float>({0, 1, 2, 3, 4});
    for (int i = 0; i < 5; i++)
    {
        REQUIRE(t1[{i}] == i);
    }
    Tensor<float> t2 = Tensor<float>::zeros({5, 5});
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            REQUIRE(t2[{i, j}] == 0);
        }
    }
    Tensor<float> t3 = Tensor<float>::ones({5, 5});
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            REQUIRE(t3[{i, j}] == 1);
        }
    }
    Tensor<float> t4 = Tensor<float>::randn({16, 16});
    float mean = 0;
    float last = -10;
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            REQUIRE_FALSE(t4[{i, j}] == last);
            last = t4[{i, j}];
            mean += t4[{i, j}];
        }
    }
    mean /= 16 * 16;
    REQUIRE(mean < 1);
    REQUIRE(mean > -1);
}

TEST_CASE("The indexing operator works correctly")
{
    Tensor<float> t1 = Tensor<float>::ones({3, 4, 5, 6, 7});
    SECTION("The indexing operator returns a reference")
    {
        REQUIRE(t1[{0, 0, 0, 0, 0}] == 1);
        t1[{0, 0, 0, 0, 0}] = 5;
        REQUIRE(t1[{0, 0, 0, 0, 0}] == 5);
    }
    SECTION("The indexing operator checks the number of indices")
    {
        CHECK_THROWS(t1[{0}]);
        CHECK_THROWS(t1[{0, 0, 0, 0, 0, 0}]);
    }
    SECTION("The indexing operator checks the shape of the tensor")
    {
        CHECK_THROWS(t1[{3, 0, 0, 0, 0}]);
        CHECK_THROWS(t1[{3, 4, 5, 6, 7}]);
        CHECK_THROWS(t1[{0, 0, 0, 0, 10000}]);
    }
    SECTION("The indexing operator works for negative values too")
    {
        t1[{2, 0, 0, 0, 0}] = 2;
        REQUIRE(t1[{-1, 0, 0, 0, 0}] == 2);
        t1[{0, 0, 0, 0, 6}] = 2;
        REQUIRE(t1[{0, 0, 0, 0, -1}] == 2);
        t1[{0, 0, 0, 5, 6}] = 2;
        REQUIRE(t1[{0, 0, 0, -1, -1}] == 2);
        t1[{2, 3, 4, 5, 6}] = 7;
        REQUIRE(t1[{-1, -1, -1, -1, -1}] == 7);
        t1[{0, 0, 0, 0, 3}] = 2;
        REQUIRE(t1[{0, 0, 0, 0, -4}] == 2);
        t1[{0, 0, 0, 0, 0}] = 2;
        REQUIRE(t1[{-3, 0, 0, 0, 0}] == 2);
    }
    SECTION("Negative indexing checks the range correctly")
    {
        CHECK_THROWS(t1[{-4, 0, 0, 0, 0}]);
        CHECK_THROWS(t1[{0, 0, -6, 0, 0}]);
        CHECK_THROWS(t1[{-4, -5, -6, -7, -8}]);
    }
}

TEST_CASE("The indexing operator works for ranges")
{
    Tensor<float> t1 = Tensor<float>::ones({3, 4, 5, 6, 7});
    SECTION("The indexing operator returns a reference")
    {
        REQUIRE(t1[{0, 0, 0, 0, 0}] == 1);
        Tensor<float> t2 = t1[{{0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}}];
        t2[{0, 0, 0, 0, 0}] = 5;
        REQUIRE(t1[{0, 0, 0, 0, 0}] == 5);
    }
    SECTION("The indexing operator calculates the shape correctly")
    {
        Tensor<float> t2 = t1[{{0, 2}, {0, 1}, {3, 5}, {2, 4}, {0, 7}}];
        REQUIRE(t2.size() == std::vector<size_t>({2, 1, 2, 2, 7}));
    }
    SECTION("The indexing operator chooses the right range")
    {
        t1[{0, 0, 3, 2, 0}] = 2;
        t1[{0, 0, 3, 2, 1}] = 3;
        t1[{0, 0, 4, 2, 0}] = 4;
        Tensor<float> t2 = t1[{{0, 2}, {0, 1}, {3, 5}, {2, 4}, {0, 7}}];
        REQUIRE(t2[{0, 0, 0, 0, 0}] == 2);
        REQUIRE(t2[{0, 0, 0, 0, 1}] == 3);
        REQUIRE(t2[{0, 0, 1, 0, 0}] == 4);
    }
    SECTION("The indexing operator works for negative values too")
    {
        t1[{0, 0, 3, 2, 0}] = 2;
        t1[{0, 0, 3, 2, 1}] = 3;
        t1[{0, 0, 4, 2, 0}] = 4;
        Tensor<float> t2 = t1[{{0, -1}, {-4, 1}, {-2, 5}, {2, 4}, {0, 7}}];
        REQUIRE(t2.size() == std::vector<size_t>({2, 1, 2, 2, 7}));
        REQUIRE(t2[{0, 0, 0, 0, 0}] == 2);
        REQUIRE(t2[{0, 0, 0, 0, 1}] == 3);
        REQUIRE(t2[{0, 0, 1, 0, 0}] == 4);
    }
    SECTION("The indexing operator works for ranges larger than its shape")
    {
        Tensor<float> t2 = t1[{{-100, 2}, {-50, 1}, {3, 5}, {2, 4}, {0, 200}}];
        REQUIRE(t2.size() == std::vector<size_t>({2, 1, 2, 2, 7}));
    }
    SECTION("Can use the indexing operator recursively")
    {
        t1[{0, 0, 3, 2, 0}] = 2;
        t1[{0, 0, 3, 2, 1}] = 3;
        t1[{0, 0, 4, 2, 0}] = 4;
        Tensor<float> t2 = t1[{{0, 2}, {0, 1}, {3, 5}, {2, 4}, {0, 7}}];
        Tensor<float> t3 = t2[{{-100, 200}, {0, 1}, {0, 2}, {0, 2}, {0, 3}}];
        REQUIRE(t3[{0, 0, 0, 0, 0}] == 2);
        REQUIRE(t3[{0, 0, 0, 0, 1}] == 3);
        REQUIRE(t3[{0, 0, 1, 0, 0}] == 4);
        REQUIRE(t3[{0, 0, 0, 0, 2}] == 1);
        REQUIRE(t3.size() == std::vector<size_t>({2, 1, 2, 2, 3}));
    }
    SECTION("Indexing operator can handle indices vector of different length than it's shape")
    {
        t1[{0, 0, 3, 2, 0}] = 2;
        t1[{0, 0, 3, 2, 1}] = 3;
        t1[{0, 0, 4, 2, 0}] = 4;
        Tensor<float> t2 = t1[{{0, 0}, {0, 2}}];
        REQUIRE(t2.size() == std::vector<size_t>({2, 5, 6, 7}));
        REQUIRE(t2[{0, 3, 2, 0}] == 2);
        REQUIRE(t2[{0, 3, 2, 1}] == 3);
        REQUIRE(t2[{0, 4, 2, 0}] == 4);
        Tensor<float> t3 = t1[{{0, 1}, {0, 0}}];
        REQUIRE(t3.size() == std::vector<size_t>({1, 5, 6, 7}));
        REQUIRE(t3[{0, 3, 2, 0}] == 2);
        REQUIRE(t3[{0, 3, 2, 1}] == 3);
        REQUIRE(t3[{0, 4, 2, 0}] == 4);
        CHECK_THROWS(t1[{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}}]);
    }
}

TEST_CASE("The size method works correctly")
{
    Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
    REQUIRE(t1.size() == std::vector<size_t>({3, 4, 5}));
    Tensor<float> t2 = Tensor<float>::zeros({5});
    REQUIRE(t2.size() == std::vector<size_t>({5}));
    Tensor<float> t3 = Tensor<float>::randn({5, 5, 6, 7});
    REQUIRE(t3.size() == std::vector<size_t>({5, 5, 6, 7}));
    Tensor<float> t4 = Tensor<float>({0, 1, 2, 3, 4});
    REQUIRE(t4.size() == std::vector<size_t>({5}));
}

TEST_CASE("The view method works correctly")
{
    SECTION("The view method throws exceptions correctly")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        CHECK_THROWS(t1.view({3, 4, 6}));
        CHECK_THROWS(t1.view({3, 4, -1, -1}));
        CHECK_THROWS(t1.view({3, 100, -1}));
        CHECK_THROWS(t1.view({1, 13, -1}));
        Tensor<float> t2 = t1[{{1, 3}, {2, 4}, {3, 5}}];
        CHECK_THROWS(t2.view({-1, 2}));
    }

    SECTION("The view method handles -1 correctly")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.view({-1, 5});
        REQUIRE(t2.size() == std::vector<size_t>({12, 5}));
        Tensor<float> t3 = t1.view({3, -1});
        REQUIRE(t3.size() == std::vector<size_t>({3, 20}));
    }

    SECTION("The view method works on references")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1[{{0, 3}, {0, 4}, {0, 5}}];
        Tensor<float> t3 = t2.view({-1, 5});
        REQUIRE(t3.size() == std::vector<size_t>({12, 5}));
        t2[{0, 0, 0}] = 2;
        REQUIRE(t3[{0, 0}] == 2);
    }

    SECTION("The view method works correctly for valid shapes")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.view({6, 2, 5});
        REQUIRE(t2.size() == std::vector<size_t>({6, 2, 5}));
        Tensor<float> t3 = t1.view({3, 20});
        REQUIRE(t3.size() == std::vector<size_t>({3, 20}));
    }
}

TEST_CASE("The clone method works correctly")
{
    Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
    Tensor<float> t2 = t1.clone();

    SECTION("The clone method returns a new object")
    {
        REQUIRE(&t1 != &t2);
    }

    SECTION("The clone method creates a deep copy")
    {
        t1[{0, 0, 0}] = 2;
        REQUIRE(t2[{0, 0, 0}] == 1);
    }

    SECTION("The cloned tensor has the same shape and values")
    {
        REQUIRE(t1.size() == t2.size());
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 5; k++)
                {
                    REQUIRE(t1[{i, j, k}] == t2[{i, j, k}]);
                }
            }
        }
    }
}

TEST_CASE("The transpose method works correctly")
{
    Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
    SECTION("The transpose method returns a reference")
    {
        Tensor<float> t2 = t1.transpose(0, 1);
        t2[{0, 0, 0}] = 2;
        REQUIRE(t1[{0, 0, 0}] == 2);
    }
    SECTION("The transpose method swaps the dimensions correctly")
    {
        Tensor<float> t2 = t1.transpose(0, 1);
        REQUIRE(t2.size() == std::vector<size_t>({4, 3, 5}));
        Tensor<float> t3 = t1.transpose(1, 2);
        REQUIRE(t3.size() == std::vector<size_t>({3, 5, 4}));
    }
    SECTION("The transpose method throws exceptions for invalid dimensions")
    {
        CHECK_THROWS(t1.transpose(0, 3));
        CHECK_THROWS(t1.transpose(-1, 1));
        CHECK_THROWS(t1.transpose(1, 1));
    }
    SECTION("The transpose method works correctly for higher dimensions")
    {
        Tensor<float> t2 = Tensor<float>::ones({2, 3, 4, 5});
        Tensor<float> t3 = t2.transpose(1, 3);
        REQUIRE(t3.size() == std::vector<size_t>({2, 5, 4, 3}));
        Tensor<float> t4 = t2.transpose(0, 2);
        REQUIRE(t4.size() == std::vector<size_t>({4, 3, 2, 5}));
    }
    SECTION("The transpose method correctly transposes the tensor elements")
    {
        Tensor<float> t2 = t1.transpose(0, 1);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 5; k++)
                {
                    REQUIRE(t2[{j, i, k}] == t1[{i, j, k}]);
                }
            }
        }
        Tensor<float> t3 = t1.transpose(1, 2);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 5; k++)
                {
                    REQUIRE(t3[{i, k, j}] == t1[{i, j, k}]);
                }
            }
        }
    }
}

TEST_CASE("Memory is managed properly")
{
    SECTION("The data doesn't get freed when one tensor is deleted")
    {
        Tensor<float> *t1 = new Tensor<float>({3, 4, 5});
        Tensor<float> t2 = t1->view({1, -1});
        delete t1;
        REQUIRE(t2[{0, 0}] == 3);
        REQUIRE(t2[{0, 1}] == 4);
        REQUIRE(t2[{0, 2}] == 5);
    }
    SECTION("The memory gets freed after deleting all references")
    {
        Tensor<float> *t1 = new Tensor<float>({3, 4, 5});
        Tensor<float> *t2 = new Tensor<float>(*t1);
        (*t2)[{0}] = 2;
        REQUIRE((*t1)[{0}] == 2);
        Tensor<float> *t3 = new Tensor<float>(*t2);
        (*t3)[{0}] = 3;
        REQUIRE((*t1)[{0}] == 3);
        delete t3;
        REQUIRE((*t2)[{0}] == 3);
        REQUIRE((*t1)[{0}] == 3);
        delete t2;
        REQUIRE((*t1)[{0}] == 3);
        delete t1;
    }
}

TEST_CASE("Tensor initialization works properly for different data types")
{
    SECTION("The Tensor works for doubles too")
    {
        Tensor<double> t1 = Tensor<double>({0, 1, 2, 3, 4});
        for (int i = 0; i < 5; i++)
        {
            REQUIRE(t1[{i}] == i);
        }
        Tensor<double> t2 = Tensor<double>::zeros({5, 5});
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                REQUIRE(t2[{i, j}] == 0);
            }
        }
        Tensor<double> t3 = Tensor<double>::ones({5, 5});
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                REQUIRE(t3[{i, j}] == 1);
            }
        }
        Tensor<double> t4 = Tensor<double>::randn({16, 16});
        double mean = 0;
        double last = -10;
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                REQUIRE_FALSE(t4[{i, j}] == last);
                last = t4[{i, j}];
                mean += t4[{i, j}];
            }
        }
        mean /= 16 * 16;
        REQUIRE(mean < 1);
        REQUIRE(mean > -1);
    }
    SECTION("The Tensor works for ints too")
    {
        Tensor<int> t1 = Tensor<int>({0, 1, 2, 3, 4});
        for (int i = 0; i < 5; i++)
        {
            REQUIRE(t1[{i}] == i);
        }
        Tensor<int> t2 = Tensor<int>::zeros({5, 5});
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                REQUIRE(t2[{i, j}] == 0);
            }
        }
        Tensor<int> t3 = Tensor<int>::ones({5, 5});
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                REQUIRE(t3[{i, j}] == 1);
            }
        }
    }
}

TEST_CASE("Arithmetic operations work correctly")
{
    SECTION("Addition works correctly")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4});
        Tensor<float> t2 = Tensor<float>::ones({3, 4});
        Tensor<float> t3 = t1 + t2;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t3[{i, j}] == 2);
            }
        }
    }

    SECTION("Subtraction works correctly")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4});
        Tensor<float> t2 = Tensor<float>::ones({3, 4});
        Tensor<float> t3 = t1 - t2;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t3[{i, j}] == 0);
            }
        }
    }

    SECTION("Unary minus works correctly")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4});
        Tensor<float> t2 = -t1;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t2[{i, j}] == -1);
            }
        }
    }

    SECTION("Unary minus works correctly for sliced tensors")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1[{{0, 2}, {0, 2}, {2, 4}}];
        Tensor<float> t3 = -t2;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    REQUIRE(t3[{i, j, k}] == -1);
                }
            }
        }
    }

    SECTION("Multiplication works correctly")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}).view({3, 4});
        Tensor<float> t2 = Tensor<float>({12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}).view({3, 4});
        Tensor<float> t3 = t1 * t2;
        std::vector<float> expected_values = {12, 22, 30, 36, 40, 42, 42, 40, 36, 30, 22, 12};
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t3[{i, j}] == expected_values[i * 4 + j]);
            }
        }
    }

    SECTION("Division works correctly")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4});
        Tensor<float> t2 = Tensor<float>::ones({3, 4});
        Tensor<float> t3 = t1 / t2;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t3[{i, j}] == 1);
            }
        }
    }

    SECTION("Broadcasting works correctly for higher dimensions")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 3, 4});
        Tensor<float> t2 = Tensor<float>::ones({4});
        Tensor<float> t3 = t1 + t2;
        REQUIRE(t3.size() == std::vector<size_t>({2, 3, 4}));
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    REQUIRE(t3[{i, j, k}] == 2);
                }
            }
        }

        Tensor<float> t4 = Tensor<float>::ones({1, 3, 1});
        Tensor<float> t5 = Tensor<float>::ones({2, 1, 4});
        Tensor<float> t6 = t4 + t5;
        REQUIRE(t6.size() == std::vector<size_t>({2, 3, 4}));
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    REQUIRE(t6[{i, j, k}] == 2);
                }
            }
        }

        Tensor<float> t7 = Tensor<float>::ones({2, 1, 4});
        Tensor<float> t8 = Tensor<float>::ones({3, 4});
        Tensor<float> t9 = t7 + t8;
        REQUIRE(t9.size() == std::vector<size_t>({2, 3, 4}));
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    REQUIRE(t9[{i, j, k}] == 2);
                }
            }
        }
    }

    SECTION("Broadcasting works correctly for tensors with many dimensions")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 3, 4, 5});
        Tensor<float> t2 = Tensor<float>::ones({5});
        Tensor<float> t3 = t1 + t2;
        REQUIRE(t3.size() == std::vector<size_t>({2, 3, 4, 5}));
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    for (int l = 0; l < 5; l++)
                    {
                        REQUIRE(t3[{i, j, k, l}] == 2);
                    }
                }
            }
        }

        Tensor<float> t4 = Tensor<float>::ones({1, 3, 1, 5});
        Tensor<float> t5 = Tensor<float>::ones({2, 1, 4, 1});
        Tensor<float> t6 = t4 + t5;
        REQUIRE(t6.size() == std::vector<size_t>({2, 3, 4, 5}));
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    for (int l = 0; l < 5; l++)
                    {
                        REQUIRE(t6[{i, j, k, l}] == 2);
                    }
                }
            }
        }

        Tensor<float> t7 = Tensor<float>::ones({2, 1, 4, 1});
        Tensor<float> t8 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t9 = t7 + t8;
        REQUIRE(t9.size() == std::vector<size_t>({2, 3, 4, 5}));
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    for (int l = 0; l < 5; l++)
                    {
                        REQUIRE(t9[{i, j, k, l}] == 2);
                    }
                }
            }
        }
    }

    SECTION("Broadcasting throws exceptions for incompatible shapes")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4});
        Tensor<float> t2 = Tensor<float>::ones({5});
        CHECK_THROWS(t1 + t2);
        Tensor<float> t3 = Tensor<float>::ones({2, 1});
        Tensor<float> t4 = Tensor<float>::ones({8, 4, 3});
        CHECK_THROWS(t3 + t4);
        Tensor<float> t5 = Tensor<float>::ones({2, 1, 4, 5, 8, 1, 2});
        Tensor<float> t6 = Tensor<float>::ones({4, 5, 2, 1, 2});
        CHECK_THROWS(t5 + t6);
    }

    SECTION("Arithmetic operations work for different data types")
    {
        Tensor<double> t1 = Tensor<double>::ones({3, 4});
        Tensor<double> t2 = Tensor<double>::ones({3, 4});
        Tensor<double> t3 = t1 + t2;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t3[{i, j}] == 2);
            }
        }

        Tensor<int> t4 = Tensor<int>::ones({3, 4});
        Tensor<int> t5 = Tensor<int>::ones({3, 4});
        Tensor<int> t6 = t4 + t5;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t6[{i, j}] == 2);
            }
        }
    }
    
    SECTION("Arithmetic operations on sliced tensors work correctly")
    {

        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t3 = t1[{{0, 2}, {0, 2}, {2, 4}}];
        Tensor<float> t4 = t2[{{0, 2}, {1, 3}, {0, 2}}];
        Tensor<float> t5 = t3 + t4;
        REQUIRE(t5.size() == std::vector<size_t>({2, 2, 2}));
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    REQUIRE(t5[{i, j, k}] == 2);
                }
            }
        }
    }

    SECTION("Broadcasting works correctly on sliced tensors")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t3 = t1[{{0, 2}, {0, 2}, {0, 2}}];
        Tensor<float> t4 = Tensor<float>::ones({2});
        Tensor<float> t5 = t3 + t4;
        REQUIRE(t5.size() == std::vector<size_t>({2, 2, 2}));
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    REQUIRE(t5[{i, j, k}] == 2);
                }
            }
        }
    }
}

TEST_CASE("Mixed arithmetic operations between tensors and primitive types work correctly")
{
    SECTION("Addition with primitive types")
    {
        Tensor<int> t1 = Tensor<int>::ones({2, 2});
        Tensor<int> t2 = t1 + 1;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE(t2[{i, j}] == 2);
            }
        }

        Tensor<double> t3 = Tensor<double>::ones({2, 2});
        Tensor<double> t4 = t3 + 1.0;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE(t4[{i, j}] == 2.0);
            }
        }
    }

    SECTION("Subtraction with primitive types")
    {
        Tensor<int> t1 = Tensor<int>::ones({2, 2});
        Tensor<int> t2 = t1 - 1;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE(t2[{i, j}] == 0);
            }
        }

        Tensor<double> t3 = Tensor<double>::ones({2, 2});
        Tensor<double> t4 = t3 - 1.0;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE(t4[{i, j}] == 0.0);
            }
        }
    }

    SECTION("Multiplication with primitive types")
    {
        Tensor<int> t1 = Tensor<int>::ones({2, 2});
        Tensor<int> t2 = t1 * 2;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE(t2[{i, j}] == 2);
            }
        }

        Tensor<double> t3 = Tensor<double>::ones({2, 2});
        Tensor<double> t4 = t3 * 2.0;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE(t4[{i, j}] == 2.0);
            }
        }
    }

    SECTION("Division with primitive types")
    {
        Tensor<int> t1 = Tensor<int>::ones({2, 2}) * 2;
        Tensor<int> t2 = t1 / 2;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE(t2[{i, j}] == 1);
            }
        }

        Tensor<double> t3 = Tensor<double>::ones({2, 2}) * 2.0;
        Tensor<double> t4 = t3 / 2.0;
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE(t4[{i, j}] == 1.0);
            }
        }
    }
}

TEST_CASE("The Tensor works correctly with multiple chained operations")
{
    Tensor<float> a = Tensor<float>::ones({3, 4});
    Tensor<float> b = a.view({2, -1});
    Tensor<float> c = -(((b + 2) * 3) - 4) / 2;
    Tensor<float> d = c[{{0, 2}, {3, 5}}];
    Tensor<float> e = d.clone();
    Tensor<float> f = e.transpose(0, 1);
    REQUIRE(f[{0, 0}] == -2.5);
    REQUIRE(f[{0, 1}] == -2.5);
    REQUIRE(f[{1, 0}] == -2.5);
    REQUIRE(f[{1, 1}] == -2.5);
}

TEST_CASE("The sum method works correctly")
{
    SECTION("Sum of a 1D tensor")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5});
        Tensor<float> t2 = t1.sum();
        REQUIRE(t2[{0}] == 15);
    }

    SECTION("Sum of a 3D tensor")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 2, 2});
        Tensor<float> t2 = t1.sum();
        REQUIRE(t2[{0}] == 8);
    }

    SECTION("Sum of a tensor with mixed values")
    {
        Tensor<float> t1 = Tensor<float>({-1, 2, -3, 4, -5, 2}).view({2, -1});
        Tensor<float> t2 = t1.sum();
        REQUIRE(t2[{0}] == -1);
    }
}
TEST_CASE("The sum method with dimension and keep_dim works correctly")
{
    SECTION("Sum along the first dimension")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.sum({0}, false);
        REQUIRE(t2.size() == std::vector<size_t>({4, 5}));
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                REQUIRE(t2[{i, j}] == 3);
            }
        }
    }

    SECTION("Sum along the third dimension")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.sum({2}, false);
        REQUIRE(t2.size() == std::vector<size_t>({3, 4}));
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t2[{i, j}] == 5);
            }
        }
    }

    SECTION("Sum along the first dimension with keep_dim")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.sum({0}, true);
        REQUIRE(t2.size() == std::vector<size_t>({1, 4, 5}));
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                REQUIRE(t2[{0, i, j}] == 3);
            }
        }
    }

    SECTION("Sum along the third dimension with keep_dim")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.sum({2}, true);
        REQUIRE(t2.size() == std::vector<size_t>({3, 4, 1}));
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t2[{i, j, 0}] == 5);
            }
        }
    }

    SECTION("Sum along a dimension with mixed values")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6}).view({2, 2, 3});
        Tensor<float> t2 = t1.sum({0}, false);
        REQUIRE(t2.size() == std::vector<size_t>({2, 3}));
        REQUIRE(t2[{0, 0}] == 0);
        REQUIRE(t2[{0, 1}] == 0);
        REQUIRE(t2[{0, 2}] == 0);
        REQUIRE(t2[{1, 0}] == 0);
        REQUIRE(t2[{1, 1}] == 0);
        REQUIRE(t2[{1, 2}] == 0);
    }
    SECTION("Sum along multiple dimensions")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.sum({0, 2}, false);
        REQUIRE(t2.size() == std::vector<size_t>({4}));
        for (int i = 0; i < 4; i++)
        {
            REQUIRE(t2[{i}] == 15);
        }
    }

    SECTION("Sum along multiple dimensions with keep_dim")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.sum({0, 2}, true);
        REQUIRE(t2.size() == std::vector<size_t>({1, 4, 1}));
        for (int i = 0; i < 4; i++)
        {
            REQUIRE(t2[{0, i, 0}] == 15);
        }
    }
}

TEST_CASE("Basic functionality of autograd works correctly.") 
{
    SECTION(".backward works for multiplication on 1 element tensor")
    {
        Tensor<float> t1 = Tensor<float>({1}, true);
        Tensor<float> t2 = Tensor<float>({2}, true);
        Tensor<float> t3 = t1 * t2;
        t3.backward();
        REQUIRE((*t1.grad)[{0}] == 2);
        REQUIRE((*t2.grad)[{0}] == 1);
        REQUIRE((*t3.grad)[{0}] == 1);
    }
    SECTION(".backward works for addition on 1 element tensor")
    {
        Tensor<float> t1 = Tensor<float>({1}, true);
        Tensor<float> t2 = Tensor<float>({2}, true);
        Tensor<float> t3 = t1 + t2;
        t3.backward();
        REQUIRE((*t1.grad)[{0}] == 1);
        REQUIRE((*t2.grad)[{0}] == 1);
        REQUIRE((*t3.grad)[{0}] == 1);
    }
    SECTION(".backward works for addition and multiplication combined on 1 element tensor")
    {
        Tensor<float> t1 = Tensor<float>({1}, true);
        Tensor<float> t2 = Tensor<float>({2}, true);
        Tensor<float> t3 = t1 + t2;
        Tensor<float> t4 = t3 * t2;
        t4.backward();
        REQUIRE((*t1.grad)[{0}] == 2);
        REQUIRE((*t2.grad)[{0}] == 5);
        REQUIRE((*t3.grad)[{0}] == 2);
        REQUIRE((*t4.grad)[{0}] == 1);
    }
}

TEST_CASE("Autograd works for higher dimensional tensors")
{
    SECTION(".backward works for multiplication")
    {
        Tensor<float> t1 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t2 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t3 = t1 * t2;
        Tensor<float> t4 = t3.sum();
        t4.backward();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 1);
                REQUIRE((*t2.grad)[{i, j}] == 1);
            }
        }
    }

    SECTION(".backward works for multiplication for more complicated tensors")
    {
        Tensor<float> t1 = Tensor<float>({5, 4, 3, 2, 1, 0}, true).view({2, 3});
        Tensor<float> t2 = Tensor<float>({0, 1, 2, 3, 4, 5}, true).view({2, 3}); 
        Tensor<float> t3 = t1 * t2;
        Tensor<float> t4 = t3.sum();
        t4.backward();
        std::vector<float> expected_grad1 = {0, 1, 2, 3, 4, 5};
        std::vector<float> expected_grad2 = {5, 4, 3, 2, 1, 0};
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == expected_grad1[i * 3 + j]);
                REQUIRE((*t2.grad)[{i, j}] == expected_grad2[i * 3 + j]);
            }
        }
    }

    SECTION(".backward works for addition with more elements")
    {
        Tensor<float> t1 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t2 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t3 = t1 + t2;
        Tensor<float> t4 = t3.sum();
        t4.backward();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 1);
                REQUIRE((*t2.grad)[{i, j}] == 1);
            }
        }
    }
}

TEST_CASE("Autograd works for subtraction and division")
{
    SECTION(".backward works for subtraction")
    {
        Tensor<float> t1 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t2 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t3 = t1 - t2;
        Tensor<float> t4 = t3.sum();
        t4.backward();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 1);
                REQUIRE((*t2.grad)[{i, j}] == -1);
            }
        }
    }

    SECTION(".backward works for division")
    {
        Tensor<float> t1 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t2 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t3 = t1 / t2;
        Tensor<float> t4 = t3.sum();
        t4.backward();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 1);
                REQUIRE((*t2.grad)[{i, j}] == -1);
            }
        }
    }

    SECTION(".backward works for subtraction and division combined")
    {
        Tensor<float> t1 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t2 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t3 = t1 - t2;
        Tensor<float> t4 = t3 / t2;
        Tensor<float> t5 = t4.sum();
        t5.backward();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 1);
                REQUIRE((*t2.grad)[{i, j}] == -1);
            }
        }
    }

    SECTION(".backward works for tensors with mixed values")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5, 6}, true).view({2, 3});
        Tensor<float> t2 = Tensor<float>({6, 5, 4, 3, 2, 1}, true).view({2, 3});
        Tensor<float> t3 = t1 - t2;
        Tensor<float> t4 = t3 / t2;
        Tensor<float> t5 = t4.sum();
        t5.backward();
        std::vector<float> expected_grad1 = {0.1667, 0.2, 0.25, 0.3333, 0.5, 1.0};
        std::vector<float> expected_grad2 = {-0.0278, -0.08, -0.1875, -0.4444, -1.25, -6.0};
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE_THAT(((*t1.grad)[{i, j}]), Catch::Matchers::WithinRel(expected_grad1[i * 3 + j], 0.01f));
                REQUIRE_THAT(((*t2.grad)[{i, j}]), Catch::Matchers::WithinRel(expected_grad2[i * 3 + j], 0.01f));
            }
        }
    }
}
TEST_CASE("Autograd works for mixed tensors and integers")
{
    SECTION(".backward works for addition with integers")
    {
        Tensor<float> t1 = Tensor<float>({1}, true);
        Tensor<float> t2 = t1 + 2;
        t2.backward();
        REQUIRE((*t1.grad)[{0}] == 1);
        REQUIRE((*t2.grad)[{0}] == 1);
    }

    SECTION(".backward works for subtraction with integers")
    {
        Tensor<float> t1 = Tensor<float>({1}, true);
        Tensor<float> t2 = t1 - 2;
        t2.backward();
        REQUIRE((*t1.grad)[{0}] == 1);
        REQUIRE((*t2.grad)[{0}] == 1);
    }

    SECTION(".backward works for multiplication with integers")
    {
        Tensor<float> t1 = Tensor<float>({1}, true);
        Tensor<float> t2 = t1 * 2;
        t2.backward();
        REQUIRE((*t1.grad)[{0}] == 2);
        REQUIRE((*t2.grad)[{0}] == 1);
    }

    SECTION(".backward works for division with integers")
    {
        Tensor<float> t1 = Tensor<float>({1}, true);
        Tensor<float> t2 = t1 / 2;
        t2.backward();
        REQUIRE((*t1.grad)[{0}] == 0.5);
        REQUIRE((*t2.grad)[{0}] == 1);
    }

    SECTION(".backward works for addition with integers and higher dimensional tensors")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 2}, true);
        Tensor<float> t2 = t1 + 2;
        Tensor<float> t3 = t2.sum();
        t3.backward();
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 1);
            }
        }
    }

    SECTION(".backward works for subtraction with integers and higher dimensional tensors")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 2}, true);
        Tensor<float> t2 = t1 - 2;
        Tensor<float> t3 = t2.sum();
        t3.backward();
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 1);
            }
        }
    }

    SECTION(".backward works for multiplication with integers and higher dimensional tensors")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 2}, true);
        Tensor<float> t2 = t1 * 2;
        Tensor<float> t3 = t2.sum();
        t3.backward();
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 2);
            }
        }
    }

    SECTION(".backward works for division with integers and higher dimensional tensors")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 2}, true);
        Tensor<float> t2 = t1 / 2;
        Tensor<float> t3 = t2.sum();
        t3.backward();
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 0.5);
            }
        }
    }
}

TEST_CASE("Autograd works for unary minus")
{
    SECTION(".backward works for unary minus on 1 element tensor")
    {
        Tensor<float> t1 = Tensor<float>({1}, true);
        Tensor<float> t2 = -t1;
        t2.backward();
        REQUIRE((*t1.grad)[{0}] == -1);
        REQUIRE((*t2.grad)[{0}] == 1);
    }

    SECTION(".backward works for unary minus on higher dimensional tensor")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 2}, true);
        Tensor<float> t2 = -t1;
        Tensor<float> t3 = t2.sum();
        t3.backward();
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == -1);
            }
        }
    }
}

TEST_CASE("Autograd works for all arithmetic operators combined")
{
    Tensor<float> t1 = Tensor<float>::ones({2, 2}, true);
    Tensor<float> t2 = Tensor<float>::ones({2, 2}, true) * 2;
    Tensor<float> t3 = t1 + t2;
    Tensor<float> t4 = t3 - t1;
    Tensor<float> t5 = t4 * t2;
    Tensor<float> t6 = t5 / t1;
    Tensor<float> t7 = -t6;
    Tensor<float> t8 = t7.sum();
    t8.backward();

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            REQUIRE((*t1.grad)[{i, j}] == 4);
            REQUIRE((*t2.grad)[{i, j}] == -4);
        }
    }
}

TEST_CASE("The max method works correctly")
{
    SECTION("Max of a 1D tensor")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5});
        Tensor<float> t2 = t1.max();
        REQUIRE(t2[{0}] == 5);
    }

    SECTION("Max of a 3D tensor")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 2, 2}) * 5;
        t1[{0, 0, 0}] = 10;
        Tensor<float> t2 = t1.max();
        REQUIRE(t2[{0}] == 10);
    }

    SECTION("Max of a tensor with mixed values")
    {
        Tensor<float> t1 = Tensor<float>({-1, 2, -3, 4, -5, 2}).view({2, -1});
        Tensor<float> t2 = t1.max();
        REQUIRE(t2[{0}] == 4);
    }
}

TEST_CASE("Autograd works for the sum method with and without dim and keepdim")
{
    SECTION(".backward works for sum without dim")
    {
        Tensor<float> t1 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t2 = t1.sum();
        t2.backward();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 1);
            }
        }
    }

    SECTION(".backward works for sum with dim")
    {
        Tensor<float> t1 = Tensor<float>::ones({4}, true);
        Tensor<float> t2 = t1.sum({0}, true);
        t2.backward();
        for (int i = 0; i < 4; i++)
        {
            REQUIRE((*t1.grad)[{i}] == 1);
        }
    }

    SECTION(".backward works for sum with dim and keepdim")
    {
        Tensor<float> t1 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t2 = t1.sum({0}, true);
        Tensor<float> t3 = t2.sum();
        t3.backward();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 1);
            }
        }
    }
}

TEST_CASE("The argmax method works correctly")
{
    SECTION("Argmax of a 1D tensor")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5});
        Tensor<int> t2 = t1.argmax();
        REQUIRE(t2[{0}] == 4);
    }

    SECTION("Argmax of a 3D tensor")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 2, 2}) * 5;
        t1[{0, 0, 0}] = 10;
        Tensor<int> t2 = t1.argmax();
        REQUIRE(t2[{0}] == 0);
    }

    SECTION("Argmax of a tensor with mixed values")
    {
        Tensor<float> t1 = Tensor<float>({-1, 2, -3, 4, -5, 2}).view({2, -1});
        Tensor<int> t2 = t1.argmax();
        REQUIRE(t2[{0}] == 3);
    }
}

TEST_CASE("The mean method works correctly")
{
    SECTION("Mean of a 1D tensor")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5});
        Tensor<float> t2 = t1.mean();
        REQUIRE(t2[{0}] == 3);
    }

    SECTION("Mean of a 3D tensor")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 2, 2}) * 4;
        Tensor<float> t2 = t1.mean();
        REQUIRE(t2[{0}] == 4);
    }
}

TEST_CASE("The mean method with dimension and keep_dim works correctly")
{
    SECTION("Mean along the first dimension")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.mean({0}, false);
        REQUIRE(t2.size() == std::vector<size_t>({4, 5}));
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                REQUIRE(t2[{i, j}] == 1);
            }
        }
    }

    SECTION("Mean along the third dimension")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.mean({2}, false);
        REQUIRE(t2.size() == std::vector<size_t>({3, 4}));
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t2[{i, j}] == 1);
            }
        }
    }

    SECTION("Mean along the first dimension with keep_dim")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.mean({0}, true);
        REQUIRE(t2.size() == std::vector<size_t>({1, 4, 5}));
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                REQUIRE(t2[{0, i, j}] == 1);
            }
        }
    }

    SECTION("Mean along the third dimension with keep_dim")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.mean({2}, true);
        REQUIRE(t2.size() == std::vector<size_t>({3, 4, 1}));
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t2[{i, j, 0}] == 1);
            }
        }
    }

    SECTION("Mean along a dimension with mixed values")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6}).view({2, 2, 3});
        Tensor<float> t2 = t1.mean({0}, false);
        REQUIRE(t2.size() == std::vector<size_t>({2, 3}));
        REQUIRE(t2[{0, 0}] == 0);
        REQUIRE(t2[{0, 1}] == 0);
        REQUIRE(t2[{0, 2}] == 0);
        REQUIRE(t2[{1, 0}] == 0);
        REQUIRE(t2[{1, 1}] == 0);
        REQUIRE(t2[{1, 2}] == 0);
    }

    SECTION("Mean with mixed values and keepdim")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6}).view({2, 2, 3});
        Tensor<float> t2 = t1.mean({0}, true);
        REQUIRE(t2.size() == std::vector<size_t>({1, 2, 3}));
        REQUIRE(t2[{0, 0, 0}] == 0);
        REQUIRE(t2[{0, 0, 1}] == 0);
        REQUIRE(t2[{0, 0, 2}] == 0);
        REQUIRE(t2[{0, 1, 0}] == 0);
        REQUIRE(t2[{0, 1, 1}] == 0);
        REQUIRE(t2[{0, 1, 2}] == 0);
    }
}

TEST_CASE("Autograd works for a copy")
{
    SECTION("Sum two tensors, copy the result, and call backward")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 2}, true);
        Tensor<float> t2 = Tensor<float>::ones({2, 2}, true) * 2;
        Tensor<float> t3 = t1 * t2;
        Tensor<float> t4 = t3.sum();
        Tensor<float> t5 = Tensor<float>(t4);
        REQUIRE(&t5 != &t4);
        t5.backward();
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 2);
                REQUIRE((*t2.grad)[{i, j}] == 1);
            }
        }
    }
}

TEST_CASE("Autograd for the mean works correctly")
{
    SECTION(".backward works for mean without dim")
    {
        Tensor<float> t1 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t2 = t1.mean();
        t2.backward();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE_THAT(((*t1.grad)[{i, j}]), Catch::Matchers::WithinRel(0.0833f, 0.01f));
            }
        }
    }
    
    SECTION(".backward works for mean with dim and keepdim")
    {
        Tensor<float> t1 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t2 = t1.mean({0}, true);
        Tensor<float> t3 = t2.sum();
        t3.backward();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE_THAT(((*t1.grad)[{i, j}]), Catch::Matchers::WithinRel(0.25f, 0.01f));
            }
        }
    }
}

TEST_CASE("The var method works correctly")
{
    SECTION("Variance of a 1D tensor")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5});
        Tensor<float> t2 = t1.var();
        REQUIRE_THAT((t2[{0}]), Catch::Matchers::WithinRel(2.5f, 0.01f));
    }

    SECTION("Variance of a 3D tensor")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 2, 2}) * 4;
        Tensor<float> t2 = t1.var();
        REQUIRE(t2[{0}] == 0);
    }

    SECTION("Variance of a tensor with mixed values")
    {
        Tensor<float> t1 = Tensor<float>({-1, 2, -3, 4, -5, 2}).view({2, -1});
        Tensor<float> t2 = t1.var();
        REQUIRE_THAT((t2[{0}]), Catch::Matchers::WithinRel(11.766f, 0.01f));
    }
}

TEST_CASE("The var method with dimension and keep_dim works correctly")
{
    SECTION("Variance along the first dimension")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.var({0}, false);
        REQUIRE(t2.size() == std::vector<size_t>({4, 5}));
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                REQUIRE(t2[{i, j}] == 0);
            }
        }
    }

    SECTION("Variance along the third dimension")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.var({2}, false);
        REQUIRE(t2.size() == std::vector<size_t>({3, 4}));
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t2[{i, j}] == 0);
            }
        }
    }

    SECTION("Variance along the first dimension with keep_dim")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.var({0}, true);
        REQUIRE(t2.size() == std::vector<size_t>({1, 4, 5}));
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                REQUIRE(t2[{0, i, j}] == 0);
            }
        }
    }

    SECTION("Variance along the third dimension with keep_dim")
    {
        Tensor<float> t1 = Tensor<float>::ones({3, 4, 5});
        Tensor<float> t2 = t1.var({2}, true);
        REQUIRE(t2.size() == std::vector<size_t>({3, 4, 1}));
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                REQUIRE(t2[{i, j, 0}] == 0);
            }
        }
    }

    SECTION("Variance along a dimension with mixed values")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6}).view({2, 2, 3});
        Tensor<float> t2 = t1.var({0}, false);
        REQUIRE(t2.size() == std::vector<size_t>({2, 3}));
        REQUIRE_THAT((t2[{0, 0}]), Catch::Matchers::WithinRel(2.0f, 0.01f));
        REQUIRE_THAT((t2[{0, 1}]), Catch::Matchers::WithinRel(8.0f, 0.01f));
        REQUIRE_THAT((t2[{0, 2}]), Catch::Matchers::WithinRel(18.0f, 0.01f));
        REQUIRE_THAT((t2[{1, 0}]), Catch::Matchers::WithinRel(32.0f, 0.01f));
        REQUIRE_THAT((t2[{1, 1}]), Catch::Matchers::WithinRel(50.0f, 0.01f));
        REQUIRE_THAT((t2[{1, 2}]), Catch::Matchers::WithinRel(72.0f, 0.01f));
    }

    SECTION("Variance with mixed values and keepdim")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6}).view({2, 2, 3});
        Tensor<float> t2 = t1.var({0}, true);
        REQUIRE(t2.size() == std::vector<size_t>({1, 2, 3}));
        REQUIRE_THAT((t2[{0, 0, 0}]), Catch::Matchers::WithinRel(2.0f, 0.01f));
        REQUIRE_THAT((t2[{0, 0, 1}]), Catch::Matchers::WithinRel(8.0f, 0.01f));
        REQUIRE_THAT((t2[{0, 0, 2}]), Catch::Matchers::WithinRel(18.0f, 0.01f));
        REQUIRE_THAT((t2[{0, 1, 0}]), Catch::Matchers::WithinRel(32.0f, 0.01f));
        REQUIRE_THAT((t2[{0, 1, 1}]), Catch::Matchers::WithinRel(50.0f, 0.01f));
        REQUIRE_THAT((t2[{0, 1, 2}]), Catch::Matchers::WithinRel(72.0f, 0.01f));
    }
}

TEST_CASE("Autograd for the var method works correctly")
{
    SECTION(".backward works for variance without dim")
    {
        Tensor<float> t1 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t2 = t1.var();
        t2.backward();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 0);
            }
        }
    }

    SECTION(".backward works for variance with dim and keepdim")
    {
        Tensor<float> t1 = Tensor<float>::ones({4, 3}, true);
        Tensor<float> t2 = t1.var({0}, true);
        Tensor<float> t3 = t2.sum();
        t3.backward();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == 0);
            }
        }
    }
}

TEST_CASE("The sqrt method works correctly")
{
    SECTION("Square root of a tensor with positive values")
    {
        Tensor<float> t1 = Tensor<float>({1, 4, 9, 16, 25});
        Tensor<float> t2 = t1.sqrt();
        REQUIRE(t2[{0}] == 1);
        REQUIRE(t2[{1}] == 2);
        REQUIRE(t2[{2}] == 3);
        REQUIRE(t2[{3}] == 4);
        REQUIRE(t2[{4}] == 5);
    }
    
    SECTION("Autograd works for sqrt")
    {
        Tensor<float> t1 = Tensor<float>({1, 4, 9, 16, 25}, true);
        Tensor<float> t2 = t1.sqrt();
        Tensor<float> t3 = t2.sum();
        t3.backward();
        std::vector<float> expected_grad = {0.5, 0.25, 0.1667, 0.125, 0.1};
        for (int i = 0; i < 5; i++)
        {
            REQUIRE_THAT(((*t1.grad)[{i}]), Catch::Matchers::WithinRel(expected_grad[i], 0.01f));
        }
    }
}

TEST_CASE("The pow method works correctly")
{
    SECTION("Power of a tensor with positive values")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5});
        Tensor<float> t2 = t1.pow(2);
        REQUIRE(t2[{0}] == 1);
        REQUIRE(t2[{1}] == 4);
        REQUIRE(t2[{2}] == 9);
        REQUIRE(t2[{3}] == 16);
        REQUIRE(t2[{4}] == 25);
    }
    
    SECTION("Autograd works for pow with positive exponent")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5}, true);
        Tensor<float> t2 = t1.pow(2);
        Tensor<float> t3 = t2.sum();
        t3.backward();
        std::vector<float> expected_grad = {2, 4, 6, 8, 10};
        for (int i = 0; i < 5; i++)
        {
            REQUIRE((*t1.grad)[{i}] == expected_grad[i]);
        }
    }

    SECTION("Autograd works for pow with negative exponent")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5}, true);
        Tensor<float> t2 = t1.pow(-1);
        Tensor<float> t3 = t2.sum();
        t3.backward();
        std::vector<float> expected_grad = {-1, -0.25, -0.1111, -0.0625, -0.04};
        for (int i = 0; i < 5; i++)
        {
            REQUIRE_THAT(((*t1.grad)[{i}]), Catch::Matchers::WithinRel(expected_grad[i], 0.01f));
        }
    }
}

TEST_CASE("The exp method works correctly")
{
    SECTION("Exponential of a tensor with positive values")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5});
        Tensor<float> t2 = t1.exp();
        REQUIRE_THAT((t2[{0}]), Catch::Matchers::WithinRel(std::exp(1.0f), 0.01f));
        REQUIRE_THAT((t2[{1}]), Catch::Matchers::WithinRel(std::exp(2.0f), 0.01f));
        REQUIRE_THAT((t2[{2}]), Catch::Matchers::WithinRel(std::exp(3.0f), 0.01f));
        REQUIRE_THAT((t2[{3}]), Catch::Matchers::WithinRel(std::exp(4.0f), 0.01f));
        REQUIRE_THAT((t2[{4}]), Catch::Matchers::WithinRel(std::exp(5.0f), 0.01f));
    }

    SECTION("Exponential of a tensor with mixed values")
    {
        Tensor<float> t1 = Tensor<float>({-1, 0, 1, 2, -2});
        Tensor<float> t2 = t1.exp();
        REQUIRE_THAT((t2[{0}]), Catch::Matchers::WithinRel(std::exp(-1.0f), 0.01f));
        REQUIRE_THAT((t2[{1}]), Catch::Matchers::WithinRel(std::exp(0.0f), 0.01f));
        REQUIRE_THAT((t2[{2}]), Catch::Matchers::WithinRel(std::exp(1.0f), 0.01f));
        REQUIRE_THAT((t2[{3}]), Catch::Matchers::WithinRel(std::exp(2.0f), 0.01f));
        REQUIRE_THAT((t2[{4}]), Catch::Matchers::WithinRel(std::exp(-2.0f), 0.01f));
    }

    SECTION("Autograd works for exp")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5}, true);
        Tensor<float> t2 = t1.exp();
        Tensor<float> t3 = t2.sum();
        t3.backward();
        std::vector<float> expected_grad = {std::exp(1.0f), std::exp(2.0f), std::exp(3.0f), std::exp(4.0f), std::exp(5.0f)};
        for (int i = 0; i < 5; i++)
        {
            REQUIRE_THAT(((*t1.grad)[{i}]), Catch::Matchers::WithinRel(expected_grad[i], 0.01f));
        }
    }
}

TEST_CASE("The sigmoid method works correctly")
{
    SECTION("Sigmoid of a tensor with positive values")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5});
        Tensor<float> t2 = Tensor<float>::sigmoid(t1);
        REQUIRE_THAT((t2[{0}]), Catch::Matchers::WithinRel(1 / (1 + std::exp(-1.0f)), 0.01f));
        REQUIRE_THAT((t2[{1}]), Catch::Matchers::WithinRel(1 / (1 + std::exp(-2.0f)), 0.01f));
        REQUIRE_THAT((t2[{2}]), Catch::Matchers::WithinRel(1 / (1 + std::exp(-3.0f)), 0.01f));
        REQUIRE_THAT((t2[{3}]), Catch::Matchers::WithinRel(1 / (1 + std::exp(-4.0f)), 0.01f));
        REQUIRE_THAT((t2[{4}]), Catch::Matchers::WithinRel(1 / (1 + std::exp(-5.0f)), 0.01f));
    }

    SECTION("Sigmoid of a tensor with mixed values")
    {
        Tensor<float> t1 = Tensor<float>({-1, 0, 1, 2, -2});
        Tensor<float> t2 = Tensor<float>::sigmoid(t1);
        REQUIRE_THAT((t2[{0}]), Catch::Matchers::WithinRel(1 / (1 + std::exp(1.0f)), 0.01f));
        REQUIRE_THAT((t2[{1}]), Catch::Matchers::WithinRel(1 / (1 + std::exp(0.0f)), 0.01f));
        REQUIRE_THAT((t2[{2}]), Catch::Matchers::WithinRel(1 / (1 + std::exp(-1.0f)), 0.01f));
        REQUIRE_THAT((t2[{3}]), Catch::Matchers::WithinRel(1 / (1 + std::exp(-2.0f)), 0.01f));
        REQUIRE_THAT((t2[{4}]), Catch::Matchers::WithinRel(1 / (1 + std::exp(2.0f)), 0.01f));
    }

    SECTION("Autograd works for sigmoid")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5}, true);
        Tensor<float> t2 = Tensor<float>::sigmoid(t1);
        Tensor<float> t3 = t2.sum();
        t3.backward();
        std::vector<float> expected_grad = {
            (1 / (1 + std::exp(-1.0f))) * (1 - (1 / (1 + std::exp(-1.0f)))),
            (1 / (1 + std::exp(-2.0f))) * (1 - (1 / (1 + std::exp(-2.0f)))),
            (1 / (1 + std::exp(-3.0f))) * (1 - (1 / (1 + std::exp(-3.0f)))),
            (1 / (1 + std::exp(-4.0f))) * (1 - (1 / (1 + std::exp(-4.0f)))),
            (1 / (1 + std::exp(-5.0f))) * (1 - (1 / (1 + std::exp(-5.0f))))
        };
        for (int i = 0; i < 5; i++)
        {
            REQUIRE_THAT(((*t1.grad)[{i}]), Catch::Matchers::WithinRel(expected_grad[i], 0.01f));
        }
    }
}

TEST_CASE("The relu method works correctly")
{
    SECTION("ReLU of a tensor with positive values")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5});
        Tensor<float> t2 = Tensor<float>::relu(t1);
        for (int i = 0; i < 5; i++)
        {
            REQUIRE(t2[{i}] == t1[{i}]);
        }
    }

    SECTION("ReLU of a tensor with mixed values")
    {
        Tensor<float> t1 = Tensor<float>({-1, 0, 1, -2, 2});
        Tensor<float> t2 = Tensor<float>::relu(t1);
        REQUIRE(t2[{0}] == 0);
        REQUIRE(t2[{1}] == 0);
        REQUIRE(t2[{2}] == 1);
        REQUIRE(t2[{3}] == 0);
        REQUIRE(t2[{4}] == 2);
    }

    SECTION("Autograd works for ReLU")
    {
        Tensor<float> t1 = Tensor<float>({-1, 0, 1, -2, 2}, true);
        Tensor<float> t2 = Tensor<float>::relu(t1);
        Tensor<float> t3 = t2.sum();
        t3.backward();
        REQUIRE((*t1.grad)[{0}] == 0);
        REQUIRE((*t1.grad)[{1}] == 0);
        REQUIRE((*t1.grad)[{2}] == 1);
        REQUIRE((*t1.grad)[{3}] == 0);
        REQUIRE((*t1.grad)[{4}] == 1);
    }
}

TEST_CASE("The softmax method works correctly")
{
    SECTION("Softmax of a tensor with positive values")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5});
        Tensor<float> t2 = Tensor<float>::softmax(t1);
        float sum_exp = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f) + std::exp(4.0f) + std::exp(5.0f);
        REQUIRE_THAT((t2[{0}]), Catch::Matchers::WithinRel(std::exp(1.0f) / sum_exp, 0.01f));
        REQUIRE_THAT((t2[{1}]), Catch::Matchers::WithinRel(std::exp(2.0f) / sum_exp, 0.01f));
        REQUIRE_THAT((t2[{2}]), Catch::Matchers::WithinRel(std::exp(3.0f) / sum_exp, 0.01f));
        REQUIRE_THAT((t2[{3}]), Catch::Matchers::WithinRel(std::exp(4.0f) / sum_exp, 0.01f));
        REQUIRE_THAT((t2[{4}]), Catch::Matchers::WithinRel(std::exp(5.0f) / sum_exp, 0.01f));
    }

    SECTION("Softmax of a tensor with mixed values")
    {
        Tensor<float> t1 = Tensor<float>({-1, 0, 1, 2, -2});
        Tensor<float> t2 = Tensor<float>::softmax(t1);
        float sum_exp = std::exp(-1.0f) + std::exp(0.0f) + std::exp(1.0f) + std::exp(2.0f) + std::exp(-2.0f);
        REQUIRE_THAT((t2[{0}]), Catch::Matchers::WithinRel(std::exp(-1.0f) / sum_exp, 0.01f));
        REQUIRE_THAT((t2[{1}]), Catch::Matchers::WithinRel(std::exp(0.0f) / sum_exp, 0.01f));
        REQUIRE_THAT((t2[{2}]), Catch::Matchers::WithinRel(std::exp(1.0f) / sum_exp, 0.01f));
        REQUIRE_THAT((t2[{3}]), Catch::Matchers::WithinRel(std::exp(2.0f) / sum_exp, 0.01f));
        REQUIRE_THAT((t2[{4}]), Catch::Matchers::WithinRel(std::exp(-2.0f) / sum_exp, 0.01f));
    }

    SECTION("Softmax with dim != 0")
    {
        Tensor<float> t1 = Tensor<float>::ones({4, 3});
        Tensor<float> t2 = Tensor<float>::softmax(t1, 1);
        for (int i = 0; i < 4; i++)
        {
            for (int k = 0; k < 3; k++)
            {

                REQUIRE_THAT((t2[{i, k}]), Catch::Matchers::WithinRel(0.333f, 0.01f));
            }
        }
    }

    SECTION("Autograd works for softmax")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5}, true);
        Tensor<float> t2 = Tensor<float>::softmax(t1);
        Tensor<float> t3 = Tensor<float>({5, 4, 3, 2, 1});
        Tensor<float> t4 = t2 * t3;
        Tensor<float> t5 = t4.sum();
        t5.backward();
        std::vector<float> expected_grad = {0.0402, 0.0777, 0.1251, 0.1058, -0.3488};
        for (int i = 0; i < 5; i++)
        {
            REQUIRE_THAT(((*t1.grad)[{i}]), Catch::Matchers::WithinRel(expected_grad[i], 0.01f));
        }
    }
}

TEST_CASE("The log method works correctly")
{
    SECTION("Logarithm of a tensor with positive values")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5});
        Tensor<float> t2 = t1.log();
        REQUIRE_THAT((t2[{0}]), Catch::Matchers::WithinRel(std::log(1.0f), 0.01f));
        REQUIRE_THAT((t2[{1}]), Catch::Matchers::WithinRel(std::log(2.0f), 0.01f));
        REQUIRE_THAT((t2[{2}]), Catch::Matchers::WithinRel(std::log(3.0f), 0.01f));
        REQUIRE_THAT((t2[{3}]), Catch::Matchers::WithinRel(std::log(4.0f), 0.01f));
        REQUIRE_THAT((t2[{4}]), Catch::Matchers::WithinRel(std::log(5.0f), 0.01f));
    }

    SECTION("Autograd works for log")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5}, true);
        Tensor<float> t2 = t1.log();
        Tensor<float> t3 = t2.sum();
        t3.backward();
        std::vector<float> expected_grad = {1.0, 0.5, 0.3333, 0.25, 0.2};
        for (int i = 0; i < 5; i++)
        {
            REQUIRE_THAT(((*t1.grad)[{i}]), Catch::Matchers::WithinRel(expected_grad[i], 0.01f));
        }
    }
}

TEST_CASE("The cross entropy loss works correctly")
{
    SECTION("Cross entropy loss for a batch of examples")
    {
        Tensor<float> logits = Tensor<float>({1.0, 2.0, 3.0, 1.0, 2.0, 3.0}, true).view({2, 3});
        Tensor<int> target = Tensor<int>({2, 1});
        Tensor<float> loss = Tensor<float>::cross_entropy(logits, target);
        REQUIRE_THAT((loss[{0}]), Catch::Matchers::WithinRel(0.9076f, 0.01f));
    }

    SECTION("Autograd works for cross entropy loss")
    {
        Tensor<float> logits = Tensor<float>({1.0, 2.0, 3.0, 1.0, 2.0, 3.0}, true).view({2, 3});
        Tensor<int> target = Tensor<int>({2, 1});
        Tensor<float> loss = Tensor<float>::cross_entropy(logits, target);
        loss.backward();
        std::vector<float> expected_grad = {0.0450, 0.1224, -0.1674, 0.0450, -0.3776, 0.3326};
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE_THAT(((*logits.grad)[{i, j}]), Catch::Matchers::WithinRel(expected_grad[i * 3 + j], 0.01f));
            }
        }
    }
}

TEST_CASE("The xavier_normal method works correctly")
{
    SECTION("Xavier normal initialization returns a tensor with the correct distribution")
    {
        Tensor<float> t1 = Tensor<float>::xavier_normal({100, 100});
        float mean = 0;
        float std = 0;
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                mean += t1[{i, j}];
            }
        }
        mean /= (100 * 100);
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                std += (t1[{i, j}] - mean) * (t1[{i, j}] - mean);
            }
        }
        std = std::sqrt(std / (100 * 100));
        float expected_std = std::sqrt(2.0f / (100 + 100));
        REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0.0f, 0.01f));
        REQUIRE_THAT(std, Catch::Matchers::WithinRel(expected_std, 0.1f));
    }

    SECTION("Kaiming normal initialization returns a tensor with the correct distribution")
    {
        Tensor<float> t1 = Tensor<float>::kaiming_normal({100, 100});
        float mean = 0;
        float std = 0;
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                mean += t1[{i, j}];
            }
        }
        mean /= (100 * 100);
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                std += (t1[{i, j}] - mean) * (t1[{i, j}] - mean);
            }
        }
        std = std::sqrt(std / (100 * 100));
        float expected_std = std::sqrt(2.0f / 100);
        REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0.0f, 0.01f));
        REQUIRE_THAT(std, Catch::Matchers::WithinRel(expected_std, 0.1f));
    }
}

TEST_CASE("The stack method works correctly")
{
    SECTION("Stacking along the first dimension")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 3});
        Tensor<float> t2 = Tensor<float>::ones({2, 3}) * 2;
        Tensor<float> t3 = Tensor<float>::ones({2, 3}) * 3;
        Tensor<float> result = Tensor<float>::stack({t1, t2, t3}, 0);
        REQUIRE(result.size() == std::vector<size_t>({3, 2, 3}));
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    REQUIRE(result[{i, j, k}] == (i + 1));
                }
            }
        }
    }

    SECTION("Stacking along the second dimension")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 3});
        Tensor<float> t2 = Tensor<float>::ones({2, 3}) * 2;
        Tensor<float> t3 = Tensor<float>::ones({2, 3}) * 3;
        Tensor<float> result = Tensor<float>::stack({t1, t2, t3}, 1);
        REQUIRE(result.size() == std::vector<size_t>({2, 3, 3}));
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    REQUIRE(result[{i, j, k}] == (j + 1));
                }
            }
        }
    }

    SECTION("Stacking along the third dimension")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 3});
        Tensor<float> t2 = Tensor<float>::ones({2, 3}) * 2;
        Tensor<float> t3 = Tensor<float>::ones({2, 3}) * 3;
        Tensor<float> result = Tensor<float>::stack({t1, t2, t3}, 2);
        REQUIRE(result.size() == std::vector<size_t>({2, 3, 3}));
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    REQUIRE(result[{i, j, k}] == (k + 1));
                }
            }
        }
    }

    SECTION("Stacking throws an exception for tensors with different shapes")
    {
        Tensor<float> t1 = Tensor<float>::ones({2, 3});
        Tensor<float> t2 = Tensor<float>::ones({3, 3});
        CHECK_THROWS(Tensor<float>::stack({t1, t2}, 0));
    }
}

TEST_CASE("The mm method works correctly")
{
    SECTION("Matrix multiplication for small n x m matrices")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5, 6}).view({2, 3});
        Tensor<float> t2 = Tensor<float>({7, 8, 9, 10, 11, 12}).view({3, 2});
        Tensor<float> result = Tensor<float>::mm(t1, t2);
        REQUIRE(result.size() == std::vector<size_t>({2, 2}));
        REQUIRE(result[{0, 0}] == 58);
        REQUIRE(result[{0, 1}] == 64);
        REQUIRE(result[{1, 0}] == 139);
        REQUIRE(result[{1, 1}] == 154);
    }

    SECTION("Matrix multiplication for 1 x n matrix times n x m matrix")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3}).view({1, 3});
        Tensor<float> t2 = Tensor<float>({4, 5, 6, 7, 8, 9}).view({3, 2});
        Tensor<float> result = Tensor<float>::mm(t1, t2);
        REQUIRE(result.size() == std::vector<size_t>({1, 2}));
        REQUIRE(result[{0, 0}] == 40);
        REQUIRE(result[{0, 1}] == 46);
    }

    SECTION("Matrix multiplication for n x m matrix times m x 1 matrix")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5, 6}).view({2, 3});
        Tensor<float> t2 = Tensor<float>({7, 8, 9}).view({3, 1});
        Tensor<float> result = Tensor<float>::mm(t1, t2);
        REQUIRE(result.size() == std::vector<size_t>({2, 1}));
        REQUIRE(result[{0, 0}] == 50);
        REQUIRE(result[{1, 0}] == 122);
    }

    SECTION("Matrix multiplication checks the shapes correctly and throws exceptions")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5, 6}).view({2, 3});
        Tensor<float> t2 = Tensor<float>({7, 8, 9, 10}).view({2, 2});
        CHECK_THROWS(Tensor<float>::mm(t1, t2));
    }

    SECTION("Autograd works for matrix multiplication for 3 x 3 matrices")
    {
        Tensor<float> t1 = Tensor<float>({1, 2, 3, 4, 5, 6, 7, 8, 9}, true).view({3, 3});
        Tensor<float> t2 = Tensor<float>({9, 8, 7, 6, 5, 4, 3, 2, 1}, true).view({3, 3});
        Tensor<float> result = Tensor<float>::mm(t1, t2);
        Tensor<float> loss = result.sum();
        loss.backward();

        std::vector<float> expected_grad_t1 = {24, 15, 6, 24, 15, 6, 24, 15, 6};
        std::vector<float> expected_grad_t2 = {12, 12, 12, 15, 15, 15, 18, 18, 18};

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                REQUIRE((*t1.grad)[{i, j}] == expected_grad_t1[i * 3 + j]);
                REQUIRE((*t2.grad)[{i, j}] == expected_grad_t2[i * 3 + j]);
            }
        }
    }
}