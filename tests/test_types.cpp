#include <doctest/doctest.h>

#include "loomcore/types.h"

using namespace loomcore;

TEST_CASE("NamedTensor::elementCount and batchDim") {
    auto t = NamedTensor::makeFloat("x", {2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK(t.elementCount() == 6);
    CHECK(t.batchDim() == 2);
}

TEST_CASE("NamedTensor::sliceBatch extracts the right rows") {
    auto t = NamedTensor::makeFloat("x", {3, 2}, {1, 2, 3, 4, 5, 6});
    auto s = t.sliceBatch(1, 3);
    CHECK(s.shape == std::vector<int64_t>{2, 2});
    CHECK(s.f32 == std::vector<float>{3, 4, 5, 6});
}

TEST_CASE("concatBatch merges rows in order") {
    auto a = NamedTensor::makeFloat("x", {1, 2}, {1, 2});
    auto b = NamedTensor::makeFloat("x", {2, 2}, {3, 4, 5, 6});
    auto merged = concatBatch({a, b});
    CHECK(merged.shape == std::vector<int64_t>{3, 2});
    CHECK(merged.f32 == std::vector<float>{1, 2, 3, 4, 5, 6});
}

TEST_CASE("concatBatch rejects mismatched non-batch dimensions") {
    auto a = NamedTensor::makeFloat("x", {1, 2}, {1, 2});
    auto b = NamedTensor::makeFloat("x", {1, 3}, {1, 2, 3});
    CHECK_THROWS_AS(concatBatch({a, b}), LoomcoreError);
}

TEST_CASE("sliceBatch is the exact inverse of concatBatch for equal-sized parts") {
    auto a = NamedTensor::makeFloat("x", {1, 3}, {1, 2, 3});
    auto b = NamedTensor::makeFloat("x", {1, 3}, {4, 5, 6});
    auto merged = concatBatch({a, b});
    CHECK(merged.sliceBatch(0, 1).f32 == a.f32);
    CHECK(merged.sliceBatch(1, 2).f32 == b.f32);
}

TEST_CASE("argmax finds the index and value of the largest element") {
    auto t = NamedTensor::makeFloat("logits", {1, 4}, {0.1f, 3.2f, -1.0f, 2.9f});
    auto [idx, val] = argmax(t);
    CHECK(idx == 1);
    CHECK(val == doctest::Approx(3.2f));
}
