#include "codegen/cpu/kernels.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace chroma;

void check(bool condition) {
    if (!condition) {
        throw std::runtime_error("Runtime core check failed");
    }
}

template <class F> void throws(F function) {
    bool caught = false;
    try {
        function();
    } catch (const std::exception&) {
        caught = true;
    }
    check(caught);
}

void check_bfloat16() {
    using BF16 = chroma::bfloat16_t;
    const std::pair<uint32_t, uint16_t> conversions[] = {
        {0x00000000, 0x0000}, {0x80000000, 0x8000}, {0x3f808000, 0x3f80}, {0x3f818000, 0x3f82},
        {0xbf808000, 0xbf80}, {0x00008000, 0x0000}, {0x00010000, 0x0001}, {0x7f800000, 0x7f80},
        {0xff800000, 0xff80}, {0x7f800001, 0x7fc0}, {0x7fffffff, 0x7fc0}, {0x7f7fffff, 0x7f80},
    };
    for (auto [source, bits] : conversions) {
        BF16 value(std::bit_cast<float>(source));
        check(value.bits == bits);
        check(std::bit_cast<uint32_t>(float(value)) == uint32_t{bits} << 16);
    }
    float one[] = {1.0f};
    auto value = view<const float, dims<1>>(one);
    auto rounded = astype<BF16>(value + Scalar<float>{1.0f / 256});
    check(float(astype<BF16>(rounded + Scalar<float>{1.0f / 256})[0]) == 1.0f);

    BF16 stored[2];
    stored[0] = std::bit_cast<BF16>(uint16_t{0x8000});
    stored[1] = std::bit_cast<BF16>(uint16_t{0x7fa1});
    BF16 copied[2];
    auto original = view<const BF16, dims<2>>(stored);
    evaluate(view<BF16, dims<2>>(copied), original);
    check(copied[0].bits == 0x8000 && copied[1].bits == 0x7fa1);

    BF16 input[] = {1, 1, 1};
    BF16 weights[] = {256, 1, -256, 1, 1.0f / 256, 0};
    BF16 bias[] = {0, 1.0f / 256}, result[4];
    linear(result, input, weights, bias, 1, 2, 3);
    check(float(result[0]) == 1 && float(result[1]) == 1.0f + 1.0f / 128);

    BF16 transposed[] = {1, 4, 2, 5, 3, 6};
    BF16 matrix[] = {1, 0, 0, 1, 0, 0};
    gemm(result, transposed, matrix, 2, 2, 3, 2, 2, 2, true, false);
    check(float(result[0]) == 1 && float(result[1]) == 2 && float(result[2]) == 4 && float(result[3]) == 5);

    BF16 rows[] = {1, 2, 3, 4, 5, 6}, experts[] = {1, 0, 0, 0, 1, 0};
    int64_t indices[] = {-1, 0};
    expert_linear(result, experts, indices, rows, 1, 2, 2, 1, 3, false);
    check(float(result[0]) == 2 && float(result[1]) == 1);
    expert_linear(result, experts, indices, rows, 1, 2, 2, 1, 3, true);
    check(float(result[0]) == 2 && float(result[1]) == 4);
    indices[0] = 2;
    throws([&] { expert_linear(result, experts, indices, rows, 1, 2, 2, 1, 3, false); });

    BF16 probabilities[3];
    auto row = view<const BF16, dims<3>>(rows);
    softmax<0>(view<BF16, dims<3>>(probabilities), row);
    const float total = std::exp(-2.0f) + std::exp(-1.0f) + 1;
    for (size_t i = 0; i < 3; ++i) {
        check(probabilities[i].bits == BF16(std::exp(float(rows[i]) - 3) / total).bits);
    }
    topk<0, true, true>(view<BF16, dims<2>>(result), view<int64_t, dims<2>>(indices), row);
    check(float(result[0]) == 3 && float(result[1]) == 2 && indices[0] == 2 && indices[1] == 1);

    BF16 triangle[6];
    triangular<true, 0>(view<BF16, dims<2, 3>>(triangle), view<const BF16, dims<2, 3>>(rows));
    check(float(triangle[3]) == 0 && float(triangle[4]) == 5 && float(triangle[5]) == 6);
}

int main() {
    check_bfloat16();
    Buffer storage(6 * sizeof(float));
    check(storage.size() == 6 * sizeof(float));
    check(reinterpret_cast<uintptr_t>(storage.data()) % 64 == 0);
    auto* x = reinterpret_cast<float*>(storage.data());
    float row[3], out[6];
    for (size_t i = 0; i < 6; ++i) {
        x[i] = static_cast<float>(i);
    }
    for (size_t i = 0; i < 3; ++i) {
        row[i] = static_cast<float>(i + 1);
    }
    auto a = view<const float, dims<2, 3>>(x);
    auto b = view<const float, dims<3>>(row);
    auto destination = view<float, dims<2, 3>>(out);
    evaluate(destination, (a + b) * Scalar<float>{2.0f});
    for (size_t i = 0; i < 6; ++i) {
        check(out[i] == (x[i] + row[i % 3]) * 2);
    }
    auto transposed = reindex<dims<3, 2>, dims<1, 3>>(a);
    check(transposed[1] == 3 && transposed[4] == 2);
    auto broadcast = reindex<dims<2, 3>, dims<0, 1>>(b);
    check(broadcast[5] == 3);
    check(numel(dims<0, 3>) == 0);
    check(numel(dims<>) == 1);

    int64_t indices[2];
    indices[0] = -1;
    indices[1] = 0;
    auto index = view<const int64_t, dims<2>>(indices);
    auto selected = gather<dims<2, 3>>(a, index);
    check(selected[0] == x[3] && selected[3] == x[0]);
    indices[0] = 2;
    throws([&] { (void)selected[0]; });
    indices[0] = -1;
    auto embedding = gather<dims<2, 3>, false>(a, index);
    throws([&] { (void)embedding[0]; });

    float input[] = {1, 2, 3, 4, 5, 6};
    float weights[] = {1, 0, 0, 0, 1, 0};
    float bias[] = {1, -1}, result[4];
    linear(result, input, weights, bias, 2, 2, 3);
    check(result[0] == 2 && result[1] == 1 && result[2] == 5 && result[3] == 4);
    auto softmax_input = view<const float, dims<2, 3>>(input);
    softmax<1>(destination, softmax_input);
    check(std::abs(out[0] + out[1] + out[2] - 1.0f) < 1e-6f);
    std::cout << "Runtime core checks passed\n";
}
