#pragma once

#include "codegen/cpu/expressions.h"
#include <algorithm>
#include <cmath>
#include <limits>

#if defined(CHROMA_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(CHROMA_CBLAS)
#include <cblas.h>
#endif

namespace chroma {
// Elements after Axis: the stride between neighbors along it in a contiguous tensor.
template <auto S, int Axis> consteval size_t inner_size() {
    size_t n = 1;
    for (size_t d = Axis + 1; d < S.size(); ++d) {
        n *= S[d];
    }
    return n;
}

template <class T>
void gemm(T* out, const T* a, const T* b, int rows, int columns, int inner, int lda, int ldb, int ldc, bool trans_a,
          bool trans_b) {
#if defined(CHROMA_ACCELERATE) || defined(CHROMA_CBLAS)
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans, rows,
                    columns, inner, 1.0f, a, lda, b, ldb, 0.0f, out, ldc);
        return;
    }
#endif
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            float sum = 0;
            for (int k = 0; k < inner; ++k) {
                sum += float(a[trans_a ? size_t(k) * lda + i : size_t(i) * lda + k]) *
                       float(b[trans_b ? size_t(j) * ldb + k : size_t(k) * ldb + j]);
            }
            out[size_t(i) * ldc + j] = sum;
        }
    }
}

inline void linear(float* out, const float* input, const float* weight, const float* bias, int rows, int columns,
                   int inner) {
#if defined(CHROMA_ACCELERATE) || defined(CHROMA_CBLAS)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, columns, inner, 1.0f, input, inner, weight, inner, 0.0f,
                out, columns);
#else
    gemm(out, input, weight, rows, columns, inner, inner, inner, columns, false, true);
#endif
    if (bias) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < columns; ++j) {
                out[static_cast<size_t>(i) * columns + j] += bias[j];
            }
        }
    }
}

inline void linear(bfloat16_t* out, const bfloat16_t* input, const bfloat16_t* weight, const bfloat16_t* bias, int rows,
                   int columns, int inner) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            float sum = 0;
            for (int k = 0; k < inner; ++k) {
                sum += float(input[size_t(i) * inner + k]) * float(weight[size_t(j) * inner + k]);
            }
            out[size_t(i) * columns + j] = sum + (bias ? float(bias[j]) : 0.0f);
        }
    }
}

template <class T>
void expert_linear(T* out, const T* weight, const int64_t* indices, const T* input, int tokens, int selected,
                   int experts, int columns, int inner, bool per_expert) {
    for (int b = 0; b < tokens; ++b) {
        for (int e = 0; e < selected; ++e) {
            const size_t row = static_cast<size_t>(b) * selected + e;
            const int64_t raw = indices[row];
            const int64_t index = raw < 0 ? raw + experts : raw;
            if (index < 0 || index >= experts) {
                throw std::out_of_range("Expert idx out of range");
            }
            const T* w = weight + static_cast<size_t>(index) * columns * inner;
            const T* x = input + (per_expert ? row : static_cast<size_t>(b)) * inner;
            T* y = out + row * columns;
#if defined(CHROMA_ACCELERATE) || defined(CHROMA_CBLAS)
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, columns, inner, 1.0f, w, inner, x, 1, 0.0f, y, 1);
                continue;
            }
#endif
            linear(y, x, w, nullptr, 1, columns, inner);
        }
    }
}

template <class T, auto S, Expression E, Expression W>
void rmsnorm(TensorView<T, S> out, E x, W weight, float epsilon) {
    constexpr size_t columns = S.back();
    for (size_t row = 0; row < numel(S) / columns; ++row) {
        float sum = 0.0f;
        for (size_t j = 0; j < columns; ++j) {
            float v = x[row * columns + j];
            sum += v * v;
        }
        float scale = 1.0f / std::sqrt(sum / float(columns) + epsilon);
        for (size_t j = 0; j < columns; ++j) {
            out.data[row * columns + j] = float(x[row * columns + j]) * scale * float(weight[j]);
        }
    }
}

template <bool Upper, int64_t Diagonal, class T, auto S, Expression E> void triangular(TensorView<T, S> out, E x) {
    constexpr size_t columns = S.back(), rows = S[S.size() - 2];
    for (size_t i = 0; i < numel(S); ++i) {
        const int64_t diagonal = int64_t(i % columns) - int64_t(i / columns % rows);
        out.data[i] = (Upper ? diagonal >= Diagonal : diagonal <= Diagonal) ? static_cast<T>(x[i]) : T(0);
    }
}

template <int Axis, class T, auto S, Expression E> void softmax(TensorView<T, S> out, E x) {
    static_assert(S == E::shape);
    constexpr size_t width = S[Axis];
    constexpr size_t inner = inner_size<S, Axis>();
    for (size_t base = 0; base < numel(S) / width; ++base) {
        const size_t start = base / inner * width * inner + base % inner;
        float maximum = -std::numeric_limits<float>::infinity();
        for (size_t k = 0; k < width; ++k) {
            maximum = std::max(maximum, float(x[start + k * inner]));
        }
        float sum = 0;
        for (size_t k = 0; k < width; ++k) {
            const float value = std::exp(float(x[start + k * inner]) - maximum);
            if constexpr (std::is_same_v<T, float>) {
                out.data[start + k * inner] = value;
            }
            sum += value;
        }
        for (size_t k = 0; k < width; ++k) {
            if constexpr (std::is_same_v<T, float>) {
                out.data[start + k * inner] /= sum;
            } else {
                out.data[start + k * inner] = std::exp(float(x[start + k * inner]) - maximum) / sum;
            }
        }
    }
}

template <int Axis, bool Largest, bool Sorted, class T, auto S, Expression E>
void topk(TensorView<T, S> values, TensorView<int64_t, S> indices, E x) {
    constexpr size_t width = E::shape[Axis], k = S[Axis];
    constexpr size_t inner = inner_size<S, Axis>();
    static_assert(k <= width);
    auto compare = [](const auto& a, const auto& b) {
        if constexpr (Largest) {
            return (std::isnan(a.first) && !std::isnan(b.first)) || a.first > b.first;
        } else {
            return (!std::isnan(a.first) && std::isnan(b.first)) || a.first < b.first;
        }
    };
    std::array<std::pair<float, int64_t>, width> scratch;
    for (size_t base = 0; base < numel(S) / k; ++base) {
        const size_t src = base / inner * width * inner + base % inner;
        const size_t dst = base / inner * k * inner + base % inner;
        for (size_t j = 0; j < width; ++j) {
            scratch[j] = {x[src + j * inner], static_cast<int64_t>(j)};
        }
        if constexpr (k * 64 <= width) {
            std::partial_sort(scratch.begin(), scratch.begin() + k, scratch.end(), compare);
        } else {
            std::nth_element(scratch.begin(), scratch.begin() + k - 1, scratch.end(), compare);
            if constexpr (Sorted) {
                std::sort(scratch.begin(), scratch.begin() + k - 1, compare);
            }
        }
        for (size_t j = 0; j < k; ++j) {
            values.data[dst + j * inner] = scratch[j].first;
            indices.data[dst + j * inner] = scratch[j].second;
        }
    }
}

template <int Axis, size_t Start, class T, auto S, Expression E> void concat_part(TensorView<T, S> out, E x) {
    constexpr size_t inner = inner_size<S, Axis>();
    constexpr size_t block = E::shape[Axis] * inner;
    for (size_t i = 0; i < numel(E::shape); ++i) {
        out.data[i / block * S[Axis] * inner + Start * inner + i % block] = x[i];
    }
}
}
