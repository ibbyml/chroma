#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace chroma {
template <int64_t... Sizes> inline constexpr auto dims = std::array<int64_t, sizeof...(Sizes)>{Sizes...};

// Shapes are fixed and nonempty by the time they get here.
template <class S> constexpr size_t numel(const S& shape) {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    return n;
}

template <auto A, auto B> consteval auto broadcast_shape() {
    constexpr size_t rank = A.size() > B.size() ? A.size() : B.size();
    std::array<int64_t, rank> res{};
    for (size_t i = 0; i < rank; ++i) {
        const int64_t a = i < rank - A.size() ? 1 : A[i - (rank - A.size())];
        const int64_t b = i < rank - B.size() ? 1 : B[i - (rank - B.size())];
        if (a != b && a != 1 && b != 1) {
            throw "Incompatible broadcast dim";
        }
        res[i] = a == 1 ? b : a;
    }
    return res;
}

template <auto From, auto To> inline size_t broadcast_index(size_t i) {
    if constexpr (From.size() == To.size()) {
        if constexpr (From == To) {
            return i;
        }
    }
    size_t offset = 0, stride = 1;
    for (size_t d = To.size(); d-- > 0;) {
        const size_t coordinate = i % To[d];
        i /= To[d];
        if (d >= To.size() - From.size()) {
            const auto extent = From[d - (To.size() - From.size())];
            if (extent != 1) {
                offset += coordinate * stride;
            }
            stride *= extent;
        }
    }
    return offset;
}
}
