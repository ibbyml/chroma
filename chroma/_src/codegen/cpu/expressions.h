#pragma once

#include "core/tensor.h"
#include <functional>
#include <type_traits>

namespace chroma {
template <class E>
concept Expression = requires(E e, size_t i) {
    E::shape;
    typename E::value_type;
    e[i];
};

template <class T> struct Scalar {
    using value_type = T;
    static constexpr auto shape = dims<>;
    T value;
    T operator[](size_t) const { return value; }
};

template <Expression E, class F> struct Unary {
    using value_type = std::invoke_result_t<F, typename E::value_type>;
    static constexpr auto shape = E::shape;
    E input;
    F op;
    value_type operator[](size_t i) const { return op(input[i]); }
};
template <Expression E, class F> auto unary(E e, F f) {
    return Unary<E, F>{e, f};
}
template <class T, Expression E> auto astype(E e) {
    return unary(e, [](auto value) { return static_cast<T>(value); });
}

template <Expression A, Expression B, class F> struct Binary {
    using value_type = std::invoke_result_t<F, typename A::value_type, typename B::value_type>;
    static constexpr auto shape = broadcast_shape<A::shape, B::shape>();
    A lhs;
    B rhs;
    F op;
    value_type operator[](size_t i) const {
        return op(lhs[broadcast_index<A::shape, shape>(i)], rhs[broadcast_index<B::shape, shape>(i)]);
    }
};
template <Expression A, Expression B, class F> auto binary(A a, B b, F f) {
    return Binary<A, B, F>{a, b, f};
}
template <Expression A, Expression B> auto operator+(A a, B b) {
    return binary(a, b, std::plus<>{});
}
template <Expression A, Expression B> auto operator-(A a, B b) {
    return binary(a, b, std::minus<>{});
}
template <Expression A, Expression B> auto operator*(A a, B b) {
    return binary(a, b, std::multiplies<>{});
}
template <Expression A, Expression B> auto operator/(A a, B b) {
    return binary(a, b, std::divides<>{});
}

template <auto Extents, auto Strides, size_t Offset, Expression E> struct Reindex {
    using value_type = typename E::value_type;
    static constexpr auto shape = Extents;
    E input;
    value_type operator[](size_t i) const {
        size_t offset = Offset;
        for (size_t d = shape.size(); d-- > 0;) {
            offset += (i % shape[d]) * Strides[d];
            i /= shape[d];
        }
        return input[offset];
    }
};
template <auto S, auto Strides, size_t Offset = 0, Expression E> auto reindex(E e) {
    return Reindex<S, Strides, Offset, E>{e};
}

template <auto Extents, Expression E> struct Reshape {
    using value_type = typename E::value_type;
    static constexpr auto shape = Extents;
    static_assert(numel(shape) == numel(E::shape));
    E input;
    value_type operator[](size_t i) const { return input[i]; }
};
template <auto S, Expression E> auto reshape(E e) {
    return Reshape<S, E>{e};
}

template <auto Extents, Expression E, Expression I, bool AllowNegative> struct Gather {
    using value_type = typename E::value_type;
    static constexpr auto shape = Extents;
    static constexpr size_t width = [] {
        auto s = E::shape;
        s[0] = 1;
        return numel(s);
    }();
    E input;
    I indices;
    value_type operator[](size_t i) const {
        int64_t row = indices[i / width];
        if constexpr (AllowNegative) {
            if (row < 0) {
                row += E::shape[0];
            }
        }
        if (row < 0 || row >= E::shape[0]) {
            throw std::out_of_range("Tensor idx out of range");
        }
        return input[static_cast<size_t>(row) * width + i % width];
    }
};
template <auto S, bool AllowNegative = true, Expression E, Expression I> auto gather(E e, I indices) {
    return Gather<S, E, I, AllowNegative>{e, indices};
}

template <class T, auto S, Expression E> void evaluate(TensorView<T, S> out, E e) {
    static_assert(S == E::shape, "Assignment shape mismatch");
    for (size_t i = 0; i < numel(S); ++i) {
        out.data[i] = e[i];
    }
}
}
