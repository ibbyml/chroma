#pragma once

#include <bit>
#include <cstdint>

namespace chroma {
struct bfloat16_t {
    uint16_t bits;

    bfloat16_t() = default;
    constexpr bfloat16_t(float value) {
        const auto word = std::bit_cast<uint32_t>(value);
        bits = (word & 0x7fffffff) > 0x7f800000 ? 0x7fc0
                                                : static_cast<uint16_t>((word + 0x7fff + ((word >> 16) & 1)) >> 16);
    }
    constexpr operator float() const { return std::bit_cast<float>(uint32_t{bits} << 16); }
};
static_assert(sizeof(bfloat16_t) == 2 && alignof(bfloat16_t) == 2);
}
