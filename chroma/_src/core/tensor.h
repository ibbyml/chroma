#pragma once

#include "bfloat16.h"
#include "layout.h"
#include <algorithm>
#include <fstream>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace chroma {
class Buffer {
public:
    explicit Buffer(size_t bytes)
        : size_(bytes),
          data_(static_cast<std::byte*>(::operator new(std::max(bytes, size_t{1}), std::align_val_t{64}))) {}
    ~Buffer() { ::operator delete(data_, std::align_val_t{64}); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    std::byte* data() { return data_; }
    const std::byte* data() const { return data_; }
    size_t size() const { return size_; }

private:
    size_t size_;
    std::byte* data_;
};

inline void read_weights(const char* filename, void* data, size_t bytes) {
    std::ifstream stream(filename, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamoff>(bytes)) {
        throw std::runtime_error("Missing weights file or incorrect byte size");
    }
    stream.seekg(0);
    if (!stream.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("Cannot read weights");
    }
}

template <class T, auto Extents> struct TensorView {
    using value_type = std::remove_const_t<T>;
    static constexpr auto shape = Extents;
    T* data;
    value_type operator[](size_t i) const { return data[i]; }
};

template <class T, auto Extents> auto view(T* data) {
    return TensorView<T, Extents>{data};
}
}
