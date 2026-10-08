#pragma once

#include "core/tensor.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <memory>
#include <vector>

namespace chroma {
namespace py = pybind11;

struct TensorSpec {
    std::vector<py::ssize_t> shape;
    const char* dtype;
};

inline py::array checked_array(py::handle value, const TensorSpec& spec, bool output) {
    if (!py::isinstance<py::array>(value)) {
        throw py::type_error("not a numpy array");
    }
    auto array = py::reinterpret_borrow<py::array>(value);
    auto dtype = py::dtype(spec.dtype);
    if (!array.dtype().equal(dtype) || array.ndim() != static_cast<py::ssize_t>(spec.shape.size()) ||
        !std::equal(spec.shape.begin(), spec.shape.end(), array.shape())) {
        throw py::value_error("shape or dtype");
    }
    const size_t alignment = static_cast<size_t>(dtype.alignment());
    if (!(array.flags() & py::array::c_style) || reinterpret_cast<uintptr_t>(array.data()) % alignment ||
        (output && !array.writeable())) {
        throw py::value_error("want C-contiguous, aligned, writable outputs");
    }
    return array;
}

template <class Model, class Factory>
void bind_model(py::module_& module, Factory factory, std::vector<TensorSpec> input_specs,
                std::vector<TensorSpec> output_specs) {
    py::class_<Model>(module, "Model", py::module_local())
        .def(py::init(factory))
        .def_property_readonly("device", &Model::device_name)
        .def("run", [input_specs, output_specs](Model& model, py::list inputs, py::list outputs) {
            if (inputs.size() != input_specs.size() || outputs.size() != output_specs.size()) {
                throw py::value_error("wrong number of arrays");
            }
            std::vector<const void*> input_ptrs;
            std::vector<void*> output_ptrs;
            std::vector<std::pair<uintptr_t, uintptr_t>> regions;
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto array = checked_array(inputs[i], input_specs[i], false);
                const auto begin = reinterpret_cast<uintptr_t>(array.data());
                input_ptrs.push_back(array.data());
                regions.emplace_back(begin, begin + array.nbytes());
            }
            for (size_t i = 0; i < outputs.size(); ++i) {
                auto array = checked_array(outputs[i], output_specs[i], true);
                const auto begin = reinterpret_cast<uintptr_t>(array.data());
                const auto end = begin + array.nbytes();
                for (auto [first, last] : regions) {
                    if (begin < last && first < end) {
                        throw py::value_error("output overlap");
                    }
                }
                output_ptrs.push_back(array.mutable_data());
                regions.emplace_back(begin, end);
            }
            py::gil_scoped_release release;
#ifdef __OBJC__
            @autoreleasepool {
                model.run(input_ptrs.data(), output_ptrs.data());
            }
#else
            model.run(input_ptrs.data(), output_ptrs.data());
#endif
        });
}
}
