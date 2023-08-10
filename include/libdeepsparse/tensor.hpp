#pragma once

// Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The tensor object holds a multidimensional array of elements.

#include "libdeepsparse/dimensions.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace deepsparse
{

/// DeepSparse supported tensor elements.
enum class element_type_t
{
    invalid,

    boolean,

    int8,
    int16,
    int32,
    int64,

    uint8,

    float32,
    float64,

    num_types

};

/// @return Minimal bytes needed for tensor not including any padding for
/// alignment.
EXPORT size_t required_bytes(element_type_t e, dimensions_t const& dims);

/// @return Minimal bytes needed for alignment.
EXPORT size_t minimum_required_alignment();

/// Custom deallocator of p_data_memory.
/// Use this to free memory and any other side effects during destruction.
using dealloc_fn = std::function<void(void*)>;

/// tensor_t points to a multidimensional array of data elements.
/// Element type and array dimensions and memory are passed into the
/// constructor.
class EXPORT tensor_t
{
public:
    /// Manually construct a tensor_t pointing to external data memory.
    ///
    /// @param element_type tensor element type
    ///
    /// @param dims tensor shape
    ///
    /// @param p_data_memory  Raw pointer to external memory
    /// aligned to minimum_required_alignment() that will be used to store this
    /// tensor.
    ///
    /// If the tensor assumes memory ownership, a dealloc data function must be
    /// provided. A no-op dealloc function is provided by default so nothing
    /// happens to p_data_memory on destruction.
    ///
    /// @param dealloc_data Function invoked on destruction, used for memory
    /// clean up.
    ///
    /// @throw std::runtime_error() if input arguments are out of range.
    tensor_t(
        element_type_t      element_type,
        dimensions_t const& dims,
        void*               p_data_memory,
        dealloc_fn          dealloc_data = [](void* p) {});

    /// Default construct an empty tensor.
    tensor_t();

    /// Destroy the tensor.
    ///
    ///  Destructor will invoke dealloc_data provided at construction when the
    ///  last reference to underlying data is removed.
    ~tensor_t() = default;

    /// @return Element type of a tensor. @see element_type_t
    element_type_t element_type() const;

    /// @return Tensor shape.
    dimensions_t const& dims() const;

    /// @return Read-only access to tensor data.
    template <typename T>
    T const* data() const
    {
        return static_cast<T const*>(data_.get());
    }

    /// @return Read/Write access to tensor data
    template <typename T>
    T* data()
    {
        return static_cast<T*>(data_.get());
    }

    /// @return Number of dimensions for tensor.
    size_t rank() const;

private:
    element_type_t element_type_ = element_type_t::invalid;

    dimensions_t dims_;

    std::shared_ptr<void> data_;
};

/// Create an uninitialized tensor of element type with specified shape.
///
/// Users may implement similar functionality, but all memory needs to be
/// aligned to minimum_required_alignment().
///
/// @param element_type Tensor element type. @see element_type_t
///
/// @param dims Tensor shape
///
/// @throw std::bad_alloc on out of memory
EXPORT tensor_t create_tensor(element_type_t      element_type,
                              dimensions_t const& dims);

/// A vector of tensors.
using tensors_t = std::vector<tensor_t>;

} // namespace deepsparse
