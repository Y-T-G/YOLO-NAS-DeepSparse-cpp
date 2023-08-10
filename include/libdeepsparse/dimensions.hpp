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

// Configuration C++ API for DeepSparse.

#include "libdeepsparse/compiler.hpp"

#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

namespace deepsparse
{

/// The tensor shape along each dimension.
class EXPORT dimensions_t : private std::vector<uint64_t>
{
public:
    /// Unsigned 64 bit integer representing extent of a dimension.
    using dim_t = uint64_t;

private:
    using base = std::vector<dim_t>;

public:
    /// Default constructor for empty dimensions_t.
    dimensions_t();

    /// Construct dimension instance with specified rank.
    ///
    /// Dimension extent values are not set in this constructor.
    /// Specific shape needs to be specified later.
    explicit dimensions_t(size_t rank);

    /// Construct shapes from data in collection.
    explicit dimensions_t(std::initializer_list<dim_t> data);

    /// Construct from specified shapes in collection.
    explicit dimensions_t(std::vector<uint64_t>&& data);

    /// Construct shapes from data in sub-range of collection.
    template <typename Itr>
    dimensions_t(Itr begin, Itr end)
        : base(begin, end)
    {
    }

    dimensions_t(const dimensions_t&)            = default;
    dimensions_t(dimensions_t&&)                 = default;
    dimensions_t& operator=(const dimensions_t&) = default;
    dimensions_t& operator=(dimensions_t&&)      = default;

    using base::at;
    using base::begin;
    using base::empty;
    using base::end;
    using base::size;

    bool operator==(dimensions_t const& other) const;
    bool operator!=(dimensions_t const& other) const;

    /// Return the number of dimensions.
    size_t rank() const;

    /// Return the number of elements along axis specified by index.
    dim_t operator[](size_t index) const;

    /// Return total number of elements in the tensor
    /// i.e.  d[0] * d[1] * .. d[rank-1]
    size_t total_num_elements() const;
};

/// DeepSparse only supports statically shaped tensors,
/// if the tensor has dynamic shape this function returns std::nullopt.
using maybe_dimensions_t = std::optional<dimensions_t>;

} // namespace deepsparse
