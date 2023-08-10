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

// Stopwatch holds functions for easily timing sections of code.

#include <chrono>

namespace deepsparse
{

/// The common recorded time format before it is converted to human-readable
/// units.
using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

/// Run the function 'func' for 'iters' times, returning the average time per
/// iteration in milliseconds.
template <typename F>
inline double measure_msecs(std::size_t iters, F&& func)
{
    auto begin = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < iters; ++i)
    {
        func();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    return static_cast<double>(duration) / 1000 / iters;
}

/// Run the function 'func' for 'iters' times with warmup iterations
/// beforehand, returning the average time per iteration in milliseconds.
template <typename F>
inline double
measure_msecs_with_warmup(std::size_t iters, std::size_t warmup_iters, F&& func)
{
    measure_msecs(warmup_iters, func);
    return measure_msecs(iters, func);
}

/// Run the function 'func' for 'iters' times with warmup iterations
/// beforehand, returning the average time per iteration in milliseconds.
template <typename F>
inline double measure_msecs_with_warmup(std::size_t iters, F&& func)
{
    return measure_msecs_with_warmup(iters, iters, func);
}

/// Return the time point for the current time.
inline auto now() { return std::chrono::high_resolution_clock::now(); }

/// Return the milliseconds elapsed between two time points.
inline double milliseconds_between(time_point const& start,
                                   time_point const& end)
{
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    return static_cast<double>(duration) / 1000;
}

/// Return the milliseconds elapsed since the provided time point.
inline double milliseconds_since(time_point const& start)
{
    auto end = std::chrono::high_resolution_clock::now();
    return milliseconds_between(start, end);
}

} // namespace deepsparse