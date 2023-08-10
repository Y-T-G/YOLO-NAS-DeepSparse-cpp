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

#include <memory>
#include <string>

namespace deepsparse
{

class engine_context_t;

/// Default value for the number of concurrent requests expected for an engine.
/// The actual default is scheduler-based.
EXPORT size_t num_streams_default();

EXPORT std::string_view scheduler_default();

/// Specifies the engine configuration.
struct EXPORT engine_config_t
{
    /// Path to the onnx model file.
    std::string model_file_path;

    /// The number of tensors to combine along the batch dimension.
    ///
    /// Parameter controlling the number of tensors concatenated so they are
    /// contiguous in memory. Increasing this number enables more parallelism.
    size_t batch_size = 1;

    // Request the number of threads to run the engine across.
    // 0 indicates the default number of threads.
    size_t num_threads = 0;

    // Number of concurrent requests to support. Default depends
    // on the scheduler.
    size_t num_streams = num_streams_default();

    // The scheduler type to use.  single_stream (default), multi_stream, or
    // elastic.
    std::string scheduler{scheduler_default()};
};

struct EXPORT multi_engine_config_t
{
    // The path to the onnx model file.
    std::string model_file_path;

    // An important tuning parameter of the engine is batching.
    // A batched tensor is the concatenation of each tensors so
    // they are contiguous in memory.
    int batch_size = 1;

    std::shared_ptr<engine_context_t> engine_context;
};
} // namespace deepsparse
