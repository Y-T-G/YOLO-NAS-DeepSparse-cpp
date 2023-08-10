/*
Licensed under the MIT License < http://opensource.org/licenses/MIT>.
SPDX - License - Identifier : MIT
Copyright(c) 2023 Mohammed Yasin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "processing.hpp"
#include "yolo-nas.hpp"
#include "utils.hpp"
#include "draw.hpp"
#include "libdeepsparse/engine.hpp"


YoloNAS::YoloNAS(std::string modelPath, std::vector<int> imgsz, bool gpu, float score, float iou)
    : postprocessor(score, iou, 1000, 300, false) // define postprocessor
{
    size_t batchSize = 1;
    size_t numThreads = 1;

    deepsparse::engine_config_t config{modelPath, batchSize, numThreads};

    engine = std::make_shared<deepsparse::engine_t>(deepsparse::engine_t(config));

    modelInputShape = deepsparse::dimensions_t({ 1, 3, static_cast<uint64_t>(imgsz[1]), static_cast<uint64_t>(imgsz[0]) });

    scoreTresh = score;
    iouTresh = iou;

}

void YoloNAS::letterbox(cv::Mat& source, cv::Mat& dst, std::vector<float>& ratios)
{
    // padding image to [n x n] dim
    int maxSize = std::max(source.cols, source.rows);
    int xPad = maxSize - source.cols;
    int yPad = maxSize - source.rows;
    float xRatio = (float)maxSize / (float)modelInputShape[3];
    float yRatio = (float)maxSize / (float)modelInputShape[2];

    cv::copyMakeBorder(source, dst, 0, yPad, 0, xPad, cv::BORDER_CONSTANT); // padding black

    cv::resize(dst, dst, cv::Size(modelInputShape[3], modelInputShape[2]), 0, 0, cv::INTER_NEAREST);

    ratios.push_back(xRatio);
    ratios.push_back(yRatio);
}

// Adapted from https://github.com/mgoin/YOLOv5-and-DeepSparse-in-CPP/blob/b505737a34cefdcba2d007fe47347406c2ea73a8/yolov5.cpp#L113C5-L113C5
std::vector<deepsparse::tensor_t> YoloNAS::pre_process_deepsparse(cv::Mat& blob, bool is_quantized = false)
{
    std::vector<deepsparse::tensor_t> inputs;

    // if model is not quantized, expects inputs from 0->1 as a float
    if (!is_quantized) {
        cv::dnn::blobFromImage(blob, blob, 1. / 255., cv::Size(), cv::Scalar(), true, false);
        assert(blob.isContinuous());

        // creates tensor with the blob raw data        
        deepsparse::tensor_t input(deepsparse::element_type_t::float32, deepsparse::dimensions_t(modelInputShape), blob.data, [](void* p) {});
        inputs.push_back(input);

        // if model is quantized, expects inputs from 0-255 as uint8_t
    }
    else {
        cv::dnn::blobFromImage(blob, blob, 1., cv::Size(), cv::Scalar(), true, false);
        assert(blob.isContinuous());

        // HACK: round down floats to uint8         
        float* float_data = reinterpret_cast<float*>(blob.data);
        uint8_t uint8_data[3 * modelInputShape[2] * modelInputShape[2]];
        for (int i = 0; i < 3 * modelInputShape[2] * modelInputShape[2]; i++) {
            assert(float_data[i] <= 255.0 && float_data[i] >= 0.);  // confirms pixel values that fit in uint8
            uint8_data[i] = static_cast<uint8_t>(float_data[i]);    // cast to uint8, rounding down
        }

        // creates tensor with the uint8 version of the blob data
        deepsparse::tensor_t input(deepsparse::element_type_t::uint8, deepsparse::dimensions_t(modelInputShape), uint8_data, [](void* p) {});
        inputs.push_back(input);
    }

    return inputs;
}

void YoloNAS::predict(cv::Mat& img)
{
    try
    {
        cv::Mat blob;
        std::vector<float> ratios;
        letterbox(img, blob, ratios);

        std::vector<deepsparse::tensor_t> inputs = pre_process_deepsparse(blob);

        auto output_tensors = engine->execute(inputs);

        float* bboxes = output_tensors[0].data<float>();
        float* scores = output_tensors[1].data<float>();

        std::vector<int> output_shape_bboxes(output_tensors[0].dims().begin(), output_tensors[0].dims().end());
        std::vector<int> output_shape_scores(output_tensors[1].dims().begin(), output_tensors[1].dims().end());

        std::vector<std::vector<Box>> results = postprocessor.forward(bboxes, scores, output_shape_bboxes, output_shape_scores);

        drawBoxes(img, results, ratios[0], ratios[1]);
    }
    catch (std::exception& ex)
    {
        LogError("Inference Error: ", ex.what());
        throw ex;
    }
    catch (...)
    {
        LogError("Inference Error: ", "Unexpected exception");
        throw;
    }
}
