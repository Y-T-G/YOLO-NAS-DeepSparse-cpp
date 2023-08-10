
![GitHub release (latest by
date)](https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Visual
Studio](https://img.shields.io/badge/Visual%20Studio-5C2D91.svg?style=for-the-badge&logo=visual-studio&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white)
[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE)

# YOLO-NAS-DeepSparse.cpp

YOLO-NAS is a state-of-the-art object detector by [Deci
AI](https://github.com/Deci-AI/super-gradients). This
project implements the YOLO-NAS object detector in C++ with
DeepSparse backend to speed up inference performance. [DeepSparse](https://github.com/neuralmagic/deepsparse) is an inference runtime by Neural Magic that can greatly speed up inference performance on CPUs by leveraging sparsity.

## Features

* Supports both **image** and **video** inference.
* **Faster** CPU inference speeds.

## Getting Started

The following instructions demonstrates how to build this
project on a Linux system. Windows is currently not supported by the DeepSparse library.

### Prerequisites

* **CMake v3.8+** - found at
[https://cmake.org/](https://cmake.org/)

* **GCC/G++ compiler** - found at [https://gcc.gnu.org/](https://gcc.gnu.org/)

* **Python 3.8+** - Python is used to install the deepsparse library which is required for the build. Download [here](https://www.python.org/downloads/).

* **OpenCV v4.0+** - Download
[here](https://github.com/opencv/opencv/releases/).

* **DeepSparse v1.6.0+** - Download [here](https://pypi.org/project/deepsparse/).

## Building the project

1. Set the `OpenCV_DIR` environment variable to point to
your `../../opencv/build` directory (if not set).
2. Run the following build commands:
    a. [Linux] Bash:

    ```bash
    cd <yolo-nas-deepsparse-cpp-directory>
    cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=Release
    cd build

    make
    ```

3. The compiled executable will be in root folder of the build directory.

## Inference

1. Export the ONNX file:

    ```python
    from super_gradients.training import models

    model = models.get("yolo_nas_s", pretrained_weights="coco")
    model.eval()
    model.prep_model_for_conversion(input_size=(1, 3, 640, 640))
    models.convert_to_onnx(model=model, prep_model_for_conversion_kwargs={"input_size":(1, 3, 640, 640)}, out_path="yolo_nas_s.onnx")
    ```

2. To run the inference, execute the following command:

    ```bash
    yolo-nas-deepsparse-cpp --model <ONNX_MODEL_PATH> [-i <IMAGE_PATH> | -v <VIDEO_PATH>] [--imgsz IMAGE_SIZE] [--gpu] [--iou-thresh IOU_THRESHOLD] [--score-thresh CONFIDENCE_THRESHOLD]
    ```

## Benchmarks

The following benchmarks were done on Google Colab using Intel® Xeon® Processor E5-2699 v4 @ 2.20GHz with 2 vCPUs.

| **Backend**               | **Latency** | **FPS** | **Implementation**                                                        |
|---------------------------|-------------|---------|---------------------------------------------------------------------------|
| PyTorch                   | 867.02ms    | 1.15    | Native (`model.predict()` in `super_gradients`)                           |
| ONNX C++ (via OpenCV DNN) | 962.27ms    | 1.04    | [Hyuotu](https://github.com/Hyuto/yolo-nas-onnx/tree/master/yolo-nas-cpp) |
| ONNX Python               | 626.37ms    | 1.59    | [Hyuotu](https://github.com/Hyuto/yolo-nas-onnx/tree/master/yolo-nas-py)  |
| OpenVINO C++              | 628.04ms    | 1.59    | [Y-T-G](https://github.com/Y-T-G/yolo-nas-openvino-cpp)                   |
| DeepSparse                | 565.75ms    | 1.83    | [Y-T-G](https://github.com/Y-T-G/yolo-nas-deepsparse-cpp)                 |

## Authors

* **Mohammed Yasin** - [@Y-T-G](https://github.com/Y-T-G)

## Acknowledgements

Thanks to [@Hyuto](https://github.com/Hyuto) for his work on
[ONNX
implementation](https://github.com/Hyuto/yolo-nas-onnx/tree/master/yolo-nas-cpp) of
YOLO-NAS in C++ which was utilized in this project.

## License

This project is licensed under the
[MIT](https://mit-license.org/) License - see the
[LICENSE](LICENSE) file for details. DeepSparse Community edition is only for evaluation, research, and non-production. See the [DeepSparse Community License](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE-NEURALMAGIC) for more details.
