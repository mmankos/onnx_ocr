#include "onnx_predictor/onnx_predictor.h"
#include <iostream>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <config_path>\n";
        return -1;
    }

    std::unique_ptr<OnnxPredictor> predictor =
        std::make_unique<OnnxPredictor>(argv[1]);
    // predictor->predict();

    return 0;
}
