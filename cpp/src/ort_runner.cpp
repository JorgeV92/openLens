#include "openlens/core.hpp"
#include <onnxruntime_cxx_api.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: openlens_ort MODEL.onnx INPUT.f32 OUTPUT_PREFIX\n";
            return 2;
        }
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "openLens");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(environment, argv[1], options);
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 2)
            throw std::runtime_error("Expected one input and two outputs from openlens export");
        // Keep TypeInfo alive while accessing its TensorTypeAndShapeInfo view.
        auto input_type = session.GetInputTypeInfo(0);
        auto input_info = input_type.GetTensorTypeAndShapeInfo();
        auto shape = input_info.GetShape();
        if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            shape.size() != 4 || shape[0] != 1 || shape[1] != 3)
            throw std::runtime_error("Expected fixed float32 NCHW input with batch=1 and RGB channels");
        std::size_t count = 1;
        for (auto dimension : shape) {
            if (dimension <= 0 || static_cast<std::size_t>(dimension) >
                std::numeric_limits<std::size_t>::max() / count)
                throw std::runtime_error("Unsupported model input dimensions");
            count *= dimension;
        }
        auto input = openlens::read_tensor(argv[2], count);
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
        const std::array<const char*, 1> input_names{"pixel_values"};
        const std::array<const char*, 2> output_names{"logits", "pred_boxes"};
        auto run = [&]() {
            return session.Run(Ort::RunOptions{nullptr}, input_names.data(), &tensor, 1,
                               output_names.data(), output_names.size());
        };
        run();  // Warm up once; this runner measures inference only.
        const auto start = std::chrono::steady_clock::now();
        auto outputs = run();
        const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const std::filesystem::path prefix(argv[3]);
        if (prefix.has_parent_path()) std::filesystem::create_directories(prefix.parent_path());
        const std::array<std::string, 2> suffixes{"_logits.f32", "_boxes.f32"};
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            auto info = outputs[i].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                throw std::runtime_error("Expected float32 output");
            openlens::write_tensor(prefix.string() + suffixes[i], outputs[i].GetTensorData<float>(), info.GetElementCount());
        }
        std::cout << "{\"inference_ms\":" << elapsed << ",\"samples\":1,\"warmup_runs\":1,"
                  << "\"scope\":\"single CPU inference, excludes preprocessing and file I/O\"}\n";
    } catch (const std::exception& error) {
        std::cerr << "openlens_ort: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
