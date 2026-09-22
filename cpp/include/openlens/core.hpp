#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace openlens {

struct Box { float x1, y1, x2, y2; };

inline double iou(const Box& a, const Box& b) {
    const double intersection = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
                                std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const double total = (a.x2 - a.x1) * (a.y2 - a.y1) +
                         (b.x2 - b.x1) * (b.y2 - b.y1) - intersection;
    return total > 0 ? intersection / total : 0.0;
}

inline Box to_pixels(float cx, float cy, float w, float h, int width, int height) {
    if (width <= 0 || height <= 0 || w < 0 || h < 0 ||
        !std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h))
        throw std::invalid_argument("Invalid box or image dimensions");
    return {std::clamp((cx - w / 2) * width, 0.0f, static_cast<float>(width)),
            std::clamp((cy - h / 2) * height, 0.0f, static_cast<float>(height)),
            std::clamp((cx + w / 2) * width, 0.0f, static_cast<float>(width)),
            std::clamp((cy + h / 2) * height, 0.0f, static_cast<float>(height))};
}

inline void check_float_format() {
    const unsigned int one = 1;
    if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559 ||
        *reinterpret_cast<const unsigned char*>(&one) != 1)
        throw std::runtime_error("Tensor bundles require little-endian IEEE float32");
}

inline std::vector<float> read_tensor(const std::string& path, std::size_t count) {
    check_float_format();
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open input: " + path);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        stream.tellg() != static_cast<std::streamoff>(count * sizeof(float)))
        throw std::runtime_error("Input byte count disagrees with model shape");
    stream.seekg(0);
    std::vector<float> result(count);
    if (!stream.read(reinterpret_cast<char*>(result.data()), count * sizeof(float)))
        throw std::runtime_error("Could not read complete tensor");
    for (float value : result)
        if (!std::isfinite(value)) throw std::runtime_error("Input contains non-finite values");
    return result;
}

inline void write_tensor(const std::string& path, const float* data, std::size_t count) {
    check_float_format();
    std::ofstream stream(path, std::ios::binary);
    if (!stream || !stream.write(reinterpret_cast<const char*>(data), count * sizeof(float)))
        throw std::runtime_error("Cannot write output: " + path);
}

}  // namespace openlens
