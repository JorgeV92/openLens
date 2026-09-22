#include "openlens/core.hpp"
#include <iostream>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        require(openlens::iou({0, 0, 10, 10}, {0, 0, 10, 10}) == 1.0, "Identical boxes");
        require(openlens::iou({0, 0, 10, 10}, {10, 0, 20, 10}) == 0.0, "Touching boxes");
        require(std::abs(openlens::iou({0, 0, 10, 10}, {5, 0, 15, 10}) - 1.0 / 3.0) < 1e-6,
                "Partial overlap");
        const auto box = openlens::to_pixels(0.5f, 0.5f, 0.5f, 0.5f, 640, 360);
        require(box.x1 == 160 && box.y1 == 90 && box.x2 == 480 && box.y2 == 270,
                "Normalized center/size to original pixels");
        const auto clipped = openlens::to_pixels(0.0f, 0.0f, 0.5f, 0.5f, 640, 360);
        require(clipped.x1 == 0 && clipped.y1 == 0, "Clip image boundary");
        bool rejected = false;
        try { openlens::to_pixels(0, 0, -1, 1, 640, 360); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Reject negative extent");
        std::cout << "openLens C++ core checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
