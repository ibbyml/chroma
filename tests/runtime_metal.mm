#include "codegen/metal/runtime.h"
#include <cstdio>
#include <iostream>
#include <unistd.h>

int main() {
    @autoreleasepool {
        char filename[] = "/tmp/chroma-metal-test-XXXXXX";
        const int file = mkstemp(filename);
        if (file < 0) {
            return 2;
        }
        close(file);
        const char* source = R"(
            #include <metal_stdlib>
            using namespace metal;
            kernel void square(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                               uint i [[thread_position_in_grid]]) {
                if (i < 7) y[i] = x[i] * x[i];
            }
        )";
        try {
            chroma::metal::Runner runner(
                filename, 0, 0, {28}, {28}, source,
                {{"square", {{0, 3, 0}, {1, 4, 0}}, MTLSizeMake(1, 1, 1), MTLSizeMake(32, 1, 1), {}}});
            std::remove(filename);
            float x[] = {1, -2, 3, -4, 5, -6, 7}, y[7];
            const void* inputs[] = {x};
            void* outputs[] = {y};
            runner.run(inputs, outputs);
            for (int i = 0; i < 7; ++i) {
                if (y[i] != x[i] * x[i]) {
                    return 1;
                }
            }
            std::cout << "Metal execution passed on " << runner.device_name() << '\n';
        } catch (const std::exception& e) {
            std::remove(filename);
            std::cerr << e.what() << '\n';
            return 1;
        }
    }
}
