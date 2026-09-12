// bench_cpp.cpp — OpenVINO C++ Runtime 最小推理基准
//
// 目的：拿到「C++ API vs Python API 端到端开销」的真实数字（简历里那一项）。
//
// 编译（先 source 好 OpenVINO 环境）：
//   source <openvino_root>/setupvars.sh
//   mkdir build && cd build && cmake .. && cmake --build . --config Release
//
// 运行：
//   ./bench_cpp <model.xml> [device=CPU] [iters=200]
//
// 注意：这里只计时「纯推理」，不含前处理(letterbox)和后处理(NMS)。
//       要对比端到端，请把 letterbox + NMS 也放进计时区间，并在 README 里写明
//       你测的是 pipeline 还是 model-only —— 面试官一定会问这个。

#include <openvino/openvino.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: bench_cpp <model.xml> [device] [iters]\n";
        return 1;
    }
    const std::string xml_path = argv[1];
    const std::string device   = (argc > 2) ? argv[2] : "CPU";
    const int         iters    = (argc > 3) ? std::stoi(argv[3]) : 200;

    ov::Core core;
    std::cout << "OpenVINO runtime : " << ov::get_openvino_version().description
              << " " << ov::get_openvino_version().buildNumber << "\n";

    auto model = core.read_model(xml_path);
    std::cout << "input  : " << model->input().get_any_name() << " "
              << model->input().get_shape() << "\n";
    std::cout << "outputs: ";
    for (const auto& o : model->outputs()) std::cout << o.get_any_name() << " ";
    std::cout << "\ndevice : " << device << "   iters: " << iters << "\n";

    ov::CompiledModel compiled = core.compile_model(model, device);
    ov::InferRequest    req     = compiled.create_infer_request();

    const ov::Shape shape = model->input().get_shape();
    const size_t    n     = ov::shape_size(shape);
    std::vector<float> buf(n, 0.5f);
    ov::Tensor input(model->input().get_element_type(), shape, buf.data());

    for (int i = 0; i < 5; ++i) {          // warm-up: 前几次包含图加载/内存分配
        req.set_input_tensor(input);
        req.infer();
    }

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        req.set_input_tensor(input);
        auto t0 = std::chrono::high_resolution_clock::now();
        req.infer();
        auto t1 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    const double mean   = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    const double p95    = samples[static_cast<size_t>(samples.size() * 0.95)];

    std::cout << "\n--- C++ runtime, model-only inference ---\n"
              << "median : " << median << " ms\n"
              << "mean   : " << mean   << " ms\n"
              << "p95    : " << p95    << " ms\n"
              << "FPS    : " << (1000.0 / median) << "\n"
              << "DEVICE=" << device << " MEDIAN_MS=" << median << " FPS=" << (1000.0 / median)
              << "\n";
    return 0;
}
