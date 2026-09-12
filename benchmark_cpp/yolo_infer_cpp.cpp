// yolo_infer_cpp.cpp — 零第三方依赖（除 OpenVINO）的完整 C++ 推理路径
//
// 目的：给简历提供 C++ 侧的真凭实据，并量化「C++ vs Python 的 host 端开销」。
//
// 本机装不了 OpenCV C++（opencv-python 的 wheel 不带头文件和 cmake config），
// 所以前处理的双线性 resize / letterbox 全部手写 —— 正好对应
// "letterbox + NMS hand-written in C++" 这件事本身。
//
// 与 Python 侧（yolo_utils.py）严格一一对应，保证两边可比：
//   letterbox(保持长宽比 + 114 填充) -> BGR2RGB -> /255 -> NCHW
//   -> 推理 -> 解码(cx,cy,w,h + 80 类) -> sigmoid(如需)
//   -> 逐类 NMS @ IoU 0.7 -> 最多 300 框 -> 还原到原图坐标
//
// 编译：
//   export OpenVINO_DIR=<site-packages>/openvino/cmake
//   cmake -S . -B build && cmake --build build -j
//
// 运行：
//   ./yolo_infer_cpp <model.xml> <raw_bgr.bin> [device] [iters] [conf] [iou]
//   raw_bgr.bin 格式：int32 H, int32 W, 随后 H*W*3 字节 BGR

#include <openvino/openvino.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int IMG_SIZE = 640;
constexpr int NUM_CLASSES = 80;
constexpr int MAX_DET = 300;
constexpr int ROW = 4 + NUM_CLASSES;  // 84

using Clock = std::chrono::high_resolution_clock;

inline double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---------------- 手写双线性 resize ----------------
void resize_bilinear(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    const double fx_scale = static_cast<double>(sw) / dw;
    const double fy_scale = static_cast<double>(sh) / dh;
    for (int y = 0; y < dh; ++y) {
        const double fy = (y + 0.5) * fy_scale - 0.5;
        const int y0raw = static_cast<int>(std::floor(fy));
        const double wy = fy - y0raw;
        const int y0 = std::clamp(y0raw, 0, sh - 1);
        const int y1 = std::clamp(y0raw + 1, 0, sh - 1);
        for (int x = 0; x < dw; ++x) {
            const double fx = (x + 0.5) * fx_scale - 0.5;
            const int x0raw = static_cast<int>(std::floor(fx));
            const double wx = fx - x0raw;
            const int x0 = std::clamp(x0raw, 0, sw - 1);
            const int x1 = std::clamp(x0raw + 1, 0, sw - 1);
            for (int c = 0; c < 3; ++c) {
                const double p00 = src[(y0 * sw + x0) * 3 + c];
                const double p01 = src[(y0 * sw + x1) * 3 + c];
                const double p10 = src[(y1 * sw + x0) * 3 + c];
                const double p11 = src[(y1 * sw + x1) * 3 + c];
                const double v = (1 - wy) * ((1 - wx) * p00 + wx * p01)
                               + wy * ((1 - wx) * p10 + wx * p11);
                dst[(y * dw + x) * 3 + c] = static_cast<uint8_t>(
                    std::clamp(static_cast<int>(std::lround(v)), 0, 255));
            }
        }
    }
}

// ---------------- 前处理：letterbox + BGR2RGB + /255 + NCHW ----------------
struct PreOut {
    std::vector<float> nchw;
    float ratio = 1.0f, pad_x = 0.0f, pad_y = 0.0f;
};

PreOut preprocess(const uint8_t* img, int h, int w) {
    const float r = std::min(IMG_SIZE / static_cast<float>(h),
                             IMG_SIZE / static_cast<float>(w));
    const int nw = static_cast<int>(std::round(w * r));
    const int nh = static_cast<int>(std::round(h * r));

    std::vector<uint8_t> resized(static_cast<size_t>(nw) * nh * 3);
    resize_bilinear(img, w, h, resized.data(), nw, nh);

    std::vector<uint8_t> canvas(static_cast<size_t>(IMG_SIZE) * IMG_SIZE * 3, 114);
    const int top = (IMG_SIZE - nh) / 2;
    const int left = (IMG_SIZE - nw) / 2;
    for (int y = 0; y < nh; ++y) {
        std::memcpy(&canvas[((y + top) * IMG_SIZE + left) * 3],
                    &resized[static_cast<size_t>(y) * nw * 3],
                    static_cast<size_t>(nw) * 3);
    }

    PreOut out;
    out.ratio = r;
    out.pad_x = static_cast<float>(left);
    out.pad_y = static_cast<float>(top);
    const size_t plane = static_cast<size_t>(IMG_SIZE) * IMG_SIZE;
    out.nchw.resize(3 * plane);
    for (size_t i = 0; i < plane; ++i) {
        // BGR -> RGB
        out.nchw[i]                = canvas[i * 3 + 2] / 255.0f;
        out.nchw[plane + i]        = canvas[i * 3 + 1] / 255.0f;
        out.nchw[2 * plane + i]    = canvas[i * 3 + 0] / 255.0f;
    }
    return out;
}

// ---------------- 检测框 ----------------
struct Det {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0, score = 0;
    int cls = 0;
};

inline float iou_xyxy(const Det& a, const Det& b) {
    const float iw = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float ih = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float inter = iw * ih;
    const float aa = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float ab = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float uni = aa + ab - inter;
    return uni > 1e-9f ? inter / uni : 0.0f;
}

std::vector<Det> nms_class(const std::vector<Det>& in, float iou_thr) {
    if (in.empty()) return {};
    std::vector<int> order(in.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int i, int j) { return in[i].score > in[j].score; });
    std::vector<char> dead(in.size(), 0);
    std::vector<Det> keep;
    for (size_t k = 0; k < order.size(); ++k) {
        const int i = order[k];
        if (dead[i]) continue;
        keep.push_back(in[i]);
        for (size_t m = k + 1; m < order.size(); ++m) {
            const int j = order[m];
            if (dead[j]) continue;
            if (iou_xyxy(in[i], in[j]) > iou_thr) dead[j] = 1;
        }
    }
    return keep;
}

// 后处理：解码 + 逐类 NMS + 坐标还原
// 输出可能是 channel-major [1,84,8400] 或 box-major [1,8400,84]，用步长统一访问
std::vector<Det> postprocess(const float* raw, int num_boxes, size_t ch_stride,
                             size_t box_stride, float ratio, float pad_x, float pad_y,
                             int ow, int oh, float conf_thr, float iou_thr) {
    const auto at = [&](int b, int i) -> float {
        return raw[static_cast<size_t>(i) * ch_stride + static_cast<size_t>(b) * box_stride];
    };

    // 不同导出版本的 YOLOv8 有的已 sigmoid，先扫一遍判定
    bool need_sigmoid = false;
    float mn = at(0, 4), mx = at(0, 4);
    for (int b = 0; b < num_boxes; b += std::max(1, num_boxes / 200)) {
        for (int c = 4; c < ROW; ++c) {
            const float v = at(b, c);
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
    }
    need_sigmoid = (mn < 0.0f || mx > 1.0f);

    std::vector<std::vector<Det>> by_class(NUM_CLASSES);
    for (int b = 0; b < num_boxes; ++b) {
        int best_cls = 0;
        float best = at(b, 4);
        for (int c = 5; c < ROW; ++c) {
            const float v = at(b, c);
            if (v > best) { best = v; best_cls = c - 4; }
        }
        if (need_sigmoid) best = 1.0f / (1.0f + std::exp(-best));
        if (best <= conf_thr) continue;

        Det d;
        d.cls = best_cls;
        d.score = best;
        const float cx = at(b, 0), cy = at(b, 1), w = at(b, 2), h = at(b, 3);
        d.x1 = (cx - w * 0.5f - pad_x) / ratio;
        d.y1 = (cy - h * 0.5f - pad_y) / ratio;
        d.x2 = (cx + w * 0.5f - pad_x) / ratio;
        d.y2 = (cy + h * 0.5f - pad_y) / ratio;
        d.x1 = std::clamp(d.x1, 0.0f, static_cast<float>(ow));
        d.y1 = std::clamp(d.y1, 0.0f, static_cast<float>(oh));
        d.x2 = std::clamp(d.x2, 0.0f, static_cast<float>(ow));
        d.y2 = std::clamp(d.y2, 0.0f, static_cast<float>(oh));
        if (d.x2 - d.x1 <= 1e-3f || d.y2 - d.y1 <= 1e-3f) continue;
        by_class[best_cls].push_back(d);
    }

    std::vector<Det> out;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        if (by_class[c].empty()) continue;
        const auto kept = nms_class(by_class[c], iou_thr);
        out.insert(out.end(), kept.begin(), kept.end());
    }
    if (static_cast<int>(out.size()) > MAX_DET) {
        std::sort(out.begin(), out.end(),
                  [](const Det& a, const Det& b) { return a.score > b.score; });
        out.resize(MAX_DET);
    }
    return out;
}

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
double p95_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(v.size() * 0.95)];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <model.xml> <raw_bgr.bin> [device=CPU] [iters=100] [conf=0.001] [iou=0.7]\n";
        return 1;
    }
    const std::string xml = argv[1];
    const std::string bin_path = argv[2];
    const std::string device = argc > 3 ? argv[3] : "CPU";
    const int iters = argc > 4 ? std::atoi(argv[4]) : 100;
    const float conf_thr = argc > 5 ? static_cast<float>(std::atof(argv[5])) : 0.001f;
    const float iou_thr = argc > 6 ? static_cast<float>(std::atof(argv[6])) : 0.7f;

    std::ifstream fin(bin_path, std::ios::binary);
    if (!fin) { std::cerr << "cannot open " << bin_path << "\n"; return 1; }
    int32_t h = 0, w = 0;
    fin.read(reinterpret_cast<char*>(&h), 4);
    fin.read(reinterpret_cast<char*>(&w), 4);
    std::vector<uint8_t> img(static_cast<size_t>(h) * w * 3);
    fin.read(reinterpret_cast<char*>(img.data()), static_cast<std::streamsize>(img.size()));

    ov::Core core;
    auto compiled = core.compile_model(core.read_model(xml), device);
    ov::InferRequest req = compiled.create_infer_request();
    const std::string out_name = compiled.output().get_any_name();

    const ov::Shape oshape = compiled.output().get_shape();
    const size_t last = oshape[oshape.size() - 1];
    const size_t mid = oshape[oshape.size() - 2];
    size_t ch_stride, box_stride;
    int num_boxes;
    if (last == ROW) { box_stride = ROW; ch_stride = 1; num_boxes = static_cast<int>(mid); }
    else             { ch_stride = last; box_stride = 1; num_boxes = static_cast<int>(last); }

    std::vector<double> t_pre, t_inf, t_post, t_total;
    int last_dets = 0;
    const int warmup = 5;

    for (int it = 0; it < iters; ++it) {
        const auto t0 = Clock::now();
        const PreOut pre = preprocess(img.data(), h, w);
        const auto t1 = Clock::now();

        ov::Tensor in_tensor(ov::element::f32, compiled.input().get_shape(),
                             const_cast<float*>(pre.nchw.data()));
        req.set_input_tensor(in_tensor);
        req.infer();
        const float* raw = req.get_tensor(out_name).data<float>();
        const auto t2 = Clock::now();

        const auto dets = postprocess(raw, num_boxes, ch_stride, box_stride, pre.ratio,
                                      pre.pad_x, pre.pad_y, w, h, conf_thr, iou_thr);
        last_dets = static_cast<int>(dets.size());
        const auto t3 = Clock::now();

        if (it >= warmup) {
            t_pre.push_back(ms_between(t0, t1));
            t_inf.push_back(ms_between(t1, t2));
            t_post.push_back(ms_between(t2, t3));
            t_total.push_back(ms_between(t0, t3));
        }
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "OpenVINO build " << ov::get_openvino_version().buildNumber << "\n"
              << "device=" << device << " iters=" << iters
              << " warmup=" << warmup << " dets=" << last_dets << "\n"
              << "stage        median_ms   p95_ms\n"
              << "preprocess   " << median_of(t_pre) << "        " << p95_of(t_pre) << "\n"
              << "inference    " << median_of(t_inf) << "        " << p95_of(t_inf) << "\n"
              << "postprocess  " << median_of(t_post) << "        " << p95_of(t_post) << "\n"
              << "total        " << median_of(t_total) << "        " << p95_of(t_total) << "\n"
              << "KV pre_ms=" << median_of(t_pre) << " infer_ms=" << median_of(t_inf)
              << " post_ms=" << median_of(t_post) << " total_ms=" << median_of(t_total)
              << " dets=" << last_dets << "\n";
    return 0;
}
