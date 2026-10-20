#include "inference_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
typedef int (*CudaSyncFn)();
inline void sync_cuda_if_active(const std::string &provider) {
  if (provider.find("CUDA") == std::string::npos) {
    return;
  }
  static CudaSyncFn sync_fn = nullptr;
  static bool probed = false;
  if (!probed) {
    void *handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) handle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) handle = dlopen("libcudart.so.13", RTLD_LAZY | RTLD_NOLOAD);
    if (!handle) handle = dlopen("libcudart.so", RTLD_LAZY);
    if (handle) {
      sync_fn = reinterpret_cast<CudaSyncFn>(dlsym(handle, "cudaDeviceSynchronize"));
    }
    probed = true;
  }
  if (sync_fn) {
    sync_fn();
  }
}
#else
inline void sync_cuda_if_active(const std::string &) {}
#endif

namespace fs = std::filesystem;

struct BenchmarkStats {
  double p50;
  double mean;
  double std_dev;
  double p95;
  double min_val;
  double max_val;
};

static BenchmarkStats compute_stats(std::vector<double> timings) {
  if (timings.empty()) {
    return {0, 0, 0, 0, 0, 0};
  }
  std::sort(timings.begin(), timings.end());
  const size_t n = timings.size();
  const double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
  const double mean = sum / static_cast<double>(n);

  double var = 0.0;
  for (double t : timings) {
    var += (t - mean) * (t - mean);
  }
  const double std_dev = std::sqrt(var / static_cast<double>(n));

  const double p50 = timings[static_cast<size_t>(n * 0.50)];
  const double p95 = timings[std::min(static_cast<size_t>(n * 0.95), n - 1)];

  return BenchmarkStats{p50, mean, std_dev, p95, timings.front(), timings.back()};
}

static void print_usage(const char *prog) {
  std::cout
      << "C++ ONNX Runtime Systems Benchmark\n"
      << "Usage: " << prog << " [options]\n\n"
      << "Options:\n"
      << "  -m, --model <path>      Path to ONNX model (default: outputs/model.onnx)\n"
      << "  -i, --image <path>      Path to input image (default: sample in data/test/original/)\n"
      << "  -d, --device <type>     Execution provider: auto, cuda, dnnl, cpu (default: auto)\n"
      << "  -r, --runs <num>        Number of timed benchmark passes (default: 100)\n"
      << "  -w, --warmup <num>      Number of un-timed warmup passes (default: 25)\n"
      << "  -t, --threads <num>     Number of intra-op threads (default: physical cores)\n"
      << "  -h, --help              Show this help message and exit\n";
}

int main(int argc, char *argv[]) {
  std::string model_path = "outputs/model.onnx";
  std::string image_path;
  std::string device = "auto";
  int runs = 100;
  int warmup_passes = 25;
  unsigned int hw_threads = std::thread::hardware_concurrency();
  int num_threads = (hw_threads >= 4) ? static_cast<int>(hw_threads / 2) : 4;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "-m" || arg == "--model") {
      if (i + 1 < argc) model_path = argv[++i];
    } else if (arg == "-i" || arg == "--image") {
      if (i + 1 < argc) image_path = argv[++i];
    } else if (arg == "-d" || arg == "--device") {
      if (i + 1 < argc) device = argv[++i];
    } else if (arg == "-r" || arg == "--runs") {
      if (i + 1 < argc) runs = std::max(1, std::stoi(argv[++i]));
    } else if (arg == "-w" || arg == "--warmup") {
      if (i + 1 < argc) warmup_passes = std::max(0, std::stoi(argv[++i]));
    } else if (arg == "-t" || arg == "--threads") {
      if (i + 1 < argc) num_threads = std::max(1, std::stoi(argv[++i]));
    }
  }

  // Auto-detect a sample image if not specified
  if (image_path.empty()) {
    std::vector<std::string> candidates = {
        "data/test/original/e02301ac-68f7-4daf-b2ba-254b917f1cca.jpg",
        "outputs/e3ade58d-086c-47fa-9120-76beacb45395.png"};
    for (const auto &c : candidates) {
      if (fs::exists(c)) {
        image_path = c;
        break;
      }
    }
  }

  if (!fs::exists(model_path)) {
    std::cerr << "Error: Model file does not exist: " << model_path << "\n";
    return 1;
  }
  if (!fs::exists(image_path)) {
    std::cerr << "Error: Image file does not exist: " << image_path << "\n";
    return 1;
  }

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Error: Failed to decode image: " << image_path << "\n";
    return 1;
  }

  std::cout << "======================================================================\n";
  std::cout << "               C++ ONNX RUNTIME SYSTEMS BENCHMARK                     \n";
  std::cout << "======================================================================\n";
  std::cout << "  Model Path      : " << model_path << "\n";
  std::cout << "  Sample Image    : " << image_path << " (" << image.cols << "x" << image.rows << ")\n";
  std::cout << "  Requested Dev   : " << device << "\n";
  std::cout << "  Intra-Op Threads: " << num_threads << "\n";
  std::cout << "  Warmup Iterations: " << warmup_passes << "\n";
  std::cout << "  Timed Iterations: " << runs << "\n";

  bcsd::SegmentationEngine engine(model_path, num_threads, device);
  const std::string active_ep = engine.active_provider();
  std::cout << "  Active Provider : " << active_ep << "\n";
  std::cout << "----------------------------------------------------------------------\n\n";

  // --------------------------------------------------------------------------
  // TIER 1: Pure ONNX Forward Pass (Tensor-to-Tensor in memory)
  // --------------------------------------------------------------------------
  std::cout << "[Tier 1] Benchmarking Pure ONNX Forward Pass (Tensor in memory)...\n";

  // Warmup passes
  for (int i = 0; i < warmup_passes; ++i) {
    (void)engine.run_pure_inference();
  }
  sync_cuda_if_active(active_ep);

  std::vector<double> tier1_timings;
  tier1_timings.reserve(runs);

  for (int i = 0; i < runs; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    (void)engine.run_pure_inference();
    sync_cuda_if_active(active_ep);
    auto t1 = std::chrono::high_resolution_clock::now();
    tier1_timings.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  BenchmarkStats s1 = compute_stats(tier1_timings);

  // --------------------------------------------------------------------------
  // TIER 2: In-Memory End-to-End Pipeline (cv::Mat -> Pre -> Infer -> Post)
  // --------------------------------------------------------------------------
  std::cout << "[Tier 2] Benchmarking Full In-Memory Pipeline (Pre + Infer + Post)...\n";

  // Warmup pipeline
  for (int i = 0; i < warmup_passes; ++i) {
    double p = 0, inf = 0, pos = 0;
    (void)engine.predict(image, 0.5f, p, inf, pos);
  }
  sync_cuda_if_active(active_ep);

  std::vector<double> pre_timings;
  std::vector<double> infer_timings;
  std::vector<double> post_timings;
  std::vector<double> total_timings;
  pre_timings.reserve(runs);
  infer_timings.reserve(runs);
  post_timings.reserve(runs);
  total_timings.reserve(runs);

  for (int i = 0; i < runs; ++i) {
    double p_ms = 0.0, inf_ms = 0.0, post_ms = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    (void)engine.predict(image, 0.5f, p_ms, inf_ms, post_ms);
    sync_cuda_if_active(active_ep);
    auto t1 = std::chrono::high_resolution_clock::now();

    const double tot_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    pre_timings.push_back(p_ms);
    infer_timings.push_back(inf_ms);
    post_timings.push_back(post_ms);
    total_timings.push_back(tot_ms);
  }

  BenchmarkStats s_pre = compute_stats(pre_timings);
  BenchmarkStats s_inf = compute_stats(infer_timings);
  BenchmarkStats s_post = compute_stats(post_timings);
  BenchmarkStats s_tot = compute_stats(total_timings);

  // --------------------------------------------------------------------------
  // Summary Results
  // --------------------------------------------------------------------------
  std::cout << "\n============================== RESULTS ==============================\n\n";

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "--- Tier 1: Pure Forward Pass ---\n";
  std::cout << "  Median (P50)  : " << std::setw(7) << s1.p50 << " ms\n";
  std::cout << "  Mean +/- Std  : " << std::setw(7) << s1.mean << " +/- " << s1.std_dev << " ms\n";
  std::cout << "  P95 Latency   : " << std::setw(7) << s1.p95 << " ms\n";
  std::cout << "  Min / Max     : " << std::setw(7) << s1.min_val << " / " << s1.max_val << " ms\n";
  std::cout << "  Effective FPS : " << std::setw(7) << (1000.0 / s1.p50) << " FPS\n\n";

  std::cout << "--- Tier 2: Full In-Memory Pipeline (Breakdown) ---\n";
  std::cout << "  Stage                  | Median (P50) | Mean +/- Std         | P95 Latency\n";
  std::cout << "  -------------------------------------------------------------------------\n";
  std::cout << "  1. Preprocessing (CV)  | " << std::setw(8) << s_pre.p50 << " ms | "
            << std::setw(6) << s_pre.mean << " +/- " << std::setw(5) << s_pre.std_dev << " ms | "
            << std::setw(8) << s_pre.p95 << " ms\n";
  std::cout << "  2. Forward Pass (ORT)  | " << std::setw(8) << s_inf.p50 << " ms | "
            << std::setw(6) << s_inf.mean << " +/- " << std::setw(5) << s_inf.std_dev << " ms | "
            << std::setw(8) << s_inf.p95 << " ms\n";
  std::cout << "  3. Postprocessing (CV) | " << std::setw(8) << s_post.p50 << " ms | "
            << std::setw(6) << s_post.mean << " +/- " << std::setw(5) << s_post.std_dev << " ms | "
            << std::setw(8) << s_post.p95 << " ms\n";
  std::cout << "  -------------------------------------------------------------------------\n";
  std::cout << "  Total In-Memory E2E    | " << std::setw(8) << s_tot.p50 << " ms | "
            << std::setw(6) << s_tot.mean << " +/- " << std::setw(5) << s_tot.std_dev << " ms | "
            << std::setw(8) << s_tot.p95 << " ms\n";
  std::cout << "  Pipeline Throughput    : " << (1000.0 / s_tot.p50) << " FPS\n";
  std::cout << "======================================================================\n";

  return 0;
}
