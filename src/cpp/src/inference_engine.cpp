#include "inference_engine.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <onnxruntime_session_options_config_keys.h>
#include <stdexcept>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace bcsd {

namespace {

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

bool has_provider(const std::vector<std::string> &providers,
                  const std::string &target) {
  return std::find(providers.begin(), providers.end(), target) !=
         providers.end();
}

void preload_cudnn9_if_needed() {
#if defined(__linux__)
  // On Linux with cuDNN 9, libcudnn.so.9 splits CNN/ops/graph symbols into
  // sub-libraries that must be loaded into the global symbol table before
  // libonnxruntime_providers_cuda.so resolves them.
  static constexpr const char *kCudnnLibs[] = {
      "libcudnn_ops.so.9", "libcudnn_cnn.so.9", "libcudnn_graph.so.9"};
  for (const char *lib : kCudnnLibs) {
    (void)dlopen(lib, RTLD_LAZY | RTLD_GLOBAL);
  }
#endif
}

} // namespace

SegmentationEngine::SegmentationEngine(const std::string &model_path,
                                       int num_threads,
                                       const std::string &device)
    : env_(ORT_LOGGING_LEVEL_WARNING, "BloodCellSegmentation"),
      memory_info_(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      run_options_(nullptr) {
  // Configure thread pool and sequential execution for single-stream U-Net
  session_options_.SetIntraOpNumThreads(num_threads);
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Enable memory arena and static memory pattern planner to avoid per-run allocations
  session_options_.EnableCpuMemArena();
  session_options_.EnableMemPattern();
  session_options_.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning,
                                  "1");

  // Select optimal Execution Provider based on requested device & system support
  const std::string dev = to_lower(device);
  const std::vector<std::string> available = Ort::GetAvailableProviders();

  auto try_enable_cuda = [&]() -> bool {
    if (!has_provider(available, "CUDAExecutionProvider")) {
      return false;
    }
    try {
      preload_cudnn9_if_needed();
      OrtCUDAProviderOptions cuda_opts{};
      cuda_opts.device_id = 0;
      cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
      cuda_opts.do_copy_in_default_stream = 1;
      session_options_.AppendExecutionProvider_CUDA(cuda_opts);
      active_provider_ = "CUDAExecutionProvider";
      return true;
    } catch (...) {
      return false;
    }
  };

  auto try_enable_dnnl = [&]() -> bool {
    if (!has_provider(available, "DnnlExecutionProvider")) {
      return false;
    }
    try {
      OrtDnnlProviderOptions *dnnl_opts = nullptr;
      const OrtApi &api = Ort::GetApi();
      Ort::ThrowOnError(api.CreateDnnlProviderOptions(&dnnl_opts));
      struct DnnlGuard {
        const OrtApi &api;
        OrtDnnlProviderOptions *ptr;
        ~DnnlGuard() {
          if (ptr)
            api.ReleaseDnnlProviderOptions(ptr);
        }
      } guard{api, dnnl_opts};
      session_options_.AppendExecutionProvider_Dnnl(*dnnl_opts);
      active_provider_ = "DnnlExecutionProvider";
      return true;
    } catch (...) {
      return false;
    }
  };

  if (dev == "cuda") {
    if (!try_enable_cuda()) {
      throw std::runtime_error(
          "Requested CUDAExecutionProvider, but it is unavailable or failed to initialize.");
    }
  } else if (dev == "dnnl" || dev == "onednn") {
    if (!try_enable_dnnl()) {
      throw std::runtime_error(
          "Requested DnnlExecutionProvider, but it is unavailable or failed to initialize.");
    }
  } else if (dev == "auto") {
    if (!try_enable_cuda()) {
      (void)try_enable_dnnl();
    }
  } else if (dev != "cpu") {
    throw std::invalid_argument("Unsupported device '" + device +
                                "'. Expected 'auto', 'cuda', 'dnnl', or 'cpu'.");
  }

  // Fallback helper: if session creation with an accelerated EP fails at graph
  // load time (e.g., missing cuDNN / oneDNN runtime libs), fall back to CPU in auto mode
  try {
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(),
                                              session_options_);
  } catch (...) {
    if (dev == "auto" && active_provider_ != "CPUExecutionProvider") {
      session_options_ = Ort::SessionOptions{};
      session_options_.SetIntraOpNumThreads(num_threads);
      session_options_.SetInterOpNumThreads(1);
      session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      session_options_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_ALL);
      session_options_.EnableCpuMemArena();
      session_options_.EnableMemPattern();
      session_options_.AddConfigEntry(
          kOrtSessionOptionsConfigAllowIntraOpSpinning, "1");
      active_provider_ = "CPUExecutionProvider";
      session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(),
                                                session_options_);
    } else {
      throw;
    }
  }

  // Cache input and output node names once at initialization
  Ort::AllocatorWithDefaultOptions allocator;
  if (session_->GetInputCount() > 0) {
    auto in_name = session_->GetInputNameAllocated(0, allocator);
    if (in_name) {
      input_name_ = in_name.get();
    }
  }
  if (session_->GetOutputCount() > 0) {
    auto out_name = session_->GetOutputNameAllocated(0, allocator);
    if (out_name) {
      output_name_ = out_name.get();
    }
  }
  input_names_[0] = input_name_.c_str();
  output_names_[0] = output_name_.c_str();

  // Query input tensor dimensions if specified in ONNX metadata
  try {
    Ort::TypeInfo type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> in_shape = tensor_info.GetShape();
    // Expected format: [batch, channels, height, width]
    if (in_shape.size() == 4) {
      if (in_shape[2] > 0) {
        input_height_ = static_cast<int>(in_shape[2]);
      }
      if (in_shape[3] > 0) {
        input_width_ = static_cast<int>(in_shape[3]);
      }
    }
  } catch (...) {
    // Retain default static dimensions (416x416) on query failure
  }

  // Pre-allocate reusable NCHW input buffer and bind persistent Ort::Value + IoBinding
  input_shape_ = {1, 3, input_height_, input_width_};
  input_buffer_.resize(static_cast<size_t>(3) * input_height_ * input_width_,
                       0.0f);
  input_tensor_ = Ort::Value::CreateTensor<float>(
      memory_info_, input_buffer_.data(), input_buffer_.size(),
      input_shape_.data(), input_shape_.size());

  io_binding_ = std::make_unique<Ort::IoBinding>(*session_);
  io_binding_->BindInput(input_name_.c_str(), input_tensor_);
  io_binding_->BindOutput(output_name_.c_str(), memory_info_);
}

void SegmentationEngine::warmup(int iterations) {
  for (int i = 0; i < iterations; ++i) {
    session_->Run(run_options_, *io_binding_);
  }
}

double SegmentationEngine::run_pure_inference() {
  auto start = std::chrono::high_resolution_clock::now();
  session_->Run(run_options_, *io_binding_);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void SegmentationEngine::preprocess_into(const cv::Mat &resized_bgr,
                                         float *dst) const {
  if (resized_bgr.empty()) {
    throw std::invalid_argument("Input BGR image is empty.");
  }
  if (resized_bgr.rows != input_height_ || resized_bgr.cols != input_width_) {
    throw std::invalid_argument(
        "Input image dimensions do not match model expectation.");
  }

  // Fused single-pass BGR uint8 -> planar RGB float32 NCHW + ImageNet normalization:
  // Avoids two intermediate cv::Mat heap allocations (cvtColor + convertTo)
  // and replaces floating-point division with precomputed FMA (pixel * scale + bias).
  const int hw = input_height_ * input_width_;
  float *__restrict__ r_plane = dst;
  float *__restrict__ g_plane = dst + hw;
  float *__restrict__ b_plane = dst + 2 * hw;

  constexpr float r_scale = kNormScale[0];
  constexpr float g_scale = kNormScale[1];
  constexpr float b_scale = kNormScale[2];
  constexpr float r_bias = kNormBias[0];
  constexpr float g_bias = kNormBias[1];
  constexpr float b_bias = kNormBias[2];

  if (resized_bgr.isContinuous()) {
    const uint8_t *__restrict__ src = resized_bgr.ptr<uint8_t>(0);
    for (int i = 0; i < hw; ++i) {
      const int idx3 = i * 3;
      b_plane[i] = static_cast<float>(src[idx3 + 0]) * b_scale + b_bias;
      g_plane[i] = static_cast<float>(src[idx3 + 1]) * g_scale + g_bias;
      r_plane[i] = static_cast<float>(src[idx3 + 2]) * r_scale + r_bias;
    }
  } else {
    for (int y = 0; y < input_height_; ++y) {
      const uint8_t *__restrict__ row_ptr = resized_bgr.ptr<uint8_t>(y);
      const int row_offset = y * input_width_;
      for (int x = 0; x < input_width_; ++x) {
        const int flat_idx = row_offset + x;
        const int idx3 = x * 3;
        b_plane[flat_idx] =
            static_cast<float>(row_ptr[idx3 + 0]) * b_scale + b_bias;
        g_plane[flat_idx] =
            static_cast<float>(row_ptr[idx3 + 1]) * g_scale + g_bias;
        r_plane[flat_idx] =
            static_cast<float>(row_ptr[idx3 + 2]) * r_scale + r_bias;
      }
    }
  }
}

std::vector<float>
SegmentationEngine::preprocess(const cv::Mat &resized_bgr) const {
  std::vector<float> input_tensor(
      static_cast<size_t>(3) * input_height_ * input_width_);
  preprocess_into(resized_bgr, input_tensor.data());
  return input_tensor;
}

void SegmentationEngine::postprocess(const float *logits, int out_h, int out_w,
                                     const cv::Mat &resized_bgr,
                                     float prob_threshold,
                                     cv::Mat &out_binary_mask,
                                     cv::Mat &out_contour_overlay) const {
  // Mathematical optimization:
  // Because sigmoid(z) = 1 / (1 + exp(-z)) is strictly monotonic,
  // sigmoid(z) > p_thresh  <==>  z > log(p_thresh / (1 - p_thresh)).
  // For p_thresh = 0.5, logit_threshold = log(1) = 0.0f!
  // This avoids calling std::exp() 51,984 times per frame.
  const float logit_threshold =
      std::log(prob_threshold / (1.0f - prob_threshold));

  // Zero-copy CV_32FC1 header over ONNX output logits + SIMD threshold via cv::compare
  const cv::Mat logits_mat(out_h, out_w, CV_32FC1,
                           const_cast<float *>(logits));
  cv::compare(logits_mat, logit_threshold, out_binary_mask, cv::CMP_GT);

  // Center-crop the resized BGR image to match the output resolution
  // (e.g. 228x228 for unpadded U-Net, or 416x416 for padded U-Net)
  const int diff_y = (resized_bgr.rows - out_h) / 2;
  const int diff_x = (resized_bgr.cols - out_w) / 2;
  cv::Rect center_roi(diff_x, diff_y, out_w, out_h);
  out_contour_overlay = resized_bgr(center_roi).clone();

  // Extract and draw predicted cell membrane contours in red (BGR: 0, 0, 255)
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(out_binary_mask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  cv::drawContours(out_contour_overlay, contours, -1, cv::Scalar(0, 0, 255), 2,
                   cv::LINE_AA);
}

SegmentationResult SegmentationEngine::predict(const cv::Mat &bgr_image,
                                               float prob_threshold) {
  double pre_ms = 0.0, infer_ms = 0.0, post_ms = 0.0;
  return predict(bgr_image, prob_threshold, pre_ms, infer_ms, post_ms);
}

SegmentationResult SegmentationEngine::predict(const cv::Mat &bgr_image,
                                               float prob_threshold,
                                               double &out_preprocess_ms,
                                               double &out_inference_ms,
                                               double &out_postprocess_ms) {
  if (bgr_image.empty()) {
    throw std::invalid_argument("Input BGR image is empty.");
  }
  if (prob_threshold <= 0.0f || prob_threshold >= 1.0f) {
    throw std::invalid_argument(
        "Probability threshold must be strictly between 0.0 and 1.0.");
  }

  // Preprocessing (Resize + fused ImageNet normalization into pinned buffer)
  auto t_pre_start = std::chrono::high_resolution_clock::now();
  cv::Mat resized_bgr;
  if (bgr_image.rows == input_height_ && bgr_image.cols == input_width_) {
    resized_bgr = bgr_image;
  } else {
    cv::resize(bgr_image, resized_bgr, cv::Size(input_width_, input_height_),
               0.0, 0.0, cv::INTER_LINEAR);
  }

  preprocess_into(resized_bgr, input_buffer_.data());
  io_binding_->BindInput(input_name_.c_str(), input_tensor_);
  auto t_pre_end = std::chrono::high_resolution_clock::now();
  out_preprocess_ms =
      std::chrono::duration<double, std::milli>(t_pre_end - t_pre_start).count();

  // Inference (Graph execution)
  auto t_infer_start = std::chrono::high_resolution_clock::now();
  session_->Run(run_options_, *io_binding_);
  auto t_infer_end = std::chrono::high_resolution_clock::now();
  out_inference_ms =
      std::chrono::duration<double, std::milli>(t_infer_end - t_infer_start).count();

  // Postprocessing (SIMD thresholding + contours)
  auto t_post_start = std::chrono::high_resolution_clock::now();
  std::vector<Ort::Value> output_tensors = io_binding_->GetOutputValues();
  auto type_info = output_tensors.front().GetTensorTypeAndShapeInfo();
  std::vector<int64_t> out_shape = type_info.GetShape();
  const int out_h = static_cast<int>(out_shape[2]);
  const int out_w = static_cast<int>(out_shape[3]);

  const float *logits_ptr = output_tensors.front().GetTensorData<float>();

  cv::Mat binary_mask;
  cv::Mat contour_overlay;
  postprocess(logits_ptr, out_h, out_w, resized_bgr, prob_threshold,
              binary_mask, contour_overlay);
  auto t_post_end = std::chrono::high_resolution_clock::now();
  out_postprocess_ms =
      std::chrono::duration<double, std::milli>(t_post_end - t_post_start).count();

  return SegmentationResult{binary_mask, contour_overlay, out_inference_ms};
}

} // namespace bcsd
