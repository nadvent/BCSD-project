#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

namespace bcsd {

struct SegmentationResult {
  cv::Mat binary_mask;     // CV_8UC1 (0 or 255) of size [out_h, out_w]
  cv::Mat contour_overlay; // CV_8UC3 BGR image [out_h, out_w] with red cell contours
  double inference_ms;     // Forward pass execution latency in milliseconds
};

class SegmentationEngine {
public:
  /**
   * @brief Initializes the ONNX Runtime environment and loads the U-Net model.
   * Inspects model input dimensions if specified in graph metadata.
   * @param model_path Path to serialized model.onnx file.
   * @param num_threads Number of intra-op CPU threads for ONNX Runtime.
   * @param device Execution provider preference: "auto", "cuda", "dnnl", or "cpu".
   */
  explicit SegmentationEngine(const std::string &model_path,
                              int num_threads = 4,
                              const std::string &device = "auto");

  ~SegmentationEngine() = default;

  // Disable copying since Ort::Session manages native runtime resources
  SegmentationEngine(const SegmentationEngine &) = delete;
  SegmentationEngine &operator=(const SegmentationEngine &) = delete;

  /**
   * @brief Runs dummy forward passes to warm up execution provider kernels and
   * memory arenas prior to latency-critical inference.
   * @param iterations Number of warm-up passes to execute.
   */
  void warmup(int iterations = 1);

  /**
   * @brief Runs end-to-end preprocessing, ONNX inference, and contour
   * postprocessing.
   * @param bgr_image Raw input image loaded via cv::imread (BGR format).
   * @param prob_threshold Probability threshold p in (0, 1); default 0.5
   * corresponds to logit > 0.0f.
   */
  SegmentationResult predict(const cv::Mat &bgr_image,
                             float prob_threshold = 0.5f);

  /**
   * @brief Runs end-to-end preprocessing, ONNX inference, and contour postprocessing
   * while measuring individual stage latencies in milliseconds.
   */
  SegmentationResult predict(const cv::Mat &bgr_image,
                             float prob_threshold,
                             double &out_preprocess_ms,
                             double &out_inference_ms,
                             double &out_postprocess_ms);

  /**
   * @brief Executes a single forward pass using currently bound input tensor buffers,
   * returning pure graph execution latency in milliseconds.
   */
  double run_pure_inference();

  int input_height() const noexcept { return input_height_; }
  int input_width() const noexcept { return input_width_; }
  const std::string &active_provider() const noexcept {
    return active_provider_;
  }

private:
  /**
   * @brief Converts a BGR uint8 OpenCV image (pre-sized to model dimensions) into
   * a flat NCHW float32 tensor normalized with ImageNet stats.
   */
  std::vector<float> preprocess(const cv::Mat &resized_bgr) const;

  /**
   * @brief Fused single-pass BGR uint8 -> planar NCHW float32 normalization
   * writing directly into a pre-allocated buffer (zero intermediate allocations).
   */
  void preprocess_into(const cv::Mat &resized_bgr, float *dst) const;

  /**
   * @brief Thresholds raw output logits and draws predicted cell membrane
   * contours on the aligned RGB crop.
   */
  void postprocess(const float *logits, int out_h, int out_w,
                   const cv::Mat &resized_bgr, float prob_threshold,
                   cv::Mat &out_binary_mask,
                   cv::Mat &out_contour_overlay) const;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;
  Ort::RunOptions run_options_;

  std::string active_provider_{"CPUExecutionProvider"};
  std::string input_name_{"input"};
  std::string output_name_{"logits"};
  std::array<const char *, 1> input_names_{nullptr};
  std::array<const char *, 1> output_names_{nullptr};

  int input_height_{416};
  int input_width_{416};
  std::array<int64_t, 4> input_shape_{1, 3, 416, 416};
  std::vector<float> input_buffer_;
  Ort::Value input_tensor_{nullptr};
  std::unique_ptr<Ort::IoBinding> io_binding_;

  static constexpr std::array<float, 3> kImageNetMean{0.485f, 0.456f, 0.406f};
  static constexpr std::array<float, 3> kImageNetStd{0.229f, 0.224f, 0.225f};

  // Precomputed scale = 1 / (255 * std) and bias = -mean / std for fused FMA
  static constexpr std::array<float, 3> kNormScale{
      1.0f / (255.0f * kImageNetStd[0]),
      1.0f / (255.0f * kImageNetStd[1]),
      1.0f / (255.0f * kImageNetStd[2])};
  static constexpr std::array<float, 3> kNormBias{
      -kImageNetMean[0] / kImageNetStd[0],
      -kImageNetMean[1] / kImageNetStd[1],
      -kImageNetMean[2] / kImageNetStd[2]};
};

} // namespace bcsd
