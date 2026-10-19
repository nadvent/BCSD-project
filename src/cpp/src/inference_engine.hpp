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
  cv::Mat contour_overlay; // CV_8UC3 BGR image [out_h, out_w] with green/red
                           // cell contours
  double inference_ms;     // Forward pass execution latency in milliseconds
};

class SegmentationEngine {
public:
  /**
   * @brief Initializes the ONNX Runtime environment and loads the static U-Net
   * model.
   * @param model_path Path to serialized model.onnx file.
   * @param num_threads Number of intra-op CPU threads for ONNX Runtime.
   */
  explicit SegmentationEngine(const std::string &model_path,
                              int num_threads = 4);

  ~SegmentationEngine() = default;

  // Disable copying since Ort::Session manages native runtime resources
  SegmentationEngine(const SegmentationEngine &) = delete;
  SegmentationEngine &operator=(const SegmentationEngine &) = delete;

  /**
   * @brief Runs end-to-end preprocessing, ONNX inference, and contour
   * postprocessing.
   * @param bgr_image Raw input image loaded via cv::imread (BGR format).
   * @param prob_threshold Probability threshold p in (0, 1); default 0.5
   * corresponds to logit > 0.0f.
   */
  SegmentationResult predict(const cv::Mat &bgr_image,
                             float prob_threshold = 0.5f);

private:
  /**
   * @brief Converts a BGR uint8 OpenCV image into a flat NCHW float32 tensor
   * normalized with ImageNet stats.
   */
  std::vector<float> preprocess(const cv::Mat &bgr_image) const;

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

  int input_height_{416};
  int input_width_{416};

  static constexpr std::array<float, 3> kImageNetMean{0.485f, 0.456f, 0.406f};
  static constexpr std::array<float, 3> kImageNetStd{0.229f, 0.224f, 0.225f};
};

} // namespace bcsd
