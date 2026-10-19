#include "inference_engine.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace bcsd {

SegmentationEngine::SegmentationEngine(const std::string &model_path,
                                       int num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "BloodCellSegmentation") {
  session_options_.SetIntraOpNumThreads(num_threads);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(),
                                            session_options_);

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
}

std::vector<float>
SegmentationEngine::preprocess(const cv::Mat &resized_bgr) const {
  if (resized_bgr.empty()) {
    throw std::invalid_argument("Input BGR image is empty.");
  }
  if (resized_bgr.rows != input_height_ || resized_bgr.cols != input_width_) {
    throw std::invalid_argument("Input image dimensions do not match model expectation.");
  }

  // 1. Convert OpenCV BGR -> RGB to match PyTorch training pipeline
  cv::Mat resized_rgb;
  cv::cvtColor(resized_bgr, resized_rgb, cv::COLOR_BGR2RGB);

  // 2. Scale uint8 [0, 255] -> float32 [0.0, 1.0]
  cv::Mat float_rgb;
  resized_rgb.convertTo(float_rgb, CV_32FC3, 1.0 / 255.0);

  // 3. HWC (Interleaved) -> NCHW (Planar) transposition + ImageNet Normalization
  const int hw = input_height_ * input_width_;
  std::vector<float> input_tensor(3 * hw);

  float *r_plane = input_tensor.data();
  float *g_plane = input_tensor.data() + hw;
  float *b_plane = input_tensor.data() + 2 * hw;

  for (int y = 0; y < input_height_; ++y) {
    const cv::Vec3f *row_ptr = float_rgb.ptr<cv::Vec3f>(y);
    for (int x = 0; x < input_width_; ++x) {
      const int flat_idx = y * input_width_ + x;
      r_plane[flat_idx] = (row_ptr[x][0] - kImageNetMean[0]) / kImageNetStd[0];
      g_plane[flat_idx] = (row_ptr[x][1] - kImageNetMean[1]) / kImageNetStd[1];
      b_plane[flat_idx] = (row_ptr[x][2] - kImageNetMean[2]) / kImageNetStd[2];
    }
  }

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

  out_binary_mask = cv::Mat(out_h, out_w, CV_8UC1);
  for (int y = 0; y < out_h; ++y) {
    uchar *mask_row = out_binary_mask.ptr<uchar>(y);
    for (int x = 0; x < out_w; ++x) {
      const float z = logits[y * out_w + x];
      mask_row[x] = (z > logit_threshold) ? 255 : 0;
    }
  }

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
  if (bgr_image.empty()) {
    throw std::invalid_argument("Input BGR image is empty.");
  }
  if (prob_threshold <= 0.0f || prob_threshold >= 1.0f) {
    throw std::invalid_argument(
        "Probability threshold must be strictly between 0.0 and 1.0.");
  }

  // Resize input to model dimensions only once
  cv::Mat resized_bgr;
  if (bgr_image.rows == input_height_ && bgr_image.cols == input_width_) {
    resized_bgr = bgr_image;
  } else {
    cv::resize(bgr_image, resized_bgr, cv::Size(input_width_, input_height_),
               0.0, 0.0, cv::INTER_LINEAR);
  }

  std::vector<float> input_data = preprocess(resized_bgr);
  std::array<int64_t, 4> input_shape{1, 3, input_height_, input_width_};

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_data.data(), input_data.size(), input_shape.data(),
      input_shape.size());

  const char *input_names[] = {"input"};
  const char *output_names[] = {"logits"};

  auto start = std::chrono::high_resolution_clock::now();
  auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names,
                                      &input_tensor, 1, output_names, 1);
  auto end = std::chrono::high_resolution_clock::now();
  double inference_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Extract output shape [1, 1, out_h, out_w] and raw float pointer
  auto type_info = output_tensors.front().GetTensorTypeAndShapeInfo();
  std::vector<int64_t> out_shape = type_info.GetShape();
  const int out_h = static_cast<int>(out_shape[2]);
  const int out_w = static_cast<int>(out_shape[3]);

  const float *logits_ptr = output_tensors.front().GetTensorData<float>();

  cv::Mat binary_mask;
  cv::Mat contour_overlay;
  postprocess(logits_ptr, out_h, out_w, resized_bgr, prob_threshold,
              binary_mask, contour_overlay);

  return SegmentationResult{binary_mask, contour_overlay, inference_ms};
}

} // namespace bcsd
