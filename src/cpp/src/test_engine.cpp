#include "inference_engine.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace fs = std::filesystem;

#define EXPECT_TRUE(cond, msg)                                                \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::cerr << "FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__  \
                << ")\n";                                                     \
      return 1;                                                               \
    }                                                                         \
  } while (0)

int main() {
  std::cout << "[BCSD C++ Test] Starting SegmentationEngine unit tests...\n";

  // Test 1: Non-existent model path throws an exception
  bool threw_on_missing_model = false;
  try {
    bcsd::SegmentationEngine invalid_engine("non_existent_model.onnx", 1, "cpu");
  } catch (const std::exception &) {
    threw_on_missing_model = true;
  }
  EXPECT_TRUE(threw_on_missing_model, "Engine should throw when model path is invalid.");
  std::cout << "  [PASS] Missing model throws exception as expected.\n";

  // Check if outputs/model.onnx exists (or search from build directory)
  std::string model_path = "outputs/model.onnx";
  if (!fs::exists(model_path)) {
    model_path = "../../../outputs/model.onnx";
  }
  if (!fs::exists(model_path)) {
    model_path = "../../outputs/model.onnx";
  }

  if (!fs::exists(model_path)) {
    std::cout << "  [SKIP] Model outputs/model.onnx not found, skipping live inference tests.\n";
    return 0;
  }

  std::cout << "  Loading model from: " << model_path << "\n";
  bcsd::SegmentationEngine engine(model_path, 2, "cpu");
  EXPECT_TRUE(engine.input_height() == 416, "Expected input height of 416.");
  EXPECT_TRUE(engine.input_width() == 416, "Expected input width of 416.");
  std::cout << "  [PASS] Engine initialized with target dimensions: "
            << engine.input_width() << "x" << engine.input_height() << "\n";

  // Test 2: Empty image throws std::invalid_argument
  bool threw_on_empty_image = false;
  try {
    cv::Mat empty_img;
    (void)engine.predict(empty_img, 0.5f);
  } catch (const std::invalid_argument &) {
    threw_on_empty_image = true;
  }
  EXPECT_TRUE(threw_on_empty_image, "Engine should throw on empty input image.");
  std::cout << "  [PASS] Empty image throws std::invalid_argument.\n";

  // Test 3: Invalid probability threshold throws std::invalid_argument
  bool threw_on_invalid_thresh = false;
  cv::Mat valid_dummy(416, 416, CV_8UC3, cv::Scalar(100, 100, 100));
  try {
    (void)engine.predict(valid_dummy, 0.0f);
  } catch (const std::invalid_argument &) {
    threw_on_invalid_thresh = true;
  }
  EXPECT_TRUE(threw_on_invalid_thresh, "Engine should throw on threshold <= 0.0.");

  threw_on_invalid_thresh = false;
  try {
    (void)engine.predict(valid_dummy, 1.0f);
  } catch (const std::invalid_argument &) {
    threw_on_invalid_thresh = true;
  }
  EXPECT_TRUE(threw_on_invalid_thresh, "Engine should throw on threshold >= 1.0.");
  std::cout << "  [PASS] Invalid threshold boundaries throw std::invalid_argument.\n";

  // Test 4: Pure forward pass latency measurement
  engine.warmup(1);
  double pure_ms = engine.run_pure_inference();
  EXPECT_TRUE(pure_ms > 0.0, "Pure inference latency should be positive.");
  std::cout << "  [PASS] Pure forward pass succeeded: " << pure_ms << " ms\n";

  // Test 5: End-to-end predict on 416x416 dummy image
  double pre_ms = 0.0, inf_ms = 0.0, post_ms = 0.0;
  bcsd::SegmentationResult res = engine.predict(valid_dummy, 0.5f, pre_ms, inf_ms, post_ms);
  EXPECT_TRUE(!res.binary_mask.empty(), "Binary mask should not be empty.");
  EXPECT_TRUE(!res.contour_overlay.empty(), "Contour overlay should not be empty.");
  EXPECT_TRUE(res.binary_mask.cols == 228, "Expected binary mask cols == 228.");
  EXPECT_TRUE(res.binary_mask.rows == 228, "Expected binary mask rows == 228.");
  EXPECT_TRUE(res.binary_mask.type() == CV_8UC1, "Expected CV_8UC1 binary mask.");
  EXPECT_TRUE(res.contour_overlay.cols == 228, "Expected contour overlay cols == 228.");
  EXPECT_TRUE(res.contour_overlay.rows == 228, "Expected contour overlay rows == 228.");
  EXPECT_TRUE(res.contour_overlay.type() == CV_8UC3, "Expected CV_8UC3 contour overlay.");
  EXPECT_TRUE(res.inference_ms > 0.0, "Inference latency should be positive.");
  EXPECT_TRUE(pre_ms > 0.0, "Preprocess latency should be positive.");
  EXPECT_TRUE(post_ms > 0.0, "Postprocess latency should be positive.");
  std::cout << "  [PASS] Predict on 416x416 produced correct 228x228 mask & overlay (infer: "
            << res.inference_ms << " ms).\n";

  // Test 6: End-to-end predict on non-standard resolution (internal resize test)
  cv::Mat non_standard(250, 350, CV_8UC3, cv::Scalar(150, 80, 200));
  bcsd::SegmentationResult res2 = engine.predict(non_standard, 0.5f);
  EXPECT_TRUE(res2.binary_mask.cols == 228, "Expected binary mask cols == 228 after resize.");
  EXPECT_TRUE(res2.binary_mask.rows == 228, "Expected binary mask rows == 228 after resize.");
  std::cout << "  [PASS] Internal resize from 250x350 to model target verified.\n";

  std::cout << "\n==================================================\n";
  std::cout << "All C++ SegmentationEngine unit tests passed!\n";
  std::cout << "==================================================\n";
  return 0;
}
