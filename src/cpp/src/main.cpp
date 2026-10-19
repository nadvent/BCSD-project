#include "inference_engine.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

int main(int argc, char *argv[]) {
  std::string model_path = "./outputs/model.onnx";
  std::string image_path;
  std::string output_prefix = "./outputs/cpp_pred";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--image" && i + 1 < argc) {
      image_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_prefix = argv[++i];
    }
  }

  if (image_path.empty()) {
    std::cerr << "Usage: " << argv[0] << " --model <path/to/model.onnx>"
              << " --image <path/to/blood_smear.jpg>"
              << " [--output <output_prefix>]\n";
    return 1;
  }

  try {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
      std::cerr << "Error: Could not read image at '" << image_path << "'.\n";
      return 1;
    }

    std::cout << "Loading ONNX model from: " << model_path << "\n";
    bcsd::SegmentationEngine engine(model_path, 4);

    bcsd::SegmentationResult result = engine.predict(image, 0.5f);

    std::string mask_out = output_prefix + "_mask.png";
    std::string overlay_out = output_prefix + "_overlay.png";

    cv::imwrite(mask_out, result.binary_mask);
    cv::imwrite(overlay_out, result.contour_overlay);

    std::cout << "--------------------------------------------------\n";
    std::cout << "C++ ONNX Runtime Inference Complete!\n";
    std::cout << "  Forward Pass Latency : " << result.inference_ms << " ms\n";
    std::cout << "  Output Mask Size     : " << result.binary_mask.cols << "x"
              << result.binary_mask.rows << "\n";
    std::cout << "  Saved Binary Mask    : " << mask_out << "\n";
    std::cout << "  Saved Contour Overlay: " << overlay_out << "\n";
    std::cout << "--------------------------------------------------\n";
  } catch (const std::exception &e) {
    std::cerr << "Inference Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
