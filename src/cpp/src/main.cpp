#include "inference_engine.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

namespace fs = std::filesystem;

static void print_usage(const char *prog_name) {
  std::cout
      << "Blood Cell Segmentation C++ Inference App\n"
      << "Usage: " << prog_name << " [options]\n\n"
      << "Options:\n"
      << "  -i, --image <path>       (Required) Path to input image (e.g. blood_smear.jpg)\n"
      << "  -m, --model <path>       Path to ONNX model (default: ./outputs/model.onnx)\n"
      << "  -o, --output <prefix>    Output path prefix (default: ./outputs/cpp_pred)\n"
      << "  -t, --threads <num>      Number of CPU threads for inference (default: 4)\n"
      << "  -p, --threshold <float>  Segmentation probability threshold in (0, 1) (default: 0.5)\n"
      << "  -h, --help               Display this help message and exit\n\n"
      << "Example:\n"
      << "  " << prog_name << " --image data/test/original/001.jpg --model outputs/model.onnx\n";
}

int main(int argc, char *argv[]) {
  std::string model_path = "./outputs/model.onnx";
  std::string image_path;
  std::string output_prefix = "./outputs/cpp_pred";
  int num_threads = 4;
  float prob_threshold = 0.5f;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "-m" || arg == "--model") {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << arg << " requires an argument.\n";
        print_usage(argv[0]);
        return 1;
      }
      model_path = argv[++i];
    } else if (arg == "-i" || arg == "--image") {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << arg << " requires an argument.\n";
        print_usage(argv[0]);
        return 1;
      }
      image_path = argv[++i];
    } else if (arg == "-o" || arg == "--output") {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << arg << " requires an argument.\n";
        print_usage(argv[0]);
        return 1;
      }
      output_prefix = argv[++i];
    } else if (arg == "-t" || arg == "--threads") {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << arg << " requires an argument.\n";
        print_usage(argv[0]);
        return 1;
      }
      try {
        num_threads = std::stoi(argv[++i]);
        if (num_threads <= 0) {
          std::cerr << "Error: Thread count must be positive.\n";
          return 1;
        }
      } catch (const std::exception &e) {
        std::cerr << "Error: Invalid thread count '" << argv[i] << "'.\n";
        return 1;
      }
    } else if (arg == "-p" || arg == "--threshold") {
      if (i + 1 >= argc) {
        std::cerr << "Error: " << arg << " requires an argument.\n";
        print_usage(argv[0]);
        return 1;
      }
      try {
        prob_threshold = std::stof(argv[++i]);
        if (prob_threshold <= 0.0f || prob_threshold >= 1.0f) {
          std::cerr << "Error: Threshold must be strictly between 0.0 and 1.0.\n";
          return 1;
        }
      } catch (const std::exception &e) {
        std::cerr << "Error: Invalid probability threshold '" << argv[i] << "'.\n";
        return 1;
      }
    } else {
      std::cerr << "Error: Unrecognized option '" << arg << "'.\n\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (image_path.empty()) {
    std::cerr << "Error: Missing required option: --image <path>\n\n";
    print_usage(argv[0]);
    return 1;
  }

  if (!fs::exists(image_path)) {
    std::cerr << "Error: Input image file does not exist: '" << image_path << "'\n";
    return 1;
  }

  if (!fs::exists(model_path)) {
    std::cerr << "Error: ONNX model file does not exist: '" << model_path << "'\n";
    return 1;
  }

  try {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
      std::cerr << "Error: OpenCV failed to decode image at '" << image_path << "'.\n";
      return 1;
    }

    std::cout << "Loading ONNX model from: " << model_path << "\n";
    bcsd::SegmentationEngine engine(model_path, num_threads);

    bcsd::SegmentationResult result = engine.predict(image, prob_threshold);

    // Ensure destination directory exists before writing images
    fs::path prefix_path(output_prefix);
    fs::path parent_dir = prefix_path.parent_path();
    if (!parent_dir.empty() && !fs::exists(parent_dir)) {
      std::error_code ec;
      if (!fs::create_directories(parent_dir, ec)) {
        std::cerr << "Error: Failed to create output directory '" << parent_dir.string()
                  << "': " << ec.message() << "\n";
        return 1;
      }
    }

    std::string mask_out = output_prefix + "_mask.png";
    std::string overlay_out = output_prefix + "_overlay.png";

    if (!cv::imwrite(mask_out, result.binary_mask)) {
      std::cerr << "Error: Failed to save binary mask to '" << mask_out << "'.\n";
      return 1;
    }
    if (!cv::imwrite(overlay_out, result.contour_overlay)) {
      std::cerr << "Error: Failed to save contour overlay to '" << overlay_out << "'.\n";
      return 1;
    }

    std::cout << "--------------------------------------------------\n";
    std::cout << "C++ ONNX Runtime Inference Complete!\n";
    std::cout << "  Input Dimensions     : " << image.cols << "x" << image.rows << "\n";
    std::cout << "  Model Target Res     : " << engine.input_width() << "x"
              << engine.input_height() << "\n";
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
