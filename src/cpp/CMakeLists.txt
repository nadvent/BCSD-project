cmake_minimum_required(VERSION 3.16)

project(blood_cell_segmentation_cpp VERSION 1.0.0 LANGUAGES CXX)



# Enforce modern C++17 standard and export compile_commands.json for clangd

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



# Default to Release build for maximum compiler optimization (-O3)

if(NOT CMAKE_BUILD_TYPE)

    set(CMAKE_BUILD_TYPE Release)

endif()



# ==============================================================================

# 1. Locate OpenCV

# ==============================================================================

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} (version: ${OpenCV_VERSION})")



# ==============================================================================

# 2. Locate ONNX Runtime Headers & Library

# ==============================================================================

# Allow user to specify custom root via: cmake -DONNXRUNTIME_ROOT=/path ..

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation prefix")



find_path(ONNXRUNTIME_INCLUDE_DIR

    NAMES

        onnxruntime_cxx_api.h

    PATHS

        # Package manager locations (Ubuntu / Debian / Arch / Fedora)

        "/usr/include/onnxruntime"

        "/usr/include/onnxruntime/core/session"

        "/usr/include"

        "/usr/local/include/onnxruntime"

        "/usr/local/include/onnxruntime/core/session"

        "/usr/local/include"

        # User-specified custom root

        "${ONNXRUNTIME_ROOT}/include"

        "${ONNXRUNTIME_ROOT}/include/onnxruntime"

        "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"

    DOC "Path to onnxruntime_cxx_api.h"

)



find_library(ONNXRUNTIME_LIB

    NAMES

        onnxruntime

        libonnxruntime.so

        libonnxruntime.dylib

        onnxruntime.lib

    PATHS

        # System & multiarch directories

        "/usr/lib/x86_64-linux-gnu"

        "/usr/lib/aarch64-linux-gnu"

        "/usr/lib"

        "/usr/lib64"

        "/usr/local/lib"

        "/usr/local/lib64"

        "/usr/lib/onnxruntime"

        # User-specified custom root

        "${ONNXRUNTIME_ROOT}/lib"

        "${ONNXRUNTIME_ROOT}/lib64"

    DOC "Path to libonnxruntime"

)



# Diagnostic validation with helpful troubleshooting hints

if(NOT ONNXRUNTIME_INCLUDE_DIR)

    message(FATAL_ERROR

        "Could not find 'onnxruntime_cxx_api.h'.\n"

        "Please provide the include directory manually via:\n"

        "  cmake -DONNXRUNTIME_INCLUDE_DIR=/usr/include/onnxruntime ..\n"

    )

endif()



if(NOT ONNXRUNTIME_LIB)

    message(FATAL_ERROR

        "Could not find 'libonnxruntime.so' in standard library paths.\n"

        "To locate it on your system, run in your terminal:\n"

        "  find /usr -name 'libonnxruntime.so*' 2>/dev/null\n"

        "Then pass the library path to cmake, e.g.:\n"

        "  cmake -DONNXRUNTIME_LIB=/path/to/libonnxruntime.so ..\n"

    )

endif()



message(STATUS "Found ONNX Runtime Include : ${ONNXRUNTIME_INCLUDE_DIR}")

message(STATUS "Found ONNX Runtime Library : ${ONNXRUNTIME_LIB}")



# ==============================================================================

# 3. Define the Inference Executable Target

# ==============================================================================

add_executable(inference_app
    src/main.cpp
    src/inference_engine.cpp
    include/inference_engine.hpp
)



target_include_directories(inference_app PRIVATE

    ${CMAKE_CURRENT_SOURCE_DIR}/include

    ${OpenCV_INCLUDE_DIRS}

    ${ONNXRUNTIME_INCLUDE_DIR}

    # Also add parent directory so both <onnxruntime_cxx_api.h> and

    # <onnxruntime/onnxruntime_cxx_api.h> resolve seamlessly

    "${ONNXRUNTIME_INCLUDE_DIR}/.."

)



target_link_libraries(inference_app PRIVATE

    ${OpenCV_LIBS}

    ${ONNXRUNTIME_LIB}

    ${CMAKE_DL_LIBS}

)



# Enable strict modern C++ warnings and aggressive Release optimizations
target_compile_options(inference_app PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -march=native -funroll-loops>
)

# Enable Link-Time Optimization (LTO / IPO) in Release builds if supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set_target_properties(inference_app PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# ==============================================================================
# 4. Define the Systems Benchmark Target
# ==============================================================================
add_executable(benchmark_ort
    src/benchmark_ort.cpp
    src/inference_engine.cpp
    include/inference_engine.hpp
)

target_include_directories(benchmark_ort PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    "${ONNXRUNTIME_INCLUDE_DIR}/.."
)

target_link_libraries(benchmark_ort PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    ${CMAKE_DL_LIBS}
)

target_compile_options(benchmark_ort PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3 -march=native -funroll-loops>
)

if(ipo_supported)
    set_target_properties(benchmark_ort PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()



