from __future__ import annotations

import argparse
import os
import time
from typing import cast

import cv2
import numpy as np
import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def sync_cuda(is_cuda: bool) -> None:
    """Synchronize CUDA GPU stream to guarantee exact timing without async skew."""
    if is_cuda and HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def compute_stats(timings: list[float]) -> dict[str, float]:
    """Compute statistical percentiles from a series of execution latencies in ms."""
    arr = np.array(timings, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def create_ort_session(
    model_path: str, device: str = "auto", num_threads: int = 8
) -> tuple[ort.InferenceSession, str]:
    """Initialize ONNX Runtime InferenceSession matching C++ session options."""
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.intra_op_num_threads = num_threads
    sess_opts.enable_cpu_mem_arena = True
    sess_opts.enable_mem_pattern = True
    sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")

    available_eps: list[str] = ort.get_available_providers()
    dev: str = device.lower()

    if dev in ("cuda", "auto") and "CUDAExecutionProvider" in available_eps:
        # Preload cuDNN 9 sub-libraries if running on Linux
        try:
            import ctypes

            for lib in ("libcudnn_ops.so.9", "libcudnn_cnn.so.9", "libcudnn_graph.so.9"):
                _ = ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

        cuda_opts = {
            "device_id": 0,
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": 1,
        }
        providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    elif dev in ("dnnl", "onednn") and "DnnlExecutionProvider" in available_eps:
        providers = ["DnnlExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, sess_opts, providers=providers)
    active_ep = cast(str, session.get_providers()[0])
    return session, active_ep


def run_benchmark(
    model_path: str,
    image_path: str,
    device: str,
    runs: int,
    warmup: int,
    num_threads: int,
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(f"Failed to decode image from: {image_path}")

    session, active_ep = create_ort_session(model_path, device, num_threads)
    is_cuda = active_ep == "CUDAExecutionProvider"

    input_meta = session.get_inputs()[0]
    input_name: str = input_meta.name
    output_name: str = session.get_outputs()[0].name

    shape = input_meta.shape
    in_h = shape[2] if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0 else 416
    in_w = shape[3] if len(shape) == 4 and isinstance(shape[3], int) and shape[3] > 0 else 416

    print("======================================================================")
    print("             PYTHON ONNX RUNTIME SYSTEMS BENCHMARK                   ")
    print("======================================================================")
    print(f"  Model Path      : {model_path}")
    print(f"  Sample Image    : {image_path} ({bgr_image.shape[1]}x{bgr_image.shape[0]})")
    print(f"  Requested Dev   : {device}")
    print(f"  Intra-Op Threads: {num_threads}")
    print(f"  Warmup Passes   : {warmup}")
    print(f"  Timed Passes    : {runs}")
    print(f"  Active Provider : {active_ep}")
    print("----------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # TIER 1: Pure Forward Pass (Pre-allocated tensor in memory)
    # -------------------------------------------------------------------------
    print("[Tier 1] Benchmarking Pure ONNX Forward Pass (Tensor in memory)...")
    dummy_input = np.zeros((1, 3, in_h, in_w), dtype=np.float32)
    ort_feed = {input_name: dummy_input}

    for _ in range(warmup):
        _ = session.run([output_name], ort_feed)
    sync_cuda(is_cuda)

    tier1_timings: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = session.run([output_name], ort_feed)
        sync_cuda(is_cuda)
        t1 = time.perf_counter()
        tier1_timings.append((t1 - t0) * 1000.0)

    s1 = compute_stats(tier1_timings)

    # -------------------------------------------------------------------------
    # TIER 2: Full In-Memory Pipeline (Preprocess + Forward + Postprocess)
    # -------------------------------------------------------------------------
    print("[Tier 2] Benchmarking Full In-Memory Pipeline (Pre + Infer + Post)...")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def pipeline_step() -> tuple[float, float, float, float]:
        # Preprocessing
        t_pre0 = time.perf_counter()
        if bgr_image.shape[0] == in_h and bgr_image.shape[1] == in_w:
            resized = bgr_image
        else:
            resized = cv2.resize(bgr_image, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - mean) / std
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        tensor = np.ascontiguousarray(tensor)
        t_pre1 = time.perf_counter()

        # Inference
        t_inf0 = time.perf_counter()
        outputs = session.run([output_name], {input_name: tensor})
        sync_cuda(is_cuda)
        t_inf1 = time.perf_counter()

        # Postprocessing
        t_post0 = time.perf_counter()
        logits: np.ndarray = outputs[0][0, 0]
        out_h, out_w = logits.shape
        mask = (logits > 0.0).astype(np.uint8) * 255

        diff_y = (resized.shape[0] - out_h) // 2
        diff_x = (resized.shape[1] - out_w) // 2
        overlay = resized[diff_y : diff_y + out_h, diff_x : diff_x + out_w].copy()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
        t_post1 = time.perf_counter()

        pre_ms = (t_pre1 - t_pre0) * 1000.0
        inf_ms = (t_inf1 - t_inf0) * 1000.0
        post_ms = (t_post1 - t_post0) * 1000.0
        tot_ms = pre_ms + inf_ms + post_ms
        return pre_ms, inf_ms, post_ms, tot_ms

    # Warmup
    for _ in range(warmup):
        _ = pipeline_step()
    sync_cuda(is_cuda)

    pre_times: list[float] = []
    inf_times: list[float] = []
    post_times: list[float] = []
    tot_times: list[float] = []
    for _ in range(runs):
        p_ms, i_ms, pos_ms, tot_ms = pipeline_step()
        pre_times.append(p_ms)
        inf_times.append(i_ms)
        post_times.append(pos_ms)
        tot_times.append(tot_ms)

    s_pre = compute_stats(pre_times)
    s_inf = compute_stats(inf_times)
    s_post = compute_stats(post_times)
    s_tot = compute_stats(tot_times)

    print("\n============================== RESULTS ==============================\n")
    print("--- Tier 1: Pure Forward Pass ---")
    print(f"  Median (P50)  : {s1['p50']:>7.2f} ms")
    print(f"  Mean +/- Std  : {s1['mean']:>7.2f} +/- {s1['std']:.2f} ms")
    print(f"  P95 Latency   : {s1['p95']:>7.2f} ms")
    print(f"  Min / Max     : {s1['min']:>7.2f} / {s1['max']:.2f} ms")
    print(f"  Effective FPS : {1000.0 / s1['p50']:>7.2f} FPS\n")

    print("--- Tier 2: Full In-Memory Pipeline (Breakdown) ---")
    print("  Stage                  | Median (P50) | Mean +/- Std         | P95 Latency")
    print("  -------------------------------------------------------------------------")
    print(
        f"  1. Preprocessing (CV)  | {s_pre['p50']:>8.2f} ms | {s_pre['mean']:>6.2f} +/- {s_pre['std']:<5.2f} ms | {s_pre['p95']:>8.2f} ms"
    )
    print(
        f"  2. Forward Pass (ORT)  | {s_inf['p50']:>8.2f} ms | {s_inf['mean']:>6.2f} +/- {s_inf['std']:<5.2f} ms | {s_inf['p95']:>8.2f} ms"
    )
    print(
        f"  3. Postprocessing (CV) | {s_post['p50']:>8.2f} ms | {s_post['mean']:>6.2f} +/- {s_post['std']:<5.2f} ms | {s_post['p95']:>8.2f} ms"
    )
    print("  -------------------------------------------------------------------------")
    print(
        f"  Total In-Memory E2E    | {s_tot['p50']:>8.2f} ms | {s_tot['mean']:>6.2f} +/- {s_tot['std']:<5.2f} ms | {s_tot['p95']:>8.2f} ms"
    )
    print(f"  Pipeline Throughput    : {1000.0 / s_tot['p50']:.2f} FPS")
    print("======================================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Python ONNX Runtime Systems Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="outputs/model.onnx", help="Path to ONNX model"
    )
    parser.add_argument(
        "--image",
        default="data/test/original/e02301ac-68f7-4daf-b2ba-254b917f1cca.jpg",
        help="Path to input test image",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "dnnl", "cpu"],
        help="Execution provider",
    )
    parser.add_argument(
        "--runs", type=int, default=100, help="Number of timed benchmark passes"
    )
    parser.add_argument(
        "--warmup", type=int, default=25, help="Number of un-timed warmup passes"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="Number of intra-op threads"
    )

    args = parser.parse_args()
    run_benchmark(
        args.model,
        args.image,
        args.device,
        args.runs,
        args.warmup,
        args.threads,
    )


if __name__ == "__main__":
    main()
