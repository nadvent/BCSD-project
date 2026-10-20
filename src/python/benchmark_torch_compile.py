from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

# Ensure local imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import UNet


def sync_cuda(device: torch.device) -> None:
    """Synchronize CUDA GPU queue to ensure exact timing without asynchronous skew."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def compute_stats(timings: List[float]) -> Dict[str, float]:
    """Calculate descriptive percentiles and latency distribution metrics in ms."""
    arr = np.array(timings, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_model(weights_path: str | None, device: torch.device, padding: int = 0) -> nn.Module:
    """Instantiate and switch model to evaluation mode."""
    model = UNet(in_channels=3, out_channels=1, padding=padding).to(device)
    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def benchmark_pure_inference(
    fn,
    warmup: int,
    runs: int,
    device: torch.device,
) -> Tuple[Dict[str, float], float]:
    """Measure compilation / first-call latency followed by steady-state inference."""
    # Measure initial execution (includes TorchDynamo tracing + Inductor compilation)
    sync_cuda(device)
    t_comp_start = time.perf_counter()
    with torch.no_grad():
        _ = fn()
    sync_cuda(device)
    t_comp_end = time.perf_counter()
    first_call_ms = (t_comp_end - t_comp_start) * 1000.0

    # Remaining warmup passes
    with torch.no_grad():
        for _ in range(max(0, warmup - 1)):
            _ = fn()
    sync_cuda(device)

    # Timed steady-state benchmark passes
    timings: List[float] = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = fn()
            sync_cuda(device)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    stats = compute_stats(timings)
    return stats, first_call_ms


def run_benchmark(
    weights_path: str | None,
    image_path: str,
    device_str: str,
    runs: int,
    warmup: int,
    padding: int,
    compare_onnx: bool,
):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'")

    bgr_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(f"OpenCV failed to read image at '{image_path}'")

    print("======================================================================")
    print("               PYTORCH TORCH.COMPILE() SYSTEMS BENCHMARK             ")
    print("======================================================================")
    print(f"  Target Device   : {device} ({device_name})")
    print(f"  Sample Image    : {image_path} ({bgr_image.shape[1]}x{bgr_image.shape[0]})")
    print(f"  Weights Checkpoint: {weights_path if weights_path else 'None (Random Init)'}")
    print(f"  U-Net Padding   : {padding} ({'416->228 unpadded' if padding == 0 else '416->416 padded'})")
    print(f"  Warmup Passes   : {warmup}")
    print(f"  Timed Passes    : {runs}")
    print("----------------------------------------------------------------------\n")

    input_tensor = torch.randn(1, 3, 416, 416, device=device, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # 1. PyTorch Eager Baseline
    # -------------------------------------------------------------------------
    print("[1/3] Benchmarking PyTorch Eager baseline...")
    model_eager = load_model(weights_path, device, padding)
    eager_stats, eager_first_ms = benchmark_pure_inference(
        lambda: model_eager(input_tensor), warmup, runs, device
    )

    # -------------------------------------------------------------------------
    # 2. torch.compile(mode="default")
    # -------------------------------------------------------------------------
    print("[2/3] Benchmarking torch.compile(mode='default')...")
    model_default = torch.compile(model_eager, mode="default")
    default_stats, default_first_ms = benchmark_pure_inference(
        lambda: model_default(input_tensor), warmup, runs, device
    )

    # -------------------------------------------------------------------------
    # 3. torch.compile(mode="reduce-overhead") (CUDA Graphs enabled)
    # -------------------------------------------------------------------------
    print("[3/3] Benchmarking torch.compile(mode='reduce-overhead')...")
    if device.type == "cuda":
        model_reduce = torch.compile(model_eager, mode="reduce-overhead")
        reduce_stats, reduce_first_ms = benchmark_pure_inference(
            lambda: model_reduce(input_tensor), warmup, runs, device
        )
    else:
        reduce_stats = None
        reduce_first_ms = 0.0

    # -------------------------------------------------------------------------
    # Optional: Python ONNX Runtime Comparison
    # -------------------------------------------------------------------------
    ort_stats = None
    if compare_onnx and os.path.exists("outputs/model.onnx"):
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.enable_cpu_mem_arena = True
            opts.add_session_config_entry("session.intra_op.allow_spinning", "1")

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device.type == "cuda"
                else ["CPUExecutionProvider"]
            )
            session = ort.InferenceSession("outputs/model.onnx", opts, providers=providers)
            active_ep = session.get_providers()[0]
            dummy_np = input_tensor.detach().cpu().numpy()
            feed = {session.get_inputs()[0].name: dummy_np}
            out_name = session.get_outputs()[0].name

            # Warmup
            for _ in range(warmup):
                _ = session.run([out_name], feed)
            sync_cuda(device)

            ort_timings = []
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = session.run([out_name], feed)
                sync_cuda(device)
                t1 = time.perf_counter()
                ort_timings.append((t1 - t0) * 1000.0)
            ort_stats = compute_stats(ort_timings)
        except Exception as e:
            print(f"Warning: ONNX comparison failed ({e})")

    # -------------------------------------------------------------------------
    # Summary Table: Pure Forward Pass
    # -------------------------------------------------------------------------
    print("\n============================== RESULTS ==============================\n")
    print("--- Tier 1: Pure Forward Pass Latency ---")
    header = f"{'Framework / Mode':<32} | {'Median (P50)':<12} | {'Mean +/- Std':<18} | {'P95 Latency':<12} | {'Compilation / 1st':<17}"
    print(header)
    print("-" * len(header))

    modes = [
        ("PyTorch Eager", eager_stats, eager_first_ms),
        ("torch.compile(default)", default_stats, default_first_ms),
    ]
    if reduce_stats is not None:
        modes.append(("torch.compile(reduce-overhead)", reduce_stats, reduce_first_ms))
    if ort_stats is not None:
        modes.append((f"Python ORT ({active_ep[:4]})", ort_stats, 0.0))

    for name, s, fst in modes:
        fst_str = f"{fst:.1f} ms" if fst > 0.0 else "N/A"
        print(
            f"{name:<32} | {s['p50']:>8.2f} ms | {s['mean']:>6.2f} +/- {s['std']:<5.2f} ms | {s['p95']:>8.2f} ms | {fst_str:>15}"
        )

    # -------------------------------------------------------------------------
    # Tier 2: In-Memory End-to-End Pipeline with torch.compile()
    # -------------------------------------------------------------------------
    print("\n--- Tier 2: In-Memory End-to-End Pipeline (Breakdown) ---")
    best_compiled = (
        model_reduce if (reduce_stats is not None) else model_default
    )

    mean_arr = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def pipeline_step():
        t_pre0 = time.perf_counter()
        resized = cv2.resize(bgr_image, (416, 416), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - mean_arr) / std_arr
        chw = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]
        t_in = torch.from_numpy(np.ascontiguousarray(chw)).to(device)
        t_pre1 = time.perf_counter()

        t_inf0 = time.perf_counter()
        logits_tensor = best_compiled(t_in)
        sync_cuda(device)
        t_inf1 = time.perf_counter()

        t_post0 = time.perf_counter()
        logits = logits_tensor[0, 0].detach().cpu().numpy()
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

    for _ in range(warmup):
        _ = pipeline_step()
    sync_cuda(device)

    pre_list, inf_list, post_list, tot_list = [], [], [], []
    for _ in range(runs):
        p_ms, i_ms, pos_ms, tot_ms = pipeline_step()
        pre_list.append(p_ms)
        inf_list.append(i_ms)
        post_list.append(pos_ms)
        tot_list.append(tot_ms)

    s_pre = compute_stats(pre_list)
    s_inf = compute_stats(inf_list)
    s_post = compute_stats(post_list)
    s_tot = compute_stats(tot_list)

    print("  Stage                  | Median (P50) | Mean +/- Std         | P95 Latency")
    print("  -------------------------------------------------------------------------")
    print(
        f"  1. Preprocessing (CV)  | {s_pre['p50']:>8.2f} ms | {s_pre['mean']:>6.2f} +/- {s_pre['std']:<5.2f} ms | {s_pre['p95']:>8.2f} ms"
    )
    print(
        f"  2. Forward Pass (Torch)| {s_inf['p50']:>8.2f} ms | {s_inf['mean']:>6.2f} +/- {s_inf['std']:<5.2f} ms | {s_inf['p95']:>8.2f} ms"
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


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch torch.compile() Systems Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        default="outputs/best_model.pth",
        help="Path to trained PyTorch weights checkpoint",
    )
    parser.add_argument(
        "--image",
        default="data/test/original/e02301ac-68f7-4daf-b2ba-254b917f1cca.jpg",
        help="Path to sample input image",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target execution device",
    )
    parser.add_argument(
        "--runs", type=int, default=100, help="Number of timed benchmark passes"
    )
    parser.add_argument(
        "--warmup", type=int, default=25, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--padding", type=int, default=0, choices=[0, 1], help="U-Net padding mode"
    )
    parser.add_argument(
        "--compare-onnx",
        action="store_true",
        default=True,
        help="Include Python ONNX Runtime in the benchmark table",
    )

    args = parser.parse_args()
    run_benchmark(
        args.weights,
        args.image,
        args.device,
        args.runs,
        args.warmup,
        args.padding,
        args.compare_onnx,
    )


if __name__ == "__main__":
    main()
