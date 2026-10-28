from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]
import torch
from model import UNet


def benchmark_fn(
    fn: Callable[[], Any],
    warmup: int = 25,
    runs: int = 100,
    sync_cuda: bool = True,
) -> dict[str, float]:
    """Measures steady-state execution time of a callable function."""
    # 1. Warmup passes
    for _ in range(warmup):
        _ = fn()
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    # 2. Timed benchmark passes
    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)  # to ms

    arr = np.array(timings, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr)),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Benchmarking on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n"
    )

    input_tensor = torch.randn(1, 3, 416, 416, device=device)
    input_np: np.ndarray = input_tensor.detach().cpu().numpy()

    # --- 1. PyTorch Eager Mode ---
    model_eager = UNet(in_channels=3, out_channels=1, padding=0).to(device)
    _ = model_eager.eval()
    with torch.no_grad():
        eager_stats = benchmark_fn(lambda: model_eager(input_tensor))

    # --- 2. PyTorch with torch.compile() ---
    # mode="reduce-overhead" uses CUDA graphs to eliminate Python-to-C++ kernel launch overhead
    model_compiled = torch.compile(model_eager, mode="reduce-overhead")
    with torch.no_grad():
        compile_stats = benchmark_fn(lambda: model_compiled(input_tensor))

    # --- 3. Python ONNX Runtime ---
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if torch.cuda.is_available()
        else ["CPUExecutionProvider"]
    )
    ort_session = ort.InferenceSession("outputs/model.onnx", providers=providers)
    ort_inputs = {"input": input_np}
    ort_stats = benchmark_fn(
        lambda: ort_session.run(["logits"], ort_inputs), sync_cuda=False
    )

    # --- Summary Table ---
    print(
        f"{'Framework':<28} | {'Median (P50)':<12} | {'Mean ± Std':<18} | {'P95 Latency':<12}"
    )
    print("-" * 78)
    for name, s in [
        ("PyTorch Eager", eager_stats),
        ("torch.compile(reduce-overhead)", compile_stats),
        ("Python ONNX Runtime", ort_stats),
    ]:
        print(
            f"{name:<28} | {s['p50']:>8.2f} ms | {s['mean']:>6.2f} ± {s['std']:<6.2f} ms | {s['p95']:>8.2f} ms"
        )


if __name__ == "__main__":
    main()
