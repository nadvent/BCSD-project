from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import onnx  # pyright: ignore[reportMissingTypeStubs]
import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]
import torch
from config_utils import load_config
from model import UNet


def export_to_onnx(
    weights_path: Path,
    output_path: Path,
    input_height: int = 416,
    input_width: int = 416,
    padding: int = 0,
    opset_version: int = 17,
) -> None:
    """Serializes trained PyTorch U-Net weights into a static, constant-folded ONNX graph
    and verifies numerical parity with ONNX Runtime.
    """
    device = torch.device("cpu")

    # 1. Initialize architecture and load trained weights on CPU for clean tracing
    model = UNet(in_channels=3, out_channels=1, padding=padding).to(device)
    if not weights_path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found at '{weights_path}'.")

    state_dict: dict[str, torch.Tensor] = cast(
        dict[str, torch.Tensor],
        torch.load(weights_path, map_location=device, weights_only=True),
    )
    _ = model.load_state_dict(state_dict)

    # CRITICAL: Switch to eval() mode before exporting so BatchNorm2d uses frozen
    # running_mean/running_var and ONNX constant-folding can fuse BatchNorm into Conv2d.
    _ = model.eval()

    # 2. Construct static dummy input tensor [1, 3, H, W]
    dummy_input: torch.Tensor = torch.randn(
        1, 3, input_height, input_width, dtype=torch.float32, device=device
    )

    with torch.no_grad():
        pytorch_out: torch.Tensor = cast(torch.Tensor, model(dummy_input))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting model from '{weights_path}' to '{output_path}'...")
    print(f"  Static Input Shape  : {tuple(dummy_input.shape)}")
    print(f"  Static Output Shape : {tuple(pytorch_out.shape)}")

    # 3. Export to ONNX with constant folding enabled
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
    )

    # 4. Structural Graph Verification via ONNX Checker
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX graph structure verified successfully (onnx.checker passed).")

    # 5. Numerical Parity Verification: PyTorch vs. ONNX Runtime
    ort_session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )

    input_np: np.ndarray = dummy_input.detach().cpu().numpy()
    ort_inputs: dict[str, np.ndarray] = {"input": input_np}
    ort_outs = ort_session.run(["logits"], ort_inputs)
    ort_logits: np.ndarray = np.asarray(ort_outs[0], dtype=np.float32)

    pytorch_logits: np.ndarray = pytorch_out.detach().cpu().numpy()
    max_abs_diff: float = float(np.max(np.abs(pytorch_logits - ort_logits)))

    np.testing.assert_allclose(
        pytorch_logits,
        ort_logits,
        rtol=1e-3,
        atol=1e-4,
    )
    print(f"Numerical parity verified! Max absolute difference: {max_abs_diff:.2e}")


def main() -> None:
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    conf_args, _ = conf_parser.parse_known_args()

    cfg = load_config(conf_args.config)
    cfg_data = cfg.get("data", {})
    cfg_model = cfg.get("model", {})
    cfg_export = cfg.get("export", {})

    parser = argparse.ArgumentParser(
        description="Export trained PyTorch U-Net to static ONNX format.",
        parents=[conf_parser],
    )
    _ = parser.add_argument(
        "--weights",
        type=str,
        default=cfg_export.get("weights", "./outputs/best_model.pth"),
        help="Path to trained PyTorch state_dict (.pth).",
    )
    _ = parser.add_argument(
        "--output",
        type=str,
        default=cfg_export.get("output", "./outputs/model.onnx"),
        help="Output path for serialized .onnx model.",
    )
    _ = parser.add_argument(
        "--height",
        type=int,
        default=int(cfg_data.get("input_height", 416)),
        help="Static input image height.",
    )
    _ = parser.add_argument(
        "--width",
        type=int,
        default=int(cfg_data.get("input_width", 416)),
        help="Static input image width.",
    )
    _ = parser.add_argument(
        "--padding",
        type=int,
        default=int(cfg_model.get("padding", 0)),
        choices=[0, 1],
        help="Convolution padding used during training (0=unpadded, 1=padded).",
    )
    _ = parser.add_argument(
        "--opset",
        type=int,
        default=int(cfg_export.get("opset_version", 17)),
        help="ONNX opset version.",
    )
    args = parser.parse_args()

    export_to_onnx(
        weights_path=Path(cast(str, args.weights)),
        output_path=Path(cast(str, args.output)),
        input_height=cast(int, args.height),
        input_width=cast(int, args.width),
        padding=cast(int, args.padding),
        opset_version=cast(int, args.opset),
    )


if __name__ == "__main__":
    main()
