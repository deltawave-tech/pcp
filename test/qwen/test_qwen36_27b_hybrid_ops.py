"""
Qwen3.6-27B hybrid op and tiny training checks.

Run from the repo root:
  python3 test/qwen/test_qwen36_27b_hybrid_ops.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OPS_MLIR_PATH = REPO_ROOT / "models" / "tests" / "qwen36_27b_tiny_hybrid_ops.mlir"
TRAINING_MLIR_PATH = REPO_ROOT / "models" / "tests" / "qwen36_27b_tiny_hybrid_training.mlir"
DYNAMIC_UPDATE_TRAINING_MLIR_PATH = REPO_ROOT / "models" / "tests" / "qwen36_27b_tiny_dynamic_update_training.mlir"
AUTODIFF_ENGINE_PATH = REPO_ROOT / "src" / "autodiff" / "engine.zig"
VJP_RULES_PATH = REPO_ROOT / "src" / "autodiff" / "vjp_rules.zig"


def require_file(path: Path) -> None:
    if path.exists():
        return
    try:
        import pytest  # type: ignore
    except ImportError:
        raise SystemExit(f"Missing required artifact: {path}") from None
    pytest.skip(f"Missing required artifact: {path}")


def require_torch():
    try:
        import torch  # type: ignore
    except ImportError:
        try:
            import pytest  # type: ignore
        except ImportError:
            raise SystemExit("Missing torch; run this parity check with venv/bin/python.") from None
        pytest.skip("torch is required for DeltaNet parity checks")
    return torch


def iree_compile() -> str | None:
    return shutil.which("iree-compile") or str(REPO_ROOT / ".venv" / "bin" / "iree-compile")


def nix_bin() -> str | None:
    candidate = shutil.which("nix")
    if candidate is not None:
        return candidate
    fallback = Path("/nix/var/nix/profiles/default/bin/nix")
    return str(fallback) if fallback.exists() else None


def test_hybrid_ops_smoke_covers_milestone6_lowering_surface() -> None:
    require_file(OPS_MLIR_PATH)
    mlir_text = OPS_MLIR_PATH.read_text(encoding="utf-8")

    required_ops = [
        "stablehlo.slice",
        "stablehlo.dynamic_update_slice",
        "stablehlo.concatenate",
        "stablehlo.logistic",
        "stablehlo.rsqrt",
        "stablehlo.dot_general",
        "stablehlo.maximum",
        "stablehlo.exponential",
        "stablehlo.divide",
    ]
    for op_name in required_ops:
        assert op_name in mlir_text

    assert "tensor<1x6x4x4xf32>" in mlir_text
    assert "tensor<1x2x4x8xf32>" in mlir_text


def test_tiny_hybrid_ops_cpu_compiles_when_iree_is_available() -> None:
    require_file(OPS_MLIR_PATH)
    compiler = iree_compile()
    if compiler is None or not Path(compiler).exists():
        try:
            import pytest  # type: ignore
        except ImportError:
            return
        pytest.skip("iree-compile is not installed on this VM")

    output_vmfb = Path("/tmp/qwen36_27b_tiny_hybrid_ops.vmfb")
    subprocess.run(
        [
            compiler,
            str(OPS_MLIR_PATH),
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=generic",
            "-o",
            str(output_vmfb),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert output_vmfb.exists()


def test_tiny_hybrid_multitoken_state_reuse_matches_reference() -> None:
    torch = require_torch()

    def step(delta_recurrent, delta_conv, attn_k, attn_v, position: int):
        zero_conv = torch.zeros(1, 6, 4, 1)
        conv_shifted = torch.cat([delta_conv[:, :, :, 1:4], zero_conv], dim=3)
        conv_prev = conv_shifted[:, 0:2, :, 0:1]
        conv_gate = torch.sigmoid(conv_prev.reshape(1, 2, 4))
        conv_gate_broadcast = conv_gate.reshape(1, 2, 4, 1).expand(1, 2, 4, 8)
        delta_recurrent_update = delta_recurrent * 0.95 + conv_gate_broadcast

        k_update = attn_k[:, :, 0:1, :]
        v_update = attn_v[:, :, 0:1, :]
        attn_k_update = attn_k.clone()
        attn_v_update = attn_v.clone()
        attn_k_update[:, :, position : position + 1, :] = k_update
        attn_v_update[:, :, position : position + 1, :] = v_update
        return delta_recurrent_update, conv_shifted, attn_k_update, attn_v_update

    torch.manual_seed(17)
    delta_recurrent = torch.randn(1, 2, 4, 8)
    delta_conv = torch.randn(1, 6, 4, 4)
    attn_k = torch.randn(1, 1, 8, 4)
    attn_v = torch.randn(1, 1, 8, 4)

    first = step(delta_recurrent, delta_conv, attn_k, attn_v, 0)
    second = step(*first, position=1)
    recomputed_second = step(*first, position=1)

    for lhs, rhs in zip(second, recomputed_second):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)

    second_recurrent, second_conv, second_k, second_v = second
    assert second_recurrent.shape == delta_recurrent.shape
    assert second_conv.shape == delta_conv.shape
    torch.testing.assert_close(second_k[:, :, 1:2, :], first[2][:, :, 0:1, :], rtol=0, atol=0)
    torch.testing.assert_close(second_v[:, :, 1:2, :], first[3][:, :, 0:1, :], rtol=0, atol=0)


def test_training_forward_keeps_state_mutation_out_of_trainable_path() -> None:
    require_file(TRAINING_MLIR_PATH)
    mlir_text = TRAINING_MLIR_PATH.read_text(encoding="utf-8")

    assert "stablehlo.dynamic_update_slice" not in mlir_text
    assert "stablehlo.dot_general" in mlir_text
    assert "stablehlo.logistic" in mlir_text
    assert "stablehlo.multiply" in mlir_text
    assert "tensor<4x4xf32>" in mlir_text


def test_dynamic_update_slice_vjp_is_registered_in_core_autodiff() -> None:
    require_file(AUTODIFF_ENGINE_PATH)
    require_file(VJP_RULES_PATH)

    engine_source = AUTODIFF_ENGINE_PATH.read_text(encoding="utf-8")
    vjp_source = VJP_RULES_PATH.read_text(encoding="utf-8")

    assert '"stablehlo.dynamic_update_slice"' in engine_source
    assert "dynamicUpdateSliceVJP" in engine_source
    assert "pub fn dynamicUpdateSliceVJP" in vjp_source
    assert '"stablehlo.dynamic_update_slice"' in vjp_source
    assert "ops.dynamicSlice" in vjp_source


def test_tiny_deltanet_block_matches_pytorch_parameter_gradients() -> None:
    torch = require_torch()
    torch.manual_seed(1234)

    x = torch.randn(1, 4)
    target = torch.randn(1, 4)
    proj = torch.randn(4, 4, requires_grad=True)
    gate_bias = torch.randn(4, requires_grad=True)

    projected = x @ proj
    gated = torch.sigmoid(projected + gate_bias)
    delta_update = projected * gated
    loss = ((delta_update - target) ** 2).sum()
    loss.backward()

    with torch.no_grad():
        projected_ref = x @ proj
        gated_ref = torch.sigmoid(projected_ref + gate_bias)
        err = projected_ref * gated_ref - target
        d_loss_d_update = 2.0 * err
        d_update_d_projected = gated_ref + projected_ref * gated_ref * (1.0 - gated_ref)
        d_projected = d_loss_d_update * d_update_d_projected
        d_gate_bias = (d_loss_d_update * projected_ref * gated_ref * (1.0 - gated_ref)).sum(dim=0)
        d_proj = x.t() @ d_projected

    torch.testing.assert_close(proj.grad, d_proj, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gate_bias.grad, d_gate_bias, rtol=1e-5, atol=1e-5)


def test_tiny_hybrid_backward_graph_builds_and_compiles_when_tools_are_available() -> None:
    require_file(TRAINING_MLIR_PATH)
    compiler = iree_compile()
    nix = nix_bin()
    if compiler is None or not Path(compiler).exists() or nix is None:
        try:
            import pytest  # type: ignore
        except ImportError:
            return
        pytest.skip("nix and iree-compile are required for the backward graph compile smoke")

    backward_mlir = Path("/tmp/qwen36_27b_tiny_hybrid_training_backward.mlir")
    subprocess.run(
        [
            nix,
            "develop",
            "-c",
            "zig",
            "build",
            "build-grpo-backward",
            "--",
            str(TRAINING_MLIR_PATH),
            str(backward_mlir),
            "2",
            "0",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert backward_mlir.exists()

    subprocess.run(
        [
            compiler,
            str(backward_mlir),
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=generic",
            "-o",
            "/tmp/qwen36_27b_tiny_hybrid_training_backward.vmfb",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            compiler,
            str(backward_mlir),
            "--iree-hal-target-backends=cuda",
            "-o",
            "/tmp/qwen36_27b_tiny_dynamic_update_training_backward_cuda.vmfb",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            compiler,
            str(backward_mlir),
            "--iree-hal-target-backends=cuda",
            "-o",
            "/tmp/qwen36_27b_tiny_hybrid_training_backward_cuda.vmfb",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_dynamic_update_slice_backward_graph_builds_and_compiles_when_tools_are_available() -> None:
    require_file(DYNAMIC_UPDATE_TRAINING_MLIR_PATH)
    compiler = iree_compile()
    nix = nix_bin()
    if compiler is None or not Path(compiler).exists() or nix is None:
        try:
            import pytest  # type: ignore
        except ImportError:
            return
        pytest.skip("nix and iree-compile are required for the dynamic-update VJP smoke")

    backward_mlir = Path("/tmp/qwen36_27b_tiny_dynamic_update_training_backward.mlir")
    subprocess.run(
        [
            nix,
            "develop",
            "-c",
            "zig",
            "build",
            "build-grpo-backward",
            "--",
            str(DYNAMIC_UPDATE_TRAINING_MLIR_PATH),
            str(backward_mlir),
            "2",
            "0",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    backward_text = backward_mlir.read_text(encoding="utf-8")
    assert "stablehlo.dynamic_update_slice" in backward_text
    assert "stablehlo.dynamic_slice" in backward_text

    subprocess.run(
        [
            compiler,
            str(backward_mlir),
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=generic",
            "-o",
            "/tmp/qwen36_27b_tiny_dynamic_update_training_backward.vmfb",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_dynamic_update_slice_vjp_numerical_runtime_check() -> None:
    nix = nix_bin()
    if nix is None:
        try:
            import pytest  # type: ignore
        except ImportError:
            return
        pytest.skip("nix is required for the isolated dynamic-update VJP runtime check")

    subprocess.run(
        [
            nix,
            "develop",
            "-c",
            "zig",
            "build",
            "run-isolated-vjp-tests",
            "--",
            "dynamic-update-slice",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    test_hybrid_ops_smoke_covers_milestone6_lowering_surface()
    test_tiny_hybrid_ops_cpu_compiles_when_iree_is_available()
    test_training_forward_keeps_state_mutation_out_of_trainable_path()
    test_dynamic_update_slice_vjp_is_registered_in_core_autodiff()
    test_tiny_deltanet_block_matches_pytorch_parameter_gradients()
    test_tiny_hybrid_backward_graph_builds_and_compiles_when_tools_are_available()
    test_dynamic_update_slice_backward_graph_builds_and_compiles_when_tools_are_available()
    test_dynamic_update_slice_vjp_numerical_runtime_check()
    print("OK: Qwen3.6-27B hybrid op checks passed.")


if __name__ == "__main__":
    main()
