import inspect
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanochat.gpt import norm, scaled_dot_product_attention


def assert_close(actual, expected, label, atol=1e-5, rtol=1e-5):
    diff = (actual - expected).abs().max().item()
    ref = expected.abs().max().item()
    print(f"{label}: max_diff={diff} ref_max={ref} tol={atol + rtol * ref}")
    if diff > atol + rtol * ref:
        raise SystemExit(f"{label} mismatch: max_diff={diff} tol={atol + rtol * ref}")


def supports_enable_gqa():
    try:
        sig = inspect.signature(F.scaled_dot_product_attention)
    except (TypeError, ValueError):
        return False
    return "enable_gqa" in sig.parameters


def test_norm():
    print("==> test_norm")
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    print(f"input shape: {tuple(x.shape)} dtype={x.dtype} device={x.device}")

    out = norm(x)
    manual = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    assert_close(out, manual, "norm vs manual")

    if hasattr(F, "rms_norm"):
        print("F.rms_norm available: yes")
        ref = F.rms_norm(x, (x.size(-1),), eps=1e-5)
        assert_close(out, ref, "norm vs F.rms_norm")
    else:
        print("F.rms_norm available: no")


def test_sdpa():
    print("==> test_sdpa")
    torch.manual_seed(0)

    q = torch.randn(2, 4, 3, 8, dtype=torch.float32)
    k = torch.randn(2, 2, 5, 8, dtype=torch.float32)
    v = torch.randn(2, 2, 5, 8, dtype=torch.float32)
    print(
        f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
        f"dtype={q.dtype} device={q.device}"
    )
    attn_mask = torch.zeros((3, 5), dtype=torch.bool)
    attn_mask[0, 4] = True

    out = scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=True
    )

    if supports_enable_gqa():
        print("F.scaled_dot_product_attention enable_gqa: yes")
        ref = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=True
        )
    else:
        print("F.scaled_dot_product_attention enable_gqa: no (using manual GQA)")
        repeat = q.size(1) // k.size(1)
        k_rep = k.repeat_interleave(repeat, dim=1)
        v_rep = v.repeat_interleave(repeat, dim=1)
        ref = F.scaled_dot_product_attention(
            q, k_rep, v_rep, attn_mask=attn_mask, is_causal=False
        )

    assert_close(out, ref, "sdpa gqa")

    q2 = torch.randn(2, 2, 3, 8, dtype=torch.float32)
    k2 = torch.randn(2, 2, 5, 8, dtype=torch.float32)
    v2 = torch.randn(2, 2, 5, 8, dtype=torch.float32)
    print(
        f"q2={tuple(q2.shape)} k2={tuple(k2.shape)} v2={tuple(v2.shape)} "
        f"dtype={q2.dtype} device={q2.device}"
    )
    out2 = scaled_dot_product_attention(q2, k2, v2, is_causal=True, enable_gqa=False)
    ref2 = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
    assert_close(out2, ref2, "sdpa non-gqa causal")


def main():
    test_norm()
    test_sdpa()
    print("OK: gpt.py compatibility checks passed.")


if __name__ == "__main__":
    main()
