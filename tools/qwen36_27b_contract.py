#!/usr/bin/env python3
"""Qwen3.6-27B text-only contract helpers for PCP.

This module intentionally has no torch dependency. Milestone 5 needs a stable
architecture and metadata contract before the full exporter starts tracing the
27B model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping


MODEL_ID = "Qwen/Qwen3.6-27B"
PCP_MODEL_ID = "qwen36-27b"
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_STATE_DTYPE = "bf16"
DEFAULT_TOKEN_DTYPE = "i64"

EXPECTED_NUM_LAYERS = 64
EXPECTED_HIDDEN_SIZE = 5120
EXPECTED_VOCAB_SIZE = 248320
EXPECTED_INTERMEDIATE_SIZE = 17408
EXPECTED_PATTERN_REPEATS = 16

DELTANET_QK_HEADS = 16
DELTANET_V_HEADS = 48
DELTANET_HEAD_DIM = 128
DELTANET_CONV_WIDTH = 4

ATTENTION_Q_HEADS = 24
ATTENTION_KV_HEADS = 4
ATTENTION_HEAD_DIM = 256
ATTENTION_ROPE_DIM = 64


@dataclass(frozen=True)
class StateSlot:
    name: str
    layer: int
    kind: str
    shape: list[int]
    update_shape: list[int]
    dtype: str
    copy_layout: str


@dataclass(frozen=True)
class OptionalFeatureGate:
    name: str
    enabled: bool
    status: str
    disabled_regression: str


@dataclass(frozen=True)
class Qwen36TextFacts:
    model_id: str
    text_only: bool
    vision_enabled: bool
    mtp_enabled: bool
    full_conditional_generation_enabled: bool
    chat_template: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    layer_pattern: list[str]
    pattern_repeats: int
    deltanet_qk_heads: int
    deltanet_v_heads: int
    deltanet_head_dim: int
    attention_q_heads: int
    attention_kv_heads: int
    attention_head_dim: int
    attention_rope_dim: int
    external_weight_dtype: str


@dataclass(frozen=True)
class Qwen36GenerationContract:
    model_id: str
    source_model_id: str
    architecture: str
    max_seq_len: int
    logits_shape: list[int]
    logits_dtype: str
    data_input_shapes: list[list[int]]
    data_input_dtypes: list[str]
    state_slots: list[StateSlot]
    output_state_slot_count: int
    approximate_parameter_count: int
    parameter_shapes: list[list[int]]
    parameter_dtypes: list[str]
    facts: Qwen36TextFacts
    optional_feature_gates: list[OptionalFeatureGate]


def layer_pattern(num_layers: int = EXPECTED_NUM_LAYERS) -> list[str]:
    if num_layers % 4 != 0:
        raise ValueError(f"Qwen3.6 hybrid layer count must be divisible by 4, got {num_layers}")
    return ["deltanet" if (idx % 4) < 3 else "attention" for idx in range(num_layers)]


def _mapping_from_config(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        text_config = config.get("text_config")
        if isinstance(text_config, Mapping):
            merged = dict(config)
            merged.update(text_config)
            return merged
        return config

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        payload = dict(getattr(config, "__dict__", {}))
        payload.update(getattr(text_config, "__dict__", {}))
        return payload
    return getattr(config, "__dict__", {})


def facts_from_config(config: Any, *, external_weight_dtype: str = DEFAULT_STATE_DTYPE) -> Qwen36TextFacts:
    cfg = _mapping_from_config(config)
    num_layers = int(cfg.get("num_hidden_layers", cfg.get("num_layers", EXPECTED_NUM_LAYERS)))
    hidden_size = int(cfg.get("hidden_size", EXPECTED_HIDDEN_SIZE))
    vocab_size = int(cfg.get("vocab_size", cfg.get("padded_vocab_size", EXPECTED_VOCAB_SIZE)))
    intermediate_size = int(cfg.get("intermediate_size", EXPECTED_INTERMEDIATE_SIZE))

    raw_pattern = cfg.get("layer_types") or cfg.get("layers_block_type") or cfg.get("hidden_layout")
    if isinstance(raw_pattern, list):
        pattern = []
        for item in raw_pattern:
            normalized = str(item).lower()
            if normalized in {"full_attention", "attention", "gated_attention"}:
                pattern.append("attention")
            elif normalized in {"linear_attention", "deltanet", "gated_deltanet"}:
                pattern.append("deltanet")
            else:
                raise ValueError(f"Unsupported Qwen3.6 layer type: {item!r}")
    else:
        pattern = layer_pattern(num_layers)

    facts = Qwen36TextFacts(
        model_id=str(cfg.get("_name_or_path", cfg.get("model_id", MODEL_ID))),
        text_only=True,
        vision_enabled=bool(cfg.get("vision_enabled", False)),
        mtp_enabled=bool(cfg.get("mtp_enabled", False)),
        full_conditional_generation_enabled=bool(cfg.get("full_conditional_generation_enabled", False)),
        chat_template=str(cfg.get("chat_template", "qwen_text_only")),
        num_layers=num_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
        layer_pattern=pattern,
        pattern_repeats=num_layers // 4,
        deltanet_qk_heads=int(cfg.get("linear_num_key_heads", cfg.get("deltanet_qk_heads", DELTANET_QK_HEADS))),
        deltanet_v_heads=int(cfg.get("linear_num_value_heads", cfg.get("deltanet_v_heads", DELTANET_V_HEADS))),
        deltanet_head_dim=int(cfg.get("linear_key_head_dim", cfg.get("deltanet_head_dim", DELTANET_HEAD_DIM))),
        attention_q_heads=int(cfg.get("num_attention_heads", cfg.get("attention_q_heads", ATTENTION_Q_HEADS))),
        attention_kv_heads=int(cfg.get("num_key_value_heads", cfg.get("attention_kv_heads", ATTENTION_KV_HEADS))),
        attention_head_dim=int(cfg.get("head_dim", cfg.get("attention_head_dim", ATTENTION_HEAD_DIM))),
        attention_rope_dim=int(
            cfg.get(
                "rope_dim",
                cfg.get(
                    "attention_rope_dim",
                    int(cfg.get("head_dim", ATTENTION_HEAD_DIM) * cfg.get("rope_parameters", {}).get("partial_rotary_factor", 0.25)),
                ),
            )
        ),
        external_weight_dtype=external_weight_dtype,
    )
    validate_facts(facts)
    return facts


def validate_facts(facts: Qwen36TextFacts) -> None:
    expected_pattern = layer_pattern(facts.num_layers)
    checks = [
        (facts.num_layers == EXPECTED_NUM_LAYERS, "num_layers"),
        (facts.hidden_size == EXPECTED_HIDDEN_SIZE, "hidden_size"),
        (facts.vocab_size == EXPECTED_VOCAB_SIZE, "vocab_size"),
        (facts.intermediate_size == EXPECTED_INTERMEDIATE_SIZE, "intermediate_size"),
        (facts.layer_pattern == expected_pattern, "layer_pattern"),
        (facts.pattern_repeats == EXPECTED_PATTERN_REPEATS, "pattern_repeats"),
        (facts.deltanet_qk_heads == DELTANET_QK_HEADS, "deltanet_qk_heads"),
        (facts.deltanet_v_heads == DELTANET_V_HEADS, "deltanet_v_heads"),
        (facts.deltanet_head_dim == DELTANET_HEAD_DIM, "deltanet_head_dim"),
        (facts.attention_q_heads == ATTENTION_Q_HEADS, "attention_q_heads"),
        (facts.attention_kv_heads == ATTENTION_KV_HEADS, "attention_kv_heads"),
        (facts.attention_head_dim == ATTENTION_HEAD_DIM, "attention_head_dim"),
        (facts.attention_rope_dim == ATTENTION_ROPE_DIM, "attention_rope_dim"),
        (facts.external_weight_dtype in {"bf16", "f16"}, "external_weight_dtype"),
        (facts.text_only is True, "text_only"),
        (facts.vision_enabled is False, "vision_enabled"),
        (facts.mtp_enabled is False, "mtp_enabled"),
        (facts.full_conditional_generation_enabled is False, "full_conditional_generation_enabled"),
        (facts.chat_template == "qwen_text_only", "chat_template"),
    ]
    bad = [name for ok, name in checks if not ok]
    if bad:
        raise ValueError(f"Qwen3.6-27B text contract mismatch: {', '.join(bad)}")


def build_state_slots(
    facts: Qwen36TextFacts,
    *,
    batch_size: int = 1,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    dtype: str | None = None,
) -> list[StateSlot]:
    state_dtype = dtype or facts.external_weight_dtype
    v_per_qk = facts.deltanet_v_heads // facts.deltanet_qk_heads
    slots: list[StateSlot] = []
    for layer_index, layer_kind in enumerate(facts.layer_pattern):
        if layer_kind == "attention":
            for kv_kind in ("k", "v"):
                slots.append(
                    StateSlot(
                        name=f"layers.{layer_index}.attention.{kv_kind}_cache",
                        layer=layer_index,
                        kind=f"attention_{kv_kind}_cache",
                        shape=[batch_size, facts.attention_kv_heads, max_seq_len, facts.attention_head_dim],
                        update_shape=[batch_size, facts.attention_kv_heads, 1, facts.attention_head_dim],
                        dtype=state_dtype,
                        copy_layout="sequence_dim=2",
                    )
                )
        else:
            slots.append(
                StateSlot(
                    name=f"layers.{layer_index}.deltanet.recurrent_state",
                    layer=layer_index,
                    kind="deltanet_recurrent_state",
                    shape=[
                        batch_size,
                        facts.deltanet_qk_heads,
                        facts.deltanet_head_dim,
                        v_per_qk * facts.deltanet_head_dim,
                    ],
                    update_shape=[
                        batch_size,
                        facts.deltanet_qk_heads,
                        facts.deltanet_head_dim,
                        v_per_qk * facts.deltanet_head_dim,
                    ],
                    dtype=state_dtype,
                    copy_layout="replace",
                )
            )
            slots.append(
                StateSlot(
                    name=f"layers.{layer_index}.deltanet.conv_state",
                    layer=layer_index,
                    kind="deltanet_conv_state",
                    shape=[
                        batch_size,
                        facts.deltanet_v_heads + 2 * facts.deltanet_qk_heads,
                        facts.deltanet_head_dim,
                        DELTANET_CONV_WIDTH,
                    ],
                    update_shape=[
                        batch_size,
                        facts.deltanet_v_heads + 2 * facts.deltanet_qk_heads,
                        facts.deltanet_head_dim,
                        DELTANET_CONV_WIDTH,
                    ],
                    dtype=state_dtype,
                    copy_layout="replace",
                )
            )
    return slots


def build_generation_contract(
    facts: Qwen36TextFacts,
    *,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    state_dtype: str | None = None,
) -> Qwen36GenerationContract:
    slots = build_state_slots(facts, max_seq_len=max_seq_len, dtype=state_dtype)
    data_input_shapes = [[1, 1], [1, 1], *[slot.shape for slot in slots]]
    data_input_dtypes = [DEFAULT_TOKEN_DTYPE, DEFAULT_TOKEN_DTYPE, *[slot.dtype for slot in slots]]
    optional_feature_gates = [
        OptionalFeatureGate(
            name="vision_encoder",
            enabled=False,
            status="deferred_until_text_generation_and_text_rl_accepted",
            disabled_regression="vision-disabled",
        ),
        OptionalFeatureGate(
            name="mtp",
            enabled=False,
            status="deferred_until_text_graph_and_state_schema_stable",
            disabled_regression="mtp-disabled",
        ),
        OptionalFeatureGate(
            name="full_conditional_generation",
            enabled=False,
            status="deferred_until_optional_feature_disabled_regressions_pass",
            disabled_regression="conditional-generation-disabled",
        ),
    ]
    return Qwen36GenerationContract(
        model_id=PCP_MODEL_ID,
        source_model_id=MODEL_ID,
        architecture="qwen36_27b_hybrid_text",
        max_seq_len=max_seq_len,
        logits_shape=[1, 1, facts.vocab_size],
        logits_dtype="f32",
        data_input_shapes=data_input_shapes,
        data_input_dtypes=data_input_dtypes,
        state_slots=slots,
        output_state_slot_count=len(slots),
        approximate_parameter_count=27_000_000_000,
        parameter_shapes=[],
        parameter_dtypes=[],
        facts=facts,
        optional_feature_gates=optional_feature_gates,
    )


def contract_to_dict(contract: Qwen36GenerationContract) -> dict[str, Any]:
    payload = asdict(contract)
    payload["state_slots"] = [asdict(slot) for slot in contract.state_slots]
    payload["facts"] = asdict(contract.facts)
    payload["optional_feature_gates"] = [asdict(gate) for gate in contract.optional_feature_gates]
    return payload


def contract_to_metadata(contract: Qwen36GenerationContract) -> dict[str, Any]:
    return {
        "parameter_names": [],
        "parameter_shapes": contract.parameter_shapes,
        "parameter_dtypes": contract.parameter_dtypes,
        "data_input_shapes": contract.data_input_shapes,
        "data_input_dtypes": contract.data_input_dtypes,
        "output_shapes": [contract.logits_shape] + [slot.update_shape for slot in contract.state_slots],
        "output_dtypes": [contract.logits_dtype] + [slot.dtype for slot in contract.state_slots],
        "generation_contract": {
            "model_id": contract.model_id,
            "architecture": contract.architecture,
            "state_slot_count": len(contract.state_slots),
            "output_state_slot_count": contract.output_state_slot_count,
            "max_seq_len": contract.max_seq_len,
        },
    }


def write_contract(contract: Qwen36GenerationContract, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract_to_dict(contract), indent=2) + "\n", encoding="utf-8")


def write_metadata(contract: Qwen36GenerationContract, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract_to_metadata(contract), indent=2) + "\n", encoding="utf-8")


def tiny_hybrid_mlir() -> str:
    return """module {
  func.func @main(%token: tensor<1x1xi64>, %position: tensor<1x1xi64>, %delta_recurrent: tensor<1x2x4x8xbf16>, %delta_conv: tensor<1x6x4x2xbf16>, %attn_k: tensor<1x1x8x4xbf16>, %attn_v: tensor<1x1x8x4xbf16>) -> (tensor<1x1x32xf32>, tensor<1x2x4x8xbf16>, tensor<1x6x4x2xbf16>, tensor<1x1x1x4xbf16>, tensor<1x1x1x4xbf16>) {
    %logits = stablehlo.constant dense<0.000000e+00> : tensor<1x1x32xf32>
    %delta_recurrent_update = stablehlo.add %delta_recurrent, %delta_recurrent : tensor<1x2x4x8xbf16>
    %delta_conv_update = stablehlo.add %delta_conv, %delta_conv : tensor<1x6x4x2xbf16>
    %k_update = stablehlo.slice %attn_k [0:1, 0:1, 0:1, 0:4] : (tensor<1x1x8x4xbf16>) -> tensor<1x1x1x4xbf16>
    %v_update = stablehlo.slice %attn_v [0:1, 0:1, 0:1, 0:4] : (tensor<1x1x8x4xbf16>) -> tensor<1x1x1x4xbf16>
    return %logits, %delta_recurrent_update, %delta_conv_update, %k_update, %v_update : tensor<1x1x32xf32>, tensor<1x2x4x8xbf16>, tensor<1x6x4x2xbf16>, tensor<1x1x1x4xbf16>, tensor<1x1x1x4xbf16>
  }
}
"""


def tiny_hybrid_ops_mlir() -> str:
    return """module {
  func.func @main(%token: tensor<1x1xi64>, %position: tensor<1x1xi64>, %delta_recurrent: tensor<1x2x4x8xf32>, %delta_conv: tensor<1x6x4x4xf32>, %attn_k: tensor<1x1x8x4xf32>, %attn_v: tensor<1x1x8x4xf32>) -> (tensor<1x1x32xf32>, tensor<1x2x4x8xf32>, tensor<1x6x4x4xf32>, tensor<1x1x8x4xf32>, tensor<1x1x8x4xf32>) {
    %c0_i64 = stablehlo.constant dense<0> : tensor<i64>
    %pos_scalar = stablehlo.reshape %position : (tensor<1x1xi64>) -> tensor<i64>
    %zero_conv = stablehlo.constant dense<0.000000e+00> : tensor<1x6x4x1xf32>
    %conv_tail = stablehlo.slice %delta_conv [0:1, 0:6, 0:4, 1:4] : (tensor<1x6x4x4xf32>) -> tensor<1x6x4x3xf32>
    %conv_shifted = stablehlo.concatenate %conv_tail, %zero_conv, dim = 3 : (tensor<1x6x4x3xf32>, tensor<1x6x4x1xf32>) -> tensor<1x6x4x4xf32>
    %conv_prev = stablehlo.slice %conv_shifted [0:1, 0:2, 0:4, 0:1] : (tensor<1x6x4x4xf32>) -> tensor<1x2x4x1xf32>
    %conv_prev_flat = stablehlo.reshape %conv_prev : (tensor<1x2x4x1xf32>) -> tensor<1x2x4xf32>
    %conv_gate = stablehlo.logistic %conv_prev_flat : tensor<1x2x4xf32>
    %conv_gate_expanded = stablehlo.reshape %conv_gate : (tensor<1x2x4xf32>) -> tensor<1x2x4x1xf32>
    %conv_gate_broadcast = stablehlo.broadcast_in_dim %conv_gate_expanded, dims = [0, 1, 2, 3] : (tensor<1x2x4x1xf32>) -> tensor<1x2x4x8xf32>
    %state_decay = stablehlo.constant dense<9.500000e-01> : tensor<1x2x4x8xf32>
    %state_scaled = stablehlo.multiply %delta_recurrent, %state_decay : tensor<1x2x4x8xf32>
    %delta_recurrent_update = stablehlo.add %state_scaled, %conv_gate_broadcast : tensor<1x2x4x8xf32>
    %rms_square = stablehlo.multiply %conv_gate_broadcast, %conv_gate_broadcast : tensor<1x2x4x8xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %rms_sum = stablehlo.reduce(%rms_square init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x2x4x8xf32>, tensor<f32>) -> tensor<1x2x4xf32>
    %rms_scale = stablehlo.rsqrt %rms_sum : tensor<1x2x4xf32>
    %q = stablehlo.slice %attn_k [0:1, 0:1, 0:1, 0:2] : (tensor<1x1x8x4xf32>) -> tensor<1x1x1x2xf32>
    %k_update = stablehlo.slice %attn_k [0:1, 0:1, 0:1, 0:4] : (tensor<1x1x8x4xf32>) -> tensor<1x1x1x4xf32>
    %v_update = stablehlo.slice %attn_v [0:1, 0:1, 0:1, 0:4] : (tensor<1x1x8x4xf32>) -> tensor<1x1x1x4xf32>
    %attn_k_update = stablehlo.dynamic_update_slice %attn_k, %k_update, %c0_i64, %c0_i64, %pos_scalar, %c0_i64 : (tensor<1x1x8x4xf32>, tensor<1x1x1x4xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x8x4xf32>
    %attn_v_update = stablehlo.dynamic_update_slice %attn_v, %v_update, %c0_i64, %c0_i64, %pos_scalar, %c0_i64 : (tensor<1x1x8x4xf32>, tensor<1x1x1x4xf32>, tensor<i64>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<1x1x8x4xf32>
    %scores_lhs = stablehlo.reshape %q : (tensor<1x1x1x2xf32>) -> tensor<1x1x2xf32>
    %scores_rhs_slice = stablehlo.slice %attn_k_update [0:1, 0:1, 0:8, 0:2] : (tensor<1x1x8x4xf32>) -> tensor<1x1x8x2xf32>
    %scores_rhs = stablehlo.reshape %scores_rhs_slice : (tensor<1x1x8x2xf32>) -> tensor<1x2x8xf32>
    %scores = stablehlo.dot_general %scores_lhs, %scores_rhs, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x1x2xf32>, tensor<1x2x8xf32>) -> tensor<1x1x8xf32>
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %scores_max = stablehlo.reduce(%scores init: %neg_inf) applies stablehlo.maximum across dimensions = [2] : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %scores_max_expanded = stablehlo.reshape %scores_max : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %scores_max_broadcast = stablehlo.broadcast_in_dim %scores_max_expanded, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x8xf32>
    %scores_centered = stablehlo.subtract %scores, %scores_max_broadcast : tensor<1x1x8xf32>
    %scores_exp = stablehlo.exponential %scores_centered : tensor<1x1x8xf32>
    %scores_sum = stablehlo.reduce(%scores_exp init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x1x8xf32>, tensor<f32>) -> tensor<1x1xf32>
    %scores_sum_expanded = stablehlo.reshape %scores_sum : (tensor<1x1xf32>) -> tensor<1x1x1xf32>
    %scores_sum_broadcast = stablehlo.broadcast_in_dim %scores_sum_expanded, dims = [0, 1, 2] : (tensor<1x1x1xf32>) -> tensor<1x1x8xf32>
    %probs = stablehlo.divide %scores_exp, %scores_sum_broadcast : tensor<1x1x8xf32>
    %v_for_attn = stablehlo.reshape %attn_v_update : (tensor<1x1x8x4xf32>) -> tensor<1x8x4xf32>
    %context = stablehlo.dot_general %probs, %v_for_attn, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<1x1x8xf32>, tensor<1x8x4xf32>) -> tensor<1x1x4xf32>
    %logits = stablehlo.constant dense<0.000000e+00> : tensor<1x1x32xf32>
    return %logits, %delta_recurrent_update, %conv_shifted, %attn_k_update, %attn_v_update : tensor<1x1x32xf32>, tensor<1x2x4x8xf32>, tensor<1x6x4x4xf32>, tensor<1x1x8x4xf32>, tensor<1x1x8x4xf32>
  }
}
"""


def tiny_hybrid_training_mlir() -> str:
    return """module {
  func.func @main(%proj: tensor<4x4xf32>, %gate_bias: tensor<4xf32>) -> tensor<1x4xf32> {
    %x = stablehlo.constant dense<[[1.000000e+00, -5.000000e-01, 2.500000e-01, 7.500000e-01]]> : tensor<1x4xf32>
    %target = stablehlo.constant dense<[[2.500000e-01, -1.000000e+00, 5.000000e-01, 1.250000e-01]]> : tensor<1x4xf32>
    %projected = stablehlo.dot_general %x, %proj, contracting_dims = [1] x [0] : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
    %gate = stablehlo.broadcast_in_dim %gate_bias, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %gated_pre = stablehlo.add %projected, %gate : tensor<1x4xf32>
    %gated = stablehlo.logistic %gated_pre : tensor<1x4xf32>
    %delta_update = stablehlo.multiply %projected, %gated : tensor<1x4xf32>
    %err = stablehlo.subtract %delta_update, %target : tensor<1x4xf32>
    %loss_vec = stablehlo.multiply %err, %err : tensor<1x4xf32>
    return %loss_vec : tensor<1x4xf32>
  }
}
"""


def tiny_dynamic_update_training_mlir() -> str:
    return """module {
  func.func @main(%base: tensor<1x4xf32>, %update: tensor<1x2xf32>) -> tensor<1x4xf32> {
    %c0_i64 = stablehlo.constant dense<0> : tensor<i64>
    %c1_i64 = stablehlo.constant dense<1> : tensor<i64>
    %updated = stablehlo.dynamic_update_slice %base, %update, %c0_i64, %c1_i64 : (tensor<1x4xf32>, tensor<1x2xf32>, tensor<i64>, tensor<i64>) -> tensor<1x4xf32>
    %target = stablehlo.constant dense<[[0.000000e+00, 2.500000e-01, -5.000000e-01, 1.000000e+00]]> : tensor<1x4xf32>
    %err = stablehlo.subtract %updated, %target : tensor<1x4xf32>
    %loss_vec = stablehlo.multiply %err, %err : tensor<1x4xf32>
    return %loss_vec : tensor<1x4xf32>
  }
}
"""


def write_tiny_hybrid_mlir(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tiny_hybrid_mlir(), encoding="utf-8")


def write_tiny_hybrid_ops_mlir(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tiny_hybrid_ops_mlir(), encoding="utf-8")


def write_tiny_hybrid_training_mlir(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tiny_hybrid_training_mlir(), encoding="utf-8")


def write_tiny_dynamic_update_training_mlir(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tiny_dynamic_update_training_mlir(), encoding="utf-8")
