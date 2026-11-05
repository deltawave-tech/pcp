"""
Qwen3.6-27B text-only contract checks.

Run from the repo root:
  python3 test/qwen/test_qwen36_27b_text_contract.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from qwen36_27b_contract import (  # noqa: E402
    ATTENTION_HEAD_DIM,
    ATTENTION_KV_HEADS,
    DEFAULT_MAX_SEQ_LEN,
    DELTANET_HEAD_DIM,
    DELTANET_QK_HEADS,
    DELTANET_V_HEADS,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_INTERMEDIATE_SIZE,
    EXPECTED_NUM_LAYERS,
    EXPECTED_PATTERN_REPEATS,
    EXPECTED_VOCAB_SIZE,
    build_generation_contract,
    contract_to_metadata,
    facts_from_config,
    layer_pattern,
)


EXPORTER_PATH = REPO_ROOT / "tools" / "export_qwen36_27b_generation.py"
CONFIG_PATH = REPO_ROOT / "experiments" / "inference_qwen36_27b.json"
CHAT_TEMPLATE_PATH = REPO_ROOT / "src" / "workloads" / "inference" / "chat_template.zig"
INFERENCE_CONTROLLER_PATH = REPO_ROOT / "src" / "nodes" / "gateway" / "controllers" / "inference_controller.zig"
TOKENIZER_PATH = REPO_ROOT / "src" / "workloads" / "inference" / "tokenizer.zig"
QWEN_TOKENIZER_PATH = REPO_ROOT / "src" / "workloads" / "inference" / "qwen_tokenizer.zig"
TOKENIZER_SERVER_PATH = REPO_ROOT / "tools" / "qwen_tokenizer_server.py"
CONTRACT_PATH = REPO_ROOT / "models" / "qwen36_27b_text_generation.contract.json"
TINY_MLIR_PATH = REPO_ROOT / "models" / "tests" / "qwen36_27b_tiny_hybrid.mlir"
TINY_OPS_MLIR_PATH = REPO_ROOT / "models" / "tests" / "qwen36_27b_tiny_hybrid_ops.mlir"
TINY_TRAINING_MLIR_PATH = REPO_ROOT / "models" / "tests" / "qwen36_27b_tiny_hybrid_training.mlir"
TINY_DYNAMIC_UPDATE_TRAINING_MLIR_PATH = REPO_ROOT / "models" / "tests" / "qwen36_27b_tiny_dynamic_update_training.mlir"


def require_file(path: Path) -> None:
    if path.exists():
        return
    try:
        import pytest  # type: ignore
    except ImportError:
        raise SystemExit(f"Missing required artifact: {path}") from None
    pytest.skip(f"Missing required artifact: {path}")


def fake_hf_config() -> dict:
    return {
        "model_id": "Qwen/Qwen3.6-27B",
        "num_hidden_layers": EXPECTED_NUM_LAYERS,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "intermediate_size": EXPECTED_INTERMEDIATE_SIZE,
        "deltanet_qk_heads": DELTANET_QK_HEADS,
        "deltanet_v_heads": DELTANET_V_HEADS,
        "deltanet_head_dim": DELTANET_HEAD_DIM,
        "attention_q_heads": 24,
        "attention_kv_heads": ATTENTION_KV_HEADS,
        "attention_head_dim": ATTENTION_HEAD_DIM,
        "attention_rope_dim": 64,
    }


def test_config_parser_recognizes_text_only_hybrid_layout() -> None:
    facts = facts_from_config(fake_hf_config())

    assert facts.text_only is True
    assert facts.vision_enabled is False
    assert facts.mtp_enabled is False
    assert facts.full_conditional_generation_enabled is False
    assert facts.chat_template == "qwen_text_only"
    assert facts.num_layers == 64
    assert facts.hidden_size == 5120
    assert facts.vocab_size == 248320
    assert facts.intermediate_size == 17408
    assert facts.pattern_repeats == EXPECTED_PATTERN_REPEATS
    assert facts.layer_pattern == layer_pattern(64)
    assert facts.layer_pattern[:8] == [
        "deltanet",
        "deltanet",
        "deltanet",
        "attention",
        "deltanet",
        "deltanet",
        "deltanet",
        "attention",
    ]


def test_generation_metadata_uses_generic_state_slots() -> None:
    facts = facts_from_config(fake_hf_config(), external_weight_dtype="bf16")
    contract = build_generation_contract(facts)
    slots = contract.state_slots

    assert contract.model_id == "qwen36-27b"
    assert contract.architecture == "qwen36_27b_hybrid_text"
    assert contract.approximate_parameter_count == 27_000_000_000
    assert contract.parameter_shapes == []
    assert contract.parameter_dtypes == []
    assert contract.logits_shape == [1, 1, EXPECTED_VOCAB_SIZE]
    assert contract.logits_dtype == "f32"
    assert [gate.name for gate in contract.optional_feature_gates] == [
        "vision_encoder",
        "mtp",
        "full_conditional_generation",
    ]
    assert all(gate.enabled is False for gate in contract.optional_feature_gates)

    assert len(slots) == 128
    assert len(contract.data_input_shapes) == 130
    assert len(contract.data_input_dtypes) == 130
    assert contract.data_input_shapes[:2] == [[1, 1], [1, 1]]
    assert contract.data_input_dtypes[:2] == ["i64", "i64"]
    assert set(contract.data_input_dtypes[2:]) == {"bf16"}

    delta_recurrent = slots[0]
    delta_conv = slots[1]
    attention_k = next(slot for slot in slots if slot.kind == "attention_k_cache")
    attention_v = next(slot for slot in slots if slot.kind == "attention_v_cache")

    assert delta_recurrent.copy_layout == "replace"
    assert delta_recurrent.shape == [1, 16, 128, 384]
    assert delta_recurrent.update_shape == delta_recurrent.shape

    assert delta_conv.copy_layout == "replace"
    assert delta_conv.shape == [1, 80, 128, 4]
    assert delta_conv.update_shape == delta_conv.shape

    assert attention_k.layer == 3
    assert attention_k.copy_layout == "sequence_dim=2"
    assert attention_k.shape == [1, ATTENTION_KV_HEADS, DEFAULT_MAX_SEQ_LEN, ATTENTION_HEAD_DIM]
    assert attention_k.update_shape == [1, ATTENTION_KV_HEADS, 1, ATTENTION_HEAD_DIM]
    assert attention_v.shape == attention_k.shape


def test_cached_text_metadata_matches_contract_state_io() -> None:
    facts = facts_from_config(fake_hf_config(), external_weight_dtype="bf16")
    contract = build_generation_contract(facts)
    metadata = contract_to_metadata(contract)

    assert metadata["parameter_shapes"] == []
    assert metadata["parameter_dtypes"] == []
    assert metadata["data_input_shapes"] == contract.data_input_shapes
    assert metadata["data_input_dtypes"] == contract.data_input_dtypes
    assert len(metadata["data_input_shapes"]) == 130
    assert len(metadata["output_shapes"]) == 129
    assert metadata["output_shapes"][0] == [1, 1, EXPECTED_VOCAB_SIZE]
    assert metadata["output_dtypes"][0] == "f32"
    assert metadata["generation_contract"]["state_slot_count"] == 128


def test_optional_full_model_features_remain_disabled_in_text_contract() -> None:
    facts = facts_from_config(fake_hf_config())
    contract = build_generation_contract(facts)

    assert facts.text_only is True
    assert facts.vision_enabled is False
    assert facts.mtp_enabled is False
    assert facts.full_conditional_generation_enabled is False
    assert facts.chat_template == "qwen_text_only"

    gates = {gate.name: gate for gate in contract.optional_feature_gates}
    assert gates["vision_encoder"].enabled is False
    assert gates["vision_encoder"].disabled_regression == "vision-disabled"
    assert gates["mtp"].enabled is False
    assert gates["mtp"].disabled_regression == "mtp-disabled"
    assert gates["full_conditional_generation"].enabled is False
    assert gates["full_conditional_generation"].disabled_regression == "conditional-generation-disabled"


def test_config_parser_rejects_unimplemented_optional_feature_enablement() -> None:
    import pytest

    feature_fields = [
        "vision_enabled",
        "mtp_enabled",
        "full_conditional_generation_enabled",
    ]
    for field in feature_fields:
        cfg = fake_hf_config()
        cfg[field] = True
        with pytest.raises(ValueError, match=field):
            facts_from_config(cfg)


def test_chat_template_runtime_is_isolated_from_gateway_controller() -> None:
    for path in (CHAT_TEMPLATE_PATH, INFERENCE_CONTROLLER_PATH, TOKENIZER_PATH, QWEN_TOKENIZER_PATH, TOKENIZER_SERVER_PATH):
        require_file(path)

    chat_template = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    controller = INFERENCE_CONTROLLER_PATH.read_text(encoding="utf-8")
    tokenizer = TOKENIZER_PATH.read_text(encoding="utf-8")
    qwen_tokenizer = QWEN_TOKENIZER_PATH.read_text(encoding="utf-8")
    tokenizer_server = TOKENIZER_SERVER_PATH.read_text(encoding="utf-8")

    assert "pub fn validateMessages" in chat_template
    assert "MultimodalMessageContentUnsupported" in chat_template
    assert "UnsupportedMessageRole" in chat_template
    assert "renderQwenTextOnly" in chat_template
    assert "self.tokenizer.renderChat" in controller
    assert "fn renderQwenPrompt" not in controller
    assert "fn renderPrompt" not in controller
    assert "chat_template.validateMessages" in tokenizer
    assert "renderChat" in qwen_tokenizer
    assert "apply_chat_template" in tokenizer_server
    assert "_fallback_qwen_text_only" in tokenizer_server


def test_qwen_tokenizer_server_matches_official_chat_template_when_available() -> None:
    require_file(TOKENIZER_SERVER_PATH)
    tokenizer_path = os.environ.get("PCP_QWEN36_27B_TOKENIZER_PATH", "Qwen/Qwen3.6-27B")
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": ""},
    ]
    request = {
        "op": "render_chat",
        "chat_template": "qwen_text_only",
        "messages": messages,
    }
    helper = subprocess.run(
        [
            sys.executable,
            str(TOKENIZER_SERVER_PATH),
            "--tokenizer-path",
            tokenizer_path,
        ],
        input=json.dumps(request) + "\n",
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if helper.returncode != 0:
        try:
            import pytest  # type: ignore
        except ImportError:
            raise AssertionError(helper.stderr) from None
        pytest.skip(f"Qwen tokenizer unavailable: {helper.stderr.strip()}")

    response = json.loads(helper.stdout.splitlines()[0])
    assert response["ok"] is True, response
    rendered = response["text"]

    reference = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from transformers import AutoTokenizer; import json; "
                "tok=AutoTokenizer.from_pretrained(r'''%s''', use_fast=True, trust_remote_code=True); "
                "messages=json.loads(r'''%s'''); "
                "print(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), end='')"
            )
            % (tokenizer_path, json.dumps(messages)),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if reference.returncode == 0:
        assert rendered == reference.stdout
        assert response["source"] == "official"
    else:
        assert rendered.endswith("<|im_start|>assistant\n")


def test_exporter_writes_contract_and_tiny_hybrid_artifact() -> None:
    require_file(EXPORTER_PATH)
    subprocess.run(
        [
            sys.executable,
            str(EXPORTER_PATH),
            "--contract-path",
            str(CONTRACT_PATH),
            "--metadata-path",
            str(CONTRACT_PATH.with_name("qwen36_27b_text_generation.mlir.meta.json")),
            "--tiny-mlir-path",
            str(TINY_MLIR_PATH),
            "--tiny-ops-mlir-path",
            str(TINY_OPS_MLIR_PATH),
            "--tiny-training-mlir-path",
            str(TINY_TRAINING_MLIR_PATH),
            "--tiny-dynamic-update-training-mlir-path",
            str(TINY_DYNAMIC_UPDATE_TRAINING_MLIR_PATH),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    tiny_mlir = TINY_MLIR_PATH.read_text(encoding="utf-8")
    tiny_ops_mlir = TINY_OPS_MLIR_PATH.read_text(encoding="utf-8")
    tiny_training_mlir = TINY_TRAINING_MLIR_PATH.read_text(encoding="utf-8")
    tiny_dynamic_update_training_mlir = TINY_DYNAMIC_UPDATE_TRAINING_MLIR_PATH.read_text(encoding="utf-8")

    assert contract["model_id"] == "qwen36-27b"
    assert contract["facts"]["num_layers"] == EXPECTED_NUM_LAYERS
    assert contract["facts"]["vision_enabled"] is False
    assert contract["facts"]["mtp_enabled"] is False
    assert contract["facts"]["full_conditional_generation_enabled"] is False
    assert contract["facts"]["chat_template"] == "qwen_text_only"
    assert [gate["name"] for gate in contract["optional_feature_gates"]] == [
        "vision_encoder",
        "mtp",
        "full_conditional_generation",
    ]
    assert all(gate["enabled"] is False for gate in contract["optional_feature_gates"])
    assert len(contract["state_slots"]) == 128
    assert len(contract["data_input_shapes"]) == 130
    assert "tensor<1x2x4x8xbf16>" in tiny_mlir
    assert "stablehlo.slice" in tiny_mlir
    assert "stablehlo.add" in tiny_mlir
    assert "stablehlo.dynamic_update_slice" in tiny_ops_mlir
    assert "stablehlo.dot_general" in tiny_ops_mlir
    assert "stablehlo.logistic" in tiny_training_mlir
    assert "stablehlo.dynamic_update_slice" in tiny_dynamic_update_training_mlir


def test_inference_config_avoids_27b_flat_f32_contract() -> None:
    require_file(CONFIG_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["model_id"] == "qwen36-27b"
    assert config["generation_contract_path"] == "models/qwen36_27b_text_generation.contract.json"
    assert config["weights_format"] == "bf16_streamed"
    assert config["weights_dtype"] == "bf16"
    assert config["chat_template"] == "qwen_text_only"
    assert config["vision_enabled"] is False
    assert config["mtp_enabled"] is False
    assert "num_gen_data_inputs" not in config
    assert config["full_conditional_generation_enabled"] is False
    assert "flat" not in config["weights_path"]
    assert config["weights_path"] != "checkpoints/initial_weights/qwen3_8b_flat.bin"


def test_tiny_hybrid_mlir_cpu_compiles_when_iree_is_available() -> None:
    require_file(TINY_MLIR_PATH)
    compiler = shutil.which("iree-compile") or str(REPO_ROOT / ".venv" / "bin" / "iree-compile")
    if compiler is None or not Path(compiler).exists():
        try:
            import pytest  # type: ignore
        except ImportError:
            return
        pytest.skip("iree-compile is not installed on this VM")

    output_vmfb = Path("/tmp/qwen36_27b_tiny_hybrid.vmfb")
    if output_vmfb.exists():
        output_vmfb.unlink()
    subprocess.run(
        [
            compiler,
            str(TINY_MLIR_PATH),
            "--iree-hal-target-backends=llvm-cpu",
            "-o",
            str(output_vmfb),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert output_vmfb.exists()


def main() -> None:
    test_config_parser_recognizes_text_only_hybrid_layout()
    test_generation_metadata_uses_generic_state_slots()
    test_exporter_writes_contract_and_tiny_hybrid_artifact()
    test_inference_config_avoids_27b_flat_f32_contract()
    test_tiny_hybrid_mlir_cpu_compiles_when_iree_is_available()
    print("OK: Qwen3.6-27B text contract checks passed.")


if __name__ == "__main__":
    main()
