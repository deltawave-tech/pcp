#!/usr/bin/env python3
import argparse
import json
import sys

from transformers import AutoTokenizer


def _write(resp):
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen tokenizer helper (stdin/stdout JSON).")
    parser.add_argument("--tokenizer-path", required=True, help="Local tokenizer dir or HF model id.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=True,
        trust_remote_code=True,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _write({"ok": False, "error": "invalid_json"})
            continue

        op = req.get("op")
        if op == "encode":
            text = req.get("text", "")
            tokens = tokenizer.encode(text, add_special_tokens=False)
            _write({"ok": True, "tokens": tokens})
        elif op == "decode":
            tokens = req.get("tokens", [])
            text = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            _write({"ok": True, "text": text})
        elif op == "eos_token_id":
            _write({"ok": True, "eos_token_id": tokenizer.eos_token_id})
        elif op == "shutdown":
            _write({"ok": True})
            return 0
        else:
            _write({"ok": False, "error": "unknown_op"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
