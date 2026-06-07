#!/usr/bin/env python3
import argparse
import json
import sys

from transformers import AutoTokenizer


def _write(resp):
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()


def _fallback_qwen_text_only(messages):
    parts = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n{message.get('content', '')}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


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
        elif op == "render_chat":
            messages = req.get("messages", [])
            template = req.get("chat_template", "qwen_text_only")
            try:
                if template == "qwen_text_only" and getattr(tokenizer, "chat_template", None):
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    _write({"ok": True, "text": text, "source": "official"})
                elif template == "qwen_text_only":
                    _write({"ok": True, "text": _fallback_qwen_text_only(messages), "source": "pcp_fallback"})
                else:
                    _write({"ok": False, "error": "unsupported_chat_template"})
            except Exception as exc:
                _write({"ok": False, "error": str(exc)})
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
