#!/usr/bin/env python3
"""
Offline pre-tokenization tool for nanochat FineWeb-style parquet shards.

This converts parquet shards containing a `text` column into a flat `.u16` token
stream plus a JSON manifest describing shard byte ranges.

Tokenization is done with the exact nanochat `tokenizer.pkl` (a pickled
`tiktoken.Encoding`) and prepends the nanochat BOS token to every document.

Example (2 shards, 1M tokens each):

  LD_LIBRARY_PATH=/nix/store/<gcc-lib>/lib:$LD_LIBRARY_PATH \\
    ./pcp/venv/bin/python pcp/tools/prepare_nanochat_parquet_to_u16.py \\
      --parquet-dir pcp/data/fineweb_parquet \\
      --parquet-glob 'shard_0000{0,1}.parquet' \\
      --tokenizer-dir pcp/tokenizer_nanochat \\
      --tokens-per-shard 1000000 \\
      --out-u16 pcp/data/fineweb_edu_2shards_1m.u16 \\
      --out-manifest pcp/data/fineweb_edu_2shards_1m.json

Note: On some Nix-based setups, importing `pyarrow` may require `libstdc++.so.6`
to be on `LD_LIBRARY_PATH` (see repo `AGENTS.MD`).
"""

from __future__ import annotations

import argparse
import array
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import pickle

import tiktoken

try:
    import pyarrow.parquet as pq
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Failed to import pyarrow. If you see `libstdc++.so.6` errors on Nix, set:\n"
        "  LD_LIBRARY_PATH=/nix/store/<gcc-lib>/lib:$LD_LIBRARY_PATH\n"
        f"Original error: {e}"
    ) from e


def _iter_parquet_files(parquet_dir: Path, parquet_glob: str | None) -> list[Path]:
    if not parquet_dir.exists():
        raise FileNotFoundError(parquet_dir)
    if parquet_glob is None:
        return sorted(parquet_dir.glob("*.parquet"))
    return sorted(parquet_dir.glob(parquet_glob))


def _load_encoding(tokenizer_dir: Path) -> tiktoken.Encoding:
    pkl_path = tokenizer_dir / "tokenizer.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)
    with pkl_path.open("rb") as f:
        enc = pickle.load(f)
    if not hasattr(enc, "encode_ordinary_batch"):
        raise TypeError(f"{pkl_path} did not contain a tiktoken.Encoding")
    return enc


def _encode_and_write_docs(
    *,
    out_f,
    enc: tiktoken.Encoding,
    docs: Sequence[str],
    bos_id: int,
    num_threads: int,
    max_tokens_to_write: int | None,
    sample_tokens_out: list[int] | None,
    sample_limit: int,
) -> int:
    ids_batch = enc.encode_ordinary_batch(list(docs), num_threads=num_threads)

    tok_buf = array.array("H")
    for ids in ids_batch:
        tok_buf.append(bos_id)
        tok_buf.extend(ids)

    if max_tokens_to_write is not None and max_tokens_to_write < len(tok_buf):
        tok_buf = tok_buf[:max_tokens_to_write]

    if sample_tokens_out is not None and len(sample_tokens_out) < sample_limit:
        remaining = sample_limit - len(sample_tokens_out)
        sample_tokens_out.extend(tok_buf[:remaining])

    write_buf = tok_buf
    if sys.byteorder != "little":
        write_buf = array.array("H", tok_buf)
        write_buf.byteswap()

    out_f.write(write_buf.tobytes())
    return len(tok_buf)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet-dir", type=Path)
    src.add_argument("--parquet-files", type=Path, nargs="+")
    parser.add_argument("--parquet-glob", type=str, default=None)
    parser.add_argument("--text-column", type=str, default="text")

    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--bos-token", type=str, default="<|bos|>")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)

    parser.add_argument("--tokens-per-shard", type=int, default=0)
    parser.add_argument("--max-docs-per-shard", type=int, default=0)

    parser.add_argument("--out-u16", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)

    parser.add_argument("--verify-first-tokens", type=int, default=0)

    args = parser.parse_args(argv)

    parquet_files: list[Path]
    if args.parquet_files is not None:
        parquet_files = [p.resolve() for p in args.parquet_files]
    else:
        parquet_files = _iter_parquet_files(args.parquet_dir.resolve(), args.parquet_glob)

    if not parquet_files:
        raise SystemExit("No parquet files found.")

    args.out_u16.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)

    enc = _load_encoding(args.tokenizer_dir.resolve())
    vocab_size = int(getattr(enc, "n_vocab"))
    bos_id = int(enc.encode_single_token(args.bos_token))
    if not (0 <= bos_id <= 0xFFFF):
        raise SystemExit(f"bos_id out of u16 range: {bos_id}")

    tokens_per_shard = int(args.tokens_per_shard)
    max_docs_per_shard = int(args.max_docs_per_shard)
    max_tokens_to_sample = int(args.verify_first_tokens)

    sample_tokens: list[int] | None = [] if max_tokens_to_sample > 0 else None

    shards: list[dict] = []
    total_tokens = 0

    with args.out_u16.open("wb") as out_f:
        for shard_index, parquet_path in enumerate(parquet_files):
            start_offset_bytes = out_f.tell()
            shard_tokens_written = 0
            shard_docs_seen = 0

            pf = pq.ParquetFile(str(parquet_path))
            for rg_idx in range(pf.num_row_groups):
                if max_docs_per_shard > 0 and shard_docs_seen >= max_docs_per_shard:
                    break
                if tokens_per_shard > 0 and shard_tokens_written >= tokens_per_shard:
                    break

                table = pf.read_row_group(rg_idx, columns=[args.text_column])
                col = table.column(0)
                docs = [d if d is not None else "" for d in col.to_pylist()]

                for i in range(0, len(docs), args.tokenizer_batch_size):
                    if max_docs_per_shard > 0 and shard_docs_seen >= max_docs_per_shard:
                        break
                    if tokens_per_shard > 0 and shard_tokens_written >= tokens_per_shard:
                        break

                    batch = docs[i : i + args.tokenizer_batch_size]
                    if max_docs_per_shard > 0:
                        batch = batch[: max_docs_per_shard - shard_docs_seen]

                    remaining_tokens = None
                    if tokens_per_shard > 0:
                        remaining_tokens = tokens_per_shard - shard_tokens_written

                    wrote = _encode_and_write_docs(
                        out_f=out_f,
                        enc=enc,
                        docs=batch,
                        bos_id=bos_id,
                        num_threads=args.num_threads,
                        max_tokens_to_write=remaining_tokens,
                        sample_tokens_out=sample_tokens,
                        sample_limit=max_tokens_to_sample,
                    )
                    shard_tokens_written += wrote
                    shard_docs_seen += len(batch)

            bytes_written = out_f.tell() - start_offset_bytes
            shards.append(
                {
                    "shard_index": shard_index,
                    "parquet_path": str(parquet_path),
                    "start_offset_bytes": int(start_offset_bytes),
                    "tokens_written": int(shard_tokens_written),
                    "bytes_written": int(bytes_written),
                }
            )
            total_tokens += shard_tokens_written

    total_bytes = args.out_u16.stat().st_size

    manifest = {
        "format": "pcp_u16_tokens_v1",
        "dtype": "u16_le",
        "vocab_size": vocab_size,
        "bos_token": args.bos_token,
        "bos_id": bos_id,
        "total_tokens": int(total_tokens),
        "total_bytes": int(total_bytes),
        "tokens_per_shard": int(tokens_per_shard),
        "chunk_size_bytes": int(tokens_per_shard * 2) if tokens_per_shard > 0 else 0,
        "created_at_unix": int(time.time()),
        "shards": shards,
    }

    args.out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    if sample_tokens is not None:
        with args.out_u16.open("rb") as f:
            buf = f.read(len(sample_tokens) * 2)
        on_disk = array.array("H")
        on_disk.frombytes(buf)
        if sys.byteorder != "little":
            on_disk.byteswap()
        if list(on_disk) != sample_tokens:
            raise SystemExit("Verification failed: first tokens on disk do not match written tokens.")

    print(f"Wrote: {args.out_u16} ({total_bytes} bytes)")
    print(f"Wrote: {args.out_manifest}")
    print(f"Tokens: {total_tokens} (bos_id={bos_id}, vocab_size={vocab_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
