# TODO

## Tokenizer determinism / checkpoint compatibility

- [ ] Make `tokenizer="char"` deterministic and checkpointed
  - **Problem:** `CharTokenizer.initFromFile` assigns token IDs by iterating a hash-map key iterator, so the `char -> id` mapping is not guaranteed stable across runs/platforms (`src/data/loader.zig:34`). Checkpoints currently persist model tensors and some training state, but not the tokenizer mapping (`src/algorithms/diloco.zig:436`, `src/algorithms/diloco.zig:720`), so resumes can rebuild a different mapping.
  - **Impact:** Resuming from a checkpoint can silently scramble token IDs (embedding rows / output logits no longer correspond to the same characters),. If the dataset has >65 unique chars, which chars get dropped can also vary (`MAX_VOCAB_SIZE=65`), changing the effective dataset.
  - **Proposed fix (choose one):**
    - Deterministic vocab construction: collect unique chars then sort (byte order) before assigning IDs, or assign IDs by first-seen order using a fixed `[256]bool` seen table.
    - Persist tokenizer artifacts: write `char_to_int`/`int_to_char` (and `vocab_size`) to the checkpoint/recovery state and ensure workers reload the same mapping on resume.
  - **Acceptance criteria:**
    - Two fresh runs on the same dataset produce identical `char -> id` mappings.
    - Resume from checkpoint reproduces identical tokenization as the original run.
    - Add a small unit/integration test that fails if mapping changes across runs (and document the behavior in `experiments/README.md`).
