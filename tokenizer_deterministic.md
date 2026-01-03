# Deterministic `CharTokenizer` vocabulary order

## What changed

PCP’s `CharTokenizer` now builds a deterministic `char -> token_id` mapping by:

1. Collecting the set of unique bytes in the dataset file
2. Sorting those bytes in ascending order
3. Assigning token IDs `0..vocab_len-1` in that sorted order

Implementation: `pcp/src/data/loader.zig` in `CharTokenizer.initFromFile`.

## Code-level change (before vs after)

This is the logic change in `CharTokenizer.initFromFile` (`pcp/src/data/loader.zig`).

### Before

- Build `char_set: AutoHashMap(u8, void)` with all unique bytes from the file.
- Iterate the hash map’s keys (`char_set.keyIterator()`).
- Assign IDs in that iteration order, stopping once you hit `MAX_VOCAB_SIZE`.

Effectively:

```zig
var it = char_set.keyIterator();
var i: u32 = 0;
while (it.next()) |char_ptr| {
    if (i >= MAX_VOCAB_SIZE) continue;
    try char_to_int.put(char_ptr.*, i);
    try int_to_char.put(i, char_ptr.*);
    i += 1;
}
// return CharTokenizer{ ..., .vocab_size = i };
```

Problem: `AutoHashMap.keyIterator()` does **not** guarantee a stable order across runs/processes, so the IDs assigned to characters were not reproducible.

### After

- Build the same `char_set` (unique bytes).
- Copy the keys into a plain list (`ArrayList(u8)`).
- Sort the list (`std.sort.heap` with `std.sort.asc(u8)`).
- Assign IDs by iterating the sorted list.

Now:

```zig
var chars = std.ArrayList(u8).init(allocator);
var it = char_set.keyIterator();
while (it.next()) |char_ptr| {
    try chars.append(char_ptr.*);
}

std.sort.heap(u8, chars.items, {}, std.sort.asc(u8));

const vocab_len: usize = @min(chars.items.len, MAX_VOCAB_SIZE);
for (chars.items[0..vocab_len], 0..) |ch, idx| {
    const token: u32 = @intCast(idx);
    try char_to_int.put(ch, token);
    try int_to_char.put(token, ch);
}
// return CharTokenizer{ ..., .vocab_size = vocab_len };
```

This guarantees that every worker (CPU/GPU) that reads the same dataset file builds the same mapping.


## Why this was needed

Previously, `CharTokenizer.initFromFile` built the vocabulary using a hash map:

- It inserted all unique bytes into `std.AutoHashMap(u8, void)`
- Then iterated the keys via `keyIterator()` and assigned IDs in that iteration order

Hash-map key iteration order is not stable (it depends on hashing/resize history and can vary across processes). In a distributed setup (e.g. 1 CPU worker + 1 CUDA worker), this could lead to:

- Worker A mapping byte `'x'` → token `3`, while Worker B maps `'x'` → token `17`
- Both workers training on “the same” dataset file, but with *different labelings*
- Aggregating updates across workers that are optimizing different objectives, which can destabilize training and break parity comparisons vs PyTorch

This non-determinism was especially problematic because:

- The nanochat model used by PCP expects a fixed vocab size (`MAX_VOCAB_SIZE = 65`), so “which 65 characters are included” could also vary if the dataset has >65 unique bytes.
- For parity work, you need CPU/GPU workers and PyTorch to interpret token IDs identically.

## What problem it caused

Symptoms from the user’s runs included:

- Inconsistent behavior across workers/backends
- Divergence and non-finite losses in real DiLoCo runs (loss becoming `NaN`)

Even if NaNs can have multiple causes, inconsistent tokenization across workers is a correctness bug by itself (it makes “distributed training on the same data” ill-defined).

## Why sorting is not “overfitting”

Sorting unique bytes is:

- Device-agnostic (CPU/GPU doesn’t matter)
- Model-agnostic (any char-level model expects a deterministic mapping)
- Dataset-agnostic (works for any byte content; it just makes ordering stable)

It is a standard, reproducible convention for char-level tokenizers.

## Behavior notes (important for parity)

- `MAX_VOCAB_SIZE` remains `65`. If the dataset has more unique bytes, bytes beyond the first 65 (in sorted order) are ignored and will encode as `0` (the existing “unknown char → 0” behavior).
- `ByteTokenizer` is unchanged (it is inherently deterministic: token = byte value).
- If you compare to PyTorch, build the same char vocab the same way (sorted unique bytes, capped to 65, OOV→0), otherwise loss curves will not match even if everything else is identical.
