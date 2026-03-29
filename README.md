### LLM Data Pipeline - Lab 1 & Lab 2 (TinyStories / OpenWebText + RoBERTa)

## Overview

This README documents the changes made to the original Lab 1 and Lab 2 LLM data pipeline notebooks, and summarizes the key learnings from both labs.

---

## What Changed

### Dataset

#### Lab 1 - Dataset Journey (In-Memory)

Three datasets were attempted before landing on a working solution:

| Attempt | Dataset | Outcome |
|---|---|---|
| 1st | `wikitext-2-raw-v1` | Original - worked but tiny/Wikipedia only |
| 2nd | `openwebtext` | ❌ ~40GB across 80 shards - 25+ min download, impractical for in-memory |
| 3rd | `bookcorpus` | ❌ `RuntimeError: Dataset scripts are no longer supported` |
| ✅ Final | `roneneldan/TinyStories` | ~475MB, modern parquet format, loads in seconds |

**Why TinyStories?**
- Stored in modern parquet format - no legacy `.py` loading scripts
- No `trust_remote_code` required
- ~475MB total, loads in seconds
- Clean, natural English text (short stories) - well-suited for LM pretraining demos
- We use `.select(range(10000))` for a 10,000-story slice that tokenizes and groups in under a minute

#### Lab 2 - Dataset (Streaming)

| | Original | Updated |
|---|---|---|
| Dataset | `wikitext-2-raw-v1` | `openwebtext` |
| Size | ~2MB | ~40GB (web-scale) |
| Content | Wikipedia articles | Reddit-curated web pages |

- OpenWebText's 40GB size is actually **ideal** for Lab 2 - streaming mode was built exactly for datasets too large to fit in RAM.
- Data is fetched shard-by-shard on the fly, so size is never a bottleneck.

---

### Tokenizer - `gpt2` → `roberta-base` (Both Labs)

| | Original | Updated |
|---|---|---|
| Model | GPT-2 | RoBERTa-base |
| Tokenizer type | BPE | BPE (improved) |
| Vocab size | 50,257 | 50,265 |
| Pad token | ❌ None (workaround needed) | ✅ Native `<pad>` token |
| Special tokens | EOS only | BOS `<s>`, EOS `</s>`, PAD `<pad>` |

- GPT-2 has no dedicated pad token, requiring the hack `tokenizer.pad_token = tokenizer.eos_token`. RoBERTa has a native `<pad>` token - no workaround needed.
- `add_special_tokens=False` is used in both labs to keep the token stream clean for LM chunking (no `<s>` / `</s>` inserted at document boundaries).

---

### Code Improvements

#### Lab 1 - In-Memory Pipeline
- `attention_mask` is now included in the `collate_fn` output (the original omitted it).
- A **decode step** is added at the end to print human-readable text from a batch as a sanity check.
- `batch_size=500` passed to `.map()` for more efficient grouping.

#### Lab 2 - Streaming Pipeline
- **Leftover token padding** added to `group_texts_streaming()` - original Lab 2 silently discarded tokens that didn't fill a full block. Now they are padded to `block_size` and yielded with a proper `attention_mask` marking real vs. padded positions:
  ```
  attention_mask: [1, 1, 1, ..., 0, 0, 0]
                   ↑ real tokens   ↑ padding
  ```
- `pad_token_id` is passed into the generator and `IterableDataset` class explicitly, making padding behavior transparent.
- A **decode step** is added at the end to inspect batch content.

---

## Key Learnings

### Lab 1 - In-Memory Pipeline
- How to load a HuggingFace dataset and slice it with `.select()` for manageable in-memory experiments.
- How `.map(batched=True)` applies tokenization efficiently across an entire dataset upfront.
- The **concatenate-then-chunk** strategy for LM training: all documents are merged into one long token stream and sliced into fixed-length blocks - avoids padding waste.
- How to build a PyTorch `DataLoader` with a custom `collate_fn` for LM training, where `labels = input_ids` (the model handles the internal next-token shift).
- That not all HuggingFace datasets are equal - legacy script-based datasets (`bookcorpus`) are no longer supported; always prefer modern parquet-based datasets.

### Lab 2 - Streaming Pipeline
- How `streaming=True` converts a HuggingFace dataset into an `IterableDataset` that fetches data lazily - enabling processing of web-scale corpora without RAM constraints.
- How `.map()` on a streaming dataset applies tokenization **on the fly** rather than upfront.
- How a **rolling buffer generator** handles document boundaries: tokens accumulate in a buffer across documents and are yielded in fixed-size chunks - no wasted tokens at boundaries.
- How to wrap a Python generator in a PyTorch `IterableDataset` to make it compatible with `DataLoader`.
- Why `shuffle=True` is **not supported** for `IterableDataset` - data must be consumed in stream order.
- How to handle **leftover tokens** at the end of a stream using padding + attention mask.

---

## Lab 1 vs Lab 2 - Side-by-Side

| Feature | Lab 1 | Lab 2 |
|---|---|---|
| Dataset | `roneneldan/TinyStories` | `openwebtext` |
| Dataset loading | Full dataset in RAM | Streaming, never in RAM |
| Tokenization | Batched upfront via `.map()` | Lazy, on-the-fly |
| Chunking strategy | One-shot `group_texts` | Rolling buffer generator |
| Leftover tokens | Discarded | Padded and yielded |
| Shuffle support | ✅ Yes | ❌ No (stream order only) |
| Scale | Small datasets | Web-scale corpora |
| Dataset class | `Dataset` | `IterableDataset` |

---

## Files

| File | Description |
|---|---|
| `lab1_tinystories_roberta.ipynb` | In-memory pipeline - TinyStories + RoBERTa |
| `lab2_openwebtext_roberta_streaming.ipynb` | Streaming pipeline - OpenWebText + RoBERTa |
| `README.md` | This file |

---

*Lab 1 Dataset: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) - Lab 2 Dataset: [openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext) - Tokenizer: [roberta-base](https://huggingface.co/roberta-base)*


Ankit Kumar Sahoo