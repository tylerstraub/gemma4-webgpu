# reference/pytorch — HF Transformers reference generator

Standalone Python environment that produces the reference tensors the
engine's `crossLabDiff` methodology verifies against. Loads
`google/gemma-4-E2B-it` via HuggingFace Transformers, runs one forward
pass on the canonical prompt with forward-hooks registered at every
capture boundary, and dumps each hook's output as an entry in an npz
file.

The outputs are the ground truth the WebGPU engine is compared against.
A passing cross-lab sweep means the engine's forward pass is
numerically equivalent to HF's bf16 reference at every intermediate
boundary — per-layer residuals, per-layer-embedding stages, and final
logits.

## When to regenerate

- After a HF Transformers version bump that might touch Gemma 4's forward
  math (check the transformers changelog for anything under `gemma4`).
- Before making a correctness claim about a modified engine against a
  reference from a different time.
- When adding a new `CapturePoint` to the engine — update the smoke
  script to dump the matching reference entry.

The committed `public/ref/smoke.npz` is generated against
`transformers 5.5.4` + `torch 2.11.0+cpu`. Matching runtime versions
pinned in `pyproject.toml` + `uv.lock` keeps regeneration deterministic.

## Setup

Requires [`uv`](https://github.com/astral-sh/uv) (Rust-based Python
package manager — fast, reproducible, handles the venv + lockfile
automatically).

```sh
cd reference/pytorch
uv sync                          # creates .venv/ and resolves from uv.lock
uv run python smoke_test.py      # auto-activates .venv, runs the script
```

First run downloads ~5 GB of HF model weights into
`~/.cache/huggingface/hub/`. Subsequent runs use the cache.

**HuggingFace auth**: `google/gemma-4-E2B-it` is not gated — no license
accept step or auth token required. (The checkpoint is distributed under
the Gemma Terms of Use, which you should review separately; `uv run` /
`huggingface_hub` handle the download mechanics.)

**CPU vs GPU**: the default dtype is bf16 on CPU. A single forward pass
for a 15-token prompt takes ~1–3 minutes. CPU keeps setup portable; no
CUDA / ROCm toolchain required.

## The canonical prompt

`"Hello, how are you?"` — 15 tokens after the Gemma chat-template is
applied, capture position is 14 (the final `\n`). Token IDs:

```
[2, 105, 2364, 107, 9259, 236764, 1217, 659, 611, 236881, 106, 107, 105, 4368, 107]
```

Both scripts `assert` the exact sequence so any divergence in the
chat-template or tokenizer surfaces immediately.

## Scripts

### `smoke_test.py` — full-layer dump

Runs the full forward pass, captures `hidden_states` at every layer
boundary, dumps to `ref_dumps/smoke.npz`. Entries:

- `tokens` — the 15 token IDs
- `position` — 14 (the capture position)
- `embed_pos14` — post-embedding activations `[hidden_size]`
- `afterLayer_00_pos14` .. `afterLayer_34_pos14` — per-layer residual
  stream after each decoder block `[hidden_size]`
- `logits_pos14` — post-LM-head + softcap `[vocab_size]`

Output size: ~1.2 MB npz. This is what the engine's `crossLabDiff`
sweeps against.

### `probe_layer0.py` — sub-block dump for layer 0

Runs the same forward but with forward-hooks on every submodule under
`language_model.layers[0]`. Dumps fine-grained intermediates to
`ref_dumps/layer_0_detail.npz`. Useful when a layer-level sweep points
at layer 0 as the first-divergence site and you need to localize which
submodule.

This is the script that originally surfaced the `v_norm` architectural
quirk — the module-tree print made the per-KV-head RMSNorm on V visible
and matched it against an engine implementation that had omitted it.

## Outputs

`ref_dumps/` is gitignored. The canonical pre-regenerated
`smoke.npz` + `layer_0_detail.npz` live under `public/ref/` (checked
into git — they're small and stable). To refresh them:

```sh
uv run python smoke_test.py
uv run python probe_layer0.py
cp ref_dumps/smoke.npz ../../public/ref/
cp ref_dumps/layer_0_detail.npz ../../public/ref/
```

## Gotchas

1. **Model class**: `Gemma4ForConditionalGeneration` (multimodal). For
   text-only hooking you access the `Gemma4TextModel` via
   `model.model.language_model` rather than instantiating
   `Gemma4ForCausalLM` directly.
2. **Tokenizer dict return**: in transformers 5.5.4,
   `apply_chat_template(msgs, tokenize=True, ...)` returns a
   `tokenizers.Encoding` dict, not a list. The two-step
   `tokenize=False` → pass through `tokenizer()` pattern is cleaner.
3. **Windows `cp1252` stdout**: multilingual vocab entries can break
   `print()` output. ASCII-escape decoded strings before printing.
   Writing JSON / npz is unaffected.
4. **Symlinks on Windows**: HF cache falls back to copy mode without
   admin / Developer Mode enabled. ~2× disk usage during download.
   Not breaking.
5. **`attn_implementation="eager"`**: required for forward hooks on
   attention internals. SDPA (the default) hides intermediates inside a
   fused kernel.
