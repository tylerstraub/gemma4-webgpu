# scripts

Standalone CLI utilities that support the main codebase without depending
on Vite, WebGPU, or the engine. Each script is a self-contained Node
module — runnable without `npm install`-ing anything in this repo.

## `audit_gguf.mjs` — GGUF header audit

Parses the prefix of a GGUF file and dumps:

- Full metadata (all `general.*` + Gemma-4 arch keys)
- Tensor-descriptor section: per-quant-type breakdown + byte totals
- Full tensor inventory grouped by type (per-layer families collapsed)
- Multimodal tensor search (`vision_tower.*`, `audio_tower.*`, `mm_*`, etc.)
- File-size consistency summary

Mirrors `src/gguf.ts` `GGUFParser` semantics so results match what the
engine's loader sees at runtime.

### When to run

- Before making public claims about what this engine loads (README
  paragraphs, blog posts, docs). GGUFs can shift silently upstream — this
  is the ground-truth check.
- Any time the runtime model URL changes — re-verify that the file is
  still Gemma 4 E2B text-only.
- After any change to `src/gguf.ts` dequant paths — confirm the tensor
  count, layer count, and per-quant distribution match the previous audit.

### How to run

```bash
# 1. Fetch the first ~20 MB of the GGUF header (enough to cover the whole
#    metadata + tensor-descriptor section for Gemma 4 E2B).
curl -sL -o /tmp/gguf_header.bin \
  -H "Range: bytes=0-20971519" \
  "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"

# 2. Run. GGUF_HEADER env var tells the script where to find the file.
#    macOS / Linux — /tmp is a native path, so:
GGUF_HEADER="/tmp/gguf_header.bin" node scripts/audit_gguf.mjs

#    Windows under MSYS / Git-Bash — Node doesn't understand the MSYS /tmp
#    path, so resolve to a real Windows path first:
#      cygpath -w /tmp/gguf_header.bin
#        → C:\Users\<you>\AppData\Local\Temp\gguf_header.bin
#    Then pass the resolved path:
#      GGUF_HEADER="C:/Users/<you>/AppData/Local/Temp/gguf_header.bin" \
#        node scripts/audit_gguf.mjs
```

For a larger GGUF with more tensors, the header prefix may need to be
larger — watch for out-of-bounds `DataView` errors and bump the Range
upper bound accordingly.
