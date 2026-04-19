"""
Layer-0 sub-block forward-hook probe.

Runs one forward pass of google/gemma-4-E2B-it on the canonical prompt
with hooks registered on every submodule under
`language_model.layers[0]`. Dumps the position-14 slice of each module's
output (and `o_proj`'s INPUT via a pre-hook, for the `attnOut` capture
point) to `ref_dumps/layer_0_detail.npz`.

This deeper per-submodule view is useful when a full-layer crossLabDiff
sweep points at layer 0 as the first-divergence site and you want to
localize which specific sub-module introduces the drift. Using this
probe is how the v_norm bug (missing per-KV-head RMSNorm on V) was
originally surfaced — the submodule tree printed by this script made the
`v_norm` module in HF's `Gemma4TextAttention` visible, revealing that a
RMSNorm was being applied on V that the engine had omitted.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "Hello, how are you?"
POSITION = 14
OUT = Path(__file__).parent / "ref_dumps"
OUT.mkdir(exist_ok=True)


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] Loading…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    lm_text = model.model.language_model
    L0 = lm_text.layers[0]
    lm_head = model.lm_head

    # Print L0's full named-module tree. Useful for mapping each capture
    # to the corresponding engine CapturePoint (and for catching missing
    # modules the engine's attention pipeline may have overlooked).
    print(f"\n[{time.time()-t0:6.1f}s] L0 module tree:")
    for name, mod in L0.named_modules():
        if name:  # skip the root ""
            cls = mod.__class__.__name__
            print(f"  {name:60s} {cls}")

    # Capture dict — filled by hooks during the forward pass.
    captures = {}

    def make_post_hook(key):
        def _hook(module, inputs, output):
            # Modules like self_attn can return (hidden, attn_weights, past_kv);
            # unwrap the tuple's first element. hf attention's forward returns
            # a tuple whose first element is the post-o_proj hidden state.
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            if torch.is_tensor(h) and h.ndim >= 2:
                # hidden-state activations are [batch, seq, dim]; scalar outputs
                # like attention weights don't fit that, so guard on ndim.
                if h.shape[0] == 1 and h.shape[1] > POSITION:
                    captures[key] = h[0, POSITION].detach().float().cpu().numpy()
        return _hook

    def make_pre_hook(key):
        def _hook(module, inputs):
            # inputs is a tuple; activations are typically inputs[0].
            if len(inputs) > 0 and torch.is_tensor(inputs[0]) and inputs[0].ndim >= 2:
                h = inputs[0]
                if h.shape[0] == 1 and h.shape[1] > POSITION:
                    captures[key] = h[0, POSITION].detach().float().cpu().numpy()
        return _hook

    # Register a forward hook on every named submodule. `named_modules()`
    # yields every descendant including nested ones. Using the module name as
    # the capture key keeps the npz keys self-documenting.
    handles = []
    for name, mod in L0.named_modules():
        if not name:
            continue
        # Skip container modules (ModuleList, etc.) — they don't produce tensor outputs.
        if mod.__class__.__name__ in ("ModuleList", "ModuleDict", "Sequential", "Gemma4DecoderLayer", "Gemma4Attention", "Gemma4MLP"):
            # Still hook attention/mlp at the module level for their aggregate output.
            if mod.__class__.__name__ in ("Gemma4Attention", "Gemma4MLP"):
                handles.append(mod.register_forward_hook(make_post_hook(f"L0_{name}_output")))
            continue
        handles.append(mod.register_forward_hook(make_post_hook(f"L0_{name}_output")))

    # Also register PRE-forward hooks on o_proj and down_proj so we
    # capture their inputs — pre-linearAttnOut (= the engine's attnOut:0
    # capture point) and pre-ffnDown-matmul (= post-geluMul / ffnAct).
    try:
        handles.append(L0.self_attn.o_proj.register_forward_pre_hook(make_pre_hook("L0_self_attn_o_proj_input")))
    except AttributeError as e:
        print(f"WARN: no self_attn.o_proj pre-hook: {e}")
    try:
        handles.append(L0.mlp.down_proj.register_forward_pre_hook(make_pre_hook("L0_mlp_down_proj_input")))
    except AttributeError as e:
        print(f"WARN: no mlp.down_proj pre-hook: {e}")

    # Also grab the input to the whole L0 block — this is the post-embed
    # residual going in (should match embed_pos14 closely).
    handles.append(L0.register_forward_pre_hook(make_pre_hook("L0_block_input")))
    # And the output of the whole L0 block (should match afterLayer_00_pos14).
    handles.append(L0.register_forward_hook(make_post_hook("L0_block_output")))

    # Tokenize (two-step: chat-template to string, then tokenize to get a
    # proper tensor dict — in transformers 5.5.4, tokenize=True returns an
    # Encoding object that's awkward to use).
    msgs = [{"role": "user", "content": PROMPT}]
    formatted = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(formatted, return_tensors="pt", add_special_tokens=False)
    tokens = enc["input_ids"][0].tolist()
    assert tokens == [2, 105, 2364, 107, 9259, 236764, 1217, 659, 611, 236881,
                      106, 107, 105, 4368, 107], f"token divergence: {tokens}"
    print(f"\n[{time.time()-t0:6.1f}s] Tokens verified. Running forward pass…")

    with torch.no_grad():
        _ = lm_text(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=False,
            use_cache=False,
        )

    for h in handles:
        h.remove()

    print(f"[{time.time()-t0:6.1f}s] Forward complete. Captured {len(captures)} intermediates:")
    for k in sorted(captures.keys()):
        v = captures[k]
        print(f"  {k:60s} shape={tuple(v.shape)}  dtype={v.dtype}  absMax={np.abs(v).max():.3f}  norm={np.linalg.norm(v):.3f}")

    # Save npz. `tokens` and `position` are included for self-
    # documentation — so the engine-side loader can sanity-check it's
    # the same prompt.
    dump = {"tokens": np.array(tokens, dtype=np.int32), "position": np.int32(POSITION)}
    dump.update(captures)
    out_path = OUT / "layer_0_detail.npz"
    np.savez(out_path, **dump)
    print(f"[{time.time()-t0:6.1f}s] Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Manifest — JSON for future-me to see what's there without loading numpy.
    manifest = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "tokens": tokens,
        "capture_position": POSITION,
        "layer_index": 0,
        "arrays": {
            k: {"shape": list(v.shape) if hasattr(v, "shape") else [],
                "dtype": str(v.dtype)}
            for k, v in dump.items()
        },
    }
    (OUT / "layer_0_detail_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{time.time()-t0:6.1f}s] Wrote manifest.")


if __name__ == "__main__":
    main()
