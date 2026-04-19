"""
Smoke test: load google/gemma-4-E2B-it, run one forward pass on the canonical
15-token prompt, verify logits shape + top-token sanity, dump per-layer hidden
states to ref_dumps/smoke.npz for next-session starting point.

Expected runtime: ~20s load + ~1-3 min forward on CPU for 15 tokens.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

MODEL_ID = "google/gemma-4-E2B-it"
PROMPT = "Hello, how are you?"
OUT = Path(__file__).parent / "ref_dumps"
OUT.mkdir(exist_ok=True)


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] Loading tokenizer + model…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # Checkpoint was saved under Gemma4ForConditionalGeneration (multimodal), which
    # nests the text model under model.language_model.*. Loading as the
    # conditional class keeps weights aligned; we then access .language_model
    # for text-only forward. Passing only input_ids (no pixel_values / audio_values)
    # makes the conditional model's forward skip the vision + audio towers.
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,        # match safetensors dtype
        attn_implementation="eager", # required for forward hooks on attention internals
        low_cpu_mem_usage=True,
    )
    model.eval()
    # Actual module structure: model.model.language_model (Gemma4TextModel)
    #                          model.model.vision_tower / audio_tower / embed_*
    #                          model.lm_head (separate at root)
    lm_text = model.model.language_model   # text-only Gemma4TextModel
    lm_head = model.lm_head
    print(f"[{time.time()-t0:6.1f}s] Model loaded. Total params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B; LM (text) params: {sum(p.numel() for p in lm_text.parameters())/1e9:.2f}B")

    # Two-step tokenize: chat-template-format string, then tokenize. The one-step
    # `apply_chat_template(tokenize=True, return_tensors='pt')` returns a
    # `tokenizers.Encoding` in transformers 5.5.4, which is awkward to work with.
    msgs = [{"role": "user", "content": PROMPT}]
    formatted = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )
    enc = tok(formatted, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    tokens = input_ids[0].tolist()
    print(f"[{time.time()-t0:6.1f}s] Tokens ({len(tokens)}): {tokens}")
    assert tokens == [2, 105, 2364, 107, 9259, 236764, 1217, 659, 611, 236881,
                      106, 107, 105, 4368, 107], "token-id divergence vs cross-verified expectation"
    print(f"[{time.time()-t0:6.1f}s] Token sequence matches the verified canonical.")

    # Text-only forward: call the Gemma4TextModel directly, skip vision/audio
    # towers entirely. Apply LM head + logit softcap manually (the full
    # ConditionalGeneration forward does this but we want isolated text-model
    # signal for the probe script).
    print(f"[{time.time()-t0:6.1f}s] Forward pass (CPU, bf16)…")
    with torch.no_grad():
        text_out = lm_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )
        last_hidden = text_out.last_hidden_state   # [1, 15, hidden_size]
        logits_raw = lm_head(last_hidden)          # [1, 15, vocab]
        softcap = float(model.config.get_text_config().final_logit_softcapping)
        logits = torch.tanh(logits_raw / softcap) * softcap
    print(f"[{time.time()-t0:6.1f}s] Forward complete. softcap={softcap}")
    out = type("Out", (), {"logits": logits, "hidden_states": text_out.hidden_states})()

    # Validate shapes.
    logits = out.logits  # [1, 15, vocab]
    assert logits.shape == (1, 15, 262144), f"unexpected logits shape {logits.shape}"
    hs = out.hidden_states  # tuple of length num_layers+1 (embedding output + after each layer)
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"hidden_states: tuple of len {len(hs)}; each shape {tuple(hs[0].shape)}")

    # Last-token argmax sanity — capture position 14 (last prompt token, \n).
    last = logits[0, 14].float()
    softcapped_max = last.max().item()
    topv, topi = torch.topk(last, k=10)
    print(f"position 14 logits max = {softcapped_max:.4f}")
    print("top-10 (token_id, logit, decoded):")
    for v, i in zip(topv.tolist(), topi.tolist()):
        # Windows cp1252 can't encode CJK glyphs in some vocab entries; replace on print.
        dec = tok.decode([i]).encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"  {i:6d}  {v:+8.4f}  {dec!r}")

    # Dump per-layer hidden state at capture position 14 (the diagnostic
    # position the workbench uses for afterLayer:L). np-side is F32 to
    # keep precision for downstream diffing against the engine's F32 captures.
    dump = {"tokens": np.array(tokens, dtype=np.int32), "position": np.int32(14)}
    for L, h in enumerate(hs):
        # hs[0] is post-embedding (before layer 0), hs[1] is afterLayer:0, …, hs[-1] is afterLayer:34.
        v = h[0, 14].float().cpu().numpy()
        if L == 0:
            dump["embed_pos14"] = v  # pre-layer-0 (post-embed)
        else:
            dump[f"afterLayer_{L-1:02d}_pos14"] = v
    # Logits at position 14 (post-LM-head, post-softcap if applied).
    dump["logits_pos14"] = last.cpu().numpy()

    out_path = OUT / "smoke.npz"
    np.savez(out_path, **dump)
    print(f"[{time.time()-t0:6.1f}s] Dumped {len(dump)} arrays to {out_path}")

    # Also emit a JSON manifest so future-me can see what's in there without loading numpy.
    manifest = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "tokens": tokens,
        "capture_position": 14,
        "arrays": {k: {"shape": list(v.shape) if hasattr(v, 'shape') else [], "dtype": str(v.dtype)}
                   for k, v in dump.items()},
        # JSON handles unicode fine (unlike Windows cp1252 stdout) — keep the raw decoded glyph.
        "logits_top10": [{"id": int(i), "logit": float(v), "decoded": tok.decode([i])}
                         for v, i in zip(topv.tolist(), topi.tolist())],
    }
    (OUT / "smoke_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{time.time()-t0:6.1f}s] Manifest at {OUT / 'smoke_manifest.json'}")


if __name__ == "__main__":
    main()
