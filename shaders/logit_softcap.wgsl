// Gemma 4 final-logit softcapping: `tanh(x / cap) * cap`, with the inner
// argument clamped to ±15 to avoid `tanh` saturation breaking autograd-style
// numerics (this is an inference-only engine, but the clamp is cheap and
// matches the reference implementation).
struct Params { size: u32, cap: f32 }
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.size) { return; }
  let x = logits[i] / params.cap;
  let clamped = clamp(x, -15.0, 15.0);
  logits[i] = tanh(clamped) * params.cap;
}
