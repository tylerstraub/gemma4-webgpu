// `sliding_window = 0` means full attention. A positive value W means
// Gemma-4-style sliding window: position p only attends to keys in
// `[current_pos - W + 1, current_pos]`. Masked positions get -inf so
// softmax drops them. `current_pos = seq_len - 1`.
//
// `scale` is always 1.0 for Gemma 4 (q_norm + k_norm already normalize
// each head to unit RMS, so the standard 1/sqrt(head_dim) compensation
// is dropped — baking it in would flatten the softmax by sqrt(HD).)
struct Params { num_q_heads: u32, num_kv_heads: u32, head_dim: u32, seq_len: u32, scale: f32, sliding_window: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.num_q_heads * params.seq_len;
  if (idx >= total) { return; }
  let head = idx / params.seq_len;
  let pos = idx % params.seq_len;
  let score_idx = head * params.seq_len + pos;
  if (params.sliding_window != 0u) {
    let current_pos = params.seq_len - 1u;
    let window_start = select(0u, current_pos + 1u - params.sliding_window, current_pos + 1u > params.sliding_window);
    if (pos < window_start) {
      scores[score_idx] = -1e30;
      return;
    }
  }
  let kv_head = head * params.num_kv_heads / params.num_q_heads;
  let q_offset = head * params.head_dim;
  let kv_stride = params.num_kv_heads * params.head_dim;
  let k_offset = pos * kv_stride + kv_head * params.head_dim;
  var dot: f32 = 0.0;
  for (var d: u32 = 0u; d < params.head_dim; d++) {
    dot += q[q_offset + d] * k_cache[k_offset + d];
  }
  scores[score_idx] = dot * params.scale;
}
