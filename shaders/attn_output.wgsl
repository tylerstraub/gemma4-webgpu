struct Params { num_q_heads: u32, num_kv_heads: u32, head_dim: u32, seq_len: u32 }
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> v_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.num_q_heads * params.head_dim;
  if (idx >= total) { return; }
  let head = idx / params.head_dim;
  let d = idx % params.head_dim;
  let kv_head = head * params.num_kv_heads / params.num_q_heads;
  let kv_stride = params.num_kv_heads * params.head_dim;
  var sum: f32 = 0.0;
  for (var pos: u32 = 0u; pos < params.seq_len; pos++) {
    let prob = probs[head * params.seq_len + pos];
    let v_idx = pos * kv_stride + kv_head * params.head_dim + d;
    sum += prob * v_cache[v_idx];
  }
  output[head * params.head_dim + d] = sum;
}
