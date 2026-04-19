struct Params { num_kv_heads: u32, head_dim: u32, position: u32, max_seq_len: u32 }
@group(0) @binding(0) var<storage, read> k_in: array<f32>;
@group(0) @binding(1) var<storage, read> v_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = params.num_kv_heads * params.head_dim;
  if (i >= total) { return; }
  let head = i / params.head_dim;
  let d = i % params.head_dim;
  let cache_idx = params.position * total + head * params.head_dim + d;
  k_cache[cache_idx] = k_in[i];
  v_cache[cache_idx] = v_in[i];
}
