struct Params { num_heads: u32, head_dim: u32, position: u32, theta: f32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let half_dim = params.head_dim / 2u;
  let total_pairs = params.num_heads * half_dim;
  if (idx >= total_pairs) { return; }
  let head = idx / half_dim;
  let i = idx % half_dim;
  let base = head * params.head_dim;
  let freq = 1.0 / pow(params.theta, f32(i * 2u) / f32(params.head_dim));
  let angle = f32(params.position) * freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);
  let x0 = data[base + i];
  let x1 = data[base + i + half_dim];
  data[base + i] = x0 * cos_a - x1 * sin_a;
  data[base + i + half_dim] = x0 * sin_a + x1 * cos_a;
}
