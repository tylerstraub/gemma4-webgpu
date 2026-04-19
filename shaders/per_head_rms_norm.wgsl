enable f16;
struct Params { num_heads: u32, head_dim: u32, eps: f32, pad: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let head = wid.x;
  if (head >= params.num_heads) { return; }
  let base = head * params.head_dim;
  var partial: f32 = 0.0;
  var i = tid;
  while (i < params.head_dim) {
    let val = data[base + i];
    partial += val * val;
    i += 256u;
  }
  shared_sum[tid] = partial;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let rms = sqrt(shared_sum[0] / f32(params.head_dim) + params.eps);
  i = tid;
  while (i < params.head_dim) {
    data[base + i] = data[base + i] * f32(weight[i]) / rms;
    i += 256u;
  }
}
