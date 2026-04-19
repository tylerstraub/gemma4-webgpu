struct Params { num_heads: u32, seq_len: u32 }
@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let head = wid.x;
  if (head >= params.num_heads) { return; }
  let base = head * params.seq_len;
  var local_max: f32 = -1e30;
  var i = tid;
  while (i < params.seq_len) {
    local_max = max(local_max, scores[base + i]);
    i += 256u;
  }
  shared_max[tid] = local_max;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]); }
    workgroupBarrier();
  }
  let max_val = shared_max[0];
  var local_sum: f32 = 0.0;
  i = tid;
  while (i < params.seq_len) {
    let e = exp(scores[base + i] - max_val);
    scores[base + i] = e;
    local_sum += e;
    i += 256u;
  }
  shared_sum[tid] = local_sum;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let sum_val = shared_sum[0];
  i = tid;
  while (i < params.seq_len) {
    scores[base + i] = scores[base + i] / sum_val;
    i += 256u;
  }
}
