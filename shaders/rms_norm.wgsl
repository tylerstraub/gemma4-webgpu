enable f16;
struct Params { hidden_size: u32, eps: f32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let hidden_size = params.hidden_size;
  var partial_sum: f32 = 0.0;
  var i = tid;
  while (i < hidden_size) {
    let val = input[i];
    partial_sum += val * val;
    i += 256u;
  }
  shared_sum[tid] = partial_sum;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let rms = sqrt(shared_sum[0] / f32(hidden_size) + params.eps);
  i = tid;
  while (i < hidden_size) {
    output[i] = input[i] * f32(weight[i]) / rms;
    i += 256u;
  }
}
