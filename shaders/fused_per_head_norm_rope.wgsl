// Fuses per-head RMSNorm + RoPE into one kernel. Dispatched once per head
// (workgroup_id.x = head index).
//
// RoPE base frequency can be attenuated via `rope_freqs`, a 256-element F16
// table loaded at init. The source GGUF values are 1.0 (first 64 entries)
// and 1e30 (entries 64..255, baking in `partial_rotary_factor=0.25`). In F16
// storage these saturate to exactly 1.0 and +Inf — that's by design.
// `base_freq / +Inf = 0` → `cos(0)=1, sin(0)=0` → no rotation, matching
// Gemma 4's proportional-RoPE behavior on the trailing pairs of GLOBAL
// (full-attention) layers. `apply_divisor=0` skips the lookup entirely for
// sliding-window layers, which use the default RoPE schedule.
enable f16;
struct Params { num_heads: u32, head_dim: u32, eps: f32, position: u32, theta: f32, apply_divisor: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> rope_freqs: array<f16>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let head = wid.x;
  if (head >= params.num_heads) { return; }
  let tid = lid.x;
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
  workgroupBarrier();
  let half_dim = params.head_dim / 2u;
  i = tid;
  while (i < half_dim) {
    let base_freq = 1.0 / pow(params.theta, f32(i * 2u) / f32(params.head_dim));
    var freq: f32 = base_freq;
    if (params.apply_divisor != 0u) {
      freq = base_freq / f32(rope_freqs[i]);
    }
    let angle = f32(params.position) * freq;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let x0 = data[base + i];
    let x1 = data[base + i + half_dim];
    data[base + i] = x0 * cos_a - x1 * sin_a;
    data[base + i + half_dim] = x0 * sin_a + x1 * cos_a;
    i += 256u;
  }
}
