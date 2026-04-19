// Single-output-row-per-workgroup matmul. All 256 threads cooperate on the
// N-long dot product: thread `tid` reads weight[m, tid..N stride 256] and
// input[tid..N stride 256] so consecutive threads within a warp hit
// consecutive addresses (coalesced). Partial sums fold via an 8-level
// shared-memory tree reduce.
//
// 2D dispatch support: callers with M > maxComputeWorkgroupsPerDimension
// (lmHead's M=262144) use a sqrt-shaped 2D grid; `m = wg.y * ng.x + wg.x`
// recovers the row index. 1D callers pass `ng.y = 1` so the formula
// reduces to `m = wg.x`.
enable f16;
struct Params { M: u32, N: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> partials: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let m = wg.y * ng.x + wg.x;
  if (m >= params.M) { return; }
  let tid = lid.x;
  let N = params.N;
  let row_off = m * N;

  var acc: f32 = 0.0;
  var k: u32 = tid;
  loop {
    if (k >= N) { break; }
    acc = acc + f32(weight[row_off + k]) * input[k];
    k = k + WG;
  }
  partials[tid] = acc;
  workgroupBarrier();

  var stride: u32 = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      partials[tid] = partials[tid] + partials[tid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (tid == 0u) {
    output[m] = partials[0];
  }
}
