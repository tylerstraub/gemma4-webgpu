// Multi-row variant: each workgroup produces R=4 consecutive output rows,
// amortizing tree-reduce + launch overhead across 4× more FMAs per
// workgroup. Caller dispatches ceil(M/4) workgroups. Bindings and Params
// layout match `matmul_quant` exactly, so bind-group layouts are
// structurally identical (only the pipeline object differs). Used for
// `ffn.linearGateUp` where M=I ∈ {6144, 12288} (both divisible by 4).
//
// Per-iteration k-loop: read `input[k]` once, multiply by 4 weights (one
// per row), accumulate 4 partials. `input[k]` fetch amortizes 4×. Weight
// fetches remain coalesced within each warp on the `tid` dimension; the
// 4 per-row fetches are N f16s apart (= 3 KB for N=1536), so they issue
// as separate memory transactions but each is independently coalesced.
//
// Reduce: all 4 rows share the 8-level tree-reduce — `partials` is 4×
// wider (1024 f32 = 4 KB shared mem), and each stride step reduces 4
// lanes in lockstep, keeping total barriers at 8.
enable f16;
struct Params { M: u32, N: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
// Pipeline-constant rows-per-workgroup. Injected at `createComputePipeline`
// time from the active tuning profile's `matmul.rowsPerWorkgroupByKernel`
// value for `ffn.linearGateUp`. The shader body is unrolled for R=4; other
// values would require a variant shader (MR2, MR8) with matching unrolling
// and a caller dispatch-count adjustment. See `src/tuning/profile.ts`.
override R: u32 = 4;
var<workgroup> partials: array<f32, 1024>; // 256 * 4

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let m0 = (wg.y * ng.x + wg.x) * R;
  let tid = lid.x;
  let N = params.N;
  let M = params.M;

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var acc2: f32 = 0.0;
  var acc3: f32 = 0.0;

  let row0 = m0 * N;
  let row1 = row0 + N;
  let row2 = row1 + N;
  let row3 = row2 + N;

  var k: u32 = tid;
  loop {
    if (k >= N) { break; }
    let inp = input[k];
    acc0 = acc0 + f32(weight[row0 + k]) * inp;
    acc1 = acc1 + f32(weight[row1 + k]) * inp;
    acc2 = acc2 + f32(weight[row2 + k]) * inp;
    acc3 = acc3 + f32(weight[row3 + k]) * inp;
    k = k + WG;
  }

  partials[0u * WG + tid] = acc0;
  partials[1u * WG + tid] = acc1;
  partials[2u * WG + tid] = acc2;
  partials[3u * WG + tid] = acc3;
  workgroupBarrier();

  var stride: u32 = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      partials[0u * WG + tid] = partials[0u * WG + tid] + partials[0u * WG + tid + stride];
      partials[1u * WG + tid] = partials[1u * WG + tid] + partials[1u * WG + tid + stride];
      partials[2u * WG + tid] = partials[2u * WG + tid] + partials[2u * WG + tid + stride];
      partials[3u * WG + tid] = partials[3u * WG + tid] + partials[3u * WG + tid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (tid < R) {
    let m = m0 + tid;
    if (m < M) {
      output[m] = partials[tid * WG];
    }
  }
}
