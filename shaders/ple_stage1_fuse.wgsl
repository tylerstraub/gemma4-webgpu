// Per-layer embedding (PLE) Stage 1. Fuses per-layer-projected norm +
// per-layer embedding lookup + rsqrt(2) blend into a single kernel.
//
// Dispatched once per layer per forward pass. Each call binds its layer's
// per_layer_token_embd slice (~128 MB on Gemma 4 E2B). The `projected`
// input comes from `per_layer_model_proj @ hidden`, reshaped as
// `[num_layers, per_layer_dim]`.
enable f16;
struct Params { layer_idx: u32, token_id: u32, per_layer_dim: u32, eps: f32 }
@group(0) @binding(0) var<storage, read> projected: array<f32>;
@group(0) @binding(1) var<storage, read> norm_weight: array<f16>;
@group(0) @binding(2) var<storage, read> embed_slice: array<f16>;
@group(0) @binding(3) var<storage, read_write> ple_inputs: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let D = params.per_layer_dim;
  let proj_base = params.layer_idx * D;
  var partial: f32 = 0.0;
  if (tid < D) {
    let v = projected[proj_base + tid];
    partial = v * v;
  }
  shared_sum[tid] = partial;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let rms = sqrt(shared_sum[0] / f32(D) + params.eps);
  if (tid < D) {
    let proj_normed = (projected[proj_base + tid] / rms) * f32(norm_weight[tid]);
    let embed_val = f32(embed_slice[params.token_id * D + tid]) * sqrt(f32(D));
    ple_inputs[proj_base + tid] = (proj_normed + embed_val) * inverseSqrt(2.0);
  }
}
