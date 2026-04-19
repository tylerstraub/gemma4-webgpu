// Per-layer embedding (PLE) Stage 2 step b: GELU the `inp_gate` result and
// multiply elementwise by the per-layer slice of `ple_inputs`. Dispatched
// once per layer (inside the block).
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const GELU_COEF_A: f32 = 0.044715;
struct Params { layer_offset: u32, size: u32 }
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> ple_inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.size) { return; }
  let x = gate[i];
  let tanh_arg = clamp(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x), -15.0, 15.0);
  let gelu = 0.5 * x * (1.0 + tanh(tanh_arg));
  output[i] = gelu * ple_inputs[params.layer_offset + i];
}
