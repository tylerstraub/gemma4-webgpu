// Per-layer embedding (PLE) block skip-scale:
//   `hidden = (hidden + ple_residual) * layer_output_scale`
//
// `layer_output_scale` is a per-layer F16 scalar (stored as a [1]-element
// buffer, padded to 4 bytes). Dispatched once per layer at end of block.
enable f16;
struct Params { size: u32 }
@group(0) @binding(0) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> post_normed: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f16>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.size) { return; }
  let s = f32(scale[0]);
  hidden[i] = (hidden[i] + post_normed[i]) * s;
}
