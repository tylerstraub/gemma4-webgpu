enable f16;
struct Params { hidden_size: u32, token_id: u32 }
@group(0) @binding(0) var<storage, read> embedding: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.hidden_size) { return; }
  let row_off = params.token_id * params.hidden_size;
  output[i] = f32(embedding[row_off + i]) * sqrt(f32(params.hidden_size));
}
