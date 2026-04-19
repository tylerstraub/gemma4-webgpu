import type { GGUFParsed, GGUFTensor, GGUFValue } from './types.js';

/**
 * GGUF v3 parser + dequantization paths for the quant types this engine
 * supports: F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K.
 *
 * Every supported source quant is dequantized to F32 once at upload;
 * callers then convert to F16 via `f32ToF16Array` before creating the
 * GPU storage buffer. The GPU only ever sees F16 weights.
 */
export class GGUFParser {
  buffer: ArrayBuffer;
  view: DataView;
  offset: number;
  textDecoder: TextDecoder;

  constructor(buffer: ArrayBuffer | Uint8Array) {
    if (buffer instanceof Uint8Array) {
      // Handle Uint8Array views with non-zero byteOffset.
      this.buffer = buffer.buffer as ArrayBuffer;
      this.view = new DataView(this.buffer, buffer.byteOffset, buffer.byteLength);
      this.offset = 0;
    } else {
      this.buffer = buffer;
      this.view = new DataView(this.buffer);
      this.offset = 0;
    }
    this.textDecoder = new TextDecoder('utf-8');
  }

  readUint32(): number {
    const val = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readUint64(): bigint {
    const val = this.view.getBigUint64(this.offset, true);
    this.offset += 8;
    return val;
  }

  readString(): string {
    const length = Number(this.readUint64());
    const bytes = new Uint8Array(this.buffer, this.offset, length);
    this.offset += length;
    return this.textDecoder.decode(bytes);
  }

  readValue(type: number): GGUFValue {
    switch (type) {
      case 0: { const val = this.view.getUint8(this.offset); this.offset += 1; return { type: 'uint8', value: val }; }
      case 1: { const val = this.view.getInt8(this.offset); this.offset += 1; return { type: 'int8', value: val }; }
      case 2: { const val = this.view.getUint16(this.offset, true); this.offset += 2; return { type: 'uint16', value: val }; }
      case 3: { const val = this.view.getInt16(this.offset, true); this.offset += 2; return { type: 'int16', value: val }; }
      case 4: return { type: 'uint32', value: this.readUint32() };
      case 5: { const val = this.view.getInt32(this.offset, true); this.offset += 4; return { type: 'int32', value: val }; }
      case 6: { const val = this.view.getFloat32(this.offset, true); this.offset += 4; return { type: 'float32', value: val }; }
      case 7: { const val = this.view.getUint8(this.offset); this.offset += 1; return { type: 'bool', value: val !== 0 }; }
      case 8: return { type: 'string', value: this.readString() };
      case 9: {
        const elemType = this.readUint32();
        const count = Number(this.readUint64());
        const arr: GGUFValue[] = [];
        for (let i = 0; i < count; i++) arr.push(this.readValue(elemType));
        return { type: 'array', value: arr };
      }
      case 10: return { type: 'uint64', value: this.readUint64() };
      case 11: { const val = this.view.getBigInt64(this.offset, true); this.offset += 8; return { type: 'int64', value: val }; }
      case 12: { const val = this.view.getFloat64(this.offset, true); this.offset += 8; return { type: 'float64', value: val }; }
      default: return { type: 'unknown', value: null };
    }
  }

  parse(): GGUFParsed {
    const magic = this.readUint32();
    if (magic !== 0x46554747) throw new Error(`Invalid GGUF magic: 0x${magic.toString(16)}`);
    const version = this.readUint32();
    const tensor_count = this.readUint64();
    const kv_count = this.readUint64();
    const kv = new Map<string, GGUFValue>();
    for (let i = 0n; i < kv_count; i++) {
      const key = this.readString();
      const valueType = this.readUint32();
      const value = this.readValue(valueType);
      kv.set(key, value);
    }
    const tensors: GGUFTensor[] = [];
    for (let i = 0n; i < tensor_count; i++) {
      const name = this.readString();
      const n_dims = this.readUint32();
      const dims: bigint[] = [];
      for (let d = 0; d < n_dims; d++) dims.push(this.readUint64());
      const type = this.readUint32();
      const offset = this.readUint64();
      tensors.push({ name, dims, type, offset });
    }
    const alignment = 32;
    const dataOffset = Math.ceil(this.offset / alignment) * alignment;
    return { version, tensor_count, kv_count, kv, tensors, dataOffset };
  }

  f16ToF32(h: number): number {
    const sign = (h >> 15) & 0x1;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x3ff;
    if (exp === 0) {
      if (mant === 0) return sign ? -0 : 0;
      return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024);
    }
    if (exp === 31) return mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
    return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
  }

  dequantizeQ8_0(offset: number, count: number): Float32Array {
    const blockSize = 32;
    const result = new Float32Array(count);
    let resultIdx = 0;
    let blockOffset = offset;
    while (resultIdx < count) {
      const scaleBits = this.view.getUint16(blockOffset, true);
      const scale = this.f16ToF32(scaleBits);
      blockOffset += 2;
      const elemsInBlock = Math.min(blockSize, count - resultIdx);
      for (let i = 0; i < elemsInBlock; i++) {
        const q = this.view.getInt8(blockOffset + i);
        result[resultIdx++] = q * scale;
      }
      blockOffset += blockSize;
    }
    return result;
  }

  dequantizeF16(offset: number, count: number): Float32Array {
    const result = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const bits = this.view.getUint16(offset + i * 2, true);
      result[i] = this.f16ToF32(bits);
    }
    return result;
  }

  /** BF16 → F32: zero-extend the 16-bit BF16 into the high half of an IEEE-754 f32. */
  dequantizeBF16(offset: number, count: number): Float32Array {
    const result = new Float32Array(count);
    const ab = new ArrayBuffer(4);
    const u32view = new Uint32Array(ab);
    const f32view = new Float32Array(ab);
    for (let i = 0; i < count; i++) {
      const bits = this.view.getUint16(offset + i * 2, true);
      u32view[0] = bits << 16;
      result[i] = f32view[0];
    }
    return result;
  }

  /**
   * Dequantize Q4_K super-blocks (type 12) to f32.
   * Each super-block = 256 elements in 144 bytes:
   *   fp16 d (super-scale)          2 bytes
   *   fp16 dmin (super-min)         2 bytes
   *   8× 6-bit packed (scale, min) 12 bytes
   *   256× 4-bit quants            128 bytes
   * Sub-block layout (per llama.cpp `dequantize_row_q4_K`):
   *   sub-block 2k   = low nibbles of qs[32k..32k+32]  (elements 64k..64k+32)
   *   sub-block 2k+1 = high nibbles of qs[32k..32k+32] (elements 64k+32..64k+64)
   */
  dequantizeQ4_K(offset: number, count: number): Float32Array {
    const BLOCK_BYTES = 144;
    const result = new Float32Array(count);
    let resultIdx = 0;
    let blockOff = offset;
    const scales = new Uint8Array(12);

    const getScaleMin = (j: number): [number, number] => {
      if (j < 4) {
        return [scales[j] & 0x3F, scales[j + 4] & 0x3F];
      }
      const d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
      const m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
      return [d, m];
    };

    while (resultIdx < count) {
      const d = this.f16ToF32(this.view.getUint16(blockOff, true));
      const dmin = this.f16ToF32(this.view.getUint16(blockOff + 2, true));
      for (let k = 0; k < 12; k++) scales[k] = this.view.getUint8(blockOff + 4 + k);
      const qsOff = blockOff + 16;

      // 4 groups of 32 bytes; each group supplies 2 sub-blocks (low and high nibbles).
      for (let group = 0; group < 4 && resultIdx < count; group++) {
        const [sc1, m1] = getScaleMin(group * 2);
        const [sc2, m2] = getScaleMin(group * 2 + 1);
        const d1 = d * sc1, min1 = dmin * m1;
        const d2 = d * sc2, min2 = dmin * m2;
        // Low nibbles → 32 elements.
        for (let l = 0; l < 32 && resultIdx < count; l++) {
          const byte = this.view.getUint8(qsOff + group * 32 + l);
          result[resultIdx++] = d1 * (byte & 0x0F) - min1;
        }
        // High nibbles → 32 elements.
        for (let l = 0; l < 32 && resultIdx < count; l++) {
          const byte = this.view.getUint8(qsOff + group * 32 + l);
          result[resultIdx++] = d2 * ((byte >> 4) & 0x0F) - min2;
        }
      }
      blockOff += BLOCK_BYTES;
    }
    return result;
  }

  /**
   * Decode one Q5_K super-block (256 elements at `byteOffset`) into
   * `out[outOffset..outOffset+256]`. Used by streaming paths that can't
   * afford a full-tensor intermediate buffer (e.g. the per-layer embedding
   * table).
   */
  decodeQ5_KBlock(byteOffset: number, out: Float32Array, outOffset: number): void {
    const scales = new Uint8Array(12);
    const getScaleMin = (j: number): [number, number] => {
      if (j < 4) return [scales[j] & 0x3F, scales[j + 4] & 0x3F];
      const d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
      const m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
      return [d, m];
    };

    const d = this.f16ToF32(this.view.getUint16(byteOffset, true));
    const dmin = this.f16ToF32(this.view.getUint16(byteOffset + 2, true));
    for (let k = 0; k < 12; k++) scales[k] = this.view.getUint8(byteOffset + 4 + k);
    const qhOff = byteOffset + 16;
    const qsOff = byteOffset + 48;

    let u1 = 1, u2 = 2;
    let dst = outOffset;
    for (let group = 0; group < 4; group++) {
      const [sc1, m1] = getScaleMin(group * 2);
      const [sc2, m2] = getScaleMin(group * 2 + 1);
      const d1 = d * sc1, min1 = dmin * m1;
      const d2 = d * sc2, min2 = dmin * m2;
      for (let l = 0; l < 32; l++) {
        const qsByte = this.view.getUint8(qsOff + group * 32 + l);
        const qhByte = this.view.getUint8(qhOff + l);
        const lo = qsByte & 0x0F;
        const hi = (qhByte & u1) ? 16 : 0;
        out[dst++] = d1 * (lo + hi) - min1;
      }
      for (let l = 0; l < 32; l++) {
        const qsByte = this.view.getUint8(qsOff + group * 32 + l);
        const qhByte = this.view.getUint8(qhOff + l);
        const lo = (qsByte >> 4) & 0x0F;
        const hi = (qhByte & u2) ? 16 : 0;
        out[dst++] = d2 * (lo + hi) - min2;
      }
      u1 <<= 2;
      u2 <<= 2;
    }
  }

  /**
   * Dequantize Q5_K super-blocks (type 13) to f32.
   * Same scale/min packing as Q4_K, plus a qh byte-per-32-elements providing
   * the high bit. Super-block = 176 bytes: fp16 d (2) + fp16 dmin (2) +
   * 12 scales + 32 qh + 128 qs.
   */
  dequantizeQ5_K(offset: number, count: number): Float32Array {
    const BLOCK_BYTES = 176;
    const result = new Float32Array(count);
    let resultIdx = 0;
    let blockOff = offset;
    const scales = new Uint8Array(12);

    const getScaleMin = (j: number): [number, number] => {
      if (j < 4) return [scales[j] & 0x3F, scales[j + 4] & 0x3F];
      const d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
      const m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
      return [d, m];
    };

    while (resultIdx < count) {
      const d = this.f16ToF32(this.view.getUint16(blockOff, true));
      const dmin = this.f16ToF32(this.view.getUint16(blockOff + 2, true));
      for (let k = 0; k < 12; k++) scales[k] = this.view.getUint8(blockOff + 4 + k);
      const qhOff = blockOff + 16;
      const qsOff = blockOff + 48;

      let u1 = 1, u2 = 2;
      // 4 groups × 64 elements (low nibbles then high nibbles). Each group advances u1/u2 bit masks.
      for (let group = 0; group < 4 && resultIdx < count; group++) {
        const [sc1, m1] = getScaleMin(group * 2);
        const [sc2, m2] = getScaleMin(group * 2 + 1);
        const d1 = d * sc1, min1 = dmin * m1;
        const d2 = d * sc2, min2 = dmin * m2;
        // Low nibbles of qs[group*32..group*32+32] + u1 bit of qh → 32 elements.
        for (let l = 0; l < 32 && resultIdx < count; l++) {
          const qsByte = this.view.getUint8(qsOff + group * 32 + l);
          const qhByte = this.view.getUint8(qhOff + l);
          const nibble = qsByte & 0x0F;
          const hi = (qhByte & u1) ? 16 : 0;
          result[resultIdx++] = d1 * (nibble + hi) - min1;
        }
        // High nibbles + u2 bit → 32 elements.
        for (let l = 0; l < 32 && resultIdx < count; l++) {
          const qsByte = this.view.getUint8(qsOff + group * 32 + l);
          const qhByte = this.view.getUint8(qhOff + l);
          const nibble = (qsByte >> 4) & 0x0F;
          const hi = (qhByte & u2) ? 16 : 0;
          result[resultIdx++] = d2 * (nibble + hi) - min2;
        }
        u1 <<= 2; u2 <<= 2;
      }
      blockOff += BLOCK_BYTES;
    }
    return result;
  }

  /**
   * Dequantize Q6_K super-blocks (type 14) to f32.
   * Super-block = 210 bytes: 128 ql + 64 qh + 16 int8 scales + fp16 d.
   * 16 sub-blocks of 16 elements each; each sub-block has its own signed
   * 8-bit scale. Each quant is 6 bits (4 low from ql + 2 high from qh),
   * centered at 32 (value = bits − 32).
   */
  dequantizeQ6_K(offset: number, count: number): Float32Array {
    const BLOCK_BYTES = 210;
    const result = new Float32Array(count);
    let resultIdx = 0;
    let blockOff = offset;

    while (resultIdx < count) {
      const qlOff = blockOff;          // 128 bytes
      const qhOff = blockOff + 128;    // 64 bytes
      const scalesOff = blockOff + 192; // 16 signed int8 bytes
      const d = this.f16ToF32(this.view.getUint16(blockOff + 208, true));

      // Process two 128-element halves. Each half uses 64 ql + 32 qh + 8 scales.
      for (let half = 0; half < 2 && resultIdx < count; half++) {
        const qlH = qlOff + half * 64;
        const qhH = qhOff + half * 32;
        const scH = scalesOff + half * 8;
        // Inner loop: l in [0, 32), produces 4 elements at positions l, l+32, l+64, l+96.
        // Accumulate in a staging array so we can write in output order.
        const stage = new Float32Array(128);
        for (let l = 0; l < 32; l++) {
          const is = l >> 4; // 0 or 1
          const ql0 = this.view.getUint8(qlH + l);
          const ql1 = this.view.getUint8(qlH + l + 32);
          const qh = this.view.getUint8(qhH + l);
          const q1 = ((ql0 & 0x0F) | (((qh >> 0) & 0x03) << 4)) - 32;
          const q2 = ((ql1 & 0x0F) | (((qh >> 2) & 0x03) << 4)) - 32;
          const q3 = ((ql0 >> 4)   | (((qh >> 4) & 0x03) << 4)) - 32;
          const q4 = ((ql1 >> 4)   | (((qh >> 6) & 0x03) << 4)) - 32;
          // Scales are signed int8.
          const sc0 = this.view.getInt8(scH + is + 0);
          const sc2 = this.view.getInt8(scH + is + 2);
          const sc4 = this.view.getInt8(scH + is + 4);
          const sc6 = this.view.getInt8(scH + is + 6);
          stage[l]      = d * sc0 * q1;
          stage[l + 32] = d * sc2 * q2;
          stage[l + 64] = d * sc4 * q3;
          stage[l + 96] = d * sc6 * q4;
        }
        const take = Math.min(128, count - resultIdx);
        for (let k = 0; k < take; k++) result[resultIdx++] = stage[k];
      }
      blockOff += BLOCK_BYTES;
    }
    return result;
  }

  getTensorData(tensor: GGUFTensor, dataOffset: number): Float32Array {
    const absOffset = dataOffset + Number(tensor.offset);
    const count = Number(tensor.dims.reduce((a, b) => a * b, 1n));
    if (tensor.type === 0) return new Float32Array(this.buffer, absOffset, count);
    if (tensor.type === 1) return this.dequantizeF16(absOffset, count);
    if (tensor.type === 8) return this.dequantizeQ8_0(absOffset, count);
    if (tensor.type === 12) return this.dequantizeQ4_K(absOffset, count);
    if (tensor.type === 13) return this.dequantizeQ5_K(absOffset, count);
    if (tensor.type === 14) return this.dequantizeQ6_K(absOffset, count);
    if (tensor.type === 30) return this.dequantizeBF16(absOffset, count);
    throw new Error(`Unsupported tensor type: ${tensor.type}`);
  }
}

/** Compute byte size of a tensor in a GGUF file. */
export function tensorByteSize(tensor: GGUFTensor): number {
  const numElements = Number(tensor.dims.reduce((a, b) => a * b, 1n));
  if (tensor.type === 0) return numElements * 4;            // F32
  if (tensor.type === 1) return numElements * 2;            // F16
  if (tensor.type === 8) return (numElements / 32) * 34;    // Q8_0: 34 bytes per 32-block
  if (tensor.type === 12) return (numElements / 256) * 144; // Q4_K: 144 bytes per 256-super-block
  if (tensor.type === 13) return (numElements / 256) * 176; // Q5_K: 176 bytes
  if (tensor.type === 14) return (numElements / 256) * 210; // Q6_K: 210 bytes
  if (tensor.type === 30) return numElements * 2;           // BF16
  throw new Error(`Unknown tensor type: ${tensor.type}`);
}

const _f32ToF16Buf = new ArrayBuffer(4);
const _f32ToF16F32 = new Float32Array(_f32ToF16Buf);
const _f32ToF16U32 = new Uint32Array(_f32ToF16Buf);

/** Scalar f32 → f16 bit pattern. Subnormal / overflow / NaN handled. */
export function f32ToF16(v: number): number {
  _f32ToF16F32[0] = v;
  const bits = _f32ToF16U32[0];
  const sign = (bits >>> 16) & 0x8000;
  const expRaw = (bits >>> 23) & 0xFF;
  const mant = bits & 0x7FFFFF;
  if (expRaw === 0xFF) return sign | 0x7C00 | (mant ? 0x200 : 0);
  if (expRaw === 0) return sign;
  const newExp = expRaw - 127 + 15;
  if (newExp >= 0x1F) return sign | 0x7C00;
  if (newExp <= 0) {
    if (newExp < -10) return sign;
    return sign | ((mant | 0x800000) >>> (1 - newExp + 13));
  }
  return sign | (newExp << 10) | (mant >>> 13);
}

/**
 * Bulk f32 → f16 conversion. Uses native `Float16Array` when available
 * (Chrome 135+), falls back to a manual bit-pattern conversion. Output is
 * a `Uint16Array` of IEEE-754 half-precision bit patterns ready to upload
 * to a WebGPU storage buffer.
 */
export function f32ToF16Array(src: Float32Array): Uint16Array {
  const F16 = (globalThis as unknown as {
    Float16Array?: new (n: number) => ArrayBufferView & {
      buffer: ArrayBufferLike;
      set: (s: Float32Array) => void;
    };
  }).Float16Array;
  if (F16) {
    const f16 = new F16(src.length);
    f16.set(src);
    return new Uint16Array(f16.buffer);
  }
  // Manual fallback — ~80 MB/s in JS on a modern laptop. Slow for huge
  // tensors; acceptable for one-time load.
  const out = new Uint16Array(src.length);
  const tmp = new ArrayBuffer(4);
  const f32v = new Float32Array(tmp);
  const u32v = new Uint32Array(tmp);
  for (let i = 0; i < src.length; i++) {
    f32v[0] = src[i];
    const bits = u32v[0];
    const sign = (bits >>> 16) & 0x8000;
    const expRaw = (bits >>> 23) & 0xFF;
    const mant = bits & 0x7FFFFF;
    if (expRaw === 0xFF) {
      // inf or NaN
      out[i] = sign | 0x7C00 | (mant ? 0x200 : 0);
      continue;
    }
    if (expRaw === 0) {
      // zero or f32-subnormal → f16 zero
      out[i] = sign;
      continue;
    }
    const newExp = expRaw - 127 + 15;
    if (newExp >= 0x1F) {
      // overflow → inf
      out[i] = sign | 0x7C00;
    } else if (newExp <= 0) {
      // f16 subnormal or underflow
      if (newExp < -10) { out[i] = sign; continue; }
      const mantWithLead = mant | 0x800000;
      out[i] = sign | (mantWithLead >>> (1 - newExp + 13));
    } else {
      out[i] = sign | (newExp << 10) | (mant >>> 13);
    }
  }
  return out;
}

/** Pick a single scalar from a GGUF KV entry (returns null for missing / array). */
export function kvNumberOrNull(gguf: GGUFParsed, key: string): number | null {
  const entry = gguf.kv.get(key);
  if (!entry) return null;
  const v = entry.value;
  if (v === null || v === undefined) return null;
  // Array-valued keys come back as { type: 'array', value: GGUFValue[] };
  // callers should use `kvArray` for those.
  if (typeof v === 'object' && 'length' in (v as object)) return null;
  return typeof v === 'bigint' ? Number(v) : Number(v);
}

/** Extract a GGUF array entry's unwrapped values (returns null if missing / not array). */
export function kvArray(gguf: GGUFParsed, key: string): unknown[] | null {
  const entry = gguf.kv.get(key);
  if (!entry || entry.type !== 'array') return null;
  return (entry.value as { value: unknown }[]).map((e) => e.value);
}
