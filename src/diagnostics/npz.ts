/**
 * NPZ reader for reference dumps produced by `np.savez`.
 *
 * Supports STORED compression only — `np.savez_compressed` (DEFLATE) throws.
 * Walks the zip local-file-header chain from offset 0, extracts each .npy
 * blob, and materializes the typed-array payload. ZIP64 extra fields are
 * honored so arrays > 4 GB work.
 *
 * One allocation per entry — we copy out of the zip buffer so the caller
 * can release it.
 */

import type { LoadedReference, ReferenceTensors } from './types.js';

export function parseNpz(buf: ArrayBuffer): ReferenceTensors {
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);
  const out: ReferenceTensors = {};
  const td = new TextDecoder('ascii');
  let off = 0;
  while (off + 4 <= buf.byteLength) {
    const sig = view.getUint32(off, true);
    if (sig === 0x02014b50) break; // central directory starts — done with local headers
    if (sig !== 0x04034b50) throw new Error(`npz: unexpected signature 0x${sig.toString(16)} at ${off}`);
    const flag = view.getUint16(off + 6, true);
    const method = view.getUint16(off + 8, true);
    let compSize: number = view.getUint32(off + 18, true);
    let uncompSize: number = view.getUint32(off + 22, true);
    const nameLen = view.getUint16(off + 26, true);
    const extraLen = view.getUint16(off + 28, true);
    if (method !== 0) throw new Error(`npz: entry uses compression method ${method}; only STORED (0) supported`);
    if (flag & 0x0008) throw new Error('npz: data-descriptor flag set; loader requires sizes in local header');
    const name = td.decode(bytes.subarray(off + 30, off + 30 + nameLen));

    // ZIP64: numpy's `np.savez` forces ZIP64 per-entry so arrays > 4 GB are
    // supported. When the 32-bit size fields are 0xFFFFFFFF the real sizes
    // live in a ZIP64 Extended Information extra field (tag 0x0001).
    if (compSize === 0xFFFFFFFF || uncompSize === 0xFFFFFFFF) {
      const extraStart = off + 30 + nameLen;
      let eOff = extraStart;
      const extraEnd = extraStart + extraLen;
      let found = false;
      while (eOff + 4 <= extraEnd) {
        const tag = view.getUint16(eOff, true);
        const tagSize = view.getUint16(eOff + 2, true);
        if (tag === 0x0001) {
          let z = eOff + 4;
          if (uncompSize === 0xFFFFFFFF) {
            uncompSize = Number(view.getBigUint64(z, true));
            z += 8;
          }
          if (compSize === 0xFFFFFFFF) {
            compSize = Number(view.getBigUint64(z, true));
            z += 8;
          }
          found = true;
          break;
        }
        eOff += 4 + tagSize;
      }
      if (!found) throw new Error(`npz: '${name}' has ZIP64 sentinel size but no 0x0001 extra field`);
    }

    const dataStart = off + 30 + nameLen + extraLen;
    const dataEnd = dataStart + compSize;
    const payload = bytes.subarray(dataStart, dataEnd);
    if (name.endsWith('.npy')) {
      const key = name.slice(0, -4);
      out[key] = parseNpyPayload(payload, key);
    }
    off = dataEnd;
    void uncompSize;
  }
  return out;
}

function parseNpyPayload(payload: Uint8Array, keyForErrors: string): Float32Array | Int32Array {
  if (
    payload.length < 10 ||
    payload[0] !== 0x93 ||
    payload[1] !== 0x4e /* N */ ||
    payload[2] !== 0x55 /* U */ ||
    payload[3] !== 0x4d /* M */ ||
    payload[4] !== 0x50 /* P */ ||
    payload[5] !== 0x59 /* Y */
  ) {
    throw new Error(`npz: '${keyForErrors}' missing NUMPY magic`);
  }
  const major = payload[6];
  const dv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  let headerLen: number;
  let dataOff: number;
  if (major === 1) {
    headerLen = dv.getUint16(8, true);
    dataOff = 10 + headerLen;
  } else if (major === 2 || major === 3) {
    headerLen = dv.getUint32(8, true);
    dataOff = 12 + headerLen;
  } else {
    throw new Error(`npz: '${keyForErrors}' unsupported npy version ${major}`);
  }
  const hdrStart = major === 1 ? 10 : 12;
  const headerStr = new TextDecoder('ascii').decode(payload.subarray(hdrStart, hdrStart + headerLen));

  const descrMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
  const fortranMatch = headerStr.match(/'fortran_order'\s*:\s*(True|False)/);
  const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
  if (!descrMatch || !fortranMatch || !shapeMatch) {
    throw new Error(`npz: '${keyForErrors}' malformed header: ${headerStr}`);
  }
  if (fortranMatch[1] !== 'False') {
    throw new Error(`npz: '${keyForErrors}' fortran_order=True not supported`);
  }
  const descr = descrMatch[1];
  const shape = shapeMatch[1]
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10));
  const numel = shape.length === 0 ? 1 : shape.reduce((a, b) => a * b, 1);
  const dataBytes = payload.subarray(dataOff);
  // Copy into a freshly-aligned buffer so the typed-array view isn't tied
  // to zip storage.
  const copy = new ArrayBuffer(dataBytes.byteLength);
  new Uint8Array(copy).set(dataBytes);
  if (descr === '<f4') return new Float32Array(copy, 0, numel);
  if (descr === '<i4') return new Int32Array(copy, 0, numel);
  throw new Error(`npz: '${keyForErrors}' unsupported dtype '${descr}' (need <f4 or <i4)`);
}

/**
 * Fetch an npz file, parse it, and return both the loaded tensors and a
 * human-readable manifest. Caller owns the storage of the returned tensors.
 */
export async function loadReferenceTensors(url: string): Promise<{
  info: LoadedReference;
  tensors: ReferenceTensors;
}> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`reference fetch failed: ${resp.status} ${resp.statusText}`);
  const buf = await resp.arrayBuffer();
  const tensors = parseNpz(buf);
  const entries = Object.entries(tensors).map(([name, arr]) => ({
    name,
    dtype: arr instanceof Float32Array ? '<f4' : '<i4',
    // Shape isn't round-tripped — caller infers from known layout. Length
    // is enough for diff verification.
    shape: [arr.length],
    length: arr.length,
  }));
  const keys = Object.keys(tensors).sort();
  return {
    info: { url, sizeBytes: buf.byteLength, entries, keys },
    tensors,
  };
}
