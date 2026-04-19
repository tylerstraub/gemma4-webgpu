// Read-only GGUF audit: parse header, dump metadata + full tensor inventory.
// Mirrors src/gguf.ts `GGUFParser` semantics but has no dependency on Vite,
// WebGPU, or the engine — runnable from the CLI on a downloaded header
// prefix. See scripts/README.md for usage.

import { readFileSync } from 'node:fs';

const GGML_TYPE_NAMES = {
  0:  'F32',
  1:  'F16',
  2:  'Q4_0',
  3:  'Q4_1',
  6:  'Q5_0',
  7:  'Q5_1',
  8:  'Q8_0',
  9:  'Q8_1',
  10: 'Q2_K',
  11: 'Q3_K',
  12: 'Q4_K',
  13: 'Q5_K',
  14: 'Q6_K',
  15: 'Q8_K',
  16: 'IQ2_XXS',
  17: 'IQ2_XS',
  18: 'IQ3_XXS',
  19: 'IQ1_S',
  20: 'IQ4_NL',
  21: 'IQ3_S',
  22: 'IQ2_S',
  23: 'IQ4_XS',
  24: 'I8',
  25: 'I16',
  26: 'I32',
  27: 'I64',
  28: 'F64',
  29: 'IQ1_M',
  30: 'BF16',
};

// Bytes-per-tensor for the types this audit understands.
function tensorByteSize(type, dims) {
  const n = Number(dims.reduce((a, b) => a * b, 1n));
  switch (type) {
    case 0:  return n * 4;           // F32
    case 1:  return n * 2;           // F16
    case 8:  return (n / 32) * 34;   // Q8_0
    case 12: return (n / 256) * 144; // Q4_K
    case 13: return (n / 256) * 176; // Q5_K
    case 14: return (n / 256) * 210; // Q6_K
    case 30: return n * 2;           // BF16
    case 11: return (n / 256) * 110; // Q3_K (approx; audit-only, not consumed at runtime)
    case 10: return (n / 256) * 84;  // Q2_K (approx)
    default: return -1;              // unknown
  }
}

const headerPath = process.env.GGUF_HEADER;
if (!headerPath) {
  console.error('Usage: GGUF_HEADER=/path/to/gguf_header.bin node scripts/audit_gguf.mjs');
  console.error('See scripts/README.md for the Range-request recipe that produces the header file.');
  process.exit(2);
}

const buf = readFileSync(headerPath);
const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
const dec = new TextDecoder('utf-8');
let off = 0;

const readU32 = () => { const v = view.getUint32(off, true); off += 4; return v; };
const readU64 = () => { const v = view.getBigUint64(off, true); off += 8; return v; };
const readI32 = () => { const v = view.getInt32(off, true); off += 4; return v; };
const readI64 = () => { const v = view.getBigInt64(off, true); off += 8; return v; };
const readU8  = () => { const v = view.getUint8(off); off += 1; return v; };
const readI8  = () => { const v = view.getInt8(off); off += 1; return v; };
const readU16 = () => { const v = view.getUint16(off, true); off += 2; return v; };
const readI16 = () => { const v = view.getInt16(off, true); off += 2; return v; };
const readF32 = () => { const v = view.getFloat32(off, true); off += 4; return v; };
const readF64 = () => { const v = view.getFloat64(off, true); off += 8; return v; };
const readStr = () => {
  const len = Number(readU64());
  const bytes = new Uint8Array(buf.buffer, buf.byteOffset + off, len);
  off += len;
  return dec.decode(bytes);
};

function readValue(type) {
  switch (type) {
    case 0:  return readU8();
    case 1:  return readI8();
    case 2:  return readU16();
    case 3:  return readI16();
    case 4:  return readU32();
    case 5:  return readI32();
    case 6:  return readF32();
    case 7:  return readU8() !== 0;
    case 8:  return readStr();
    case 9:  {
      const elemType = readU32();
      const count = Number(readU64());
      const arr = new Array(count);
      for (let i = 0; i < count; i++) arr[i] = readValue(elemType);
      return arr;
    }
    case 10: return readU64();
    case 11: return readI64();
    case 12: return readF64();
    default: throw new Error(`Unknown value type: ${type}`);
  }
}

const magic = readU32();
if (magic !== 0x46554747) throw new Error(`Invalid GGUF magic: 0x${magic.toString(16)}`);
const version = readU32();
const tensorCount = Number(readU64());
const kvCount = Number(readU64());

console.log('=== GGUF HEADER ===');
console.log('magic:          GGUF');
console.log('version:       ', version);
console.log('tensor_count:  ', tensorCount);
console.log('kv_count:      ', kvCount);

const kv = new Map();
for (let i = 0; i < kvCount; i++) {
  const key = readStr();
  const valueType = readU32();
  const value = readValue(valueType);
  kv.set(key, { type: valueType, value });
}

console.log('\n=== METADATA KEYS (sorted) ===');
const keys = Array.from(kv.keys()).sort();
for (const k of keys) console.log(' ', k);

console.log('\n=== METADATA VALUES (provenance + model-identity fields) ===');
const provenanceKeys = [
  'general.architecture', 'general.type', 'general.name', 'general.finetune',
  'general.basename', 'general.quantization_version', 'general.file_type',
  'general.size_label', 'general.license', 'general.license.link',
  'general.source.url', 'general.source.repo', 'general.source.huggingface.repository',
  'general.author', 'general.organization', 'general.version',
  'general.tags', 'general.languages',
];
for (const k of provenanceKeys) {
  if (kv.has(k)) {
    const entry = kv.get(k);
    let v = entry.value;
    if (Array.isArray(v)) v = `[${v.slice(0, 8).join(', ')}${v.length > 8 ? `, ... (${v.length} items)` : ''}]`;
    if (typeof v === 'bigint') v = v.toString() + 'n';
    console.log(`  ${k} = ${v}  (type=${entry.type})`);
  }
}

console.log('\n=== ALL general.* METADATA (raw) ===');
for (const k of keys.filter((k) => k.startsWith('general.'))) {
  const entry = kv.get(k);
  let v = entry.value;
  if (Array.isArray(v)) v = `[${v.slice(0, 16).map((x) => typeof x === 'bigint' ? x.toString() + 'n' : JSON.stringify(x)).join(', ')}${v.length > 16 ? ` ... +${v.length - 16}` : ''}]`;
  if (typeof v === 'bigint') v = v.toString() + 'n';
  console.log(`  ${k} = ${JSON.stringify(v)}`);
}

// Parse tensor descriptors.
const tensors = [];
for (let i = 0; i < tensorCount; i++) {
  const name = readStr();
  const nDims = readU32();
  const dims = [];
  for (let d = 0; d < nDims; d++) dims.push(readU64());
  const type = readU32();
  const offset = readU64();
  tensors.push({ name, dims, type, offset });
}
const alignment = 32;
const dataOffset = Math.ceil(off / alignment) * alignment;

console.log('\n=== TENSOR DESCRIPTOR SECTION ===');
console.log('data_offset (absolute):', dataOffset);
console.log('tensors:               ', tensors.length);

// Per-quant breakdown.
const byType = new Map(); // type → { count, bytes, names[] }
for (const t of tensors) {
  const bytes = tensorByteSize(t.type, t.dims);
  if (!byType.has(t.type)) byType.set(t.type, { count: 0, bytes: 0, names: [] });
  const rec = byType.get(t.type);
  rec.count++;
  rec.bytes += bytes;
  if (rec.names.length < 8) rec.names.push(t.name);
}

console.log('\n=== PER-QUANT BREAKDOWN ===');
let totalBytes = 0;
const sorted = [...byType.entries()].sort((a, b) => b[1].bytes - a[1].bytes);
for (const [type, rec] of sorted) {
  const name = GGML_TYPE_NAMES[type] ?? `type_${type}`;
  console.log(`  ${name.padEnd(8)} (type=${type})  count=${String(rec.count).padStart(4)}  bytes=${rec.bytes.toLocaleString().padStart(14)}  (${(rec.bytes / 1e9).toFixed(3)} GB)  sample=${rec.names.slice(0, 3).join(', ')}`);
  totalBytes += rec.bytes;
}
console.log(`  TOTAL                    count=${String(tensors.length).padStart(4)}  bytes=${totalBytes.toLocaleString().padStart(14)}  (${(totalBytes / 1e9).toFixed(3)} GB)`);

// Multimodal tensor search.
console.log('\n=== MULTIMODAL TENSOR SEARCH ===');
const mmPatterns = [/^vision_tower/, /^audio_tower/, /^multi_modal_projector/, /^mm_/, /^vision/, /^audio/, /^mmproj/, /img_/, /_img/];
const mmHits = tensors.filter((t) => mmPatterns.some((rx) => rx.test(t.name)));
if (mmHits.length === 0) {
  console.log('  (none) — no tensors matching vision_tower.*, audio_tower.*, multi_modal_projector.*, mm_*, vision*, audio*');
} else {
  let mmBytes = 0;
  for (const t of mmHits) {
    const b = tensorByteSize(t.type, t.dims);
    mmBytes += b;
    console.log(`  ${t.name}  dims=[${t.dims.map(String).join(',')}]  type=${GGML_TYPE_NAMES[t.type]}  bytes=${b.toLocaleString()}`);
  }
  console.log(`  MM total bytes: ${mmBytes.toLocaleString()} (${(mmBytes / 1e9).toFixed(3)} GB)`);
}

// Full inventory grouped by type.
console.log('\n=== FULL TENSOR INVENTORY (grouped by quant type) ===');
for (const [type, rec] of sorted) {
  const name = GGML_TYPE_NAMES[type] ?? `type_${type}`;
  console.log(`\n--- ${name} (type=${type}, ${rec.count} tensors) ---`);
  const group = tensors.filter((t) => t.type === type);
  // Collapse per-layer repeats: blk.{N}.xxx appearing for all layers gets shown once.
  const families = new Map();
  for (const t of group) {
    const m = /^blk\.(\d+)\.(.+)$/.exec(t.name);
    if (m) {
      const fam = `blk.N.${m[2]}`;
      if (!families.has(fam)) families.set(fam, { layers: new Set(), dims: t.dims, type: t.type });
      families.get(fam).layers.add(parseInt(m[1]));
      // Track dim variation across layers (e.g. ffn_* with double-wide).
      if (families.get(fam).dims.some((d, i) => d !== t.dims[i])) {
        families.get(fam).dimsVary = true;
        families.get(fam).dimsAll = families.get(fam).dimsAll || [families.get(fam).dims];
        families.get(fam).dimsAll.push(t.dims);
      }
    } else {
      families.set(t.name, { single: true, dims: t.dims });
    }
  }
  for (const [famName, info] of families) {
    if (info.single) {
      console.log(`  ${famName}  dims=[${info.dims.map(String).join(',')}]`);
    } else {
      const layers = [...info.layers].sort((a, b) => a - b);
      const dimsStr = info.dimsVary
        ? `dims VARY across layers: ${info.dimsAll.map((d) => '[' + d.map(String).join(',') + ']').filter((v, i, a) => a.indexOf(v) === i).join(' / ')}`
        : `dims=[${info.dims.map(String).join(',')}]`;
      console.log(`  ${famName}  layers=[${layers[0]}..${layers[layers.length - 1]}] (${layers.length} layers)  ${dimsStr}`);
    }
  }
}

// Names of all global (non-blk.N) tensors — useful for loader-consumption checks.
console.log('\n=== GLOBAL (non-blk.N) TENSOR NAMES ===');
const globals = tensors.filter((t) => !/^blk\.\d+\./.test(t.name));
for (const t of globals) {
  console.log(`  ${t.name}  dims=[${t.dims.map(String).join(',')}]  type=${GGML_TYPE_NAMES[t.type]}  bytes=${tensorByteSize(t.type, t.dims).toLocaleString()}`);
}

// Per-layer uniqueness check — what families exist per layer?
console.log('\n=== PER-LAYER TENSOR FAMILIES (unique suffixes) ===');
const perLayerSuffixes = new Set();
for (const t of tensors) {
  const m = /^blk\.(\d+)\.(.+)$/.exec(t.name);
  if (m) perLayerSuffixes.add(m[2]);
}
for (const suf of [...perLayerSuffixes].sort()) console.log(`  blk.N.${suf}`);

// File-size consistency summary (compare against HF Content-Length externally).
console.log('\n=== FILE SIZE CONSISTENCY ===');
const expectedFileSize = dataOffset + totalBytes;
console.log(`  dataOffset + sum(tensor bytes) = ${expectedFileSize.toLocaleString()}`);
console.log('  Compare this value against the HF Content-Length for the GGUF URL');
console.log('  (for unsloth/gemma-4-E2B-it-GGUF Q4_K_M: 3,106,735,776 bytes).');
