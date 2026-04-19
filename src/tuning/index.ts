/**
 * Tuning: device profiles and per-kernel knobs that shape engine init.
 *
 * A profile selects matmul workgroup shape and decode-loop depth. The
 * engine reads these at init time — profiles are immutable for the life
 * of an engine instance. To try a different profile, dispose and
 * re-create the engine with a different `tuning` option.
 */

export type {
  TuningProfile,
  TuningProfileOverrides,
  KernelName,
  FeatureIntent,
} from './profile.js';
export { rowsPerWorkgroupFor, overrideProfile } from './profile.js';
export { PROFILES, nvidiaBlackwell, appleMSeries, generic } from './devices.js';
export type { SelectedProfile, TuningOverride } from './detect.js';
export { selectDeviceProfile } from './detect.js';
