import type { AudioBuffer } from "./audio.js";
import {
  buildAudioFeatureIndex,
  normalizeFeatureIndexes,
  frameFeatureDistance,
} from "./features.js";
import type { MidiNote } from "./midi.js";
import { defaultAnalyzeOptions, type AnalyzeOptions } from "./pitch.js";

export interface SpliceOptions {
  /** Hann grain length (seconds) overlap-added at each analysis frame. */
  grainSeconds: number;
  /** Silence appended after the reference/MIDI content, in seconds. */
  tailSeconds: number;
  /** Pitch analysis bounds. */
  analyze: AnalyzeOptions;
  /**
   * Octave-folded semitone tolerance when constraining a grain to the MIDI
   * pitch. Grains outside this are only used when nothing closer carries the
   * required phoneme.
   */
  pitchTolerance: number;
  /**
   * Bonus (in phonetic-distance units) for continuing the previous source run
   * one frame forward, which keeps articulation contiguous and natural.
   */
  continuityBonus: number;
}

export const defaultSpliceOptions: SpliceOptions = {
  grainSeconds: 0.03,
  tailSeconds: 0.25,
  analyze: defaultAnalyzeOptions,
  pitchTolerance: 2,
  continuityBonus: 0.15,
};

// ---- lightweight FFT-autocorrelation pitch tracker (aligned to feature frames) ----

const pitchWorkRate = 16000;

function nextPow2(value: number): number {
  let power = 1;
  while (power < value) power <<= 1;
  return power;
}

function fftInPlace(real: Float64Array, imag: Float64Array, inverse: boolean): void {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = real[i] ?? 0;
      const ti = imag[i] ?? 0;
      real[i] = real[j] ?? 0;
      imag[i] = imag[j] ?? 0;
      real[j] = tr;
      imag[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(angle);
    const wi = Math.sin(angle);
    for (let start = 0; start < n; start += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < half; k++) {
        const a = start + k;
        const b = a + half;
        const xr = (real[b] ?? 0) * cr - (imag[b] ?? 0) * ci;
        const xi = (real[b] ?? 0) * ci + (imag[b] ?? 0) * cr;
        real[b] = (real[a] ?? 0) - xr;
        imag[b] = (imag[a] ?? 0) - xi;
        real[a] = (real[a] ?? 0) + xr;
        imag[a] = (imag[a] ?? 0) + xi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) {
      real[i] = (real[i] ?? 0) / n;
      imag[i] = (imag[i] ?? 0) / n;
    }
  }
}

function resampleLinear(samples: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return samples;
  const ratio = to / from;
  const length = Math.max(1, Math.floor(samples.length * ratio));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const src = i / ratio;
    const i0 = Math.floor(src);
    const frac = src - i0;
    const a = samples[i0] ?? 0;
    const b = samples[i0 + 1] ?? a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

/**
 * MIDI pitch (or NaN) for each feature frame of the source, via FFT-based
 * autocorrelation on a downsampled copy. `frameCount`/`hopSeconds` match the
 * source feature index so pitch and phonetics are frame-aligned.
 */
function buildPitchTrack(
  samples: Float32Array,
  sampleRate: number,
  frameCount: number,
  hopSeconds: number,
  analyze: AnalyzeOptions,
): Float32Array {
  const work = resampleLinear(samples, sampleRate, pitchWorkRate);
  const windowSize = Math.min(2048, nextPow2(Math.max(256, Math.floor(0.06 * pitchWorkRate))));
  const fftSize = nextPow2(windowSize);
  const hop = hopSeconds * pitchWorkRate;
  const minHz = 440 * Math.pow(2, (analyze.minMidi - 69) / 12);
  const maxHz = 440 * Math.pow(2, (analyze.maxMidi - 69) / 12);
  const minLag = Math.max(1, Math.floor(pitchWorkRate / maxHz));
  const maxLag = Math.min(windowSize - 1, Math.floor(pitchWorkRate / minHz));
  const track = new Float32Array(frameCount);
  const real = new Float64Array(fftSize);
  const imag = new Float64Array(fftSize);
  const hann = new Float64Array(windowSize);
  for (let i = 0; i < windowSize; i++) {
    hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (windowSize - 1));
  }
  for (let f = 0; f < frameCount; f++) {
    const start = Math.floor(f * hop);
    let mean = 0;
    for (let i = 0; i < windowSize; i++) mean += work[start + i] ?? 0;
    mean /= windowSize;
    real.fill(0);
    imag.fill(0);
    let energy = 0;
    for (let i = 0; i < windowSize; i++) {
      const centered = (work[start + i] ?? 0) - mean;
      real[i] = centered * (hann[i] ?? 0);
      energy += centered * centered;
    }
    if (energy < 1e-6) {
      track[f] = NaN;
      continue;
    }
    fftInPlace(real, imag, false);
    for (let i = 0; i < fftSize; i++) {
      real[i] = (real[i] ?? 0) * (real[i] ?? 0) + (imag[i] ?? 0) * (imag[i] ?? 0);
      imag[i] = 0;
    }
    fftInPlace(real, imag, true);
    const zero = real[0] ?? 1e-9;
    let bestLag = -1;
    let bestValue = 0.2;
    for (let lag = minLag; lag <= maxLag; lag++) {
      const value = (real[lag] ?? 0) / zero;
      if (value > bestValue) {
        bestValue = value;
        bestLag = lag;
      }
    }
    track[f] = bestLag > 0 ? 69 + 12 * Math.log2(pitchWorkRate / bestLag / 440) : NaN;
  }
  return track;
}

function foldedDistance(a: number, b: number): number {
  let d = a - b;
  while (d > 6) d -= 12;
  while (d < -6) d += 12;
  return Math.abs(d);
}

/** Top (highest) active MIDI note at each source/reference frame time, or NaN. */
function buildMidiPitchTargets(
  notes: MidiNote[],
  frameCount: number,
  hopSeconds: number,
): Float32Array {
  const targets = new Float32Array(frameCount).fill(NaN);
  for (const note of notes) {
    const startFrame = Math.max(0, Math.floor(note.time / hopSeconds));
    const endFrame = Math.min(frameCount, Math.ceil((note.time + note.duration) / hopSeconds));
    for (let f = startFrame; f < endFrame; f++) {
      const current = targets[f];
      if (Number.isNaN(current) || note.midi > (current ?? -Infinity)) {
        targets[f] = note.midi;
      }
    }
  }
  return targets;
}

export interface SpliceInput {
  source: AudioBuffer;
  /** Reference recording carrying the target pronunciation, aligned to the MIDI. */
  reference?: AudioBuffer;
  notes: MidiNote[];
  midiDuration: number;
  options?: Partial<SpliceOptions>;
}

/**
 * Vocal cross-synthesis. For every frame of the reference (which carries the
 * words/pronunciation, aligned to the MIDI), pick the source grain whose timbre
 * best matches that reference frame while sitting near the MIDI's pitch, and
 * overlap-add it. The result speaks the reference's phonemes in the source's
 * voice at the MIDI's pitches. Grain selection is contiguous where possible so
 * articulation stays natural.
 */
export function spliceAudioFromMidi(input: SpliceInput): AudioBuffer {
  const options: SpliceOptions = {
    ...defaultSpliceOptions,
    ...input.options,
    analyze: { ...defaultSpliceOptions.analyze, ...input.options?.analyze },
  };
  const { source, reference, notes, midiDuration } = input;
  const sampleRate = source.sampleRate;

  const sourceIndex = buildAudioFeatureIndex(source.samples, sampleRate);
  if (!reference) {
    throw new Error(
      "A reference recording is required: it supplies the pronunciation the output must follow.",
    );
  }
  const referenceIndex = buildAudioFeatureIndex(reference.samples, reference.sampleRate);
  normalizeFeatureIndexes([sourceIndex, referenceIndex]);

  const hopSeconds = sourceIndex.hopSeconds;
  const sourcePitch = buildPitchTrack(
    source.samples,
    sampleRate,
    sourceIndex.frameCount,
    hopSeconds,
    options.analyze,
  );
  const pitchTargets = buildMidiPitchTargets(notes, referenceIndex.frameCount, hopSeconds);

  const outputLength = Math.ceil((midiDuration + options.tailSeconds) * sampleRate);
  const output = new Float32Array(outputLength);
  const weight = new Float32Array(outputLength);

  const grainSamples = Math.max(2, Math.floor(options.grainSeconds * sampleRate));
  const grainWindow = new Float64Array(grainSamples);
  for (let i = 0; i < grainSamples; i++) {
    grainWindow[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (grainSamples - 1));
  }
  const sourceHopSamples = sourceIndex.hopSize;

  const referenceFrameMax = Math.min(
    referenceIndex.frameCount,
    Math.floor(midiDuration / hopSeconds),
  );

  const frameCount = sourceIndex.frameCount;
  let previousFrame = -1;
  for (let rf = 0; rf < referenceFrameMax; rf++) {
    const targetPitch = pitchTargets[rf] ?? NaN;
    let chosen = -1;
    let bestScore = Infinity;

    if (!Number.isNaN(targetPitch)) {
      // Among source frames near the target pitch, take the closest phoneme.
      // Cheap pitch test first, so the phonetic distance is only computed for
      // the handful of candidates that carry the right note.
      for (let sf = 0; sf < frameCount; sf++) {
        const sourceMidi = sourcePitch[sf] ?? NaN;
        if (Number.isNaN(sourceMidi)) continue;
        if (foldedDistance(sourceMidi, targetPitch) > options.pitchTolerance) continue;
        const distance = frameFeatureDistance(sourceIndex, sf, referenceIndex, rf);
        const score = distance - (sf === previousFrame + 1 ? options.continuityBonus : 0);
        if (score < bestScore) {
          bestScore = score;
          chosen = sf;
        }
      }
    }

    // No note active, or the source has nothing at the required pitch: fall back
    // to the best phonetic match regardless of pitch so the word still lands.
    if (chosen < 0) {
      for (let sf = 0; sf < frameCount; sf++) {
        const distance = frameFeatureDistance(sourceIndex, sf, referenceIndex, rf);
        const score = distance - (sf === previousFrame + 1 ? options.continuityBonus : 0);
        if (score < bestScore) {
          bestScore = score;
          chosen = sf;
        }
      }
    }
    if (chosen < 0) continue;
    previousFrame = chosen;

    const sourceStart = chosen * sourceHopSamples;
    const destStart = Math.floor(rf * hopSeconds * sampleRate);
    for (let i = 0; i < grainSamples; i++) {
      const dst = destStart + i;
      if (dst >= output.length) break;
      const value = source.samples[sourceStart + i] ?? 0;
      const w = grainWindow[i] ?? 0;
      output[dst] = (output[dst] ?? 0) + value * w;
      weight[dst] = (weight[dst] ?? 0) + w;
    }
  }

  for (let i = 0; i < output.length; i++) {
    const w = weight[i] ?? 0;
    if (w > 1e-6) output[i] = (output[i] ?? 0) / w;
  }

  let peak = 0;
  for (let i = 0; i < output.length; i++) {
    const abs = Math.abs(output[i] ?? 0);
    if (abs > peak) peak = abs;
  }
  if (peak > 1) {
    const scale = 0.98 / peak;
    for (let i = 0; i < output.length; i++) output[i] = (output[i] ?? 0) * scale;
  }

  return { samples: output, sampleRate };
}
