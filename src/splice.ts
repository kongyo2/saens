import type { AudioBuffer } from "./audio.js";
import {
  buildAudioFeatureIndex,
  normalizeFeatureIndexes,
  frameFeatureDistance,
} from "./features.js";
import type { MidiNote } from "./midi.js";
import { defaultAnalyzeOptions, type AnalyzeOptions } from "./pitch.js";

export interface SpliceOptions {
  /**
   * Equal-power crossfade (seconds) applied only at cut points between two
   * source slices. Kept short so each slice is heard as a raw, verbatim piece
   * of the input — the hand-cut character — rather than a resynthesis.
   */
  crossfadeSeconds: number;
  /** Silence appended after the reference/MIDI content, in seconds. */
  tailSeconds: number;
  /** Pitch analysis bounds. */
  analyze: AnalyzeOptions;
  /**
   * Octave-folded semitone tolerance used when *jumping* to a fresh source
   * position. Grains outside this are only used when nothing closer carries the
   * required phoneme.
   */
  pitchTolerance: number;
  /**
   * Hysteresis margin. Keep playing the current source run forward unless
   * jumping elsewhere beats "continue" by more than this (phonetic-distance
   * units). Larger → longer contiguous cut-and-paste chunks of the source's own
   * words (the meme character); 0 → re-pick the best phoneme every frame.
   */
  continuityMargin: number;
  /**
   * How strongly a held run is pulled back toward the melody. A running chunk
   * accrues this penalty per octave-folded semitone of pitch error, so it stays
   * on the source's own words for a while but still breaks to chase the tune.
   */
  pitchDriftPenalty: number;
  /**
   * Consonant-joining strength (0..1). At reference frames where the spectrum
   * changes sharply — consonant onsets / syllable attacks — continuity is
   * suppressed by this much so a fresh matching consonant grain is stitched in,
   * the way 人力 (hand-made) edits splice a crisp consonant onto a vowel. Steady
   * vowels keep full continuity, so this sharpens articulation without flutter.
   * Off by default: it fragments the raw slices that carry the meme character.
   */
  onsetSensitivity: number;
  /**
   * Minimum length (seconds) of a verbatim source slice before a cut is allowed.
   * Long slices keep each piece recognisable as a raw chunk of the input (high
   * "it's really them" character); the hand-made ideals sit around 0.15-0.25 s.
   */
  minSliceSeconds: number;
}

export const defaultSpliceOptions: SpliceOptions = {
  crossfadeSeconds: 0.006,
  tailSeconds: 0.25,
  analyze: defaultAnalyzeOptions,
  pitchTolerance: 2,
  continuityMargin: 0.6,
  pitchDriftPenalty: 0.05,
  onsetSensitivity: 0,
  minSliceSeconds: 0.15,
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
  const sourceHopSamples = sourceIndex.hopSize;

  const referenceFrameMax = Math.min(
    referenceIndex.frameCount,
    Math.floor(midiDuration / hopSeconds),
  );

  // Onset strength per reference frame = how much its spectrum differs from the
  // previous frame. Peaks mark consonant attacks / syllable starts, where we
  // want to break continuity and stitch a fresh matching consonant grain.
  const onsetStrength = new Float32Array(referenceFrameMax);
  let onsetMean = 0;
  for (let rf = 1; rf < referenceFrameMax; rf++) {
    const flux = frameFeatureDistance(referenceIndex, rf, referenceIndex, rf - 1);
    onsetStrength[rf] = flux;
    onsetMean += flux;
  }
  onsetMean /= Math.max(1, referenceFrameMax - 1);
  let onsetVariance = 0;
  for (let rf = 1; rf < referenceFrameMax; rf++) {
    const d = (onsetStrength[rf] ?? 0) - onsetMean;
    onsetVariance += d * d;
  }
  const onsetScale = onsetMean + 1.5 * Math.sqrt(onsetVariance / Math.max(1, referenceFrameMax - 1));

  const frameCount = sourceIndex.frameCount;
  // Source frame chosen for each output frame. `cursor` is the frame currently
  // playing; it advances one frame at a time so the source's own words play
  // through contiguously, jumping only when a fresh position matches markedly
  // better. Actual audio is synthesised verbatim from these choices afterwards.
  const chosenFrames = new Int32Array(referenceFrameMax).fill(-1);
  const minSliceFrames = Math.max(1, Math.round(options.minSliceSeconds / hopSeconds));
  let cursor = -1;
  let runLength = 0;
  let contiguousFrames = 0;
  let renderedFrames = 0;
  for (let rf = 0; rf < referenceFrameMax; rf++) {
    const targetPitch = pitchTargets[rf] ?? NaN;

    // Best fresh position: closest phoneme among grains near the target pitch.
    let bestJump = -1;
    let bestJumpScore = Infinity;
    if (!Number.isNaN(targetPitch)) {
      for (let sf = 0; sf < frameCount; sf++) {
        const sourceMidi = sourcePitch[sf] ?? NaN;
        if (Number.isNaN(sourceMidi)) continue;
        if (foldedDistance(sourceMidi, targetPitch) > options.pitchTolerance) continue;
        const distance = frameFeatureDistance(sourceIndex, sf, referenceIndex, rf);
        if (distance < bestJumpScore) {
          bestJumpScore = distance;
          bestJump = sf;
        }
      }
    }
    if (bestJump < 0) {
      for (let sf = 0; sf < frameCount; sf++) {
        const distance = frameFeatureDistance(sourceIndex, sf, referenceIndex, rf);
        if (distance < bestJumpScore) {
          bestJumpScore = distance;
          bestJump = sf;
        }
      }
    }

    // Continuity is relaxed at consonant onsets so a fresh consonant grain is
    // stitched in (the 人力 consonant-join), and full through steady vowels.
    const onsetNorm = onsetScale > 0 ? Math.min(1, (onsetStrength[rf] ?? 0) / onsetScale) : 0;
    const effectiveMargin =
      options.continuityMargin * Math.max(0, 1 - options.onsetSensitivity * onsetNorm);

    // Continue the current raw slice forward, cutting to a fresh position only
    // when the slice has reached its minimum length AND jumping matches clearly
    // better. A gentle pitch-drift pull keeps a held slice near the tune.
    const prevCursor = cursor;
    let chosen = bestJump;
    if (prevCursor >= 0 && prevCursor + 1 < frameCount) {
      const contFrame = prevCursor + 1;
      let contScore = frameFeatureDistance(sourceIndex, contFrame, referenceIndex, rf);
      const contMidi = sourcePitch[contFrame] ?? NaN;
      if (!Number.isNaN(targetPitch) && !Number.isNaN(contMidi)) {
        contScore += foldedDistance(contMidi, targetPitch) * options.pitchDriftPenalty;
      }
      const tooShortToCut = runLength < minSliceFrames;
      if (tooShortToCut || contScore <= bestJumpScore + effectiveMargin) {
        chosen = contFrame;
      }
    }
    if (chosen < 0) continue;
    if (prevCursor >= 0 && chosen === prevCursor + 1) {
      runLength++;
      contiguousFrames++;
    } else {
      runLength = 0;
    }
    cursor = chosen;
    renderedFrames++;
    chosenFrames[rf] = chosen;
  }

  // Synthesise VERBATIM: group the choices into contiguous runs and copy raw
  // source samples for each run, so the output is literally hand-cut slices of
  // the input. Only the cut points are crossfaded (equal power), so the middle
  // of every slice is bit-identical to the source — this is what makes it read
  // as "the source itself, forced" rather than a resynthesis.
  const framesPerHop = hopSeconds * sampleRate;
  const xfade = Math.max(1, Math.floor(options.crossfadeSeconds * sampleRate));
  let rf = 0;
  while (rf < referenceFrameMax) {
    if ((chosenFrames[rf] ?? -1) < 0) {
      rf++;
      continue;
    }
    const runStart = rf;
    while (
      rf + 1 < referenceFrameMax &&
      (chosenFrames[rf + 1] ?? -1) === (chosenFrames[rf] ?? -1) + 1
    ) {
      rf++;
    }
    const runEnd = rf; // inclusive
    const sourceStart = (chosenFrames[runStart] ?? 0) * sourceHopSamples;
    const destStart = Math.floor(runStart * framesPerHop);
    const runSamples = Math.max(1, Math.round((runEnd - runStart + 1) * framesPerHop));
    // Copy verbatim; fade in over the first xfade (crossfading the previous
    // run's tail) and fade out over an extra xfade past the end.
    for (let i = 0; i < runSamples + xfade; i++) {
      const dst = destStart + i;
      if (dst < 0 || dst >= output.length) break;
      let gain = 1;
      if (i < xfade) gain = Math.sin((Math.PI / 2) * (i / xfade));
      if (i >= runSamples) gain = Math.sin((Math.PI / 2) * ((runSamples + xfade - i) / xfade));
      output[dst] = (output[dst] ?? 0) + (source.samples[sourceStart + i] ?? 0) * gain;
    }
    rf++;
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

  if (process.env.SPLICE_DEBUG === "1") {
    const contiguity = renderedFrames > 0 ? contiguousFrames / renderedFrames : 0;
    let slices = 0, backJumps = 0, jumpMag = 0, prevStart = -1;
    for (let i = 0; i < referenceFrameMax; ) {
      if ((chosenFrames[i] ?? -1) < 0) { i++; continue; }
      const start = chosenFrames[i] ?? 0;
      slices++;
      if (prevStart >= 0) { const d = start - prevStart; if (d < 0) backJumps++; jumpMag += Math.abs(d); }
      prevStart = start;
      while (i + 1 < referenceFrameMax && (chosenFrames[i + 1] ?? -1) === (chosenFrames[i] ?? -1) + 1) i++;
      i++;
    }
    process.stderr.write(
      `[splice] contiguity=${contiguity.toFixed(3)} frames=${renderedFrames} slices=${slices} avgSlice=${(renderedFrames / Math.max(1, slices) * hopSeconds * 1000).toFixed(0)}ms backJumpFrac=${(backJumps / Math.max(1, slices)).toFixed(2)} avgJump=${(jumpMag / Math.max(1, slices) * hopSeconds).toFixed(2)}s\n`,
    );
  }

  return { samples: output, sampleRate };
}
