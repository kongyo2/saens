import type { AudioBuffer } from "./audio.js";
import {
  buildAudioFeatureIndex,
  buildReferenceFeatureSegment,
  normalizeFeatureIndexes,
  scoreFeatureSegment,
  type AudioFeatureIndex,
  type ReferenceFeatureSegment,
} from "./features.js";
import type { MidiNote } from "./midi.js";
import {
  analyzePitches,
  defaultAnalyzeOptions,
  type AnalyzeOptions,
  type PitchWindow,
} from "./pitch.js";

export interface SpliceOptions {
  fadeSeconds: number;
  tailSeconds: number;
  analyze: AnalyzeOptions;
}

export const defaultSpliceOptions: SpliceOptions = {
  fadeSeconds: 0.005,
  tailSeconds: 0.25,
  analyze: defaultAnalyzeOptions,
};

const waveformCandidateCount = 384;
const maxWaveformPitchDiff = 1.8;
const pitchScoreWeight = 0.06;

interface PitchCandidate {
  window: PitchWindow;
  pitchDiff: number;
}

function collectPitchCandidates(
  windows: PitchWindow[],
  targetMidi: number,
): PitchCandidate[] {
  const candidates: PitchCandidate[] = [];
  for (const window of windows) {
    const pitchDiff = Math.abs(window.midi - targetMidi);
    const candidate = { window, pitchDiff };
    let index = 0;
    while (index < candidates.length) {
      const existing = candidates[index];
      if (!existing || existing.pitchDiff > pitchDiff) break;
      index++;
    }
    if (index < waveformCandidateCount) {
      candidates.splice(index, 0, candidate);
      if (candidates.length > waveformCandidateCount) candidates.pop();
    }
  }
  return candidates;
}

function pickBestWindow(
  windows: PitchWindow[],
  targetMidi: number,
  sourceFeatures?: AudioFeatureIndex,
  reference?: ReferenceFeatureSegment,
): PitchWindow | undefined {
  const candidates = collectPitchCandidates(windows, targetMidi);
  const nearest = candidates[0];
  if (!nearest) return undefined;
  if (!sourceFeatures || !reference) return nearest.window;

  const pitchGuard = Math.min(
    maxWaveformPitchDiff,
    Math.max(0.8, nearest.pitchDiff + 1),
  );
  let best = nearest;
  let bestScore = Infinity;

  for (const candidate of candidates) {
    if (candidate.pitchDiff > pitchGuard) {
      continue;
    }
    const featureScore = scoreFeatureSegment(
      sourceFeatures,
      candidate.window.startSample / sourceFeatures.sampleRate,
      reference,
    );
    if (!Number.isFinite(featureScore)) continue;

    const pitchScore =
      Math.pow(Math.max(0, candidate.pitchDiff), 1.25) * pitchScoreWeight;
    const score = featureScore + pitchScore;
    if (
      score < bestScore ||
      (score === bestScore && candidate.pitchDiff < best.pitchDiff)
    ) {
      bestScore = score;
      best = candidate;
    }
  }

  return Number.isFinite(bestScore) ? best.window : nearest.window;
}

function copyWithLoop(
  source: Float32Array,
  sourceStart: number,
  length: number,
): Float32Array {
  const available = Math.max(0, source.length - sourceStart);
  const out = new Float32Array(length);
  if (available === 0) return out;
  if (available >= length) {
    out.set(source.subarray(sourceStart, sourceStart + length));
    return out;
  }
  let written = 0;
  while (written < length) {
    const chunk = Math.min(available, length - written);
    out.set(source.subarray(sourceStart, sourceStart + chunk), written);
    written += chunk;
  }
  return out;
}

function applyFade(buffer: Float32Array, fadeSamples: number): void {
  const fade = Math.min(fadeSamples, Math.floor(buffer.length / 2));
  if (fade <= 0) return;
  for (let i = 0; i < fade; i++) {
    const gain = i / fade;
    const head = buffer[i] ?? 0;
    const tailIndex = buffer.length - 1 - i;
    const tail = buffer[tailIndex] ?? 0;
    buffer[i] = head * gain;
    buffer[tailIndex] = tail * gain;
  }
}

function buildFeaturePair(
  source: AudioBuffer,
  reference: AudioBuffer | undefined,
):
  | {
      sourceFeatures: AudioFeatureIndex;
      referenceFeatures: AudioFeatureIndex;
    }
  | undefined {
  if (!reference) return undefined;
  const sourceFeatures = buildAudioFeatureIndex(
    source.samples,
    source.sampleRate,
  );
  const referenceFeatures = buildAudioFeatureIndex(
    reference.samples,
    reference.sampleRate,
  );
  normalizeFeatureIndexes([sourceFeatures, referenceFeatures]);
  return { sourceFeatures, referenceFeatures };
}

export interface SpliceInput {
  source: AudioBuffer;
  reference?: AudioBuffer;
  notes: MidiNote[];
  midiDuration: number;
  options?: Partial<SpliceOptions>;
}

export function spliceAudioFromMidi(input: SpliceInput): AudioBuffer {
  const options: SpliceOptions = {
    ...defaultSpliceOptions,
    ...input.options,
    analyze: { ...defaultSpliceOptions.analyze, ...input.options?.analyze },
  };
  const { source, notes, midiDuration } = input;
  const sampleRate = source.sampleRate;
  const windows = analyzePitches(source.samples, sampleRate, options.analyze);
  if (windows.length === 0) {
    throw new Error(
      "No pitched windows detected in the input audio. Provide louder or more tonal audio.",
    );
  }

  const featurePair = buildFeaturePair(source, input.reference);
  const outputLength = Math.ceil(
    (midiDuration + options.tailSeconds) * sampleRate,
  );
  const output = new Float32Array(outputLength);
  const fadeSamples = Math.max(1, Math.floor(options.fadeSeconds * sampleRate));

  for (const note of notes) {
    const noteSamples = Math.max(1, Math.floor(note.duration * sampleRate));
    const reference =
      featurePair &&
      buildReferenceFeatureSegment(
        featurePair.referenceFeatures,
        note.time,
        note.duration,
      );
    const best = pickBestWindow(
      windows,
      note.midi,
      featurePair?.sourceFeatures,
      reference || undefined,
    );
    if (!best) continue;

    const segment = copyWithLoop(source.samples, best.startSample, noteSamples);
    const gain = Math.max(0, Math.min(1, note.velocity));
    for (let i = 0; i < segment.length; i++) {
      segment[i] = (segment[i] ?? 0) * gain;
    }
    applyFade(segment, fadeSamples);

    const offset = Math.floor(note.time * sampleRate);
    const end = Math.min(output.length, offset + segment.length);
    for (let i = 0, j = offset; j < end; i++, j++) {
      output[j] = (output[j] ?? 0) + (segment[i] ?? 0);
    }
  }

  let peak = 0;
  for (let i = 0; i < output.length; i++) {
    const abs = Math.abs(output[i] ?? 0);
    if (abs > peak) peak = abs;
  }
  if (peak > 1) {
    const scale = 0.98 / peak;
    for (let i = 0; i < output.length; i++) {
      output[i] = (output[i] ?? 0) * scale;
    }
  }

  return { samples: output, sampleRate };
}
