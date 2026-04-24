import { YIN } from "pitchfinder";
import { frequencyToMidi } from "./midi.js";

export interface PitchWindow {
  startSample: number;
  midi: number;
}

export interface AnalyzeOptions {
  windowSeconds: number;
  hopSeconds: number;
  minMidi: number;
  maxMidi: number;
}

export const defaultAnalyzeOptions: AnalyzeOptions = {
  windowSeconds: 0.05,
  hopSeconds: 0.025,
  minMidi: 21,
  maxMidi: 108,
};

export function analyzePitches(
  samples: Float32Array,
  sampleRate: number,
  options: AnalyzeOptions = defaultAnalyzeOptions,
): PitchWindow[] {
  const windowSize = Math.max(
    512,
    Math.floor(options.windowSeconds * sampleRate),
  );
  const hopSize = Math.max(1, Math.floor(options.hopSeconds * sampleRate));
  const detector = YIN({ sampleRate, threshold: 0.1 });
  const windows: PitchWindow[] = [];

  for (let start = 0; start + windowSize <= samples.length; start += hopSize) {
    const slice = samples.subarray(start, start + windowSize);
    let rms = 0;
    for (let i = 0; i < slice.length; i++) {
      const value = slice[i] ?? 0;
      rms += value * value;
    }
    rms = Math.sqrt(rms / slice.length);
    if (rms < 1e-4) continue;
    const frequency = detector(slice);
    if (frequency == null || !Number.isFinite(frequency) || frequency <= 0) {
      continue;
    }
    const midi = frequencyToMidi(frequency);
    if (midi < options.minMidi || midi > options.maxMidi) continue;
    windows.push({ startSample: start, midi });
  }

  return windows;
}
