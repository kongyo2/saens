export { loadMidiNotes, midiToFrequency, frequencyToMidi } from "./midi.js";
export type { MidiNote } from "./midi.js";
export { loadWav, writeWav } from "./audio.js";
export type { AudioBuffer } from "./audio.js";
export {
  analyzePitches,
  defaultAnalyzeOptions,
  type AnalyzeOptions,
  type PitchWindow,
} from "./pitch.js";
export {
  spliceAudioFromMidi,
  defaultSpliceOptions,
  type SpliceInput,
  type SpliceOptions,
} from "./splice.js";
export { bundledMidiPath } from "./defaults.js";

import { bundledMidiPath } from "./defaults.js";
import { loadMidiNotes } from "./midi.js";
import { loadWav, writeWav } from "./audio.js";
import { spliceAudioFromMidi, type SpliceOptions } from "./splice.js";

export interface RenderOptions {
  midiPath?: string;
  audioPath: string;
  outputPath: string;
  splice?: Partial<SpliceOptions>;
}

export async function renderMidiFromAudio(
  options: RenderOptions,
): Promise<void> {
  const midiPath = options.midiPath ?? bundledMidiPath;
  const [{ notes, duration }, source] = await Promise.all([
    loadMidiNotes(midiPath),
    loadWav(options.audioPath),
  ]);
  const result = spliceAudioFromMidi({
    source,
    notes,
    midiDuration: duration,
    ...(options.splice ? { options: options.splice } : {}),
  });
  await writeWav(options.outputPath, result);
}
