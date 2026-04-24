import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { writeWav } from "../src/audio.js";
import { midiToFrequency, loadMidiNotes } from "../src/midi.js";
import { renderMidiFromAudio } from "../src/index.js";

const sampleRate = 44100;
const noteDuration = 0.4;
const startMidi = 48;
const endMidi = 84;

async function ensureDir(path: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
}

async function main(): Promise<void> {
  const midiPath = "元データ.mid";
  const audioPath = "tmp/source.wav";
  const outputPath = "tmp/output.wav";
  await ensureDir(audioPath);

  const total = Math.floor(
    (endMidi - startMidi + 1) * noteDuration * sampleRate,
  );
  const samples = new Float32Array(total);
  let cursor = 0;
  for (let midi = startMidi; midi <= endMidi; midi++) {
    const freq = midiToFrequency(midi);
    const len = Math.floor(noteDuration * sampleRate);
    for (let i = 0; i < len; i++) {
      const t = i / sampleRate;
      const envelope =
        Math.min(1, i / (sampleRate * 0.01)) *
        Math.min(1, (len - i) / (sampleRate * 0.01));
      samples[cursor + i] = 0.3 * envelope * Math.sin(2 * Math.PI * freq * t);
    }
    cursor += len;
  }
  await writeWav(audioPath, { samples, sampleRate });

  const { notes, duration } = await loadMidiNotes(midiPath);
  process.stdout.write(
    `Loaded ${notes.length} notes across ${duration.toFixed(2)}s\n`,
  );

  await renderMidiFromAudio({ midiPath, audioPath, outputPath });
  process.stdout.write(`Rendered to ${outputPath}\n`);
}

main().catch((error: unknown) => {
  process.stderr.write(`${String(error)}\n`);
  process.exit(1);
});
