import { readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { WaveFile } = require("wavefile") as typeof import("wavefile");

export interface AudioBuffer {
  samples: Float32Array;
  sampleRate: number;
}

export async function loadWav(path: string): Promise<AudioBuffer> {
  const data = await readFile(path);
  const wav = new WaveFile(data);
  wav.toBitDepth("32f");
  const rawSamples = wav.getSamples(false, Float32Array);
  let mono: Float32Array;
  if (Array.isArray(rawSamples) || rawSamples instanceof Array) {
    const channels = rawSamples as Float32Array[];
    const length = channels[0]?.length ?? 0;
    mono = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      let sum = 0;
      for (const channel of channels) {
        sum += channel[i] ?? 0;
      }
      mono[i] = sum / channels.length;
    }
  } else {
    mono = rawSamples as unknown as Float32Array;
  }
  const fmt = wav.fmt as { sampleRate: number };
  return { samples: mono, sampleRate: fmt.sampleRate };
}

export async function writeWav(
  path: string,
  buffer: AudioBuffer,
): Promise<void> {
  const wav = new WaveFile();
  wav.fromScratch(1, buffer.sampleRate, "32f", buffer.samples);
  await writeFile(path, wav.toBuffer());
}
