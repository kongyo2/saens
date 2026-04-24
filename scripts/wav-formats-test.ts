// Exercise loadWav against a battery of WAV variants to make sure it can
// accept any real-world file. Run with: npx tsx scripts/wav-formats-test.ts
import { mkdir, writeFile, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { loadWav } from "../src/audio.js";

const require = createRequire(import.meta.url);
const { WaveFile } = require("wavefile") as typeof import("wavefile");
const ffmpegPath = require("ffmpeg-static") as string;

const sampleRate = 22050;
const duration = 0.2;
const freq = 440;

function makeSine(
  channels: number,
  bits: number,
): Int16Array | Int32Array | Float32Array | Uint8Array {
  const n = Math.floor(sampleRate * duration);
  const peak = bits === 8 ? 127 : Math.pow(2, bits - 1) - 1;
  if (bits === 8) {
    const out = new Uint8Array(n * channels);
    for (let i = 0; i < n; i++) {
      const s = Math.sin((2 * Math.PI * freq * i) / sampleRate);
      const v = 128 + Math.round(s * 100);
      for (let c = 0; c < channels; c++) out[i * channels + c] = v;
    }
    return out;
  }
  if (bits === 32) {
    const out = new Int32Array(n * channels);
    for (let i = 0; i < n; i++) {
      const s = Math.sin((2 * Math.PI * freq * i) / sampleRate);
      const v = Math.round(s * peak * 0.7);
      for (let c = 0; c < channels; c++) out[i * channels + c] = v;
    }
    return out;
  }
  const out = new Int16Array(n * channels);
  for (let i = 0; i < n; i++) {
    const s = Math.sin((2 * Math.PI * freq * i) / sampleRate);
    const v = Math.round(s * peak * 0.7);
    for (let c = 0; c < channels; c++) out[i * channels + c] = v;
  }
  return out;
}

function makeSineFloat(channels: number): Float32Array {
  const n = Math.floor(sampleRate * duration);
  const out = new Float32Array(n * channels);
  for (let i = 0; i < n; i++) {
    const s = 0.7 * Math.sin((2 * Math.PI * freq * i) / sampleRate);
    for (let c = 0; c < channels; c++) out[i * channels + c] = s;
  }
  return out;
}

async function writeWith(
  path: string,
  numChannels: number,
  bitDepth: string,
  samples: ArrayLike<number>,
): Promise<void> {
  const wav = new WaveFile();
  wav.fromScratch(numChannels, sampleRate, bitDepth, samples as number[]);
  await writeFile(path, Buffer.from(wav.toBuffer()));
}

async function writeMuLaw(path: string, numChannels: number): Promise<void> {
  const wav = new WaveFile();
  wav.fromScratch(
    numChannels,
    sampleRate,
    "16",
    makeSine(numChannels, 16) as unknown as number[],
  );
  wav.toMuLaw();
  await writeFile(path, Buffer.from(wav.toBuffer()));
}

async function writeALaw(path: string, numChannels: number): Promise<void> {
  const wav = new WaveFile();
  wav.fromScratch(
    numChannels,
    sampleRate,
    "16",
    makeSine(numChannels, 16) as unknown as number[],
  );
  wav.toALaw();
  await writeFile(path, Buffer.from(wav.toBuffer()));
}

async function writeExtensibleFloat(
  path: string,
  numChannels: number,
): Promise<void> {
  // Build a WAVE_FORMAT_EXTENSIBLE file with IEEE float subformat by hand.
  const samples = makeSineFloat(numChannels);
  const dataBytes = samples.length * 4;
  const fmtSize = 40;
  const totalBody = 4 + (8 + fmtSize) + (8 + dataBytes);
  const buf = Buffer.alloc(8 + totalBody);
  let o = 0;
  buf.write("RIFF", o);
  o += 4;
  buf.writeUInt32LE(totalBody, o);
  o += 4;
  buf.write("WAVE", o);
  o += 4;
  buf.write("fmt ", o);
  o += 4;
  buf.writeUInt32LE(fmtSize, o);
  o += 4;
  buf.writeUInt16LE(0xfffe, o);
  o += 2; // audioFormat = extensible
  buf.writeUInt16LE(numChannels, o);
  o += 2;
  buf.writeUInt32LE(sampleRate, o);
  o += 4;
  buf.writeUInt32LE(sampleRate * numChannels * 4, o);
  o += 4;
  buf.writeUInt16LE(numChannels * 4, o);
  o += 2; // blockAlign
  buf.writeUInt16LE(32, o);
  o += 2; // bitsPerSample
  buf.writeUInt16LE(22, o);
  o += 2; // cbSize
  buf.writeUInt16LE(32, o);
  o += 2; // validBitsPerSample
  buf.writeUInt32LE(0, o);
  o += 4; // channel mask
  // SubFormat GUID: first 2 bytes = 0x0003 (IEEE float), then fixed tail
  buf.writeUInt16LE(0x0003, o);
  o += 2;
  buf.writeUInt16LE(0x0000, o);
  o += 2;
  buf.writeUInt32LE(0x00100000, o);
  o += 4;
  // 8 bytes: 80 00 00 AA 00 38 9B 71
  Buffer.from([0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71]).copy(buf, o);
  o += 8;
  buf.write("data", o);
  o += 4;
  buf.writeUInt32LE(dataBytes, o);
  o += 4;
  for (let i = 0; i < samples.length; i++) {
    buf.writeFloatLE(samples[i] ?? 0, o);
    o += 4;
  }
  await writeFile(path, buf);
}

function runFfmpeg(args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(ffmpegPath, args, {
      stdio: ["ignore", "ignore", "pipe"],
    });
    let stderr = "";
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg failed (${code}): ${stderr.trim()}`));
    });
  });
}

async function writeViaFfmpeg(
  path: string,
  numChannels: number,
  codec: string,
): Promise<void> {
  // Build a PCM seed via ffmpeg's lavfi sine generator, then re-encode it into
  // the requested codec. This lets us exercise formats wavefile can't author.
  await runFfmpeg([
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-f",
    "lavfi",
    "-i",
    `sine=frequency=${freq}:sample_rate=${sampleRate}:duration=${duration}`,
    "-ac",
    String(numChannels),
    "-c:a",
    codec,
    path,
  ]);
}

async function writeLegacy14ByteFmt(
  path: string,
  audioFormat: number,
  bytesPerSample: number,
  bytes: Uint8Array,
): Promise<void> {
  // Hand-roll a WAVE file with a 14-byte WAVEFORMAT header (no bitsPerSample).
  const fmtSize = 14;
  const dataBytes = bytes.length;
  const totalBody = 4 + (8 + fmtSize) + (8 + dataBytes);
  const buf = Buffer.alloc(8 + totalBody);
  let o = 0;
  buf.write("RIFF", o);
  o += 4;
  buf.writeUInt32LE(totalBody, o);
  o += 4;
  buf.write("WAVE", o);
  o += 4;
  buf.write("fmt ", o);
  o += 4;
  buf.writeUInt32LE(fmtSize, o);
  o += 4;
  buf.writeUInt16LE(audioFormat, o);
  o += 2;
  buf.writeUInt16LE(1, o);
  o += 2; // numChannels
  buf.writeUInt32LE(sampleRate, o);
  o += 4;
  buf.writeUInt32LE(sampleRate * bytesPerSample, o);
  o += 4; // byteRate
  buf.writeUInt16LE(bytesPerSample, o);
  o += 2; // blockAlign
  buf.write("data", o);
  o += 4;
  buf.writeUInt32LE(dataBytes, o);
  o += 4;
  Buffer.from(bytes).copy(buf, o);
  await writeFile(path, buf);
}

interface Case {
  label: string;
  build: (path: string) => Promise<void>;
}

const cases: Case[] = [
  {
    label: "8-bit PCM mono",
    build: (p) => writeWith(p, 1, "8", makeSine(1, 8) as unknown as number[]),
  },
  {
    label: "16-bit PCM mono",
    build: (p) => writeWith(p, 1, "16", makeSine(1, 16) as unknown as number[]),
  },
  {
    label: "16-bit PCM stereo",
    build: (p) => writeWith(p, 2, "16", makeSine(2, 16) as unknown as number[]),
  },
  {
    label: "24-bit PCM mono",
    build: async (p) => {
      const n = Math.floor(sampleRate * duration);
      const samples: number[] = [];
      for (let i = 0; i < n; i++) {
        const s = Math.sin((2 * Math.PI * freq * i) / sampleRate);
        samples.push(Math.round(s * (Math.pow(2, 23) - 1) * 0.7));
      }
      await writeWith(p, 1, "24", samples);
    },
  },
  {
    label: "32-bit PCM mono",
    build: async (p) => {
      const n = Math.floor(sampleRate * duration);
      const samples: number[] = [];
      for (let i = 0; i < n; i++) {
        const s = Math.sin((2 * Math.PI * freq * i) / sampleRate);
        samples.push(Math.round(s * (Math.pow(2, 31) - 1) * 0.7));
      }
      await writeWith(p, 1, "32", samples);
    },
  },
  {
    label: "32-bit float mono",
    build: async (p) => {
      const arr = Array.from(makeSineFloat(1));
      await writeWith(p, 1, "32f", arr);
    },
  },
  {
    label: "32-bit float stereo",
    build: async (p) => {
      const arr = Array.from(makeSineFloat(2));
      await writeWith(p, 2, "32f", arr);
    },
  },
  {
    label: "mu-Law mono",
    build: (p) => writeMuLaw(p, 1),
  },
  {
    label: "A-Law mono",
    build: (p) => writeALaw(p, 1),
  },
  {
    label: "WAVE_FORMAT_EXTENSIBLE float stereo",
    build: (p) => writeExtensibleFloat(p, 2),
  },
  {
    label: "legacy 14-byte mu-Law (WAVEFORMAT, no bitsPerSample)",
    build: async (p) => {
      // Encode real mu-Law bytes via wavefile, extract the data, and drop them
      // into a hand-rolled WAVEFORMAT-sized header.
      const wav = new WaveFile();
      wav.fromScratch(
        1,
        sampleRate,
        "16",
        makeSine(1, 16) as unknown as number[],
      );
      wav.toMuLaw();
      const samples = wav.getSamples(false, Uint8Array) as unknown as
        | Uint8Array
        | Uint8Array[];
      const bytes = Array.isArray(samples)
        ? (samples[0] ?? new Uint8Array())
        : samples;
      await writeLegacy14ByteFmt(p, 0x0007, 1, bytes);
    },
  },
  {
    label: "MS ADPCM mono (via ffmpeg)",
    build: (p) => writeViaFfmpeg(p, 1, "adpcm_ms"),
  },
  {
    label: "MS ADPCM stereo (via ffmpeg)",
    build: (p) => writeViaFfmpeg(p, 2, "adpcm_ms"),
  },
  {
    label: "IMA ADPCM mono (via ffmpeg)",
    build: (p) => writeViaFfmpeg(p, 1, "adpcm_ima_wav"),
  },
  {
    label: "Yamaha ADPCM mono (via ffmpeg)",
    build: (p) => writeViaFfmpeg(p, 1, "adpcm_yamaha"),
  },
  {
    label: "Yamaha ADPCM stereo (via ffmpeg)",
    build: (p) => writeViaFfmpeg(p, 2, "adpcm_yamaha"),
  },
];

async function main(): Promise<void> {
  await mkdir("tmp/fmt", { recursive: true });
  let pass = 0;
  let fail = 0;
  for (const c of cases) {
    const path = `tmp/fmt/${c.label.replace(/[^a-z0-9]+/gi, "_")}.wav`;
    try {
      await c.build(path);
      const { samples, sampleRate: sr } = await loadWav(path);
      let peak = 0;
      for (let i = 0; i < samples.length; i++) {
        const v = Math.abs(samples[i] ?? 0);
        if (v > peak) peak = v;
      }
      const ok =
        sr === sampleRate && samples.length > 0 && peak > 0.1 && peak <= 1.01;
      if (ok) {
        process.stdout.write(
          `PASS ${c.label}  (sr=${sr}, n=${samples.length}, peak=${peak.toFixed(3)})\n`,
        );
        pass++;
      } else {
        process.stdout.write(
          `FAIL ${c.label}  (sr=${sr}, n=${samples.length}, peak=${peak.toFixed(3)})\n`,
        );
        fail++;
      }
    } catch (e) {
      process.stdout.write(`FAIL ${c.label}: ${String(e)}\n`);
      fail++;
    }
  }
  await rm("tmp/fmt", { recursive: true, force: true });
  process.stdout.write(`\n${pass} passed, ${fail} failed\n`);
  if (fail > 0) process.exit(1);
}

main().catch((e) => {
  process.stderr.write(String(e) + "\n");
  process.exit(1);
});
