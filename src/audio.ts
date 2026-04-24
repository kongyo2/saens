import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const { WaveFile } = require("wavefile") as typeof import("wavefile");

export interface AudioBuffer {
  samples: Float32Array;
  sampleRate: number;
}

const FMT_PCM = 0x0001;
const FMT_FLOAT = 0x0003;
const FMT_ALAW = 0x0006;
const FMT_MULAW = 0x0007;
const FMT_EXTENSIBLE = 0xfffe;

interface FmtChunk {
  audioFormat: number;
  numChannels: number;
  sampleRate: number;
  bitsPerSample: number;
  blockAlign: number;
  subFormat?: number;
}

interface ParsedWav {
  fmt: FmtChunk;
  dataOffset: number;
  dataLength: number;
  littleEndian: boolean;
}

function readChunkId(view: DataView, offset: number): string {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );
}

function parseWav(input: Uint8Array): ParsedWav {
  if (input.length < 12) {
    throw new Error("File is too small to be a WAV");
  }
  const view = new DataView(input.buffer, input.byteOffset, input.byteLength);
  const riff = readChunkId(view, 0);
  if (riff !== "RIFF" && riff !== "RIFX" && riff !== "RF64") {
    throw new Error(`Unsupported WAV container: ${riff}`);
  }
  const littleEndian = riff !== "RIFX";
  const format = readChunkId(view, 8);
  if (format !== "WAVE") {
    throw new Error(`Unsupported RIFF format: ${format}`);
  }

  let fmt: FmtChunk | undefined;
  let dataOffset = -1;
  let dataLength = 0;
  let ds64DataSize: bigint | undefined;
  let offset = 12;

  while (offset + 8 <= input.length) {
    const id = readChunkId(view, offset);
    let size = view.getUint32(offset + 4, littleEndian);
    const body = offset + 8;

    if (id === "ds64") {
      // RF64: real sizes live here
      const dataSize = view.getBigUint64(body + 8, littleEndian);
      ds64DataSize = dataSize;
    } else if (id === "fmt ") {
      // Legacy WAVEFORMAT is 14 bytes (no bitsPerSample field). PCMWAVEFORMAT
      // is 16, WAVEFORMATEX is 18, WAVEFORMATEXTENSIBLE is 40. Accept all.
      if (size < 14) throw new Error("fmt chunk is too short");
      const audioFormat = view.getUint16(body, littleEndian);
      const numChannels = view.getUint16(body + 2, littleEndian);
      const sampleRate = view.getUint32(body + 4, littleEndian);
      const blockAlign = view.getUint16(body + 12, littleEndian);
      // bitsPerSample only exists from PCMWAVEFORMAT (size >= 16) onward. For
      // legacy 14-byte headers assume 8-bit (typical for A-law / mu-law).
      const bitsPerSample =
        size >= 16 ? view.getUint16(body + 14, littleEndian) : 8;
      let subFormat: number | undefined;
      if (audioFormat === FMT_EXTENSIBLE && size >= 40) {
        subFormat = view.getUint16(body + 24, littleEndian);
      }
      fmt = {
        audioFormat,
        numChannels,
        sampleRate,
        bitsPerSample,
        blockAlign,
        ...(subFormat !== undefined ? { subFormat } : {}),
      };
    } else if (id === "data") {
      dataOffset = body;
      // RF64 encodes 0xFFFFFFFF in the header and the real size in ds64.
      dataLength =
        size === 0xffffffff && ds64DataSize !== undefined
          ? Number(ds64DataSize)
          : size;
      // Cap to remaining file bytes to tolerate truncated/mis-sized headers.
      const maxRemaining = input.length - body;
      if (dataLength > maxRemaining) dataLength = maxRemaining;
      break;
    }

    if (size === 0xffffffff && ds64DataSize !== undefined && id === "data") {
      size = Number(ds64DataSize);
    }
    // Chunks are word-aligned (pad byte for odd sizes).
    offset = body + size + (size & 1);
  }

  if (!fmt) throw new Error('Missing "fmt " chunk');
  if (dataOffset < 0) throw new Error('Missing "data" chunk');
  if (fmt.numChannels < 1) throw new Error("Invalid channel count");
  if (fmt.sampleRate < 1) throw new Error("Invalid sample rate");

  return { fmt, dataOffset, dataLength, littleEndian };
}

function muLawDecode(byte: number): number {
  const u = ~byte & 0xff;
  const sign = u & 0x80;
  const exponent = (u >> 4) & 0x07;
  const mantissa = u & 0x0f;
  let magnitude = ((mantissa << 3) + 0x84) << exponent;
  magnitude -= 0x84;
  const value = sign ? -magnitude : magnitude;
  return value / 32768;
}

function aLawDecode(byte: number): number {
  const a = byte ^ 0x55;
  const sign = a & 0x80;
  const exponent = (a >> 4) & 0x07;
  const mantissa = a & 0x0f;
  let magnitude: number;
  if (exponent === 0) {
    magnitude = (mantissa << 4) + 8;
  } else {
    magnitude = ((mantissa << 4) + 0x108) << (exponent - 1);
  }
  const value = sign ? magnitude : -magnitude;
  return value / 32768;
}

function readInt(
  view: DataView,
  offset: number,
  bytes: number,
  littleEndian: boolean,
): number {
  // Little-endian signed integer of arbitrary byte width up to 4.
  // The high byte provides sign extension.
  let value = 0;
  if (littleEndian) {
    for (let i = 0; i < bytes - 1; i++) {
      value |= view.getUint8(offset + i) << (8 * i);
    }
    value |= view.getInt8(offset + bytes - 1) << (8 * (bytes - 1));
  } else {
    for (let i = 1; i < bytes; i++) {
      value |= view.getUint8(offset + i) << (8 * (bytes - 1 - i));
    }
    value |= view.getInt8(offset) << (8 * (bytes - 1));
  }
  return value;
}

function decodePcmInt(
  view: DataView,
  dataOffset: number,
  frameCount: number,
  numChannels: number,
  bytesPerSample: number,
  blockAlign: number,
  littleEndian: boolean,
): Float32Array {
  const mono = new Float32Array(frameCount);
  // Normalize by the max positive value for the given bit depth.
  const bits = bytesPerSample * 8;
  const scale = 1 / Math.pow(2, bits - 1);
  for (let f = 0; f < frameCount; f++) {
    const frameStart = dataOffset + f * blockAlign;
    let sum = 0;
    for (let c = 0; c < numChannels; c++) {
      const sampleOffset = frameStart + c * bytesPerSample;
      let sample: number;
      if (bytesPerSample === 1) {
        // 8-bit WAV PCM is unsigned [0, 255] centered at 128.
        sample = (view.getUint8(sampleOffset) - 128) / 128;
      } else {
        sample =
          readInt(view, sampleOffset, bytesPerSample, littleEndian) * scale;
      }
      sum += sample;
    }
    mono[f] = sum / numChannels;
  }
  return mono;
}

function decodeFloat(
  view: DataView,
  dataOffset: number,
  frameCount: number,
  numChannels: number,
  bytesPerSample: number,
  blockAlign: number,
  littleEndian: boolean,
): Float32Array {
  const mono = new Float32Array(frameCount);
  for (let f = 0; f < frameCount; f++) {
    const frameStart = dataOffset + f * blockAlign;
    let sum = 0;
    for (let c = 0; c < numChannels; c++) {
      const sampleOffset = frameStart + c * bytesPerSample;
      const sample =
        bytesPerSample === 8
          ? view.getFloat64(sampleOffset, littleEndian)
          : view.getFloat32(sampleOffset, littleEndian);
      sum += sample;
    }
    mono[f] = sum / numChannels;
  }
  return mono;
}

function decodeLog(
  view: DataView,
  dataOffset: number,
  frameCount: number,
  numChannels: number,
  blockAlign: number,
  decoder: (byte: number) => number,
): Float32Array {
  const mono = new Float32Array(frameCount);
  for (let f = 0; f < frameCount; f++) {
    const frameStart = dataOffset + f * blockAlign;
    let sum = 0;
    for (let c = 0; c < numChannels; c++) {
      sum += decoder(view.getUint8(frameStart + c));
    }
    mono[f] = sum / numChannels;
  }
  return mono;
}

export async function loadWav(path: string): Promise<AudioBuffer> {
  const buf = await readFile(path);
  const input = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  return decodeAudio(input, path);
}

async function decodeAudio(
  input: Uint8Array,
  sourcePath: string,
): Promise<AudioBuffer> {
  // Try the manual parser first. It succeeds on the vast majority of real-world
  // WAV files (PCM, float, A-law, mu-law, WAVE_FORMAT_EXTENSIBLE).
  try {
    const parsed = parseWav(input);
    const decoded = tryDecodeParsed(input, parsed);
    if (decoded) return decoded;
  } catch {
    // Not a WAV we can understand — fall through.
  }

  // Next: wavefile covers IMA ADPCM and some other formats with a permissive
  // parser.
  try {
    return loadWavViaWaveFile(input);
  } catch {
    // Fall through to ffmpeg.
  }

  // Last resort: shell out to ffmpeg (bundled via ffmpeg-static). This handles
  // MS ADPCM, GSM 6.10, MP3-in-WAV, truncated headers, and non-WAV audio.
  return loadViaFfmpeg(sourcePath);
}

function tryDecodeParsed(
  input: Uint8Array,
  parsed: ParsedWav,
): AudioBuffer | undefined {
  const { fmt, dataOffset, dataLength, littleEndian } = parsed;
  const view = new DataView(input.buffer, input.byteOffset, input.byteLength);

  const effectiveFormat =
    fmt.audioFormat === FMT_EXTENSIBLE && fmt.subFormat !== undefined
      ? fmt.subFormat
      : fmt.audioFormat;

  // Derive block alignment: trust the header, but fall back to a computed
  // value when it is obviously wrong (some encoders write 0).
  const bytesPerSample = Math.max(1, Math.ceil(fmt.bitsPerSample / 8));
  const blockAlign =
    fmt.blockAlign && fmt.blockAlign >= bytesPerSample * fmt.numChannels
      ? fmt.blockAlign
      : bytesPerSample * fmt.numChannels;
  const frameCount = Math.floor(dataLength / blockAlign);

  switch (effectiveFormat) {
    case FMT_PCM: {
      const samples = decodePcmInt(
        view,
        dataOffset,
        frameCount,
        fmt.numChannels,
        bytesPerSample,
        blockAlign,
        littleEndian,
      );
      return { samples, sampleRate: fmt.sampleRate };
    }
    case FMT_FLOAT: {
      if (fmt.bitsPerSample !== 32 && fmt.bitsPerSample !== 64) {
        return undefined;
      }
      const samples = decodeFloat(
        view,
        dataOffset,
        frameCount,
        fmt.numChannels,
        bytesPerSample,
        blockAlign,
        littleEndian,
      );
      return { samples, sampleRate: fmt.sampleRate };
    }
    case FMT_MULAW: {
      const samples = decodeLog(
        view,
        dataOffset,
        frameCount,
        fmt.numChannels,
        blockAlign,
        muLawDecode,
      );
      return { samples, sampleRate: fmt.sampleRate };
    }
    case FMT_ALAW: {
      const samples = decodeLog(
        view,
        dataOffset,
        frameCount,
        fmt.numChannels,
        blockAlign,
        aLawDecode,
      );
      return { samples, sampleRate: fmt.sampleRate };
    }
    default:
      return undefined;
  }
}

function loadWavViaWaveFile(buf: Uint8Array): AudioBuffer {
  const wav = new WaveFile(buf);
  wav.toBitDepth("32f");
  const rawSamples = wav.getSamples(false, Float32Array);
  let mono: Float32Array;
  if (Array.isArray(rawSamples)) {
    const channels = rawSamples as unknown as Float32Array[];
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

function resolveFfmpegPath(): string | undefined {
  try {
    const mod = require("ffmpeg-static") as unknown;
    if (typeof mod === "string") return mod;
    if (mod && typeof mod === "object" && "default" in mod) {
      const def = (mod as { default: unknown }).default;
      if (typeof def === "string") return def;
    }
  } catch {
    // ffmpeg-static not installed (shouldn't happen — it's a dep).
  }
  // Final fallback: rely on ffmpeg from PATH.
  return "ffmpeg";
}

async function loadViaFfmpeg(sourcePath: string): Promise<AudioBuffer> {
  const ffmpegPath = resolveFfmpegPath();
  if (!ffmpegPath) {
    throw new Error("Unable to decode audio: no ffmpeg binary available");
  }
  const scratchDir = await mkdtemp(join(tmpdir(), "saens-wav-"));
  const outPath = join(scratchDir, "decoded.wav");
  try {
    await runFfmpeg(ffmpegPath, [
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-i",
      sourcePath,
      "-vn",
      "-ac",
      "1",
      "-c:a",
      "pcm_f32le",
      "-f",
      "wav",
      outPath,
    ]);
    const raw = await readFile(outPath);
    const input = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
    const parsed = parseWav(input);
    const decoded = tryDecodeParsed(input, parsed);
    if (!decoded) {
      throw new Error("ffmpeg output was not decodable");
    }
    return decoded;
  } finally {
    await rm(scratchDir, { recursive: true, force: true });
  }
}

function runFfmpeg(binary: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(binary, args, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(
          new Error(
            `ffmpeg exited with code ${code}: ${stderr.trim() || "no stderr"}`,
          ),
        );
      }
    });
  });
}

export async function writeWav(
  path: string,
  buffer: AudioBuffer,
): Promise<void> {
  const wav = new WaveFile();
  wav.fromScratch(1, buffer.sampleRate, "32f", buffer.samples);
  await writeFile(path, wav.toBuffer());
}
