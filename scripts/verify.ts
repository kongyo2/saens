import { loadWav } from "../src/audio.js";

async function main(): Promise<void> {
  const { samples, sampleRate } = await loadWav("tmp/output.wav");
  let peak = 0;
  let rmsSum = 0;
  let nonZero = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i] ?? 0;
    const abs = Math.abs(v);
    if (abs > peak) peak = abs;
    if (abs > 1e-5) nonZero++;
    rmsSum += v * v;
  }
  const rms = Math.sqrt(rmsSum / samples.length);
  process.stdout.write(
    `length=${samples.length} (${(samples.length / sampleRate).toFixed(2)}s) peak=${peak.toFixed(4)} rms=${rms.toFixed(4)} nonZeroRatio=${(nonZero / samples.length).toFixed(3)}\n`,
  );
}

main().catch((error: unknown) => {
  process.stderr.write(`${String(error)}\n`);
  process.exit(1);
});
