import { renderMidiFromAudio, bundledMidiPath } from "./index.js";

interface CliArgs {
  audio: string;
  out: string;
}

function parseArgs(argv: string[]): CliArgs {
  const args: Partial<CliArgs> = {};
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!key || !value) continue;
    if (key === "--audio") {
      args.audio = value;
      i++;
    } else if (key === "--out") {
      args.out = value;
      i++;
    }
  }
  if (!args.audio || !args.out) {
    throw new Error("Usage: tsx src/cli.ts --audio <path> --out <path>");
  }
  return args as CliArgs;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const args = parseArgs(argv);
  await renderMidiFromAudio({
    audioPath: args.audio,
    outputPath: args.out,
  });
  process.stdout.write(`Wrote ${args.out} (MIDI: ${bundledMidiPath})\n`);
}

main().catch((error: unknown) => {
  process.stderr.write(`${String(error)}\n`);
  process.exit(1);
});
