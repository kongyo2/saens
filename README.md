# midi-audio-splicer

Experimental TypeScript library that takes a reference MIDI file and an input
audio (WAV) file, finds the nearest-pitch moment in the audio for each MIDI
note, chops that moment into a note-long slice, and splices the slices back
together on the MIDI's timeline.

The repository's `元データ.mid` is used as the canonical reference MIDI.

## Install

```bash
npm install
```

## Usage

### Programmatic

```ts
import { renderMidiFromAudio } from "midi-audio-splicer";

await renderMidiFromAudio({
  midiPath: "元データ.mid",
  audioPath: "voice.wav",
  outputPath: "out.wav",
});
```

### CLI

```bash
npm run render -- --midi 元データ.mid --audio voice.wav --out out.wav
```

## How it works

1. `@tonejs/midi` parses the MIDI into `{ midi, time, duration, velocity }`
   notes, sorted by start time.
2. `wavefile` loads the WAV, down-mixes to mono, and converts to 32-bit float.
3. `pitchfinder`'s YIN detector walks the audio with a sliding window
   (50 ms window, 25 ms hop by default) and records the fundamental pitch of
   each sufficiently loud window.
4. For each MIDI note, the window whose detected MIDI pitch is closest to the
   note is selected. A `note.duration * sampleRate` long slice is copied
   (looping back into the same window if the duration exceeds what's left).
5. A short linear fade-in/out is applied to every slice to prevent clicks,
   the slice is scaled by `note.velocity`, and it is summed into the output at
   `note.time * sampleRate`. The final mix is peak-normalized if it clips.

## Scripts

- `npm run build` – compile TypeScript to `dist/`
- `npm run dev` / `npm run render` – run the CLI via `tsx`
- `npm run typecheck`
- `npm run lint` / `npm run lint:strict` – oxlint
- `npm run format` / `npm run format:check` – prettier

## Layout

- `src/midi.ts` – MIDI loading and pitch conversion helpers
- `src/audio.ts` – WAV I/O (mono, float32)
- `src/pitch.ts` – YIN-based sliding-window pitch analysis
- `src/splice.ts` – nearest-pitch search and concatenative synthesis
- `src/index.ts` – public entry + `renderMidiFromAudio` convenience
- `src/cli.ts` – minimal CLI wrapper

## Notes

- Input audio must be WAV. Any bit depth accepted by `wavefile` is converted
  to 32-bit float internally.
- Output is always 32-bit float mono WAV at the input's sample rate.
- This is an experiment and prioritises clarity over production robustness.
