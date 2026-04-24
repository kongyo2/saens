import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { Midi } = require("@tonejs/midi") as typeof import("@tonejs/midi");

export interface MidiNote {
  midi: number;
  time: number;
  duration: number;
  velocity: number;
}

export async function loadMidiNotes(path: string): Promise<{
  notes: MidiNote[];
  duration: number;
}> {
  const data = await readFile(path);
  const midi = new Midi(data);
  const notes: MidiNote[] = [];
  for (const track of midi.tracks) {
    for (const note of track.notes) {
      notes.push({
        midi: note.midi,
        time: note.time,
        duration: note.duration,
        velocity: note.velocity,
      });
    }
  }
  notes.sort((a, b) => a.time - b.time);
  return { notes, duration: midi.duration };
}

export function midiToFrequency(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function frequencyToMidi(frequency: number): number {
  return 69 + 12 * Math.log2(frequency / 440);
}
