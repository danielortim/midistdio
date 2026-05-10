"""
rechannel.py

Take any multi-track MIDI and assign each track its own MIDI channel, so the
HTML player can show / mute them independently.

Tracks whose name contains "drum" go to channel 9 (the GM drum channel).
Everything else gets channels 0, 1, 2, ... (skipping 9).

Designed to post-process MIDI from:
  - YourMT3+        (https://huggingface.co/spaces/mimbres/YourMT3)
  - audio_to_multitrack.py
  - any other multi-track MIDI source

Usage:
    python rechannel.py input.mid
    python rechannel.py input.mid -o output.mid
    python rechannel.py input.mid --in-place
"""

import argparse
import shutil
import sys
from pathlib import Path

import mido


CHANNELED_TYPES = {
    "note_on", "note_off", "control_change", "program_change",
    "pitchwheel", "aftertouch", "polytouch",
}

# General-MIDI program numbers picked by track-name keyword. First match wins.
PROGRAM_HINTS = [
    ("vocal",      53),  # Voice Oohs
    ("voice",      53),
    ("sing",       53),
    ("bass",       33),  # Electric Bass (finger)
    ("guitar",     24),  # Acoustic Guitar (nylon)
    ("piano",       0),  # Acoustic Grand Piano
    ("organ",      19),
    ("string",     48),
    ("violin",     40),
    ("brass",      61),
    ("trumpet",    56),
    ("sax",        65),
    ("flute",      73),
    ("synth",      80),
    ("pad",        88),
    ("lead",       80),
]


def pick_program(track_name: str) -> int | None:
    name = (track_name or "").lower()
    for keyword, prog in PROGRAM_HINTS:
        if keyword in name:
            return prog
    return None


def rechannel(in_path: Path, out_path: Path) -> Path:
    mid = mido.MidiFile(in_path)

    next_ch = 0  # channels we'll hand out to non-drum tracks

    def take_next_channel() -> int:
        nonlocal next_ch
        if next_ch == 9:           # reserve 9 for drums
            next_ch = 10
        if next_ch > 15:
            raise RuntimeError("More than 15 non-drum tracks — out of MIDI channels")
        ch = next_ch
        next_ch += 1
        return ch

    for i, track in enumerate(mid.tracks):
        # Skip a pure-meta track (no channeled messages anywhere in it).
        if not any(msg.type in CHANNELED_TYPES for msg in track):
            continue

        track_name = next(
            (m.name for m in track if m.type == "track_name"), f"Track {i}"
        )
        is_drums = "drum" in track_name.lower()
        ch = 9 if is_drums else take_next_channel()

        # Insert a program_change at the start so each channel sounds distinct
        # in any synth that respects program changes. Drums stay on the kit.
        if not is_drums:
            program = pick_program(track_name)
            if program is not None:
                # Find where to insert (after track_name if present, else at 0)
                insert_idx = 0
                for j, m in enumerate(track):
                    if m.type == "track_name":
                        insert_idx = j + 1
                        break
                track.insert(
                    insert_idx,
                    mido.Message("program_change", channel=ch,
                                 program=program, time=0),
                )

        # Reassign every channeled message in this track.
        for msg in track:
            if msg.type in CHANNELED_TYPES:
                msg.channel = ch

        print(f"  Track {i}: {track_name!r}  ->  channel {ch + 1}")

    mid.save(out_path)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Assign each track in a multi-track MIDI to its own MIDI channel."
    )
    p.add_argument("input", type=Path, help="Path to input .mid")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output path (default: <input>_chan.mid)")
    p.add_argument("--in-place", action="store_true",
                   help="Overwrite input (a backup '<name>_orig.mid' is saved first)")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Error: not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.in_place:
        backup = args.input.with_name(args.input.stem + "_orig.mid")
        if not backup.exists():
            shutil.copy(args.input, backup)
            print(f"Backup saved: {backup.name}")
        out = args.input
    else:
        out = args.output or args.input.with_name(args.input.stem + "_chan.mid")

    print(f"Rechanneling {args.input.name} -> {out.name}")
    rechannel(args.input, out)
    print(f"Done.")


if __name__ == "__main__":
    main()
