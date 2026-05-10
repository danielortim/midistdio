"""
mp3_to_midi.py

Convert an audio file (mp3, wav, m4a, flac, ogg) into a MIDI file using
Spotify's basic-pitch model. Optionally splits the output into two tracks
(left hand / right hand) based on pitch, so a player that detects hands
from track metadata will treat them independently.

Usage:
    python mp3_to_midi.py path/to/song.mp3
    python mp3_to_midi.py path/to/song.mp3 --split-hands
    python mp3_to_midi.py path/to/song.mp3 --split-hands --split-note 60

Setup (one time):
    pip install basic-pitch mido
"""

import argparse
import sys
from pathlib import Path

import mido
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict_and_save


def transcribe(audio_path: Path, output_dir: Path) -> Path:
    """Run basic-pitch on the audio file and return the path to the output MIDI."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Transcribing {audio_path.name} ... this can take a minute or two.")
    predict_and_save(
        audio_path_list=[str(audio_path)],
        output_directory=str(output_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )

    # basic-pitch writes its output as "<stem>_basic_pitch.mid"
    midi_path = output_dir / f"{audio_path.stem}_basic_pitch.mid"
    if not midi_path.exists():
        raise FileNotFoundError(f"Expected MIDI not found at {midi_path}")
    return midi_path


def split_hands(midi_path: Path, split_note: int = 60) -> Path:
    """
    Split a single-track MIDI into two tracks by pitch.
        notes >= split_note  -> Right Hand
        notes <  split_note  -> Left Hand
    Returns the path to the new (split) MIDI file.
    """
    mid = mido.MidiFile(midi_path)

    # Walk every track, convert delta-time to absolute ticks, and collect
    # all note_on / note_off events into one timeline.
    events = []  # list of (absolute_tick, message)
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type in ("note_on", "note_off"):
                events.append((abs_tick, msg.copy()))

    events.sort(key=lambda e: e[0])

    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

    # Track 0: meta info (tempo, time signature, etc.) copied from original
    meta_track = mido.MidiTrack()
    meta_track.append(mido.MetaMessage("track_name", name="Meta", time=0))
    for msg in mid.tracks[0]:
        if msg.is_meta and msg.type != "end_of_track":
            meta_track.append(msg.copy())
    meta_track.append(mido.MetaMessage("end_of_track", time=0))
    new_mid.tracks.append(meta_track)

    # Track 1 = Right Hand, Track 2 = Left Hand
    rh_track = mido.MidiTrack()
    rh_track.append(mido.MetaMessage("track_name", name="Right Hand", time=0))
    lh_track = mido.MidiTrack()
    lh_track.append(mido.MetaMessage("track_name", name="Left Hand", time=0))

    rh_last_tick = 0
    lh_last_tick = 0
    for abs_tick, msg in events:
        if msg.note >= split_note:
            rh_track.append(msg.copy(time=abs_tick - rh_last_tick))
            rh_last_tick = abs_tick
        else:
            lh_track.append(msg.copy(time=abs_tick - lh_last_tick))
            lh_last_tick = abs_tick

    rh_track.append(mido.MetaMessage("end_of_track", time=0))
    lh_track.append(mido.MetaMessage("end_of_track", time=0))
    new_mid.tracks.append(rh_track)
    new_mid.tracks.append(lh_track)

    out_path = midi_path.with_name(f"{midi_path.stem}_split.mid")
    new_mid.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an audio file to MIDI using basic-pitch."
    )
    parser.add_argument("audio", type=Path, help="Path to audio file (mp3, wav, etc.)")
    parser.add_argument("--output-dir", type=Path, default=Path("midi_output"))
    parser.add_argument(
        "--split-hands",
        action="store_true",
        help="Also produce a 2-track version split into left/right hand",
    )
    parser.add_argument(
        "--split-note",
        type=int,
        default=60,
        help="MIDI note number that divides hands (default 60 = middle C)",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    midi_path = transcribe(args.audio, args.output_dir)
    print(f"\nMIDI saved to: {midi_path}")

    if args.split_hands:
        split_path = split_hands(midi_path, args.split_note)
        print(f"Split MIDI saved to: {split_path}")
        print(
            f"  (notes >= {args.split_note} -> right hand, "
            f"< {args.split_note} -> left hand)"
        )


if __name__ == "__main__":
    main()
