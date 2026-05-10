"""
audio_to_multitrack.py

Convert a song (local audio file OR URL) into a multi-track MIDI file with
separate tracks for Vocals, Bass, and Other Instruments. Drums are skipped
because basic-pitch can't meaningfully transcribe non-pitched percussion.

Pipeline:
  1. (URL only) yt-dlp downloads audio as MP3
  2. Demucs separates the audio into 4 stems (vocals/drums/bass/other)
  3. basic-pitch transcribes each pitched stem to a single-track MIDI
  4. mido merges them into one multi-track MIDI with named tracks

Usage:
    python audio_to_multitrack.py "song.mp3"
    python audio_to_multitrack.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
    python audio_to_multitrack.py "song.mp3" --keep-stems
    python audio_to_multitrack.py "song.mp3" --include-drums

Requirements (already installed):
    pip install basic-pitch mido onnxruntime demucs yt-dlp
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import mido


def is_url(source: str) -> bool:
    return "://" in source or source.startswith("www.")


def download_audio_from_url(url: str, output_dir: Path) -> Path:
    """Use yt-dlp to download the best audio track and convert it to MP3."""
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / "%(title).100s.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-x",                       # extract audio
        "--audio-format", "mp3",
        "--audio-quality", "0",     # best quality
        "-o", template,
        "--no-playlist",
        url,
    ]
    print(f"Downloading audio from {url} ...")
    subprocess.run(cmd, check=True)

    mp3s = list(output_dir.glob("*.mp3"))
    if not mp3s:
        raise FileNotFoundError("yt-dlp didn't produce an mp3")
    return max(mp3s, key=lambda p: p.stat().st_mtime)


def separate_stems(audio_path: Path, output_dir: Path) -> dict:
    """Use Demucs to split audio into vocals/drums/bass/other WAVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "-o", str(output_dir),
        "-n", "htdemucs",
        str(audio_path),
    ]
    print(f"Separating stems from {audio_path.name} (this is the slow step)...")
    subprocess.run(cmd, check=True)

    stem_dir = output_dir / "htdemucs" / audio_path.stem
    stems = {
        "vocals": stem_dir / "vocals.wav",
        "drums":  stem_dir / "drums.wav",
        "bass":   stem_dir / "bass.wav",
        "other":  stem_dir / "other.wav",
    }
    for name, path in stems.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected stem {name!r} at {path}")
    return stems


def transcribe_stem(audio_path: Path, output_dir: Path) -> Path:
    """Run basic-pitch on a single audio file and return path to its MIDI."""
    # Imported here so basic-pitch's startup warnings only appear once we
    # actually need it (and after demucs has finished its noisy progress bars).
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict_and_save

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Transcribing {audio_path.name} ...")
    predict_and_save(
        audio_path_list=[str(audio_path)],
        output_directory=str(output_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )

    midi_path = output_dir / f"{audio_path.stem}_basic_pitch.mid"
    if not midi_path.exists():
        raise FileNotFoundError(f"Expected MIDI not found at {midi_path}")
    return midi_path


# Pretty names for the MIDI track headers (so your HTML player picks them up).
TRACK_NAMES = {
    "vocals": "Vocals",
    "bass":   "Bass",
    "other":  "Other Instruments",
    "drums":  "Drums (raw)",
}

# Distinct MIDI channel per stem so the player can mute them independently.
# Channel 9 is the General-MIDI drum channel.
CHANNEL_MAP = {
    "vocals": 0,
    "bass":   1,
    "other":  2,
    "drums":  9,
}

# General-MIDI program numbers — gives each channel a different default voice.
PROGRAM_MAP = {
    "vocals": 53,  # Voice Oohs
    "bass":   33,  # Electric Bass (finger)
    "other":  0,   # Acoustic Grand Piano
    "drums":  0,   # ignored on channel 9
}


def merge_tracks(stem_midis: dict, output_path: Path) -> Path:
    """Combine each stem's single-track MIDI into one multi-track MIDI."""
    first_midi = mido.MidiFile(next(iter(stem_midis.values())))
    merged = mido.MidiFile(ticks_per_beat=first_midi.ticks_per_beat)

    # Track 0 carries tempo/meta info.
    meta_track = mido.MidiTrack()
    meta_track.append(mido.MetaMessage("track_name", name="Meta", time=0))
    for msg in first_midi.tracks[0]:
        if msg.is_meta and msg.type != "end_of_track":
            meta_track.append(msg.copy())
    meta_track.append(mido.MetaMessage("end_of_track", time=0))
    merged.tracks.append(meta_track)

    # One track per stem, on its own MIDI channel.
    for stem_name, midi_path in stem_midis.items():
        ch = CHANNEL_MAP[stem_name]
        new_track = mido.MidiTrack()
        new_track.append(
            mido.MetaMessage("track_name", name=TRACK_NAMES[stem_name], time=0)
        )
        new_track.append(
            mido.Message("program_change", channel=ch,
                         program=PROGRAM_MAP[stem_name], time=0)
        )
        src_midi = mido.MidiFile(midi_path)
        for src_track in src_midi.tracks:
            for msg in src_track:
                if msg.type in ("note_on", "note_off"):
                    new_track.append(msg.copy(channel=ch))
        new_track.append(mido.MetaMessage("end_of_track", time=0))
        merged.tracks.append(new_track)

    merged.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert audio (file or URL) to a multi-track MIDI file."
    )
    parser.add_argument(
        "source",
        help="Path to audio file OR URL (YouTube, SoundCloud, etc.)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("midi_output"))
    parser.add_argument(
        "--keep-stems",
        action="store_true",
        help="Keep the separated audio stems and intermediate files",
    )
    parser.add_argument(
        "--include-drums",
        action="store_true",
        help="Try to transcribe drums (results will be noisy)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get the audio
    if is_url(args.source):
        audio_path = download_audio_from_url(args.source, work_dir / "downloads")
    else:
        audio_path = Path(args.source)
        if not audio_path.exists():
            print(f"Error: file not found: {audio_path}", file=sys.stderr)
            sys.exit(1)

    print(f"Processing: {audio_path.name}\n")

    # 2. Separate stems
    stems = separate_stems(audio_path, work_dir / "stems")

    # 3. Transcribe each pitched stem
    pitched = {k: v for k, v in stems.items() if k != "drums" or args.include_drums}

    print("\nTranscribing stems to MIDI ...")
    stem_midis = {}
    for stem_name, stem_path in pitched.items():
        stem_midis[stem_name] = transcribe_stem(stem_path, work_dir / "midis")

    # 4. Merge
    out_path = args.output_dir / f"{audio_path.stem}_multitrack.mid"
    merge_tracks(stem_midis, out_path)

    print(f"\nDone. Multi-track MIDI saved to:\n  {out_path}")
    print(f"Tracks included: {', '.join(TRACK_NAMES[k] for k in stem_midis)}")

    # 5. Cleanup
    if not args.keep_stems:
        shutil.rmtree(work_dir, ignore_errors=True)
        print("(Cleaned up intermediate files. Use --keep-stems to preserve them.)")


if __name__ == "__main__":
    main()
