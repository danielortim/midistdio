"""
transcribe.py

End-to-end transcription pipeline:

    audio file (mp3 / wav / m4a / flac / ogg)
      [optional: --separate -> Demucs splits into vocals/bass/drums/other]
        -> chunked + sent to YourMT3+ (HuggingFace Space, GPU)
        -> per-chunk multi-track MIDIs
        -> merged into one timeline (track names prefixed by stem if --separate)
        -> rechannel.py (one MIDI channel per track)
        -> midi_output/<name>_yourmt3_chan.mid

Auth:
    Requires a HuggingFace token (free signup). Set it via env var:
        $env:HF_TOKEN = "hf_xxx..."
    or pass it on the command line:
        python transcribe.py song.mp3 --token hf_xxx...

Usage:
    python transcribe.py song.mp3
    python transcribe.py song.mp3 --separate                # better for dense mixes
    python transcribe.py song.mp3 --separate --skip-vocals  # rap / instrumental
    python transcribe.py song.mp3 --max-chunks 2            # first 2 min only (cheap test)
"""

import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import librosa
import mido
import soundfile as sf
from gradio_client import Client, handle_file

import rechannel  # local module


SPACE = "mimbres/YourMT3"
SPACE_BASE = "https://mimbres-yourmt3.hf.space"

# Output time base for the merged MIDI.
MERGED_TPB = 480
MERGED_TEMPO = 500_000  # 120 BPM


def extract_midi_bytes(html: str) -> bytes:
    """Pull the MIDI bytes out of the Space's HTML iframe response."""
    m = re.search(r'data:audio/midi;base64,([A-Za-z0-9+/=]+)', html)
    if m:
        return base64.b64decode(m.group(1))
    url_m = re.search(r'(?:href|src)="([^"]+\.mid[^"]*)"', html, re.I)
    if not url_m:
        raise RuntimeError("No MIDI in Space response (see _yourmt3_response.html).")
    url = url_m.group(1)
    if url.startswith("/"):
        url = SPACE_BASE + url
    with urllib.request.urlopen(url) as r:
        return r.read()


def transcribe_chunk(client: Client, wav_path: Path) -> mido.MidiFile:
    html = client.predict(
        audio_filepath=handle_file(str(wav_path)),
        api_name="/process_audio",
    )
    Path("_yourmt3_response.html").write_text(html, encoding="utf-8")
    midi_bytes = extract_midi_bytes(html)
    tmp = wav_path.with_suffix(".mid")
    tmp.write_bytes(midi_bytes)
    return mido.MidiFile(tmp)


def chunk_audio(audio: Path, chunk_sec: float, work_dir: Path) -> list[Path]:
    """Load audio, return list of mono-WAV chunk paths of `chunk_sec` each."""
    y, sr = librosa.load(audio, sr=22050, mono=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    samples = int(chunk_sec * sr)
    paths = []
    i = 0
    while i * samples < len(y):
        clip = y[i*samples:(i+1)*samples]
        if len(clip) < sr * 1.0:
            break
        out = work_dir / f"chunk_{i:03d}.wav"
        sf.write(out, clip, sr)
        paths.append(out)
        i += 1
    return paths


def separate_stems(audio: Path, work_dir: Path) -> dict[str, Path]:
    """Run Demucs (htdemucs) to split audio into vocals/drums/bass/other WAVs."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs",
           "-o", str(work_dir), "-n", "htdemucs", str(audio)]
    print(f"Separating stems with Demucs (slow on CPU; once per song)...")
    subprocess.run(cmd, check=True)
    stem_dir = work_dir / "htdemucs" / audio.stem
    out = {}
    for name in ("vocals", "bass", "drums", "other"):
        p = stem_dir / f"{name}.wav"
        if p.exists():
            out[name] = p
    if not out:
        raise RuntimeError(f"Demucs produced no stems in {stem_dir}")
    return out


def prefix_track_names(mid: mido.MidiFile, prefix: str) -> None:
    """In-place: prepend `prefix: ` to every track_name in the file."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == "track_name":
                msg.name = f"{prefix}: {msg.name}"
                break


def transcribe_audio_to_chunks(
    client: Client, audio: Path, chunk_sec: float, max_chunks: int | None,
    work_dir: Path, label: str = "",
) -> list[tuple[float, mido.MidiFile]]:
    """Chunk an audio file and transcribe each chunk via YourMT3+.
    Returns list of (start_seconds, MidiFile)."""
    chunks_paths = chunk_audio(audio, chunk_sec, work_dir)
    if max_chunks:
        chunks_paths = chunks_paths[:max_chunks]
    out: list[tuple[float, mido.MidiFile]] = []
    for i, wav in enumerate(chunks_paths):
        start = i * chunk_sec
        tag = f"[{label}] " if label else ""
        print(f"  {tag}[{i+1}/{len(chunks_paths)}] @ {start:.0f}s ...", end="", flush=True)
        t0 = time.time()
        try:
            mid = transcribe_chunk(client, wav)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        if label:
            prefix_track_names(mid, label)
        print(f" {len(mid.tracks)} tracks ({time.time()-t0:.1f}s)")
        out.append((start, mid))
    return out


def merge_chunks(chunks: list[tuple[float, mido.MidiFile]],
                 out_path: Path) -> Path:
    """Merge per-chunk MIDIs by track-name into one timeline."""
    by_track: dict[str, list[tuple[float, mido.Message]]] = {}
    program_for: dict[str, mido.Message] = {}

    for start_sec, mid in chunks:
        tpb = mid.ticks_per_beat
        for track in mid.tracks:
            name_msg = next((m for m in track if m.type == "track_name"), None)
            if name_msg is None:
                continue
            name = name_msg.name
            tempo = MERGED_TEMPO
            abs_tick = 0
            evt_list = by_track.setdefault(name, [])
            for msg in track:
                abs_tick += msg.time
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                    continue
                sec = mido.tick2second(abs_tick, tpb, tempo) + start_sec
                if msg.type == "program_change":
                    program_for.setdefault(name, msg)
                elif msg.type in ("note_on", "note_off"):
                    evt_list.append((sec, msg))

    merged = mido.MidiFile(ticks_per_beat=MERGED_TPB)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Meta", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=MERGED_TEMPO, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    merged.tracks.append(meta)

    for name, evts in by_track.items():
        if not evts:
            continue
        evts.sort(key=lambda e: e[0])
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        if name in program_for:
            track.append(program_for[name].copy(time=0))
        last_tick = 0
        for sec, msg in evts:
            tick = int(round(mido.second2tick(sec, MERGED_TPB, MERGED_TEMPO)))
            delta = max(0, tick - last_tick)
            last_tick = tick
            track.append(msg.copy(time=delta))
        track.append(mido.MetaMessage("end_of_track", time=0))
        merged.tracks.append(track)

    merged.save(out_path)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("audio", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--token", default=None,
                   help="HuggingFace token (or read from $HF_TOKEN)")
    p.add_argument("--chunk-sec", type=float, default=60.0)
    p.add_argument("--max-chunks", type=int, default=None,
                   help="Only process first N chunks (per stem if --separate)")
    p.add_argument("--separate", action="store_true",
                   help="Run Demucs first, then transcribe each pitched stem "
                        "separately. Better quality on dense mixes; ~3x quota.")
    p.add_argument("--skip-vocals", action="store_true",
                   help="With --separate: skip the vocals stem (good for rap)")
    p.add_argument("--keep-raw", action="store_true",
                   help="Keep the unchanneled merged MIDI")
    args = p.parse_args()

    if not args.audio.exists():
        print(f"Error: not found: {args.audio}", file=sys.stderr); sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: no HuggingFace token. Set $HF_TOKEN or pass --token.",
              file=sys.stderr); sys.exit(1)

    out_dir = (args.output.parent if args.output else Path("midi_output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # When --separate, output a bundle directory with stems + MIDI so the
    # player can load it as a multi-track DAW view.
    bundle_dir = (out_dir / args.audio.stem) if args.separate else None
    if bundle_dir:
        bundle_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {SPACE} ...")
    client = Client(SPACE, token=token)

    with tempfile.TemporaryDirectory(prefix="yourmt3_") as work:
        work = Path(work)
        all_chunks: list[tuple[float, mido.MidiFile]] = []

        if args.separate:
            stems = separate_stems(args.audio, work / "stems")
            print(f"Got stems: {', '.join(stems)}")
            # Copy stems to the bundle directory
            for name, p in stems.items():
                dest = bundle_dir / f"{name}.wav"
                shutil.copy(p, dest)
            print(f"Stems saved to: {bundle_dir}")
            for stem_name, stem_path in stems.items():
                if stem_name == "drums":
                    print(f"Skipping {stem_name} (non-pitched)")
                    continue
                if stem_name == "vocals" and args.skip_vocals:
                    print(f"Skipping {stem_name} (--skip-vocals)")
                    continue
                print(f"Transcribing stem: {stem_name}")
                all_chunks.extend(transcribe_audio_to_chunks(
                    client, stem_path,
                    args.chunk_sec, args.max_chunks,
                    work / f"chunks_{stem_name}",
                    label=stem_name.capitalize(),
                ))
        else:
            y, sr = librosa.load(args.audio, sr=22050, mono=True)
            print(f"Duration: {len(y)/sr:.1f}s ({len(y)/sr/60:.2f} min)")
            print(f"Transcribing whole mix")
            all_chunks = transcribe_audio_to_chunks(
                client, args.audio,
                args.chunk_sec, args.max_chunks,
                work / "chunks",
            )

        if not all_chunks:
            print("Nothing transcribed.", file=sys.stderr); sys.exit(2)

        raw_merged = (bundle_dir or out_dir) / f"{args.audio.stem}_yourmt3.mid"
        print(f"Merging {len(all_chunks)} chunks -> {raw_merged.name}")
        merge_chunks(all_chunks, raw_merged)

        if args.output:
            final = args.output
        elif bundle_dir:
            final = bundle_dir / "song.mid"
        else:
            final = out_dir / f"{args.audio.stem}_yourmt3_chan.mid"
        print(f"Rechanneling -> {final.name}")
        rechannel.rechannel(raw_merged, final)

        if not args.keep_raw and raw_merged != final:
            raw_merged.unlink()

    if bundle_dir:
        print(f"\nDone. Drop the whole folder into the player:\n  {bundle_dir}")
    else:
        print(f"\nDone. Drop into the player:\n  {final}")


if __name__ == "__main__":
    main()
