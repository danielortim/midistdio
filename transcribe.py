"""
transcribe.py

End-to-end transcription pipeline:

    audio file (mp3 / wav / m4a / flac / ogg)
        -> chunked + sent to YourMT3+ (HuggingFace Space, GPU)
        -> per-chunk multi-track MIDIs
        -> merged into one timeline
        -> rechannel.py (one MIDI channel per instrument)
        -> midi_output/<name>_yourmt3_chan.mid

Auth:
    Requires a HuggingFace token (free signup). Set it via env var:
        setx HF_TOKEN "hf_xxx..."     (then reopen the terminal)
    or pass it on the command line:
        python transcribe.py song.mp3 --token hf_xxx...

Usage:
    python transcribe.py path/to/song.mp3
    python transcribe.py path/to/song.mp3 --chunk-sec 60
    python transcribe.py path/to/song.mp3 --max-chunks 2     # quick test
"""

import argparse
import base64
import os
import re
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
    """The Space returns an HTML iframe with the MIDI embedded as a
    base64 data URI: href="data:audio/midi;base64,...". Pull the bytes out."""
    m = re.search(r'data:audio/midi;base64,([A-Za-z0-9+/=]+)', html)
    if m:
        return base64.b64decode(m.group(1))
    url_m = re.search(r'(?:href|src)="([^"]+\.mid[^"]*)"', html, re.I)
    if not url_m:
        raise RuntimeError(
            "No MIDI data URI or .mid URL found in Space response. "
            "See _yourmt3_response.html for details."
        )
    url = url_m.group(1)
    if url.startswith("/"):
        url = SPACE_BASE + url
    with urllib.request.urlopen(url) as r:
        return r.read()


def transcribe_chunk(client: Client, wav_path: Path) -> mido.MidiFile:
    """Send one chunk to YourMT3+ and return the resulting MidiFile."""
    html = client.predict(
        audio_filepath=handle_file(str(wav_path)),
        api_name="/process_audio",
    )
    Path("_yourmt3_response.html").write_text(html, encoding="utf-8")
    midi_bytes = extract_midi_bytes(html)
    tmp = wav_path.with_suffix(".mid")
    tmp.write_bytes(midi_bytes)
    return mido.MidiFile(tmp)


def merge_chunks(chunks: list[tuple[float, mido.MidiFile]],
                 out_path: Path) -> Path:
    """Merge per-chunk MIDIs into one timeline.
    chunks = [(start_seconds, MidiFile), ...]
    Tracks are merged by track-name across chunks."""
    # absolute (seconds) events keyed by track name
    by_track: dict[str, list[tuple[float, mido.Message]]] = {}
    program_for: dict[str, mido.Message] = {}

    for start_sec, mid in chunks:
        # Each chunk has its own tempo map; flatten to seconds.
        tpb = mid.ticks_per_beat
        for track in mid.tracks:
            name_msg = next(
                (m for m in track if m.type == "track_name"), None
            )
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

    # Build merged MIDI
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
            pc = program_for[name].copy(time=0)
            track.append(pc)
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


def chunk_audio(audio: Path, chunk_sec: float, work_dir: Path) -> list[Path]:
    """Load the audio and split into mono WAVs of `chunk_sec` seconds each."""
    print(f"Loading {audio.name} ...")
    y, sr = librosa.load(audio, sr=22050, mono=True)
    total = len(y) / sr
    print(f"  Duration: {total:.1f}s ({total/60:.2f} min) at {sr} Hz")
    work_dir.mkdir(parents=True, exist_ok=True)
    samples_per_chunk = int(chunk_sec * sr)
    paths = []
    i = 0
    while i * samples_per_chunk < len(y):
        clip = y[i*samples_per_chunk:(i+1)*samples_per_chunk]
        if len(clip) < sr * 1.0:  # skip <1s tail
            break
        out = work_dir / f"chunk_{i:03d}.wav"
        sf.write(out, clip, sr)
        paths.append(out)
        i += 1
    return paths


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("audio", type=Path, help="Audio file to transcribe")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Final MIDI output path "
                        "(default: midi_output/<stem>_yourmt3_chan.mid)")
    p.add_argument("--token", default=None,
                   help="HuggingFace token (otherwise read from $HF_TOKEN)")
    p.add_argument("--chunk-sec", type=float, default=60.0,
                   help="Audio chunk length in seconds (default 60)")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="Process only the first N chunks (for quick tests)")
    p.add_argument("--keep-raw", action="store_true",
                   help="Keep the unchanneled merged MIDI and per-chunk files")
    args = p.parse_args()

    if not args.audio.exists():
        print(f"Error: not found: {args.audio}", file=sys.stderr); sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: no HuggingFace token. Set $HF_TOKEN or pass --token.",
              file=sys.stderr)
        sys.exit(1)

    out_dir = (args.output.parent if args.output else Path("midi_output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="yourmt3_") as work:
        work = Path(work)
        chunks_paths = chunk_audio(args.audio, args.chunk_sec, work)
        if args.max_chunks:
            chunks_paths = chunks_paths[:args.max_chunks]
        print(f"Chunks to process: {len(chunks_paths)}")

        print(f"Connecting to {SPACE} ...")
        client = Client(SPACE, token=token)

        chunks: list[tuple[float, mido.MidiFile]] = []
        for i, wav in enumerate(chunks_paths):
            start = i * args.chunk_sec
            print(f"[{i+1}/{len(chunks_paths)}] Transcribing chunk @ {start:.0f}s ...")
            t0 = time.time()
            try:
                mid = transcribe_chunk(client, wav)
            except Exception as e:
                print(f"  ! chunk {i} failed: {e}")
                continue
            print(f"  -> {len(mid.tracks)} tracks  ({time.time()-t0:.1f}s)")
            chunks.append((start, mid))

        if not chunks:
            print("Nothing transcribed.", file=sys.stderr); sys.exit(2)

        raw_merged = out_dir / f"{args.audio.stem}_yourmt3.mid"
        print(f"Merging {len(chunks)} chunks -> {raw_merged.name}")
        merge_chunks(chunks, raw_merged)

        final = args.output or out_dir / f"{args.audio.stem}_yourmt3_chan.mid"
        print(f"Rechanneling -> {final.name}")
        rechannel.rechannel(raw_merged, final)

        if not args.keep_raw and raw_merged != final:
            raw_merged.unlink()

    print(f"\nDone. Drop this into the player:\n  {final}")


if __name__ == "__main__":
    main()
