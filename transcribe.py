"""
transcribe.py

End-to-end transcription pipeline:

    audio file (mp3 / wav / m4a / flac / ogg)
        -> YourMT3+ (HuggingFace Space, GPU)
        -> multi-track MIDI
        -> rechannel.py (one MIDI channel per instrument)
        -> midi_output/<name>_yourmt3_chan.mid

Auth:
    Requires a HuggingFace token (free signup). Set it via env var:
        setx HF_TOKEN "hf_xxx..."     (then reopen the terminal)
    or pass it on the command line:
        python transcribe.py song.mp3 --token hf_xxx...

Usage:
    python transcribe.py path/to/song.mp3
    python transcribe.py path/to/song.mp3 -o midi_output/custom_name.mid
"""

import argparse
import os
import re
import sys
import urllib.request
from pathlib import Path

from gradio_client import Client, handle_file

import rechannel  # local module


SPACE = "mimbres/YourMT3"
SPACE_BASE = "https://mimbres-yourmt3.hf.space"


def find_midi_url(html: str) -> str:
    """The Space returns an HTML snippet that links/embeds the produced MIDI.
    Pull the first .mid URL out of it. Tolerate absolute or relative links."""
    candidates = re.findall(r'(?:href|src)="([^"]+\.mid[^"]*)"', html, re.I)
    if not candidates:
        # Some Gradio components embed the file path differently — fall back to
        # any .mid mention.
        candidates = re.findall(r'([^"\s]+\.mid[^"\s]*)', html, re.I)
    if not candidates:
        raise RuntimeError(
            "No .mid URL found in Space response. Raw HTML saved to "
            "_yourmt3_response.html for inspection."
        )
    url = candidates[0]
    if url.startswith("/"):
        url = SPACE_BASE + url
    return url


def download(url: str, dest: Path) -> None:
    print(f"  Downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path, help="Audio file to transcribe")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Final MIDI output path (default: midi_output/<stem>_yourmt3_chan.mid)")
    p.add_argument("--token", default=None,
                   help="HuggingFace token (otherwise read from $HF_TOKEN)")
    p.add_argument("--keep-raw", action="store_true",
                   help="Keep the unchanneled MIDI from the Space")
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

    print(f"Connecting to {SPACE} ...")
    client = Client(SPACE, token=token)

    print(f"Transcribing {args.audio.name} (this can take 1-5 min on a free GPU)...")
    html = client.predict(
        audio_filepath=handle_file(str(args.audio)),
        api_name="/process_audio",
    )

    # Always save the raw HTML so we can debug if URL extraction fails.
    Path("_yourmt3_response.html").write_text(html, encoding="utf-8")

    midi_url = find_midi_url(html)
    raw_midi = out_dir / f"{args.audio.stem}_yourmt3.mid"
    download(midi_url, raw_midi)
    print(f"  Raw MIDI: {raw_midi}")

    final = args.output or out_dir / f"{args.audio.stem}_yourmt3_chan.mid"
    print(f"Rechanneling -> {final.name}")
    rechannel.rechannel(raw_midi, final)

    if not args.keep_raw and raw_midi != final:
        raw_midi.unlink()

    print(f"\nDone. Drop this into the player:\n  {final}")


if __name__ == "__main__":
    main()
