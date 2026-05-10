"""
server.py

Local web server that:
  - serves index.html (so the player loads at http://localhost:8000/)
  - POST /separate  : accepts an uploaded audio file, runs Demucs on it,
                      returns a JSON list of stem URLs the player can fetch.
  - GET  /stems/<sid>/<name>.wav : serves a separated stem.
  - GET  /stems/<sid>/original.<ext> : serves the original audio.

Stems are cached on disk in ./stems_cache/<song-hash>/ so re-uploading the
same audio is instant.

Run:
    venv\\Scripts\\python.exe server.py
    -> open http://localhost:8000/

Dependencies (already installed): demucs, flask
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

# Windows: force UTF-8 stdout/stderr so libraries that print emoji
# (basic_pitch, demucs progress bars) don't crash on cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from flask import Flask, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).parent
CACHE = ROOT / "stems_cache"
CACHE.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=None)

# Separation jobs and transcription jobs.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_trx_jobs: dict[str, dict] = {}
_trx_lock = threading.Lock()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _run_demucs(song_id: str, src_audio: Path) -> None:
    job_dir = CACHE / song_id
    try:
        with _jobs_lock:
            _jobs[song_id] = {"status": "running", "progress": 5, "stems": {},
                              "error": None, "name": src_audio.name}
        # demucs writes to <out>/htdemucs/<stem>/<wav>
        out = job_dir / "_work"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "demucs",
               "-o", str(out), "-n", "htdemucs_6s", str(src_audio)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        # Poll demucs output for progress %, scan for "%" lines
        for line in proc.stdout:
            line = line.strip()
            if "%" in line:
                # demucs prints lines like "  37%|███... | ..."
                try:
                    pct = int(line.split("%")[0].split()[-1])
                    with _jobs_lock:
                        _jobs[song_id]["progress"] = max(5, min(95, pct))
                except (ValueError, IndexError):
                    pass
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"demucs exited {proc.returncode}")

        stem_src = out / "htdemucs_6s" / src_audio.stem
        stems = {}
        for name in ("vocals", "drums", "bass", "other", "piano", "guitar"):
            p = stem_src / f"{name}.wav"
            if p.exists():
                dest = job_dir / f"{name}.wav"
                shutil.move(str(p), str(dest))
                stems[name] = f"/stems/{song_id}/{name}.wav"

        # Keep the original audio next to the stems for reference playback
        orig_dest = job_dir / f"original{src_audio.suffix}"
        shutil.copy(src_audio, orig_dest)
        stems["original"] = f"/stems/{song_id}/{orig_dest.name}"

        # Cleanup the demucs work dir
        shutil.rmtree(out, ignore_errors=True)

        with _jobs_lock:
            _jobs[song_id]["status"] = "done"
            _jobs[song_id]["progress"] = 100
            _jobs[song_id]["stems"] = stems
    except Exception as e:
        with _jobs_lock:
            _jobs[song_id] = {"status": "error", "progress": 0, "stems": {},
                              "error": str(e)}


@app.route("/")
def index():
    return send_file(ROOT / "index.html")


@app.route("/<path:filename>")
def static_files(filename: str):
    # Only serve files that exist next to index.html (rechannel.py etc. excluded)
    p = ROOT / filename
    if p.is_file() and p.suffix.lower() in {".html", ".css", ".js", ".png", ".jpg",
                                             ".svg", ".ico", ".woff", ".woff2"}:
        return send_from_directory(ROOT, filename)
    return ("Not found", 404)


@app.route("/separate", methods=["POST"])
def separate():
    if "audio" not in request.files:
        return jsonify({"error": "no audio file"}), 400
    f = request.files["audio"]
    # Save upload to a temp path so we can hash and feed to demucs
    tmp_dir = CACHE / "_uploads"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / f.filename
    f.save(tmp_path)

    song_id = _hash_file(tmp_path)
    job_dir = CACHE / song_id

    # Cache hit: stems already exist
    if job_dir.exists():
        cached = {}
        for name in ("vocals", "drums", "bass", "other", "piano", "guitar"):
            if (job_dir / f"{name}.wav").exists():
                cached[name] = f"/stems/{song_id}/{name}.wav"
        for orig in job_dir.glob("original.*"):
            cached["original"] = f"/stems/{song_id}/{orig.name}"
            break
        if cached:
            tmp_path.unlink(missing_ok=True)
            with _jobs_lock:
                _jobs[song_id] = {"status": "done", "progress": 100,
                                  "stems": cached, "error": None,
                                  "name": f.filename, "cached": True}
            return jsonify({"song_id": song_id, "cached": True})

    # Move upload into the job dir so we can clean up the work dir later
    job_dir.mkdir(parents=True, exist_ok=True)
    persistent = job_dir / f"_input{tmp_path.suffix}"
    shutil.move(str(tmp_path), str(persistent))

    # Kick off demucs in a thread; client polls /status/<song_id>
    th = threading.Thread(target=_run_demucs,
                          args=(song_id, persistent), daemon=True)
    th.start()
    return jsonify({"song_id": song_id, "cached": False})


@app.route("/status/<song_id>")
def status(song_id: str):
    with _jobs_lock:
        job = _jobs.get(song_id)
    if not job:
        job_dir = CACHE / song_id
        if job_dir.exists():
            cached = {}
            for name in ("vocals", "drums", "bass", "other", "piano", "guitar"):
                if (job_dir / f"{name}.wav").exists():
                    cached[name] = f"/stems/{song_id}/{name}.wav"
            for orig in job_dir.glob("original.*"):
                cached["original"] = f"/stems/{song_id}/{orig.name}"
                break
            if cached:
                return jsonify({"status": "done", "progress": 100,
                                "stems": cached, "error": None})
        return jsonify({"status": "unknown"}), 404
    return jsonify(job)


@app.route("/stems/<song_id>/<path:fname>")
def serve_stem(song_id: str, fname: str):
    return send_from_directory(CACHE / song_id, fname)


def _find_original(song_id: str) -> Path | None:
    job_dir = CACHE / song_id
    if not job_dir.exists():
        return None
    for cand in list(job_dir.glob("original.*")) + list(job_dir.glob("_input.*")):
        if cand.is_file():
            return cand
    return None


def _run_transcribe(song_id: str, audio_path: Path) -> None:
    """Local, offline transcription:
       - Reuse the Demucs stems already cached for this song_id
       - Run Basic Pitch on each pitched stem (skip drums)
       - Merge into a multi-track MIDI, one track per stem
       - Rechannel so each track gets its own MIDI channel
    No HuggingFace, no quotas. Quality is lower than YourMT3+ but it works
    fully offline."""
    try:
        with _trx_lock:
            _trx_jobs[song_id] = {"status": "running", "progress": 2,
                                  "detail": "Loading basic-pitch", "error": None}
        # Lazy import so server starts fast and tensorflow only loads on demand.
        import mido
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict_and_save
        import rechannel

        job_dir = CACHE / song_id
        stem_files = []
        for name in ("vocals", "bass", "other", "piano", "guitar"):
            p = job_dir / f"{name}.wav"
            if p.exists():
                stem_files.append((name, p))
        if not stem_files:
            stem_files.append(("mix", audio_path))

        with tempfile.TemporaryDirectory(prefix="bp_") as work:
            work = Path(work)
            n = len(stem_files)
            stem_midis: dict[str, Path] = {}
            for i, (name, src) in enumerate(stem_files):
                with _trx_lock:
                    _trx_jobs[song_id]["progress"] = 5 + int(80 * i / n)
                    _trx_jobs[song_id]["detail"] = f"Transcribing {name} ({i+1}/{n})"
                stem_out = work / name
                stem_out.mkdir()
                predict_and_save(
                    audio_path_list=[str(src)],
                    output_directory=str(stem_out),
                    save_midi=True,
                    sonify_midi=False,
                    save_model_outputs=False,
                    save_notes=False,
                    model_or_model_path=ICASSP_2022_MODEL_PATH,
                )
                produced = next(stem_out.glob("*_basic_pitch.mid"), None)
                if produced:
                    stem_midis[name] = produced

            if not stem_midis:
                raise RuntimeError("Basic Pitch produced no MIDI for any stem")

            # Merge stems into one multitrack MIDI
            with _trx_lock:
                _trx_jobs[song_id]["progress"] = 90
                _trx_jobs[song_id]["detail"] = "Merging tracks"
            first = mido.MidiFile(next(iter(stem_midis.values())))
            merged = mido.MidiFile(ticks_per_beat=first.ticks_per_beat)
            meta = mido.MidiTrack()
            meta.append(mido.MetaMessage("track_name", name="Meta", time=0))
            for msg in first.tracks[0]:
                if msg.is_meta and msg.type != "end_of_track":
                    meta.append(msg.copy())
            meta.append(mido.MetaMessage("end_of_track", time=0))
            merged.tracks.append(meta)
            for name, mid_path in stem_midis.items():
                track = mido.MidiTrack()
                track.append(mido.MetaMessage("track_name",
                                               name=name.capitalize(), time=0))
                src_mid = mido.MidiFile(mid_path)
                for src_track in src_mid.tracks:
                    for msg in src_track:
                        if msg.type in ("note_on", "note_off"):
                            track.append(msg.copy())
                track.append(mido.MetaMessage("end_of_track", time=0))
                merged.tracks.append(track)

            raw = job_dir / "_raw.mid"
            merged.save(raw)
            with _trx_lock:
                _trx_jobs[song_id]["progress"] = 96
                _trx_jobs[song_id]["detail"] = "Rechanneling"
            out_mid = job_dir / "transcription.mid"
            rechannel.rechannel(raw, out_mid)
            try: raw.unlink()
            except Exception: pass

        with _trx_lock:
            _trx_jobs[song_id]["status"] = "done"
            _trx_jobs[song_id]["progress"] = 100
            _trx_jobs[song_id]["detail"] = "Ready"
            _trx_jobs[song_id]["midi_url"] = f"/stems/{song_id}/transcription.mid"
    except Exception as e:
        with _trx_lock:
            _trx_jobs[song_id] = {"status": "error", "progress": 0,
                                  "error": str(e), "detail": str(e)}


@app.route("/transcribe/<song_id>", methods=["POST"])
def start_transcribe(song_id: str):
    audio = _find_original(song_id)
    if audio is None:
        return jsonify({"error": "no original audio for that song_id"}), 404
    # If already done, return cached
    cached = CACHE / song_id / "transcription.mid"
    if cached.exists():
        with _trx_lock:
            _trx_jobs[song_id] = {"status": "done", "progress": 100,
                                  "detail": "Cached",
                                  "midi_url": f"/stems/{song_id}/transcription.mid"}
        return jsonify({"started": True, "cached": True})
    th = threading.Thread(target=_run_transcribe,
                          args=(song_id, audio), daemon=True)
    th.start()
    return jsonify({"started": True})


@app.route("/transcribe/<song_id>", methods=["GET"])
def status_transcribe(song_id: str):
    cached = CACHE / song_id / "transcription.mid"
    with _trx_lock:
        job = dict(_trx_jobs.get(song_id) or {})
    if not job and cached.exists():
        return jsonify({"status": "done", "progress": 100, "detail": "Cached",
                        "midi_url": f"/stems/{song_id}/transcription.mid"})
    if not job:
        return jsonify({"status": "unknown"}), 404
    return jsonify(job)


if __name__ == "__main__":
    print(" * Open http://localhost:8000/ in Chrome")
    app.run(host="127.0.0.1", port=8000, threaded=True, debug=False)
