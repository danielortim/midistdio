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
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).parent
CACHE = ROOT / "stems_cache"
CACHE.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=None)

# {song_id: {"status": "running"|"done"|"error", "progress": 0..100, "stems": {...}, "error": str|None}}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


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
               "-o", str(out), "-n", "htdemucs", str(src_audio)]
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

        stem_src = out / "htdemucs" / src_audio.stem
        stems = {}
        for name in ("vocals", "drums", "bass", "other"):
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
        for name in ("vocals", "drums", "bass", "other"):
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
        # Maybe it's a cached song server-restarted; check disk
        job_dir = CACHE / song_id
        if job_dir.exists():
            cached = {}
            for name in ("vocals", "drums", "bass", "other"):
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


if __name__ == "__main__":
    print(" * Open http://localhost:8000/ in Chrome")
    app.run(host="127.0.0.1", port=8000, threaded=True, debug=False)
