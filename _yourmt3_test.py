"""Quick test: call YourMT3 Space, dump the HTML response so we can see how to
extract the MIDI download URL."""

import sys
from pathlib import Path
from gradio_client import Client, handle_file

audio = Path(sys.argv[1])
print(f"Uploading {audio.name} to YourMT3 Space (this can take several minutes)...")
client = Client("mimbres/YourMT3")
result = client.predict(
    audio_filepath=handle_file(str(audio)),
    api_name="/process_audio",
)
print("---HTML RESULT---")
print(result)
print("---END---")
Path("_yourmt3_response.html").write_text(result, encoding="utf-8")
print("Saved to _yourmt3_response.html")
