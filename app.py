"""
VoiceRadar web demo — local Flask app.
Upload an MP3/WAV/FLAC, get back Human / AI-Generated + confidence.
Progress is streamed to the browser via Server-Sent Events (SSE).
"""

import io
import json
import os
import queue
import tempfile
import threading
import subprocess
import numpy as np
import torch
import torchaudio
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context
from transformers import HubertModel, AutoFeatureExtractor

from micro_frequencies import compute_micro_frequencies
from doppler import compute_fo
from train import VoiceRadarMLP

# ---------------------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------------------

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading HuBERT on {DEVICE}...")
extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert    = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(DEVICE)
hubert.eval()

print("Loading VoiceRadar MLP...")
mlp = VoiceRadarMLP(input_dim=768).to(DEVICE)
mlp.load_state_dict(torch.load("voiceradar_best.pt", map_location=DEVICE))
mlp.eval()

print("Ready.")

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(audio_bytes: bytes, ext: str) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        in_path = tmp.name
    out_path = in_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path,
             "-ac", "1", "-ar", "16000", "-f", "wav", out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        import soundfile as sf
        samples, _ = sf.read(out_path, dtype="float32")
        return torch.from_numpy(samples).unsqueeze(0)   # [1, T]
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)

# ---------------------------------------------------------------------------
# Inference with progress callbacks
# ---------------------------------------------------------------------------

def predict_with_progress(audio_bytes: bytes, ext: str, progress):
    """
    progress(pct, message) sends an SSE update to the browser.
    Stages:
      10% — decoding audio
      30% — extracting HuBERT embedding
      70% — computing micro-frequencies
      90% — running MLP
     100% — done
    """
    progress(10, "Decoding audio...")
    waveform = load_audio(audio_bytes, ext)

    progress(30, "Extracting HuBERT embeddings...")
    inputs = extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        E = hubert(inputs.input_values.to(DEVICE)).last_hidden_state.cpu().numpy()

    progress(70, "Computing micro-frequencies...")
    mf = compute_micro_frequencies(E)

    progress(90, "Running classifier...")
    emb_vec = torch.from_numpy(E[0].mean(axis=0).astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = mlp(emb_vec).item()

    label      = "Human" if prob >= 0.5 else "AI-Generated"
    confidence = prob if prob >= 0.5 else 1.0 - prob

    progress(100, "Done", result={
        "label": label,
        "confidence": round(confidence * 100, 1),
        "raw": round(prob, 4),
    })

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VoiceRadar</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 16px;
    padding: 48px 40px;
    width: 100%;
    max-width: 480px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }

  h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 6px; color: #fff; }
  .subtitle { font-size: 0.85rem; color: #666; margin-bottom: 36px; }

  .drop-zone {
    border: 2px dashed #2a2d3a;
    border-radius: 12px;
    padding: 40px 24px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: #5b6af0;
    background: rgba(91,106,240,0.05);
  }
  .drop-zone input[type="file"] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
  }
  .drop-icon { font-size: 2rem; margin-bottom: 12px; }
  .drop-text { font-size: 0.9rem; color: #888; }
  .drop-text span { color: #5b6af0; }
  .file-name { margin-top: 12px; font-size: 0.8rem; color: #5b6af0; min-height: 18px; }

  .btn {
    width: 100%; margin-top: 20px; padding: 14px;
    background: #5b6af0; color: #fff; border: none;
    border-radius: 10px; font-size: 0.95rem; font-weight: 600;
    cursor: pointer; transition: background 0.2s, opacity 0.2s;
  }
  .btn:hover { background: #4a58d4; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* ── Progress section ── */
  .progress-section {
    display: none;
    margin-top: 24px;
  }
  .progress-label {
    font-size: 0.82rem;
    color: #888;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
  }
  .progress-track {
    background: #2a2d3a;
    border-radius: 99px;
    height: 6px;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    border-radius: 99px;
    background: #5b6af0;
    width: 0%;
    transition: width 0.4s ease;
  }
  .progress-step {
    margin-top: 10px;
    font-size: 0.78rem;
    color: #555;
    min-height: 16px;
  }

  /* ── Result section ── */
  .result {
    display: none;
    margin-top: 28px;
    padding: 24px;
    border-radius: 12px;
    text-align: center;
  }
  .result.human { background: rgba(52,199,89,0.08);  border: 1px solid rgba(52,199,89,0.25); }
  .result.ai    { background: rgba(255,69,58,0.08);  border: 1px solid rgba(255,69,58,0.25); }

  .result-emoji  { font-size: 2.4rem; margin-bottom: 8px; }
  .result-label  { font-size: 1.3rem; font-weight: 700; margin-bottom: 6px; }
  .result.human .result-label { color: #34c759; }
  .result.ai    .result-label { color: #ff453a; }
  .result-sub { font-size: 0.82rem; color: #888; margin-bottom: 16px; }

  .conf-track {
    background: #2a2d3a; border-radius: 99px; height: 8px; overflow: hidden;
  }
  .conf-fill {
    height: 100%; border-radius: 99px; transition: width 0.6s ease;
  }
  .result.human .conf-fill { background: #34c759; }
  .result.ai    .conf-fill { background: #ff453a; }
  .conf-labels {
    display: flex; justify-content: space-between;
    font-size: 0.75rem; color: #666; margin-top: 6px;
  }
</style>
</head>
<body>
<div class="card">
  <h1>VoiceRadar</h1>
  <p class="subtitle">Detects AI-generated speech using micro-frequency analysis</p>

  <div class="drop-zone" id="dropZone">
    <input type="file" id="fileInput" accept=".mp3,.wav,.flac">
    <div class="drop-icon">🎙️</div>
    <div class="drop-text">Drop an audio file here or <span>browse</span></div>
    <div class="drop-text" style="margin-top:4px;font-size:0.75rem;">MP3 · WAV · FLAC</div>
    <div class="file-name" id="fileName"></div>
  </div>

  <button class="btn" id="analyzeBtn" disabled>Analyze</button>

  <div class="progress-section" id="progressSection">
    <div class="progress-label">
      <span>Analyzing...</span>
      <span id="progressPct">0%</span>
    </div>
    <div class="progress-track">
      <div class="progress-fill" id="progressFill"></div>
    </div>
    <div class="progress-step" id="progressStep"></div>
  </div>

  <div class="result" id="result">
    <div class="result-emoji" id="resultEmoji"></div>
    <div class="result-label" id="resultLabel"></div>
    <div class="result-sub"   id="resultSub"></div>
    <div class="conf-track"><div class="conf-fill" id="confFill" style="width:0%"></div></div>
    <div class="conf-labels"><span>0%</span><span id="confLabel"></span><span>100%</span></div>
  </div>
</div>

<script>
  const fileInput      = document.getElementById("fileInput");
  const fileName       = document.getElementById("fileName");
  const analyzeBtn     = document.getElementById("analyzeBtn");
  const progressSection= document.getElementById("progressSection");
  const progressFill   = document.getElementById("progressFill");
  const progressPct    = document.getElementById("progressPct");
  const progressStep   = document.getElementById("progressStep");
  const resultDiv      = document.getElementById("result");
  const dropZone       = document.getElementById("dropZone");

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) {
      fileName.textContent  = fileInput.files[0].name;
      analyzeBtn.disabled   = false;
      resultDiv.style.display      = "none";
      progressSection.style.display= "none";
    }
  });

  dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault(); dropZone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) {
      fileInput.files = e.dataTransfer.files;
      fileName.textContent = f.name;
      analyzeBtn.disabled  = false;
      resultDiv.style.display = "none";
    }
  });

  analyzeBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    analyzeBtn.disabled = true;
    resultDiv.style.display = "none";
    progressSection.style.display = "block";
    setProgress(0, "Uploading...");

    const form = new FormData();
    form.append("file", file);

    // Upload first, get a job ID back
    const uploadRes = await fetch("/upload", { method: "POST", body: form });
    const { job_id } = await uploadRes.json();

    // Stream progress via SSE
    const evtSource = new EventSource(`/progress/${job_id}`);
    evtSource.onmessage = e => {
      const data = JSON.parse(e.data);
      setProgress(data.pct, data.message);
      if (data.result) {
        evtSource.close();
        progressSection.style.display = "none";
        showResult(data.result);
        analyzeBtn.disabled = false;
      }
      if (data.error) {
        evtSource.close();
        progressSection.style.display = "none";
        alert("Error: " + data.error);
        analyzeBtn.disabled = false;
      }
    };
  });

  function setProgress(pct, msg) {
    progressFill.style.width = pct + "%";
    progressPct.textContent  = pct + "%";
    progressStep.textContent = msg;
  }

  function showResult(data) {
    const isHuman = data.label === "Human";
    resultDiv.className = "result " + (isHuman ? "human" : "ai");
    document.getElementById("resultEmoji").textContent = isHuman ? "✅" : "🤖";
    document.getElementById("resultLabel").textContent = data.label;
    document.getElementById("resultSub").textContent   = `${data.confidence}% confidence`;
    document.getElementById("confFill").style.width    = data.confidence + "%";
    document.getElementById("confLabel").textContent   = data.confidence + "%";
    resultDiv.style.display = "block";
  }
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

# In-memory job store: job_id → queue of SSE events
_jobs: dict[str, queue.Queue] = {}


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f   = request.files["file"]
    ext = f.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("mp3", "wav", "flac"):
        return jsonify({"error": "unsupported format"}), 400

    audio_bytes = f.read()
    job_id = os.urandom(8).hex()
    q: queue.Queue = queue.Queue()
    _jobs[job_id] = q

    def run():
        try:
            def progress(pct, message, result=None):
                q.put({"pct": pct, "message": message, "result": result})
            predict_with_progress(audio_bytes, ext, progress)
        except Exception as e:
            q.put({"pct": 0, "message": "", "error": str(e)})
        finally:
            q.put(None)   # sentinel — stream done

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/progress/<job_id>")
def progress_stream(job_id):
    q = _jobs.get(job_id)
    if q is None:
        return jsonify({"error": "unknown job"}), 404

    def generate():
        while True:
            item = q.get()
            if item is None:
                _jobs.pop(job_id, None)
                break
            yield f"data: {json.dumps(item)}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
