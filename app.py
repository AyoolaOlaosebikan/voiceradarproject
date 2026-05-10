"""
VoiceRadar web demo — local Flask app.
Upload an MP3/WAV/FLAC, record your voice, generate AI speech via OpenRouter TTS,
or convert your voice via ElevenLabs Speech-to-Speech — all run through the detector.

Env vars (put in .env and load with python-dotenv, or export manually):
  OPENROUTER_API_KEY   — for TTS
  ELEVENLABS_API_KEY   — for voice conversion
"""

import base64
import io
import json
import os
import queue
import tempfile
import threading
import subprocess
import requests
import numpy as np
import torch
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string, Response, stream_with_context
from transformers import HubertModel, AutoFeatureExtractor

from micro_frequencies import compute_micro_frequencies
from doppler import compute_fo
from train import VoiceRadarMLP

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel

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

def load_audio(audio_bytes: bytes, ext: str) -> "torch.Tensor":
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

    print(f"[debug] waveform shape={waveform.shape} duration={waveform.shape[-1]/16000:.2f}s "
          f"E shape={E.shape} prob={prob:.4f}")

    label      = "Human" if prob >= 0.5 else "AI-Generated"
    confidence = prob if prob >= 0.5 else 1.0 - prob

    progress(100, "Done", result={
        "label": label,
        "confidence": round(confidence * 100, 1),
        "raw": round(prob, 4),
    })

# ---------------------------------------------------------------------------
# TTS via OpenRouter
# ---------------------------------------------------------------------------

def generate_tts(text: str) -> bytes:
    """Call ElevenLabs TTS, return MP3 bytes."""
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY not set")

    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "output_format": "mp3_44100_128",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Voice conversion via ElevenLabs Speech-to-Speech
# ---------------------------------------------------------------------------

def convert_voice(audio_bytes: bytes, ext: str) -> bytes:
    """Send audio to ElevenLabs STS, return MP3 bytes of converted voice."""
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY not set")

    # Convert to WAV first so ElevenLabs gets a clean format
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
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)

    resp = requests.post(
        f"https://api.elevenlabs.io/v1/speech-to-speech/{ELEVENLABS_VOICE_ID}",
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
        data={
            "model_id": "eleven_english_sts_v2",
            "output_format": "mp3_44100_128",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content  # MP3 bytes

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
    align-items: flex-start;
    justify-content: center;
    padding: 40px 16px;
  }

  .container { width: 100%; max-width: 520px; display: flex; flex-direction: column; gap: 16px; }

  .card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 16px;
    padding: 32px 28px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }

  h1 { font-size: 1.6rem; font-weight: 700; color: #fff; }
  .subtitle { font-size: 0.85rem; color: #666; margin-top: 4px; margin-bottom: 0; }

  h2 { font-size: 0.95rem; font-weight: 600; color: #aaa; margin-bottom: 16px; letter-spacing: 0.05em; text-transform: uppercase; }

  /* tabs */
  .tabs { display: flex; gap: 8px; margin-bottom: 20px; }
  .tab {
    flex: 1; padding: 8px; border-radius: 8px; border: 1px solid #2a2d3a;
    background: transparent; color: #888; font-size: 0.82rem; cursor: pointer;
    transition: all 0.2s;
  }
  .tab.active { background: #5b6af0; border-color: #5b6af0; color: #fff; font-weight: 600; }
  .tab:hover:not(.active) { border-color: #5b6af0; color: #ccc; }

  .panel { display: none; }
  .panel.active { display: block; }

  /* drop zone */
  .drop-zone {
    border: 2px dashed #2a2d3a; border-radius: 12px; padding: 32px 20px;
    text-align: center; cursor: pointer; transition: border-color 0.2s, background 0.2s;
    position: relative;
  }
  .drop-zone:hover, .drop-zone.drag-over { border-color: #5b6af0; background: rgba(91,106,240,0.05); }
  .drop-zone input[type="file"] { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
  .drop-icon { font-size: 1.8rem; margin-bottom: 10px; }
  .drop-text { font-size: 0.88rem; color: #888; }
  .drop-text span { color: #5b6af0; }
  .file-name { margin-top: 10px; font-size: 0.8rem; color: #5b6af0; min-height: 16px; }

  /* record */
  .record-controls { display: flex; gap: 10px; align-items: center; }
  .rec-btn {
    width: 48px; height: 48px; border-radius: 50%; border: none; cursor: pointer;
    font-size: 1.2rem; transition: all 0.2s; flex-shrink: 0;
  }
  .rec-btn.idle    { background: #e53e3e; }
  .rec-btn.recording { background: #2a2d3a; animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(229,62,62,0.4); } 50% { box-shadow: 0 0 0 8px rgba(229,62,62,0); } }
  .rec-status { font-size: 0.85rem; color: #888; }
  audio.preview { width: 100%; margin-top: 14px; border-radius: 8px; display: none; }

  /* tts */
  textarea {
    width: 100%; background: #12141c; border: 1px solid #2a2d3a; border-radius: 10px;
    color: #e0e0e0; font-size: 0.88rem; padding: 12px; resize: vertical; min-height: 80px;
    font-family: inherit;
  }
  textarea:focus { outline: none; border-color: #5b6af0; }
  .tts-audio { margin-top: 12px; display: none; }
  .tts-audio audio { width: 100%; border-radius: 8px; }

  /* voice convert */
  .vc-note { font-size: 0.8rem; color: #666; margin-bottom: 14px; }

  /* button */
  .btn {
    width: 100%; margin-top: 16px; padding: 13px;
    background: #5b6af0; color: #fff; border: none;
    border-radius: 10px; font-size: 0.92rem; font-weight: 600;
    cursor: pointer; transition: background 0.2s, opacity 0.2s;
  }
  .btn:hover { background: #4a58d4; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn.secondary { background: #2a2d3a; }
  .btn.secondary:hover { background: #363a4a; }

  /* progress */
  .progress-section { display: none; margin-top: 20px; }
  .progress-label { font-size: 0.82rem; color: #888; margin-bottom: 8px; display: flex; justify-content: space-between; }
  .progress-track { background: #2a2d3a; border-radius: 99px; height: 6px; overflow: hidden; }
  .progress-fill { height: 100%; border-radius: 99px; background: #5b6af0; width: 0%; transition: width 0.4s ease; }
  .progress-step { margin-top: 10px; font-size: 0.78rem; color: #555; min-height: 16px; }

  /* result */
  .result { display: none; margin-top: 24px; padding: 22px; border-radius: 12px; text-align: center; }
  .result.human { background: rgba(52,199,89,0.08);  border: 1px solid rgba(52,199,89,0.25); }
  .result.ai    { background: rgba(255,69,58,0.08);  border: 1px solid rgba(255,69,58,0.25); }
  .result-emoji  { font-size: 2.2rem; margin-bottom: 8px; }
  .result-label  { font-size: 1.25rem; font-weight: 700; margin-bottom: 6px; }
  .result.human .result-label { color: #34c759; }
  .result.ai    .result-label { color: #ff453a; }
  .result-sub { font-size: 0.82rem; color: #888; margin-bottom: 14px; }
  .conf-track { background: #2a2d3a; border-radius: 99px; height: 8px; overflow: hidden; }
  .conf-fill  { height: 100%; border-radius: 99px; transition: width 0.6s ease; }
  .result.human .conf-fill { background: #34c759; }
  .result.ai    .conf-fill { background: #ff453a; }
  .conf-labels { display: flex; justify-content: space-between; font-size: 0.75rem; color: #666; margin-top: 6px; }
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="card">
    <h1>VoiceRadar</h1>
    <p class="subtitle">Detects AI-generated speech using micro-frequency analysis &amp; Doppler effect</p>
  </div>

  <!-- Input panel -->
  <div class="card">
    <div class="tabs">
      <button class="tab active" onclick="switchTab('upload')">Upload</button>
      <button class="tab" onclick="switchTab('record')">Record</button>
      <button class="tab" onclick="switchTab('tts')">AI Speech</button>
      <button class="tab" onclick="switchTab('vc')">Voice Convert</button>
    </div>

    <!-- Upload -->
    <div class="panel active" id="panel-upload">
      <div class="drop-zone" id="dropZone">
        <input type="file" id="fileInput" accept=".mp3,.wav,.flac">
        <div class="drop-icon">🎙️</div>
        <div class="drop-text">Drop an audio file or <span>browse</span></div>
        <div class="drop-text" style="margin-top:4px;font-size:0.75rem;">MP3 · WAV · FLAC</div>
        <div class="file-name" id="fileName"></div>
      </div>
      <button class="btn" id="uploadBtn" disabled onclick="analyzeUpload()">Analyze</button>
    </div>

    <!-- Record -->
    <div class="panel" id="panel-record">
      <div class="record-controls">
        <button class="rec-btn idle" id="recBtn" onclick="toggleRecord()">🎤</button>
        <span class="rec-status" id="recStatus">Click to start recording</span>
      </div>
      <audio class="preview" id="recPreview" controls></audio>
      <button class="btn" id="recAnalyzeBtn" disabled onclick="analyzeRecording()">Analyze Recording</button>
    </div>

    <!-- TTS -->
    <div class="panel" id="panel-tts">
      <textarea id="ttsText" placeholder="Type something to generate AI speech...">Hello, this is an AI-generated voice speaking to you right now.</textarea>
      <button class="btn secondary" id="ttsGenBtn" onclick="generateTTS()">Generate AI Speech</button>
      <div class="tts-audio" id="ttsAudioWrap">
        <audio id="ttsAudio" controls></audio>
        <button class="btn" id="ttsAnalyzeBtn" onclick="analyzeTTS()">Analyze This Audio</button>
      </div>
    </div>

    <!-- Voice Convert -->
    <div class="panel" id="panel-vc">
      <p class="vc-note">Record your voice, convert it to an AI voice via ElevenLabs, then analyze.</p>
      <div class="record-controls">
        <button class="rec-btn idle" id="vcRecBtn" onclick="toggleVCRecord()">🎤</button>
        <span class="rec-status" id="vcRecStatus">Click to start recording</span>
      </div>
      <audio class="preview" id="vcRecPreview" controls></audio>
      <button class="btn secondary" id="vcConvertBtn" disabled onclick="convertVoice()">Convert Voice</button>
      <div class="tts-audio" id="vcConvertedWrap">
        <p style="font-size:0.8rem;color:#888;margin-bottom:8px;">Converted audio:</p>
        <audio id="vcConvertedAudio" controls></audio>
        <button class="btn" id="vcAnalyzeBtn" onclick="analyzeConverted()">Analyze Converted Audio</button>
      </div>
    </div>
  </div>

  <!-- Progress + Result (shared) -->
  <div class="card" id="outputCard" style="display:none;">
    <div class="progress-section" id="progressSection">
      <div class="progress-label">
        <span>Analyzing...</span><span id="progressPct">0%</span>
      </div>
      <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
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

</div>

<script>
// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t, i) => {
    const names = ['upload','record','tts','vc'];
    t.classList.toggle('active', names[i] === name);
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
}

// ---------------------------------------------------------------------------
// Shared: submit audio bytes to /upload and stream progress
// ---------------------------------------------------------------------------
async function submitAudio(blob, filename) {
  document.getElementById('outputCard').style.display = 'block';
  document.getElementById('progressSection').style.display = 'block';
  document.getElementById('result').style.display = 'none';
  setProgress(0, 'Uploading...');

  const form = new FormData();
  form.append('file', blob, filename);
  const uploadRes = await fetch('/upload', { method: 'POST', body: form });
  if (!uploadRes.ok) {
    const err = await uploadRes.json();
    alert('Upload error: ' + (err.error || uploadRes.status));
    return;
  }
  const { job_id } = await uploadRes.json();

  const evtSource = new EventSource('/progress/' + job_id);
  evtSource.onmessage = e => {
    const data = JSON.parse(e.data);
    setProgress(data.pct, data.message);
    if (data.result) {
      evtSource.close();
      document.getElementById('progressSection').style.display = 'none';
      showResult(data.result);
    }
    if (data.error) {
      evtSource.close();
      document.getElementById('progressSection').style.display = 'none';
      alert('Error: ' + data.error);
    }
  };
}

function setProgress(pct, msg) {
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressPct').textContent  = pct + '%';
  document.getElementById('progressStep').textContent = msg;
}

function showResult(data) {
  const isHuman = data.label === 'Human';
  const r = document.getElementById('result');
  r.className = 'result ' + (isHuman ? 'human' : 'ai');
  document.getElementById('resultEmoji').textContent = isHuman ? '✅' : '🤖';
  document.getElementById('resultLabel').textContent = data.label;
  document.getElementById('resultSub').textContent   = data.confidence + '% confidence';
  document.getElementById('confFill').style.width    = data.confidence + '%';
  document.getElementById('confLabel').textContent   = data.confidence + '%';
  r.style.display = 'block';
}

// ---------------------------------------------------------------------------
// Upload panel
// ---------------------------------------------------------------------------
const fileInput  = document.getElementById('fileInput');
const dropZone   = document.getElementById('dropZone');
let uploadedFile = null;

fileInput.addEventListener('change', () => {
  uploadedFile = fileInput.files[0];
  if (uploadedFile) {
    document.getElementById('fileName').textContent = uploadedFile.name;
    document.getElementById('uploadBtn').disabled   = false;
  }
});
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  uploadedFile = e.dataTransfer.files[0];
  if (uploadedFile) {
    document.getElementById('fileName').textContent = uploadedFile.name;
    document.getElementById('uploadBtn').disabled   = false;
  }
});

async function analyzeUpload() {
  if (!uploadedFile) return;
  await submitAudio(uploadedFile, uploadedFile.name);
}

// ---------------------------------------------------------------------------
// Record panel
// ---------------------------------------------------------------------------
let mediaRecorder = null;
let recChunks     = [];
let recBlob       = null;

async function toggleRecord() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => recChunks.push(e.data);
  mediaRecorder.onstop = () => {
    recBlob = new Blob(recChunks, { type: 'audio/webm' });
    const url = URL.createObjectURL(recBlob);
    const preview = document.getElementById('recPreview');
    preview.src = url;
    preview.style.display = 'block';
    document.getElementById('recAnalyzeBtn').disabled = false;
    document.getElementById('recBtn').className = 'rec-btn idle';
    document.getElementById('recBtn').textContent = '🎤';
    document.getElementById('recStatus').textContent = 'Recording saved. Click analyze or re-record.';
    stream.getTracks().forEach(t => t.stop());
  };
  mediaRecorder.start();
  document.getElementById('recBtn').className = 'rec-btn recording';
  document.getElementById('recBtn').textContent = '⏹';
  document.getElementById('recStatus').textContent = 'Recording... click to stop';
}

async function analyzeRecording() {
  if (!recBlob) return;
  await submitAudio(recBlob, 'recording.webm');
}

// ---------------------------------------------------------------------------
// TTS panel
// ---------------------------------------------------------------------------
let ttsBlob = null;

async function generateTTS() {
  const text = document.getElementById('ttsText').value.trim();
  if (!text) return;
  const btn = document.getElementById('ttsGenBtn');
  btn.disabled = true;
  btn.textContent = 'Generating...';
  try {
    const resp = await fetch('/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!resp.ok) {
      const err = await resp.json();
      alert('TTS error: ' + (err.error || resp.status));
      return;
    }
    const audioBytes = await resp.arrayBuffer();
    ttsBlob = new Blob([audioBytes], { type: 'audio/mpeg' });
    const url = URL.createObjectURL(ttsBlob);
    document.getElementById('ttsAudio').src = url;
    document.getElementById('ttsAudioWrap').style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate AI Speech';
  }
}

async function analyzeTTS() {
  if (!ttsBlob) return;
  await submitAudio(ttsBlob, 'tts.mp3');
}

// ---------------------------------------------------------------------------
// Voice Convert panel
// ---------------------------------------------------------------------------
let vcMediaRecorder = null;
let vcChunks        = [];
let vcBlob          = null;

async function toggleVCRecord() {
  if (vcMediaRecorder && vcMediaRecorder.state === 'recording') {
    vcMediaRecorder.stop();
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  vcChunks = [];
  vcMediaRecorder = new MediaRecorder(stream);
  vcMediaRecorder.ondataavailable = e => vcChunks.push(e.data);
  vcMediaRecorder.onstop = () => {
    vcBlob = new Blob(vcChunks, { type: 'audio/webm' });
    const url = URL.createObjectURL(vcBlob);
    const preview = document.getElementById('vcRecPreview');
    preview.src = url;
    preview.style.display = 'block';
    document.getElementById('vcConvertBtn').disabled = false;
    document.getElementById('vcRecBtn').className = 'rec-btn idle';
    document.getElementById('vcRecBtn').textContent = '🎤';
    document.getElementById('vcRecStatus').textContent = 'Recorded. Click convert to run through ElevenLabs.';
    stream.getTracks().forEach(t => t.stop());
  };
  vcMediaRecorder.start();
  document.getElementById('vcRecBtn').className = 'rec-btn recording';
  document.getElementById('vcRecBtn').textContent = '⏹';
  document.getElementById('vcRecStatus').textContent = 'Recording... click to stop';
}

let vcConvertedBlob = null;

async function convertVoice() {
  if (!vcBlob) return;
  const btn = document.getElementById('vcConvertBtn');
  btn.disabled = true;
  btn.textContent = 'Converting...';
  document.getElementById('vcConvertedWrap').style.display = 'none';

  try {
    const form = new FormData();
    form.append('file', vcBlob, 'voice.webm');
    const resp = await fetch('/voice-convert', { method: 'POST', body: form });
    if (!resp.ok) {
      const err = await resp.json();
      alert('Voice conversion error: ' + (err.error || resp.status));
      return;
    }
    const mp3Bytes = await resp.arrayBuffer();
    vcConvertedBlob = new Blob([mp3Bytes], { type: 'audio/mpeg' });
    const url = URL.createObjectURL(vcConvertedBlob);
    document.getElementById('vcConvertedAudio').src = url;
    document.getElementById('vcConvertedWrap').style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Convert Voice';
  }
}

async function analyzeConverted() {
  if (!vcConvertedBlob) return;
  await submitAudio(vcConvertedBlob, 'converted.mp3');
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
    if ext not in ("mp3", "wav", "flac", "webm", "ogg", "m4a"):
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
            q.put(None)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "no text"}), 400
    try:
        mp3_bytes = generate_tts(text)
        return Response(mp3_bytes, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/voice-convert", methods=["POST"])
def voice_convert():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f   = request.files["file"]
    ext = f.filename.rsplit(".", 1)[-1].lower()
    audio_bytes = f.read()
    try:
        mp3_bytes = convert_voice(audio_bytes, ext)
        return Response(mp3_bytes, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
