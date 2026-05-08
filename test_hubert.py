import torch
import torchaudio
from transformers import HubertModel, AutoFeatureExtractor
import soundfile as sf

MODEL_NAME = "facebook/hubert-base-ls960"

# M2 device detection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# Load model + feature extractor (auto-downloads ~1.2GB on first run)
extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = HubertModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Load your test audio
waveform_np, sr = sf.read("absolute_end.wav")
waveform = torch.from_numpy(waveform_np).float()
if waveform.ndim == 1:
    waveform = waveform.unsqueeze(0)  # shape: [1, samples]
else:
    waveform = waveform.T  # soundfile returns [samples, channels], we want [channels, samples]
# HuBERT requires 16kHz mono
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Extract embeddings
inputs = extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    outputs = model(inputs.input_values.to(device))

embeddings = outputs.last_hidden_state
print(f"Embedding shape: {embeddings.shape}")
print(f"First few values: {embeddings[0, 0, :5]}")