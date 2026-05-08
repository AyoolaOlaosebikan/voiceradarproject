"""
Stage 3: Wave-based analysis — Doppler equation (Section IV-B).

Paper: VoiceRadar (Kumari et al., NDSS 2025)

  f_o = ((cv + vo) / (cv - vs)) * fs

  vo = 0
  vs = y * abs(var(E(x)))
  cv = var(E(x)) * FJ0(0, 1)
  fs = delta_f_total from Stage 2

FJ0(0,1) is the first zero of J0 ≈ 2.4048.
"""

from scipy.special import jn_zeros

FJ0_0_1 = float(jn_zeros(0, 1)[-1])   # ≈ 2.4048, constant across all samples


def compute_fo(fs: float, variance: float, y: float) -> float:
    """
    Compute the observed frequency f_o for one audio sample.

    Parameters
    ----------
    fs       : delta_f_total from Stage 2 (source frequency)
    variance : var(E(x)) from Stage 2 stats
    y        : label — 0 for human, 1 for AI-generated

    Returns
    -------
    f_o : observed frequency (float)
    """
    vo = 0.0
    vs = y * abs(variance)
    cv = variance * FJ0_0_1

    return ((cv + vo) / (cv - vs)) * fs


if __name__ == "__main__":
    import torch
    import torchaudio
    import soundfile as sf
    from transformers import HubertModel, AutoFeatureExtractor
    from micro_frequencies import compute_micro_frequencies

    MODEL_NAME = "facebook/hubert-base-ls960"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model     = HubertModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    waveform_np, sr = sf.read("absolute_end.wav")
    waveform = torch.from_numpy(waveform_np).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    inputs = extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs.input_values.to(device))

    E = outputs.last_hidden_state.cpu().numpy()
    mf = compute_micro_frequencies(E)

    fs       = mf["delta_f_total"]
    variance = mf["variance"]

    fo_human = compute_fo(fs, variance, y=0.0)
    fo_ai    = compute_fo(fs, variance, y=1.0)

    print(f"\nFJ0(0,1)  = {FJ0_0_1:.6f}")
    print(f"variance  = {variance:.6f}")
    print(f"cv        = {variance * FJ0_0_1:.6f}")
    print(f"fs        = {fs:.4f}")
    print(f"\nf_o (y=0, human) = {fo_human:.6f}")
    print(f"f_o (y=1, AI)    = {fo_ai:.6f}")
    print(f"\nRatio fo(AI)/fo(human) = {fo_ai/fo_human:.6f}")
