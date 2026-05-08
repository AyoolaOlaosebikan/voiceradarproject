"""
Stage 2: Micro-frequency computation (Section IV-B, Algorithm 1).

Paper: VoiceRadar (Kumari et al., NDSS 2025)

Algorithm 1 (page 6):
  Input:  m=0, k=number of unique values in E(x)
  Output: fs = FJ0(0,k) / FJ0(0,1)
  where FJ0(0,n) is the nth zero of the zeroth-order Bessel function J0.

The three micro-frequencies differ only in what they use as n:
  Translational: n = k              (count of unique values)
  Rotational:    n = k * r * sin(θ) (k=count, r=max unique value, θ=rotation angle)
  Vibrational:   n = k * r          (k=count, r=max unique value)

"Unique values" = float-distinct values in the flattened embedding vector,
excluding duplicates. The paper treats these as radii in a concentric circle
model. Divergences from paper are noted inline and in notes/divergences.md.
"""

import numpy as np
from scipy.special import jn_zeros  # zeros of Bessel J_n


# ---------------------------------------------------------------------------
# Algorithm 1 — Bessel zero ratio
# ---------------------------------------------------------------------------

def _bessel_zero(n: int) -> float:
    """Return the nth zero of J0 (1-indexed). Uses scipy.special.jn_zeros."""
    if n < 1:
        # n < 1 can happen with rotational mode when k*r*sin(θ) rounds down
        # to zero. Return 0 as a safe fallback (logged in divergences.md).
        return 0.0
    return float(jn_zeros(0, n)[-1])


def drumhead_frequency(n: int) -> float:
    """
    Algorithm 1: fs = FJ0(0, n) / FJ0(0, 1).

    This is the nth zero of J0 normalised by the fundamental (1st) zero.
    The normalisation removes the unknown physical constants cd and a
    from the drum resonance formula (paper page 7).
    """
    f1 = _bessel_zero(1)   # ≈ 2.4048, the fundamental zero of J0
    fn = _bessel_zero(n)
    return fn / f1


# ---------------------------------------------------------------------------
# Embedding statistics (computed once, shared across all three modes)
# ---------------------------------------------------------------------------

def embedding_stats(E: np.ndarray) -> dict:
    """
    Extract the quantities Algorithm 1 requires from a HuBERT embedding.

    Parameters
    ----------
    E : np.ndarray, shape [T, D]  — one sample's last_hidden_state

    Returns
    -------
    dict with:
      unique_vals : sorted array of distinct float values in E (duplicates removed)
      k           : count of unique values (integer)
      r           : largest unique value = max(unique_vals), sorted ascending
      theta       : rotation angle (see divergence D5 below)
      variance    : var(E), used in Stage 3 Doppler computation
    """
    flat = E.flatten()
    unique_vals = np.unique(flat)   # sorted ascending, duplicates removed

    k = len(unique_vals)            # count of distinct values
    r = float(unique_vals[-1])      # paper: "last value in E(x) sorted ascending"
    variance = float(np.var(flat))

    # D5 — θ (rotation angle): paper states each unique value "may be rotated
    # by angle θ from the x-axis" but never defines how θ is extracted from
    # the embedding. We use θ = arctan(r / k) as a proxy: it is determined
    # entirely by the two quantities the paper does define (r and k), gives a
    # value in (0, π/2), and is geometrically interpretable as the angle
    # subtended by the largest radius at the observer distance k.
    # Logged in notes/divergences.md.
    theta = float(np.arctan2(r, k))  # radians ∈ (0, π/2)

    return dict(
        unique_vals=unique_vals,
        k=k,
        r=r,
        theta=theta,
        variance=variance,
    )


# ---------------------------------------------------------------------------
# Three micro-frequency modes
# ---------------------------------------------------------------------------

def translational_delta_f(stats: dict) -> float:
    """
    Translational: n = k (count of unique values).
    Waves travel straight along the x-axis; critical points cut the x-axis.
    """
    n = stats["k"]
    return drumhead_frequency(n)


def rotational_delta_f(stats: dict) -> float:
    """
    Rotational: n = k * r * sin(θ).
    Waves travel at angle θ from the x-axis; accounts for tangential velocity.

    D6 — n must be a positive integer for jn_zeros. We round to nearest int
    and clamp to ≥1. Logged in notes/divergences.md.
    """
    k     = stats["k"]
    r     = stats["r"]
    theta = stats["theta"]
    n_float = k * r * np.sin(theta)
    n = max(1, round(n_float))
    return drumhead_frequency(n)


def vibrational_delta_f(stats: dict) -> float:
    """
    Vibrational: n = k * r.
    Points oscillate along x-axis; only spatial distribution matters (no θ).

    D6 — same integer clamping as rotational. Logged in notes/divergences.md.
    """
    k = stats["k"]
    r = stats["r"]
    n_float = k * r
    n = max(1, round(n_float))
    return drumhead_frequency(n)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_micro_frequencies(E: np.ndarray) -> dict:
    """
    Compute Δft, Δfr, Δfv and their sum Δf_total = fs for one audio sample.

    Parameters
    ----------
    E : np.ndarray, shape [T, D] or [1, T, D]

    Returns
    -------
    dict with individual Δf values, their sum (= fs fed into Doppler),
    and the intermediate stats needed by Stage 3.
    """
    if E.ndim == 3:
        E = E[0]   # drop batch dim → [T, D]

    stats = embedding_stats(E)

    df_t = translational_delta_f(stats)
    df_r = rotational_delta_f(stats)
    df_v = vibrational_delta_f(stats)
    df_total = df_t + df_r + df_v      # = fs in Doppler equation

    return dict(
        delta_f_translational=df_t,
        delta_f_rotational=df_r,
        delta_f_vibrational=df_v,
        delta_f_total=df_total,         # this is fs
        # stats forwarded to Stage 3
        variance=stats["variance"],
        k=stats["k"],
        r=stats["r"],
        theta=stats["theta"],
    )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    import torchaudio
    import soundfile as sf
    from transformers import HubertModel, AutoFeatureExtractor

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
    print(f"Embedding shape: {E.shape}")

    result = compute_micro_frequencies(E)

    print("\n--- Micro-frequency results ---")
    print(f"  k (unique values):        {result['k']}")
    print(f"  r (max unique value):     {result['r']:.6f}")
    print(f"  theta:                    {result['theta']:.6f} rad")
    print(f"  variance:                 {result['variance']:.6f}")
    print(f"  Δf_translational (fs_t):  {result['delta_f_translational']:.6f}")
    print(f"  Δf_rotational    (fs_r):  {result['delta_f_rotational']:.6f}")
    print(f"  Δf_vibrational   (fs_v):  {result['delta_f_vibrational']:.6f}")
    print(f"  Δf_total         (fs):    {result['delta_f_total']:.6f}")

    # Cross-check: first zero of J0 ≈ 2.4048, so drumhead_frequency(1) == 1.0
    from micro_frequencies import drumhead_frequency
    assert abs(drumhead_frequency(1) - 1.0) < 1e-10, "fundamental must equal 1"
    print("\nSanity check passed: drumhead_frequency(1) == 1.0")
