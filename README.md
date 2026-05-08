# VoiceRadar

A reproduction of **VoiceRadar: Detecting AI-Generated Speech via Micro-Frequency Analysis and Doppler Effect** (Kumari et al., NDSS 2025).

## Paper

> Kumari, N., et al. "VoiceRadar: Detecting AI-Generated Speech via Micro-Frequency Analysis and Doppler Effect."
> *Network and Distributed System Security Symposium (NDSS)*, 2025.

This repository is an independent reproduction of the paper's four-stage pipeline for a university honors project. The original authors did not release code.

## Pipeline

1. **HuBERT embeddings** — audio → `[T, 768]` feature vectors via `facebook/hubert-base-ls960`
2. **Micro-frequency analysis** — Algorithm 1 from the paper: translational, rotational, and vibrational frequency modes derived from Bessel function zeros
3. **Doppler equation** — observed frequency `f_o` computed from embedding variance and predicted label
4. **MLP classifier** — 6-layer network (512→256→128→64→32→1) trained with BCE + physics loss

## Results

| Dataset | Accuracy | TPR | TNR | EER |
|---|---|---|---|---|
| ASVspoof 2019 LA eval (unseen attacks) | 0.865 | 0.997 | 0.733 | — |
| In-the-Wild (Muller et al. 2022) | 0.614 | 0.292 | 0.936 | 38.6% |

Paper reports EER 0.10% on ASVspoof 2019 using `hubert-large` and the full dataset. We use `hubert-base` (768-dim vs 1024-dim) and a 5K subsample due to M2 MacBook hardware constraints.

## Usage

```bash
# 1. Extract embeddings
python extract_embeddings.py   # ASVspoof 2019 LA
python extract_itw.py          # In-the-Wild (optional)

# 2. Precompute micro-frequencies
python precompute_fo.py

# 3. Train
python train.py

# 4. Run web demo
python app.py   # → http://localhost:5000
```

## Datasets

- [ASVspoof 2019](https://datashare.ed.ac.uk/handle/10283/3336)
- [In-the-Wild](https://deepfake-total.com/in_the_wild) (Muller et al. 2022)

## Divergences from Paper

## Stage 1 — HuBERT model

**D1. Model size**
Paper uses `hubert-large-ls960-ft` (1024-dim). We use `hubert-base-ls960` (768-dim)
due to 8GB M2 memory constraint.

---

## Stage 2 — Micro-frequency computation (Algorithm 1)

**What Algorithm 1 actually does (confirmed from paper):**
- `FJ0(0, n)` = the nth zero of the zeroth-order Bessel function J0
  (i.e., `scipy.special.jn_zeros(0, n)[-1]`)
- `fs = FJ0(0, k) / FJ0(0, 1)` — the drumhead frequency is the kth Bessel
  zero divided by the fundamental (1st) zero. This normalises out the unknown
  physical constants cd and a from the drum resonance formula.
- k, r, θ are scalars derived from the embedding (not per-element arrays).
- The three modes each compute one scalar fs via Algorithm 1 with different n:
    Translational: n = k
    Rotational:    n = k · r · sin(θ)
    Vibrational:   n = k · r

**D2. "Unique values" = float-exact distinct values**
The paper is explicit: "we ignore repetitive values" and "k = count of only
distinct values in E(x)." For a continuous float32 HuBERT embedding of shape
[156, 768] this gives k ≈ 119,714 (out of 119,808 total; only 94 duplicates).
We use numpy.unique() on the flattened embedding, which matches the paper's
intent exactly. No binning is needed — that was a misunderstanding from before
reading the paper.

**D3. n must be a positive integer for jn_zeros**
`scipy.special.jn_zeros(0, n)` requires integer n ≥ 1. For rotational
(n = k·r·sin θ) and vibrational (n = k·r), n is a float. We round to the
nearest integer and clamp to ≥ 1. Given k ≈ 119,714 and r ≈ 5.8, these
values are very large integers. jn_zeros handles them correctly.

**D5. Rotation angle θ definition**
Paper states each unique value "may be rotated by angle θ from the x-axis"
but never defines how θ is extracted from the embedding. We use:
  θ = arctan(r / k)
This is determined entirely by the two quantities the paper does define (r and
k), gives θ ∈ (0, π/2), and is geometrically interpretable as the angle
subtended by the largest radius at observer distance k. For our test sample:
θ ≈ 0.000049 rad (very small, since k ≈ 119K >> r ≈ 5.8).

**Consequence of small θ:**
sin(θ) ≈ θ ≈ 4.9×10⁻⁵, so rotational n = k·r·sin(θ) ≈ 119714·5.8·4.9e-5
≈ 34. Rotational Δf is therefore much smaller than translational or
vibrational (which have n = k·r ≈ 694,341 and n = k ≈ 119,714 respectively).
This is physically sensible: the tiny rotation angle produces a small frequency.

---

## Stage 3 — Doppler equation

No divergences. The paper's formula is fully specified:
  f_o = ((cv + vo) / (cv - vs)) * fs
  vo = 0, vs = y·|var(E(x))|, cv = var(E(x))·FJ0(0,1)
All variables are defined and unambiguous.

**Note on y during inference:** vs requires the true label y, which is
unavailable at inference time. During inference, y is replaced by the model's
predicted label F(E(x)) — this is explicit in the paper's loss function
(Equation 2), where fo(x, F(E(x))) uses the predicted label.

---

## Stage 4 — MLP training

**D6. Mean pooling for MLP input**
The paper feeds E(x) into the MLP but doesn't specify how a variable-length
[T, 768] sequence becomes a fixed-size vector. We use mean pooling over the T
time frames → [768]. This is the standard approach for HuBERT-based classifiers.

**D7. Physics term normalization**
The paper's loss (Equation 2) is:
  loss = BCE + 0.6 * (fo(true) - fo(predicted))
fo values are in the range ~200K–6M. Without normalization, the physics term
dominates BCE by ~6 orders of magnitude and the model cannot learn (dev_acc
stuck at 0.5, confirmed experimentally). We normalize by dividing by fo_true:
  physics_term = mean((fo_true - fo_pred) / fo_true)
This makes the term dimensionless and O(1), matching the scale of BCE.
The paper trained on 4× NVIDIA RTX 8000 (48GB each) where batch statistics
may have naturally kept this term bounded; our single-sample scale exposes
the issue.

**D8. fo_pred uses continuous pred, not rounded label**
Equation 2 uses F(E(x)) (the model's output class) to compute fo_predicted.
Rounding pred to 0/1 would make the loss non-differentiable. We use the raw
sigmoid output as a continuous y substitute so gradients flow through the
physics term.

**D9. Evaluation on unseen attack types (eval split)**
The paper evaluates on ASVspoof 2019 but does not specify whether it uses
the dev or eval split for its Table IV numbers. We evaluate on the eval split
(attack types A11–A19), which is the standard benchmark split and provides a
true out-of-distribution test. Dev accuracy (~1.000) is inflated by same-attack
familiarity and is not reported as a primary result.

**D10. Dataset subsample**
Paper trains on the full ASVspoof 2019 LA dataset (~25K train files). We use
5000 train / 2000 dev / 3000 eval due to M2 memory and time constraints.
This likely accounts for a portion of the gap between our eval accuracy (0.865)
and the paper's reported near-zero EER.
