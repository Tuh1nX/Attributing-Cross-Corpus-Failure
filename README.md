<div align="center">

# ShiftLens

### Topic Shift and Label Noise: A Factorial Intervention Study of Performance Drops in Bangla Hate-Speech Detection

[![Paper](https://img.shields.io/badge/paper-under%20review-lightgrey?style=flat-square)](#citation)
[![Dataset](https://img.shields.io/badge/dataset-Zenodo-2ea44f?style=flat-square)](https://doi.org/10.5281/zenodo.21848661)
[![License](https://img.shields.io/badge/data-CC%20BY%204.0-blue?style=flat-square)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/python-3.13-3776ab?style=flat-square&logo=python&logoColor=white)](#quick-start)
[![PyTorch](https://img.shields.io/badge/PyTorch%20%7C%20Transformers-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](#quick-start)
[![Language](https://img.shields.io/badge/language-Bangla-orange?style=flat-square)](#dataset-youtube-bn)

<br>

<img width="1033" height="672" alt="image" src="https://github.com/user-attachments/assets/9a2c0c2f-bdea-46e7-abc8-a354f7364527" />

<sub><i>Pipeline: one base corpus → declared content and protocol operators → 2×2 factorial → 10 model arms → exact Shapley attribution</i></sub>

</div>

---

## Overview

A model trained on one corpus usually does worse on another, but a cross-corpus drop mixes many things at once: topic, platform, annotation practice, label definitions, class balance. It does not tell you *which* of them caused the drop.

**ShiftLens** takes a narrower, answerable question. We apply two *declared* interventions to a single Bangla corpus and measure how much each one costs:

| Intervention | What changes | What stays fixed |
|---|---|---|
| **Content** (topic shift) | Class-preserving re-sampling, 3× weight on Religious / Geopolitical / Gender Abusive | Labels, class counts, sample sizes |
| **Protocol** (label shift) | Training labels pass through `T = [[0.90, 0.10], [0.20, 0.80]]` | Texts; test labels stay clean |

Every cell of the 2×2 design is measured by retraining, and the Macro-F1 drop is split with **exact two-factor Shapley values**, with no causal graph or density-ratio assumption needed.

> `T` is declared by us as a stress test. It is **not** an estimate of real annotator noise.

<div align="center">
<img width="6870" height="3210" alt="fig_attribution_map" src="https://github.com/user-attachments/assets/f98b415a-86ec-450a-a251-7875710447d9" />
<br><sub><i>Per-arm attribution: content (x) vs protocol (y). Points above the diagonal are protocol-dominated.</i></sub>
</div>

---

## Key Findings

| # | Finding |
|:-:|---|
| 1 | **Protocol dominates content.** Every label-learning arm loses **0.018–0.021** Macro-F1 to the label shift, while the topic shift costs **≤ 0.002**. |
| 2 | **Not a single-model artefact.** The pattern holds for XLM-R, BanglaBERT, mDeBERTa-v3 and TF-IDF, and under all six metrics for every label-learning arm. |
| 3 | **Significant where well powered.** TF-IDF, 20 seeds: Holm-corrected `p = 4×10⁻⁶`, `d_z = 3.65`, 20/20 seeds. Neural arms use 5 seeds, so their evidence is descriptive (intervals and sign consistency). |
| 4 | **Correction does not help.** DANN, CORAL and forward correction (given the *true* `T`) do not recover the loss. |
| 5 | **Synthetic noise is a worst-case bound.** On English HateXplain, real annotator disagreement is about **2.5× less harmful** than synthetic symmetric noise at a similar flip rate. |
| 6 | **Controlled drops are much smaller than the real gap.** Synthetic operators cost at most ~0.022 Macro-F1, while YouTube-BN → BD-SHS drops by ~0.20. |
| 7 | **No label leakage.** The zero-shot LLM never sees training labels, so its `φ_protocol` is exactly `0.0`, a null control for the pipeline. |

### Factorial summary

`φ_content` and `φ_protocol` are Macro-F1 points attributed to each factor. A positive value means switching the factor on lowers performance. Interaction `I` stays within ±0.004 everywhere.

| Arm | Seeds | Clean Macro-F1 | `φ_content` | `φ_protocol` |
|---|:-:|:-:|:-:|:-:|
| Majority baseline | 20 | 0.340 | 0.0000 | 0.0000 |
| TF-IDF logistic regression | 20 | 0.771 | 0.0018 | **0.0213** |
| ERM (XLM-RoBERTa) | 5 | 0.933 | 0.0017 | **0.0194** |
| DANN | 5 | 0.931 | 0.0003 | **0.0190** |
| CORAL | 5 | 0.933 | −0.0009 | **0.0182** |
| Forward correction (oracle `T`) | 5 | 0.931 | 0.0004 | **0.0199** |
| ERM (BanglaBERT) | 5 | 0.934 | 0.0001 | **0.0187** |
| ERM (mDeBERTa-v3) | 5 | 0.933 | −0.0002 | **0.0207** |
| Qwen2.5-1.5B zero-shot | 5 | 0.341 | 0.0006 | 0.0000 |
| Qwen2.5-1.5B few-shot | 5 | 0.624 | −0.0036 | 0.0015 |

Primary metric: **Macro-F1**. Also reported: F1-positive, balanced accuracy, AUROC, AUPRC and ECE.

<details>
<summary><b>More figures</b></summary>
<br>
<div align="center">

<img width="4281" height="1605" alt="factorial_macro_f1" src="https://github.com/user-attachments/assets/26f17a6f-0693-4e42-96d7-3d46c827ff41" />
<br><sub>Complete 2×2 factorial</sub>
<br><br>
<img width="4279" height="2223" alt="fig_leaderboard_reversal" src="https://github.com/user-attachments/assets/7f61990e-e46c-4f13-b3d5-9b0fa4bf8eaf" />
<br><sub>Rank change between the clean (0,0) and fully shifted (1,1) cells (Kendall τ = 0.76, 95% CI [0.66, 0.86]). The small changes suggest practical ties, not a robust reversal.</sub>
<img width="7168" height="2555" alt="fig_model_panel_macro_f1" src="https://github.com/user-attachments/assets/608533e0-1f7f-4c73-acf6-5ed01b239c9e" />
<br><sub>Macro-F1 across all ten arms</sub>
<br><br>

<img width="2318" height="1834" alt="fig_robustness_curve" src="https://github.com/user-attachments/assets/860028c5-1b6f-442e-a34d-9fae9f513b62" />
<br><sub>Robustness to intervention strength: φ_protocol rises with noise scale, φ_content stays near zero</sub>
<br><br>

<img width="4281" height="3033" alt="shapley_contributions" src="https://github.com/user-attachments/assets/f9390eae-8676-41fb-ad0a-e26abaf5c34f" />
<br><sub>Shapley contributions</sub>
<br><br>



</div>
</details>

---

## Experimental Design

**Models (10 arms)**

| Group | Arms | Seeds |
|---|---|:-:|
| Baselines | Majority, TF-IDF logistic regression | 20 |
| Encoders | ERM (XLM-R), DANN, CORAL, Forward correction, ERM (BanglaBERT), ERM (mDeBERTa-v3) | 5 |
| Small LLM | Qwen2.5-1.5B-Instruct, zero-shot and few-shot (6 exemplars) | 5 |

**Statistics.** Percentile bootstrap CIs over seeds (5,000 resamples) and exact paired sign-flip tests. With 5 seeds the smallest attainable two-sided `p` is 0.0625, so neural-arm results are reported as intervals and sign consistency, and significance rests on the 20-seed TF-IDF arm.

---

## Dataset: YouTube-BN

| | |
|---|---|
| **Size** | 67,982 Bangla social-media comments |
| **Labels** | 6 topical labels, binarised to abusive / non-abusive (positive rate 0.5155) |
| **Classes** | Non-abusive 32,938 · Religious 9,155 · Personal 9,057 · Political 7,006 · Geopolitical 5,746 · Gender Abusive 4,080 |
| **Collection** | Mainly YouTube Data API v3 (May–July 2026), plus a small set from Facebook, X/Twitter and Instagram. Public content only. |
| **Split** | Hash-grouped 0.55 / 0.15 / 0.10 / 0.15 / 0.05 (train / val / adapt / test / reserve). Train 37,390, test 10,197. Model selection uses validation only. |
| **Privacy** | Usernames, channel IDs and direct identifiers removed. Comment IDs kept for take-down requests. |
| **Release** | [Zenodo 10.5281/zenodo.21848661](https://doi.org/10.5281/zenodo.21848661), CC BY 4.0 (texts, labels, split manifests, code) |

**Annotation.** Two Bangla speakers labelled each comment jointly and settled uncertain cases by discussion. There was no independent double coding, so **no agreement statistic is reported**.

**Label audits** (review candidates, not an error rate):
- 67,981 unique texts among 67,982 rows; the one duplicate pair has the same label.
- 5-fold out-of-fold TF-IDF disagreement: 2.45% (confidence ≥ 0.8), 0.74% (confidence ≥ 0.9).

**Leakage.** Only exact-text duplicates are removed; paraphrases and transliterations are not.

---

## Scope and Limitations

This study does **not** claim that:

- protocol is universally more important than content. It holds only for these operators, strengths and this corpus;
- `T` models real annotator noise. It is a declared stress test;
- the YouTube-BN → BD-SHS drop is a platform-only effect. It changes many things at once;
- the results causally explain arbitrary cross-corpus degradation;
- the few-shot LLM arm supports the main effect. It is a boundary condition;
- anything about hosted-API LLMs (Gemini, GPT, etc.).

Other limitations: single split, binarised target, labels without independent double annotation, TF-IDF-only controls, and a second-language replication on HateXplain only (not a full factorial). See Section 9 of the paper.

---

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers tokenizers scikit-learn pandas numpy matplotlib seaborn \
            scipy jupyter ipykernel tqdm joblib
```

### Sanity check on result tables

Run after loading `FACTORIAL_RESULTS` and `SHAPLEY`:

```python
expected = {"MAJORITY": 20, "TFIDF_LOGREG": 20, "ERM": 5, "DANN": 5,
            "CORAL": 5, "PROTOCOL_CORRECTED": 5,
            "ERM_BANGLABERT": 5, "ERM_MDEBERTA_V3": 5,
            "LLM_ZEROSHOT": 5, "LLM_FEWSHOT": 5}

counts = FACTORIAL_RESULTS.groupby("model").size().to_dict()
assert counts == {k: 4 * v for k, v in expected.items()}          # 4 cells per seed
assert FACTORIAL_RESULTS[["model", "seed", "content_on", "protocol_on"]].duplicated().sum() == 0
assert (SHAPLEY.query("model == 'LLM_ZEROSHOT'")["phi_protocol"] == 0.0).all()  # leakage null control
```

---

## Citation

The paper is under review. Until it is published, please cite the dataset:

```bibtex
@dataset{tuhin_2026_21848661,
  author       = {Tuhin, Ashabul Yamin Tuhin},
  title        = {YouTube-BN},
  month        = aug,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.21848661},
  url          = {https://doi.org/10.5281/zenodo.21848661},
}
```

## Author

**Ashabul Yamin Tuhin**, Department of CSE, International Islamic University Chittagong, Bangladesh
 c233188@ugrad.iiuc.ac.bd

## Ethics

The corpus contains abusive and hateful language. It should **not** be used to automatically sanction individuals; label noise and the binary mapping limit its suitability for such use.
