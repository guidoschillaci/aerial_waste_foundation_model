# Foundation Model Benchmark for illegal landfill detection

**Illegal landfill detection with Prithvi-EO-2.0, benchmarked against the original AerialWaste ResNet50+FPN baseline.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: CC BY-NC-ND 4.0](https://img.shields.io/badge/Data-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## Overview

This repository benchmarks Prithvi-EO-2.0 (IBM/NASA geospatial foundation model) against the original [AerialWaste](https://aerialwaste.org/) ResNet50+FPN baseline (Torres & Fraternali, *Scientific Data* 2023) on binary illegal landfill detection. Four Prithvi configurations are evaluated:

- **Linear probe 300M** — backbone frozen, only classification head trained
- **Linear probe 600M** — backbone frozen, only classification head trained
- **LoRA 300M** — LoRA adapters injected into attention layers (~1.3% trainable parameters)
- **LoRA 600M** — LoRA adapters injected into attention layers (~1.3% trainable parameters)

The baseline uses the authors' original checkpoint (CC BY-NC-ND 4.0), evaluated directly on our test split.

---

## Dataset

[AerialWaste](https://aerialwaste.org/) — illegal landfill discovery from aerial and satellite imagery.

| Property | Value |
|---|---|
| Total images | 11,703 |
| Positive (waste sites) | 3,478 |
| Negative | 6,956+ |
| Sources | AGEA orthophotos (~20 cm GSD), WorldView-3 (~30 cm GSD), Google Earth (~50 cm GSD) |
| Annotations | Binary labels, multi-label waste types (15 classes), segmentation masks (subset) |
| Region | Lombardy, Italy |

Google Earth RGB images have been originally downloaded using the Google Maps API. Their use must respect the Google Earth [terms and conditions](https://about.google/brand-resource-center/products-and-services/geo-guidelines/).

### Download

```bash
python scripts/download_dataset.py --output data/raw

# Annotations only (no images):
python scripts/download_dataset.py --output data/raw --annotations-only

# Manual download: https://zenodo.org/records/12607190
```

---

## Results

All models evaluated on the same 70/15/15 stratified split. AP = Average Precision (area under PR curve).

| Model | AP | F1 | Precision | Recall | Params trained |
|---|---|---|---|---|---|
| ResNet50+FPN (Torres & Fraternali 2023)¹ | 87.99% | 80.70% | 81.89% | 79.54% | 100% |
| ResNet50+FPN (paper checkpoint, our split)² | 89.45% | 89.87% | 86.17% | 93.90% | 100% |
| Prithvi-300M linear probe | 83.90% | 75.84% | 63.03% | 95.17% | ~0% (head only) |
| Prithvi-600M linear probe | 84.62% | 76.31% | 64.16% | 94.13% | ~0% (head only) |
| Prithvi-300M + LoRA | 89.74% | 85.10% | 79.38% | 91.71% | 1.28% |
| Prithvi-600M + LoRA | 93.06% | 86.39% | 79.25% | 94.94% | 1.03% |

¹ Reported by Torres & Fraternali on a 75/25 split — not directly comparable to our split.  
² Authors' checkpoint evaluated on our 70/15/15 split. Training data may overlap our test set, so this row is an upper bound for the baseline.

See `results/figures/` for PR curves, ROC curves, confusion matrices, and parameter-efficiency plots.

---

## Installation

```bash
git clone https://github.com/yourname/aerialwaste-foundation
cd aerialwaste-foundation
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.10
- CUDA ≥ 11.8 or Apple MPS (M-series)
- ~12 GB VRAM for Prithvi-300M + LoRA at batch size 16

---

## Quick Start

```bash
# 1. Download and preprocess AerialWaste
python scripts/download_dataset.py --output data/raw
python scripts/prepare_dataset.py --input data/raw --output data/processed

# 2. Evaluate the original paper baseline (download checkpoint first)
#    Checkpoint: https://drive.google.com/drive/folders/1xy9BDFWWFkyaw3P8npEZxpTDFxkzA3NK
#    Place at: checkpoints/aerialwaste_paper/aerialwaste-model/checkpoint.pth
python scripts/eval_paper_baseline.py

# 3. Train Prithvi variants (Ctrl+C safe: resumes automatically)
python scripts/train_prithvi.py --config configs/frozen/prithvi_frozen_300.yaml
python scripts/train_prithvi.py --config configs/frozen/prithvi_frozen_600.yaml
python scripts/train_prithvi.py --config configs/lora/prithvi_lora_300.yaml
python scripts/train_prithvi.py --config configs/lora/prithvi_lora_600.yaml

# 4. Compare all models
python scripts/evaluate.py --results-dir results/metrics --output results/figures
```

Training saves a `checkpoints/<run>_latest.pt` after every epoch. If interrupted, re-running the same command resumes from the last completed epoch.

---

## Repository Structure

```
aerialwaste-foundation/
├── configs/
│   ├── frozen/
│   │   ├── prithvi_frozen_300.yaml   # Prithvi-300M linear probe
│   │   └── prithvi_frozen_600.yaml   # Prithvi-600M linear probe
│   └── lora/
│       ├── prithvi_lora_300.yaml     # Prithvi-300M + LoRA
│       └── prithvi_lora_600.yaml     # Prithvi-600M + LoRA
├── data/
│   ├── raw/                          # Downloaded from Zenodo
│   ├── processed/                    # Images + labels per split
│   └── splits/                       # train/val/test JSON manifests
├── models/
│   ├── prithvi/
│   │   └── backbone.py               # Prithvi backbone + LoRA wrapper
│   ├── baseline/
│   │   └── resnet50_fpn.py           # Paper architecture reconstruction
│   └── dataset.py                    # AerialWaste PyTorch dataset
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── eval_paper_baseline.py        # Evaluates authors' checkpoint
│   ├── train_baseline.py             # Trains ResNet50+FPN reproduction
│   ├── train_prithvi.py              # Trains all Prithvi variants
│   └── evaluate.py                   # Comparison plots and tables
├── checkpoints/                      # Saved model weights
├── results/
│   ├── figures/                      # Comparison plots
│   └── metrics/                      # JSON metric files
└── tests/
```

---

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

The suite covers the dataset pipeline, model architecture, evaluate utilities, and dataset preparation. Tests that require `terratorch` (Prithvi backbone) are automatically skipped if the package is not installed.

---

## Citation

A paper describing this work is available here:

If you use this code, please cite the original AerialWaste paper:

```bibtex
@article{torres2023aerialwaste,
  title={AerialWaste dataset for landfill discovery in aerial and satellite images},
  author={Torres, Rocio Nahime and Fraternali, Piero},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={63},
  year={2023},
  publisher={Nature Publishing Group}
}
```

And the foundation model:

```bibtex
@article{szwarcman2024prithvi,
  title={Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications},
  author={Szwarcman, Daniela and Roy, Sujit and Fraccaro, Paolo and others},
  journal={arXiv preprint arXiv:2412.02732},
  year={2024}
}
```

---

## License

**Code in this repository**: [MIT](LICENSE) — free to use, modify, and redistribute.

**Data and weights**: this repository contains no dataset files or model weights. All external assets must be downloaded separately and must not be committed to this repo.

| Asset | License | Source |
|---|---|---|
| This code | [MIT](LICENSE) | this repo |
| AerialWaste dataset | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) — non-commercial only | [aerialwaste.org](https://aerialwaste.org) |
| AerialWaste model weights | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) — non-commercial only | [nahitorres/aerialwaste-model](https://github.com/nahitorres/aerialwaste-model) |
| Prithvi-EO-2.0 weights | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [ibm-nasa-geospatial/Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) |
| Trained weights (fine-tuned on AerialWaste) | CC BY-NC-ND 4.0 (derived from dataset) — non-commercial only | this repo |
| Google Earth imagery (subset of AerialWaste) | [Google Earth ToS](https://about.google/brand-resource-center/products-and-services/geo-guidelines/) — **do not store or redistribute** | via AerialWaste |


---
## Disclaimer

This work was carried out by the author independently, outside of and unrelated to the author's employment. The work and the views expressed herein are solely the author's own and do not reflect those of any affiliated institution. This research was conducted in the author's personal time using exclusively personal resources.}