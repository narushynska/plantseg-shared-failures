# PlantSeg shared failures

Code and result tables for the paper **"What plant lesion segmenters miss: shared failures and edge inference cost"** (D. Boiko, O. Narushynska, Lviv Polytechnic National University; submitted to the *Journal of Edge Computing*).

The study trains four segmentation decoders (U-Net, U-Net with scSE attention, FPN, DeepLabV3+) on a shared ResNet-34 encoder on the PlantSeg v2 dataset, analyses which test images all models fail on, audits the reference masks of those images, detects near-duplicate images across the official splits, and measures CPU inference cost in ONNX Runtime.

## Data

The images and masks are **not** included. Download PlantSeg v2 from Zenodo (doi:10.5281/zenodo.13762907, CC BY-NC) and place it in `plantseg/` with `images/{train,test}` and `annotations/{train,test}`. The validation subset used here (1,247 images held out from the official training split, stratified by disease, seed 42) is listed in `data_lists/val_split_seed42.csv`; move these files into `images/val` and `annotations/val`.

## Contents

| Path | Content |
|---|---|
| `code/run_all.py` | Training and test evaluation of the four models (35 epochs, batch 4, Adam 1e-4, BCE+Dice, 256×256, seeds 0–2). Writes per-image TP/FP/FN/TN and Dice. |
| `code/analyze.py` | Seed aggregation, paired bootstrap, failure co-occurrence, failure by lesion size and disease, logistic regression. |
| `code/edge_benchmark.py` | Parameters, MACs, ONNX export and CPU latency (ONNX Runtime, 1 and 4 threads). |
| `code/soft_dice_check.py` | Soft vs. hard batch Dice (comparison with the thesis protocol). |
| `code/setup_and_run.ps1` | Windows setup (CUDA PyTorch venv) and full run. |
| `data_lists/hard_subset_shared_failures.csv` | 363 test images with per-image Dice < 0.5 for all four models (seed 0), with lesion fraction, components and disease. |
| `data_lists/annotation_audit_labels.csv` | Annotation-audit classes for the shared failures: `screen` = automatic pre-sort by a vision–language model; `human` = second inspection (A missing lesions, B coarse/over-inclusive mask, C mask plausible, D undecidable). |
| `data_lists/plantseg_v2_duplicate_pairs.csv` | 1,042 near-duplicate image pairs (perceptual hash ≤ 4 bits and thumbnail correlation > 0.95) with their splits, disease labels and mask Dice. |
| `data_lists/test_images_without_trainval_duplicate.csv` | 2,016 test images without a copy in train/val (leak-free test subset). |
| `data_lists/test_images_with_trainval_duplicate.txt` | 279 test images with a copy in train or val. |
| `data_lists/test_images_kept_in_2025_release.txt` | 1,561 test images that are kept in the 2025 PlantSeg release (doi:10.5281/zenodo.17719108). |
| `results/seed0/` | Per-image results and summaries for seed 0, aggregated tables (`results.json`, `accuracy_per_seed.csv`, `failure_by_size.csv`, `failure_cooccurrence.csv`), leakage and audit statistics. |
| `results/benchmark/edge_benchmark.csv` | CPU latency, parameters, MACs and ONNX file size. |

`results/seed1/` holds an additional U-Net run with seed 1. The remaining seed-1 and seed-2 runs are still training and will be added.

## Reproduce

```
pip install -r requirements.txt
python code/run_all.py --data plantseg --out results          # training + per-image test results
python code/analyze.py results plantseg/Metadata.csv data_lists/test_images_kept_in_2025_release.txt out
python code/edge_benchmark.py                                 # CPU latency
```

Trained weights (~95 MB per model) are available from the corresponding author on request.

## Licence

Code: MIT (see `LICENSE`). The image lists and derived tables refer to PlantSeg, which is distributed under CC BY-NC by its authors (Wei et al., *Scientific Data* 13, 205, 2026, doi:10.1038/s41597-025-06513-4).

## Contact

Olga Narushynska, Department of Automated Control Systems, Lviv Polytechnic National University — Olha.O.Narushynska@lpnu.ua
