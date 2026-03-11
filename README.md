# QualiSpeech -> SAP Cross-Dataset Augmentation

HF-only experiment pipeline for SAP speech assessment and QualiSpeech auxiliary transfer. The repository is structured so that data preparation, baseline training, JT/FT augmentation, ratio ablations, reviewer controls, summarization, table export, figure export, and report packaging can be driven from YAML suites.

## Constraints

- Hugging Face Transformers / Hugging Face Hub only for SSL encoders
- No `fairseq`
- No `torchaudio` model bundles as encoder loaders
- No `speechbrain` / `s3prl` wrapper paths

Supported encoder aliases:

- `wavlm_base`
- `wavlm_base_plus`
- `wavlm_large`
- `w2v2_base`
- `w2v2_large_lv60`
- `w2v2_large_robust`
- `hubert_base`
- `hubert_large`

Supported SAP targets:

- `sap_naturalness`
- `sap_inappropriate_silences`
- `sap_distorted_vowels`
- `sap_imprecise_consonants`
- `sap_intelligibility`

Supported QualiSpeech -> SAP pairs:

- `qs_nat_to_sap_nat`
- `qs_cont_to_sap_sil`
- `qs_dist_to_sap_vowel`
- `qs_dist_to_sap_cons`
- `qs_effort_to_sap_intel`
- `qs_overall_to_sap_intel`

Reviewer negative-control pairs:

- `qs_cont_to_sap_nat_neg`
- `qs_dist_to_sap_sil_neg`

## Raw Data Layout

Put the raw corpora under `data/raw/` and adjust [configs/paths.yaml](/Users/zrjin/git/ssl_assessment/configs/paths.yaml) if your local layout differs.

```text
data/raw/
├── sap/
│   ├── train/
│   │   ├── <speaker_id>/
│   │   │   ├── <speaker_id>.json
│   │   │   └── *.wav
│   │   └── ...
│   └── dev/
│       ├── <speaker_id>/
│       │   ├── <speaker_id>.json
│       │   └── *.wav
│       ├── <duplicate-json-at-root>.json
│       └── ...
└── qualispeech/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── train/
    ├── val/
    └── test/
```

`elements/` contains the reference paper, sample SAP JSON, directory sketches, and the original implementation brief used to define this repository.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Commands

```bash
make prepare
make smoke
make baselines
make main
make ablation
make reviewer
make summarize
make tables
make figures
make report
make all
```

Equivalent script entry points:

```bash
python scripts/prepare_all.py --config configs/paths.yaml
python scripts/run_suite.py --suite configs/suite/main.yaml
python scripts/summarize_results.py
python scripts/export_tables.py
python scripts/export_figures.py
python scripts/package_report.py
python scripts/run_pipeline.py --suite configs/suite/all.yaml
```

## Configuration Layout

- [configs/defaults.yaml](/Users/zrjin/git/ssl_assessment/configs/defaults.yaml): shared optimizer/data/eval/model/result defaults
- [configs/paths.yaml](/Users/zrjin/git/ssl_assessment/configs/paths.yaml): raw/processed/results locations
- [configs/models/](/Users/zrjin/git/ssl_assessment/configs/models): per-encoder overrides
- [configs/tasks/](/Users/zrjin/git/ssl_assessment/configs/tasks): SAP target definitions
- [configs/pairs/](/Users/zrjin/git/ssl_assessment/configs/pairs): QualiSpeech -> SAP pair definitions
- [configs/experiments/](/Users/zrjin/git/ssl_assessment/configs/experiments): baseline/JT/FT/reviewer defaults
- [configs/suite/](/Users/zrjin/git/ssl_assessment/configs/suite): runnable experiment matrices

Reviewer-specific knobs:

- [configs/experiments/reviewer_controls.yaml](/Users/zrjin/git/ssl_assessment/configs/experiments/reviewer_controls.yaml): dual-head JT, SAP multi-task, shuffled labels, speaker-disjoint reruns, head-reset, freeze schedule, negative-pair, and Huber-loss supplementary settings
- [configs/suite/reviewer.yaml](/Users/zrjin/git/ssl_assessment/configs/suite/reviewer.yaml): narrower reviewer matrix plus explicit reviewer-only protocol and control lists

## Suggested Flow

1. Run `make prepare` to parse SAP/QualiSpeech and build processed manifests.
2. Run `make smoke` to verify the end-to-end path on a single encoder/task/pair/seed.
3. Run `make baselines` and `make main`.
4. Run `make ablation` and `make reviewer`.
5. Run `make summarize`, `make tables`, `make figures`, and `make report`.

## Notes

- The suite YAMLs are explicit. The runner should not infer hidden defaults beyond [configs/defaults.yaml](/Users/zrjin/git/ssl_assessment/configs/defaults.yaml).
- `paper_faithful` is the only main-matrix protocol. `speaker_disjoint` is declared separately for reviewer reruns only.
- Large-model reviewer defaults in [configs/suite/reviewer.yaml](/Users/zrjin/git/ssl_assessment/configs/suite/reviewer.yaml) are intentionally narrower than the main suite so they remain tractable.
- Reviewer controls also reserve explicit hooks for negative-pair controls and Huber-loss supplementary runs; these do not belong in the main matrix.
