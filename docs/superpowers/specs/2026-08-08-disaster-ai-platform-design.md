# Disaster AI Platform Design

## Goal

Turn the current disaster image classifier into a resume-worthy, reproducible ML engineering project. The finished repo should train, evaluate, serve, and document a four-class post-disaster satellite tile classifier using only the data already present in this project folder.

Twitter sentiment analysis is out of scope for the core project story.

## Resume Story

The project should be presented as an end-to-end PyTorch ML pipeline and lightweight inference service for classifying dominant visible damage in xBD-derived post-disaster satellite image tiles.

The truthful claim is tile-level classification:

- Input: one post-disaster satellite tile.
- Output: one of `no_damage`, `minor_damage`, `major_damage`, or `destroyed`.
- Meaning: the dominant damage class assigned to that image tile by the existing sorted folder labels.

The project should not claim building-instance detection, segmentation, geospatial deployment, or live disaster response operations.

## Dataset

The usable local dataset lives at `data/images/<class_name>/*.png`.

Known counts:

- `no_damage`: 1421
- `destroyed`: 363
- `major_damage`: 275
- `minor_damage`: 182
- total: 2241 images

Known disaster events:

- `guatemala-volcano`
- `hurricane-florence`
- `hurricane-harvey`
- `hurricane-matthew`
- `hurricane-michael`
- `mexico-earthquake`
- `midwest-flooding`
- `palu-tsunami`
- `santa-rosa-wildfire`
- `socal-fire`

Filenames follow:

```text
<event>_<8-digit tile id>_post_disaster.png
```

The repo should generate a manifest with `image_path`, `label`, `event`, and `split` rather than relying on hidden state.

## Split Design

Use deterministic event-aware splits to reduce leakage between train, validation, and test. The fixed split is:

```text
train events:
  hurricane-florence
  hurricane-harvey
  hurricane-matthew
  midwest-flooding
  palu-tsunami
  socal-fire

val events:
  guatemala-volcano
  hurricane-michael

test events:
  mexico-earthquake
  santa-rosa-wildfire
```

The resulting split sizes are:

- train: 1552 images
- validation: 350 images
- test: 339 images

The test split is intentionally honest but imperfect: it has only 1 `minor_damage` image and 2 `major_damage` images because the local dataset is event-skewed. Documentation and generated metrics must show class counts per split so this limitation is clear.

## Architecture

Keep the project Python-first:

- `src/disaster_ai/config.py`: shared constants and paths.
- `src/disaster_ai/data.py`: manifest generation, split parsing, image dataset loading.
- `src/disaster_ai/model.py`: CNN architecture factory.
- `src/disaster_ai/metrics.py`: accuracy, balanced accuracy, macro F1, per-class metrics, confusion matrix.
- `src/disaster_ai/training.py`: training loop, validation loop, checkpoint saving.
- `src/disaster_ai/inference.py`: shared preprocessing and prediction API.
- `src/train.py`: CLI wrapper for training.
- `src/evaluate.py`: CLI wrapper for evaluating a checkpoint.
- `src/predict_api.py`: Flask API using the shared inference layer.
- `src/dashboard.py`: Streamlit dashboard using the shared inference layer.

Legacy scripts may remain if useful, but the README should direct users to the new core commands. Twitter files should be described as archived or non-core.

## Training Behavior

Training must be reproducible:

- seed defaults to `42`
- model architecture defaults to the existing lightweight CNN family
- image size defaults to `64`
- augmentation is applied only to training images
- validation and test transforms are deterministic
- imbalance is handled with class-weighted cross entropy or a weighted sampler
- the best checkpoint is selected by validation macro F1

The checkpoint format should be a dictionary, not a raw state dict. It must include:

- `model_name`
- `state_dict`
- `class_to_idx`
- `idx_to_class`
- `image_size`
- `normalization`
- `seed`
- `best_epoch`
- `val_metrics`
- `train_config`

## Evaluation Behavior

Evaluation must load a checkpoint and a manifest, run deterministic inference on a requested split, and write artifacts under `artifacts/`:

- `manifest.csv`
- `metrics.json`
- `confusion_matrix.csv`
- `classification_report.csv`
- optionally `confusion_matrix.png` if plotting dependencies are available

Metrics must include:

- total examples
- accuracy
- balanced accuracy
- macro precision
- macro recall
- macro F1
- weighted F1
- per-class precision, recall, F1, and support

Final README claims should use generated metrics from this pipeline only.

## Inference Interfaces

Both Streamlit and Flask must call `disaster_ai.inference.DamageClassifier`.

Flask:

- route: `GET /health`
- route: `POST /predict`
- accepts multipart form field `image` and, for compatibility, `file`
- returns JSON with `predicted_class`, `confidence`, and `probabilities`
- returns clear `4xx` JSON errors for missing file or unsupported images

Streamlit:

- supports image upload
- displays predicted class, confidence, class probabilities, and dataset/model metadata
- does not include Twitter sentiment UI in the core app

## Packaging And Repo Hygiene

Use a compact dependency set in `requirements.txt`:

- `torch`
- `torchvision`
- `pillow`
- `numpy`
- `pandas`
- `scikit-learn`
- `flask`
- `streamlit`
- `pytest`

The full image dataset should stay gitignored. A small sample set may be added for tests/demo if needed. The trained checkpoint may be tracked if it is small enough for normal Git and enables the demo to run without retraining.

The Dockerfile should run a real command and should not point at nonexistent files.

## Testing

Add focused tests for:

- event parsing from filenames
- manifest generation and split assignment
- dataset class mapping
- metric calculation on a known tiny prediction set
- checkpoint metadata load/save
- inference output shape and probability sum
- Flask `/health` and `/predict` behavior using a small generated image

Tests should avoid requiring the full dataset or a long training run.

## Documentation

Rewrite `README.md` around:

- what the project does
- what it does not claim
- dataset layout
- setup
- train
- evaluate
- run Flask API
- run Streamlit dashboard
- run tests
- current measured results
- limitations
- suggested resume bullets

The README should mention that the local full dataset is required for training and that the repo can still run tests without it.

