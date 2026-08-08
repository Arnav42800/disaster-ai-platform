# Disaster Damage Classifier

A reproducible PyTorch pipeline for classifying post-disaster satellite image tiles by visible damage level. The project includes deterministic dataset manifests, event-aware train/validation/test splits, model training, evaluation reports, a Flask prediction API, and a Streamlit dashboard.

The model predicts one label for an entire image tile:

- `destroyed`
- `major_damage`
- `minor_damage`
- `no_damage`

It does not perform building detection, instance segmentation, or per-building damage assessment.

## Highlights

- Event-aware splits keep whole disaster events in a single split to reduce leakage.
- Training uses class-weighted cross entropy to account for imbalance.
- Checkpoints store model weights plus class mapping, image size, normalization, seed, best epoch, validation metrics, and training config.
- Evaluation exports machine-readable metrics, a classification report, and a confusion matrix.
- Flask and Streamlit share the same inference layer, so local app predictions and API predictions stay consistent.
- Tests cover manifest generation, metrics, checkpoints, CLI wiring, and API behavior.

## Repository Layout

```text
src/
  disaster_ai/
    config.py      constants, labels, split definitions
    data.py        manifest generation, transforms, dataset class
    inference.py   shared checkpoint loading and prediction
    metrics.py     sklearn metric/report helpers
    model.py       lightweight CNN
    training.py    seed, evaluation, checkpoint helpers
  dashboard.py     Streamlit app
  evaluate.py      evaluation CLI
  predict_api.py   Flask API
  train.py         training CLI
tests/
Dockerfile
requirements.txt
```

## Data

The training data is expected locally at:

```text
data/images/
  destroyed/
  major_damage/
  minor_damage/
  no_damage/
```

The local dataset used during development contains 2,241 xBD-derived post-disaster tiles from 10 disaster events:

| Class | Images |
| --- | ---: |
| `no_damage` | 1421 |
| `destroyed` | 363 |
| `major_damage` | 275 |
| `minor_damage` | 182 |

The full dataset is not committed to Git because of size and licensing concerns. Unit tests generate their own small fixtures and do not require the dataset.

## Split Policy

`src/disaster_ai/data.py` parses filenames in this format:

```text
<event>_<8-digit tile id>_post_disaster.png
```

The manifest assigns whole events to each split:

| Split | Events | Images |
| --- | --- | ---: |
| train | `hurricane-florence`, `hurricane-harvey`, `hurricane-matthew`, `midwest-flooding`, `palu-tsunami`, `socal-fire` | 1552 |
| validation | `guatemala-volcano`, `hurricane-michael` | 350 |
| test | `mexico-earthquake`, `santa-rosa-wildfire` | 339 |

The held-out test split is intentionally reported with its limitations: it contains only 1 `minor_damage` tile and 2 `major_damage` tiles. Macro metrics and per-class reports are more informative than accuracy alone.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run commands from the repository root.

## Run Tests

```bash
PYTHONPATH=src pytest -q
```

## Train

```bash
PYTHONPATH=src python src/train.py \
  --data-dir data/images \
  --epochs 10 \
  --batch-size 32 \
  --output artifacts/disaster_cnn.pt
```

Training writes:

- `artifacts/manifest.csv`
- `artifacts/disaster_cnn.pt`
- `artifacts/training_history.json`

## Evaluate

```bash
PYTHONPATH=src python src/evaluate.py \
  --checkpoint artifacts/disaster_cnn.pt \
  --split test
```

Evaluation writes:

- `artifacts/test/metrics.json`
- `artifacts/test/classification_report.csv`
- `artifacts/test/confusion_matrix.csv`

## Smoke-Tested Result

The full workflow was verified locally with a 1-epoch CPU smoke run:

```bash
PYTHONPATH=src python src/train.py --data-dir data/images --epochs 1 --batch-size 64 --output artifacts/disaster_cnn.pt
PYTHONPATH=src python src/evaluate.py --checkpoint artifacts/disaster_cnn.pt --split test --batch-size 64
```

Held-out test metrics from that run:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.519 |
| Balanced accuracy | 0.328 |
| Macro F1 | 0.191 |
| Weighted F1 | 0.471 |

This is a reproducibility smoke result, not a tuned benchmark. The low macro F1 reflects the small, skewed held-out test distribution.

## Flask API

Start the API:

```bash
PYTHONPATH=src flask --app src.predict_api run --port 5000
```

Health check:

```bash
curl http://127.0.0.1:5000/health
```

Predict:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "image=@path/to/tile.png"
```

Example response:

```json
{
  "predicted_class": "no_damage",
  "confidence": 0.91,
  "probabilities": {
    "destroyed": 0.01,
    "major_damage": 0.03,
    "minor_damage": 0.05,
    "no_damage": 0.91
  }
}
```

## Streamlit Dashboard

```bash
PYTHONPATH=src streamlit run src/dashboard.py
```

The dashboard accepts an uploaded satellite tile and displays the predicted class, confidence, probability chart, and checkpoint metadata.

## Docker

```bash
docker build -t disaster-ai-platform .
docker run --rm -p 8501:8501 disaster-ai-platform
```

By default, the container starts the Streamlit dashboard. Mount or create `artifacts/disaster_cnn.pt` before using the dashboard for real predictions.

## Limitations

- Labels are tile-level classes from the existing sorted folders.
- The classifier is a compact CNN baseline, not a state-of-the-art remote-sensing model.
- The event-held-out test split is small for `major_damage` and `minor_damage`.
- Full training requires the local `data/images` folder, which is intentionally ignored by Git.
