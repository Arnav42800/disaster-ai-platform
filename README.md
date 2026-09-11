# Disaster Damage Classifier

A reproducible PyTorch pipeline for classifying post-disaster satellite image tiles by visible damage level. The project includes deterministic dataset manifests, leakage-aware evaluation, configurable model training, evaluation reports, a Flask prediction API, and a Streamlit dashboard.

The model predicts one label for an entire image tile:

- `destroyed`
- `major_damage`
- `minor_damage`
- `no_damage`

It does not perform building detection, instance segmentation, or per-building damage assessment.

## Highlights

- Event-aware splits keep whole disaster events in a single split to measure generalization to unseen disasters.
- A stratified split mode provides a comparable benchmark for the visual classification task when every class is represented in each split.
- The default ResNet-18 model is compared with the original lightweight CNN baseline.
- ResNet-18 can optionally start from ImageNet weights with `--pretrained`; the default remains offline-friendly.
- Training uses class-weighted cross entropy, label smoothing, augmentation, learning-rate reduction, and early stopping.
- Checkpoints store model weights plus class mapping, image size, normalization, seed, best epoch, validation metrics, and training config.
- Evaluation exports metrics, a classification report, a confusion matrix, and per-image predictions with confidence scores.
- A majority-class baseline uses training labels and reports accuracy, balanced accuracy, and macro F1 on the same evaluation split.
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
    model.py       CNN baseline and ResNet-18 model factory
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

For a second, complementary measurement, use `--split-strategy stratified`. This randomly assigns images while preserving class proportions, so it is useful for measuring whether the model can learn the visual categories. It is not a substitute for the event-held-out result because related images from the same disaster can appear across splits.

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
  --epochs 30 \
  --batch-size 32 \
  --model resnet18 \
  --split-strategy event \
  --output artifacts/disaster_resnet18.pt
```

The default training run selects the checkpoint with the best validation macro F1 and stops early when validation performance stops improving. To measure the class-balanced visual-learning baseline:

```bash
PYTHONPATH=src python src/train.py \
  --data-dir data/images \
  --epochs 30 \
  --batch-size 32 \
  --model resnet18 \
  --split-strategy stratified \
  --output artifacts/disaster_resnet18_stratified.pt
```

Use `--model cnn` to reproduce the compact baseline. The two split strategies answer different questions and should be reported separately.

When run interactively, training shows a live batch progress bar with loss and throughput, followed by a compact epoch summary. Use `--no-progress` when writing plain logs or running in CI.

Add `--pretrained` to the ResNet command when network access is available. The first run downloads the torchvision ImageNet weights; subsequent runs use the local cache.

Training writes:

- `artifacts/manifest.csv`
- `artifacts/disaster_resnet18.pt`
- `artifacts/training_history.json`

## Evaluate

```bash
PYTHONPATH=src python src/evaluate.py \
  --checkpoint artifacts/disaster_resnet18.pt \
  --split test
```

The evaluator uses the shared inference loader to restore the checkpoint's model, class mapping, image size, and normalization. Older checkpoints without preprocessing metadata use the original defaults. The split strategy and seed come from the saved training configuration. Pass `--split-strategy event`, `--split-strategy stratified`, or `--image-size` to override those settings deliberately. An override changes the evaluation conditions and must be reported with the result.

Evaluation writes:

- `artifacts/test/metrics.json`
- `artifacts/test/classification_report.csv`
- `artifacts/test/confusion_matrix.csv`
- `artifacts/test/predictions.csv`

`metrics.json` also records the checkpoint path, preprocessing settings, split seed, majority-class baseline, and metric differences from that baseline. The baseline always predicts the most frequent training class. It never chooses its class from validation or test labels. `predictions.csv` records each image path, disaster event, true class, predicted class, and softmax confidence. Confidence is not calibrated.

Use a separate output directory to preserve earlier results:

```bash
PYTHONPATH=src python src/evaluate.py \
  --checkpoint artifacts/disaster_resnet18_stratified.pt \
  --artifacts-dir artifacts/resume_review/stratified
```

## Measured Benchmark Snapshot

The local dataset was trained with the same class-weighted objective and seed under both evaluation strategies. The event-held-out CNN run stopped after 5 epochs; the heavier stratified ResNet-18 run was evaluated at its best saved checkpoint after 7 completed epochs.

| Model | Split strategy | Test accuracy | Balanced accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| CNN | event-held-out | 0.563 | 0.347 | 0.208 |
| ResNet-18 | stratified | 0.611 | 0.492 | 0.478 |

The event test contains only 1 `minor_damage` and 2 `major_damage` examples, so its macro F1 is unstable. The stratified result is the better measure of visual learning capacity, while the event result is the better measure of robustness to new disaster distributions. These values are reported as a local benchmark snapshot; rerun the commands above to regenerate them.

The stratified test contains 337 tiles, including 214 `no_damage` tiles. Stratification preserves class proportions; it does not balance class counts. Always predicting the training majority class, `no_damage`, scores 63.5% accuracy and 0.194 macro F1 on this test. ResNet-18 scores below that baseline on accuracy but above it on macro F1. The two model rows use different splits and do not establish that one architecture outperforms the other.

## Reproducibility Smoke Test

The full workflow was verified locally with a 1-epoch CPU smoke run:

```bash
PYTHONPATH=src python src/train.py --data-dir data/images --epochs 1 --batch-size 64 --model cnn --output artifacts/disaster_cnn.pt
PYTHONPATH=src python src/evaluate.py --checkpoint artifacts/disaster_cnn.pt --split test --batch-size 64
```

Held-out test metrics from that run:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.519 |
| Balanced accuracy | 0.328 |
| Macro F1 | 0.191 |
| Weighted F1 | 0.471 |

This is a pipeline smoke result, not a tuned benchmark. The low macro F1 reflects the small, skewed held-out test distribution. Run the 30-epoch commands above for the actual model comparison; report accuracy, balanced accuracy, macro F1, and each class's support together.

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

By default, the container starts the Streamlit dashboard. Mount or create `artifacts/disaster_resnet18.pt` before using the dashboard for real predictions.

## Limitations

- Labels are tile-level classes from the existing sorted folders.
- The model is a supervised image-classification baseline, not a state-of-the-art remote-sensing model.
- The event-held-out test split is small for `major_damage` and `minor_damage`.
- Full training requires the local `data/images` folder, which is intentionally ignored by Git.
