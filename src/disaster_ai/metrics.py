from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | tuple[str, ...],
) -> dict:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }

    return {
        "total_examples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
    }


def classification_report_frame(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def confusion_matrix_frame(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return pd.DataFrame(matrix, index=class_names, columns=class_names)


def write_evaluation_artifacts(
    output_dir: Path,
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | tuple[str, ...],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    pd.Series(metrics).to_json(output_dir / "metrics.json", indent=2)
    classification_report_frame(y_true, y_pred, class_names).to_csv(
        output_dir / "classification_report.csv"
    )
    confusion_matrix_frame(y_true, y_pred, class_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    return metrics
