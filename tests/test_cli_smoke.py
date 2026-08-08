import subprocess
import sys


def test_train_help_runs():
    result = subprocess.run(
        [sys.executable, "src/train.py", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--epochs" in result.stdout
    assert "--split-strategy" in result.stdout
    assert "--model" in result.stdout


def test_evaluate_help_runs():
    result = subprocess.run(
        [sys.executable, "src/evaluate.py", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--checkpoint" in result.stdout
    assert "--split-strategy" in result.stdout
