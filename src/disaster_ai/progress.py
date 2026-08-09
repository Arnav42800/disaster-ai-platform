import sys
import time
from typing import TextIO


def format_progress(current: int, total: int, width: int = 28) -> str:
    """Render a bounded ASCII progress bar for terminal output."""
    completed = min(max(current, 0), max(total, 1))
    fraction = completed / max(total, 1)
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {fraction:>4.0%}"


class TerminalProgress:
    def __init__(
        self,
        total: int,
        label: str = "train",
        enabled: bool = True,
        stream: TextIO | None = None,
    ):
        self.total = total
        self.label = label
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.started_at = time.perf_counter()

    def update(self, current: int, loss: float | None = None) -> None:
        if not self.enabled:
            return
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        rate = current / elapsed if current else 0.0
        loss_text = f" | loss {loss:.4f}" if loss is not None else ""
        self.stream.write(
            f"\r  {self.label:<5} {format_progress(current, self.total)}"
            f" | {rate:>5.1f} it/s{loss_text}"
        )
        self.stream.flush()

    def finish(self) -> None:
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()
