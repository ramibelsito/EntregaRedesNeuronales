#!/usr/bin/env python3
"""Convierte los registros de corridas del proyecto en un log compatible con TensorBoard.

Este script busca de forma recursiva en la carpeta del proyecto:
- archivos CSV con métricas de corridas (por ejemplo: *_results.csv, *combined*.csv,
  *repeated*.csv, *metrics*.csv)
- archivos de eventos de TensorBoard ya existentes (events.out.tfevents*)

Luego genera un directorio con un archivo events.out.tfevents... que puede abrirse con:

    tensorboard --logdir tensorboard_logs/combined_runs
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class SummaryWriter:
    """Writer mínimo compatible con TensorBoard para exportar scalars."""

    def __init__(self, log_dir: str | Path):
        self._writer = EventFileWriter(logdir=str(log_dir))

    def add_scalar(self, tag: str, value: float, global_step: int = 0) -> None:
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
        event = Event(wall_time=time.time(), step=global_step, summary=summary)
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DEFAULT_ROOT / "tensorboard_logs" / "combined_runs"


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", value).strip("._") or "run"


def find_event_files(root: Path) -> list[Path]:
    return sorted(root.rglob("events.out.tfevents*"))


def find_result_csvs(root: Path) -> list[Path]:
    candidates = []
    for path in root.rglob("*.csv"):
        if any(part in {".git", "__pycache__", ".ipynb_checkpoints"} for part in path.parts):
            continue
        if path.name.startswith("."):
            continue
        candidates.append(path)
    return sorted(candidates)


def is_metric_column(name: str) -> bool:
    lowered = name.lower()
    if lowered in {"epoch", "step", "seed", "trial", "trial_id", "run_id", "run"}:
        return False
    if lowered.startswith("best_"):
        return False
    keywords = (
        "loss",
        "acc",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "balanced",
        "macro",
        "gap",
        "error",
        "lr",
        "score",
        "param",
        "params",
    )
    return any(keyword in lowered for keyword in keywords)


def infer_run_name(row: dict[str, str], file_path: Path) -> str:
    for col in ("run_label", "run_name", "trial_name", "name", "model"):
        value = row.get(col, "")
        if value:
            return str(value)
    return file_path.stem


def log_csv_results(writer: SummaryWriter, csv_path: Path) -> int:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return 0

    if not rows:
        return 0

    headers = rows[0].keys()
    metric_cols = [col for col in headers if is_metric_column(col)]
    if not metric_cols:
        return 0

    base_name = sanitize_name(csv_path.stem.replace("_results", "").replace("_summary", "").replace("_metrics", ""))
    logged = 0

    for row in rows:
        run_name = infer_run_name(row, csv_path)
        run_label = sanitize_name(f"{base_name}/{run_name}")

        step = 0
        if "epoch" in row and row.get("epoch"):
            try:
                step = int(float(row["epoch"]))
            except ValueError:
                step = 0
        elif "step" in row and row.get("step"):
            try:
                step = int(float(row["step"]))
            except ValueError:
                step = 0

        for col in metric_cols:
            value = row.get(col, "")
            if not value:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            writer.add_scalar(f"csv/{run_label}/{col}", numeric_value, step)
            logged += 1

    return logged


def write_summary_from_csvs(root: Path, output_dir: Path) -> tuple[Path, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = find_result_csvs(root)
    event_files = find_event_files(root)

    writer = SummaryWriter(log_dir=str(output_dir))

    logged_metrics = 0
    logged_event_refs = 0

    for csv_path in csv_files:
        logged_metrics += log_csv_results(writer, csv_path)

    for event_path in event_files:
        rel_path = event_path.relative_to(root).as_posix()
        writer.add_scalar(f"events/available/{sanitize_name(event_path.parent.name)}/path", 1.0, 0)
        writer.add_scalar(f"events/available/{sanitize_name(event_path.parent.name)}/exists", 1.0, 0)
        logged_event_refs += 1

    writer.flush()
    writer.close()

    return output_dir, logged_metrics, logged_event_refs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generar un log de TensorBoard a partir de resultados del proyecto")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Carpeta del repositorio a inspeccionar")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directorio donde escribir el log de TensorBoard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()

    if not root.exists():
        raise FileNotFoundError(f"No existe la carpeta: {root}")

    output_dir, logged_metrics, logged_event_refs = write_summary_from_csvs(root, output_dir)
    print(f"Directorio de TensorBoard: {output_dir}")
    print(f"CSV procesados: {len(find_result_csvs(root))}")
    print(f"Métricas exportadas: {logged_metrics}")
    print(f"Referencias a eventos encontradas: {logged_event_refs}")
    print("Abrir con:")
    print(f"  tensorboard --logdir {output_dir}")


if __name__ == "__main__":
    main()
