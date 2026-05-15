from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PREFERRED_SPLITS = ("holdout", "all", "train")
MODALITIES = ("FLAIR", "T1", "T2", "CT1")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Aggregate Neural ODE run_summary.json files into publication-ready tables."
    )
    parser.add_argument("--runs-dir", type=Path, default=repo_root / "runs")
    parser.add_argument("--pattern", type=str, default="*/run_summary.json")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "results")
    parser.add_argument("--prefix", type=str, default="run_metrics")
    parser.add_argument(
        "--split",
        choices=PREFERRED_SPLITS,
        default=None,
        help="Metric split to summarize. Defaults to holdout when present, then all, then train.",
    )
    return parser.parse_args()


def metric_summary_for_split(summary: dict[str, Any], split: str | None) -> tuple[str, dict[str, Any], dict[str, Any]]:
    candidate_splits = (split,) if split is not None else PREFERRED_SPLITS
    for candidate in candidate_splits:
        model_metrics = summary.get(f"{candidate}_metric_summary") or {}
        baseline_metrics = summary.get(f"baseline_{candidate}_metric_summary") or {}
        if model_metrics.get("count", 0) and baseline_metrics.get("count", 0):
            return candidate, model_metrics, baseline_metrics
    raise ValueError(f"No non-empty model/baseline metric summary found for run {summary.get('run_name')}")


def relative_improvement(model_mse: float | None, baseline_mse: float | None) -> float | None:
    if model_mse is None or baseline_mse is None or baseline_mse == 0:
        return None
    return (baseline_mse - model_mse) / baseline_mse


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def flatten_run_summary(path: Path, split: str | None = None) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    metric_split, model_metrics, baseline_metrics = metric_summary_for_split(summary, split)
    patients = summary.get("patients") or []
    patient_id = patients[0] if len(patients) == 1 else ",".join(str(patient) for patient in patients)
    patient_weeks = summary.get("patient_weeks", {}).get(patient_id, [])
    model_mse = model_metrics.get("avg_mse")
    baseline_mse = baseline_metrics.get("avg_mse")
    row: dict[str, Any] = {
        "run_name": summary.get("run_name") or path.parent.name,
        "patient_id": patient_id,
        "metric_split": metric_split,
        "history_timepoint_count": max(len(patient_weeks) - 1, 0),
        "sample_count": model_metrics.get("count"),
        "model_mse": model_mse,
        "baseline_mse": baseline_mse,
        "relative_improvement": relative_improvement(model_mse, baseline_mse),
        "model_mae": model_metrics.get("avg_mae"),
        "baseline_mae": baseline_metrics.get("avg_mae"),
        "model_relative_flair_volume_diff": model_metrics.get("avg_relative_flair_volume_diff"),
        "baseline_relative_flair_volume_diff": baseline_metrics.get("avg_relative_flair_volume_diff"),
        "epochs": summary.get("epochs"),
        "model_size": summary.get("model_size"),
        "history_mode": summary.get("history_mode"),
        "holdout_last_pair": summary.get("holdout_last_pair"),
        "summary_path": display_path(path),
    }
    for modality in MODALITIES:
        row[f"model_{modality.lower()}_mse"] = (model_metrics.get("avg_per_modality_mse") or {}).get(modality)
        row[f"baseline_{modality.lower()}_mse"] = (baseline_metrics.get("avg_per_modality_mse") or {}).get(modality)
    return row


def collect_rows(runs_dir: Path, pattern: str, split: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for summary_path in sorted(runs_dir.glob(pattern)):
        try:
            rows.append(flatten_run_summary(summary_path, split=split))
        except ValueError:
            continue
    return sorted(
        rows,
        key=lambda row: (
            row["relative_improvement"] is None,
            -(row["relative_improvement"] or float("-inf")),
            row["patient_id"],
        ),
    )


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improvements = [row["relative_improvement"] for row in rows if row["relative_improvement"] is not None]
    model_mses = [row["model_mse"] for row in rows if row["model_mse"] is not None]
    baseline_mses = [row["baseline_mse"] for row in rows if row["baseline_mse"] is not None]
    return {
        "run_count": len(rows),
        "mean_model_mse": sum(model_mses) / len(model_mses) if model_mses else None,
        "mean_baseline_mse": sum(baseline_mses) / len(baseline_mses) if baseline_mses else None,
        "mean_relative_improvement": sum(improvements) / len(improvements) if improvements else None,
        "patients_with_positive_improvement": sum(1 for value in improvements if value > 0),
        "patients_with_nonpositive_improvement": sum(1 for value in improvements if value <= 0),
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, Any]], summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


def write_markdown(rows: list[dict[str, Any]], summary: dict[str, Any], output_path: Path, top_n: int = 10) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Aggregated Run Metrics",
        "",
        "Generated from local `run_summary.json` files. The `metric_split` column records which summary split was used for each run.",
        "",
        f"- Run count: {summary['run_count']}",
        f"- Mean model MSE: {summary['mean_model_mse']:.6g}" if summary["mean_model_mse"] is not None else "- Mean model MSE: n/a",
        f"- Mean baseline MSE: {summary['mean_baseline_mse']:.6g}" if summary["mean_baseline_mse"] is not None else "- Mean baseline MSE: n/a",
        (
            f"- Mean relative improvement: {summary['mean_relative_improvement']:+.1%}"
            if summary["mean_relative_improvement"] is not None
            else "- Mean relative improvement: n/a"
        ),
        f"- Positive-improvement patients: {summary['patients_with_positive_improvement']}",
        f"- Nonpositive-improvement patients: {summary['patients_with_nonpositive_improvement']}",
        "",
        f"## Top {min(top_n, len(rows))} Runs by Relative Improvement",
        "",
        "| Patient | Split | Samples | Model MSE | Baseline MSE | Improvement |",
        "| :--- | :---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows[:top_n]:
        improvement = row["relative_improvement"]
        lines.append(
            "| {patient_id} | {metric_split} | {sample_count} | {model_mse:.6g} | {baseline_mse:.6g} | {improvement} |".format(
                patient_id=row["patient_id"],
                metric_split=row["metric_split"],
                sample_count=row["sample_count"],
                model_mse=row["model_mse"],
                baseline_mse=row["baseline_mse"],
                improvement=f"{improvement:+.1%}" if improvement is not None else "n/a",
            )
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.runs_dir.resolve(), args.pattern, split=args.split)
    if not rows:
        raise RuntimeError(f"No run summaries with metrics matched {args.runs_dir / args.pattern}")

    summary = aggregate_rows(rows)
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / f"{args.prefix}.csv"
    json_path = output_dir / f"{args.prefix}.json"
    markdown_path = output_dir / f"{args.prefix}.md"
    write_csv(rows, csv_path)
    write_json(rows, summary, json_path)
    write_markdown(rows, summary, markdown_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
