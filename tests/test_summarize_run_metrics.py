from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_run_metrics import aggregate_rows, collect_rows, flatten_run_summary


def metric_summary(count: int, mse: float, mae: float = 0.1) -> dict[str, object]:
    return {
        "count": count,
        "avg_mse": mse,
        "avg_mae": mae,
        "avg_relative_flair_volume_diff": 0.0,
        "avg_per_modality_mse": {"FLAIR": mse, "T1": mse, "T2": mse, "CT1": mse},
        "avg_per_modality_mae": {"FLAIR": mae, "T1": mae, "T2": mae, "CT1": mae},
        "by_patient": {},
    }


class SummarizeRunMetricsTest(unittest.TestCase):
    def write_summary(self, root: Path, name: str, payload: dict[str, object]) -> Path:
        run_dir = root / name
        run_dir.mkdir(parents=True)
        path = run_dir / "run_summary.json"
        path.write_text(json.dumps(payload))
        return path

    def test_flatten_prefers_non_empty_holdout_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_summary(
                Path(tmp),
                "run_a",
                {
                    "run_name": "run_a",
                    "patients": ["Patient-001"],
                    "patient_weeks": {"Patient-001": [0, 4, 8]},
                    "holdout_metric_summary": metric_summary(1, 0.2),
                    "baseline_holdout_metric_summary": metric_summary(1, 0.5),
                    "all_metric_summary": metric_summary(2, 0.1),
                    "baseline_all_metric_summary": metric_summary(2, 0.2),
                },
            )

            row = flatten_run_summary(path)

        self.assertEqual(row["metric_split"], "holdout")
        self.assertEqual(row["history_timepoint_count"], 2)
        self.assertAlmostEqual(row["relative_improvement"], 0.6)

    def test_flatten_falls_back_to_all_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_summary(
                Path(tmp),
                "run_b",
                {
                    "run_name": "run_b",
                    "patients": ["Patient-002"],
                    "patient_weeks": {"Patient-002": [0, 4]},
                    "holdout_metric_summary": metric_summary(0, 0.0),
                    "baseline_holdout_metric_summary": metric_summary(0, 0.0),
                    "all_metric_summary": metric_summary(2, 0.25),
                    "baseline_all_metric_summary": metric_summary(2, 0.5),
                },
            )

            row = flatten_run_summary(path)

        self.assertEqual(row["metric_split"], "all")
        self.assertAlmostEqual(row["relative_improvement"], 0.5)

    def test_collect_rows_sorts_by_improvement_and_aggregate_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_summary(
                root,
                "run_a",
                {
                    "run_name": "run_a",
                    "patients": ["Patient-001"],
                    "patient_weeks": {"Patient-001": [0, 1]},
                    "all_metric_summary": metric_summary(1, 0.1),
                    "baseline_all_metric_summary": metric_summary(1, 0.2),
                },
            )
            self.write_summary(
                root,
                "run_b",
                {
                    "run_name": "run_b",
                    "patients": ["Patient-002"],
                    "patient_weeks": {"Patient-002": [0, 1]},
                    "all_metric_summary": metric_summary(1, 0.3),
                    "baseline_all_metric_summary": metric_summary(1, 0.2),
                },
            )

            rows = collect_rows(root, "*/run_summary.json")
            summary = aggregate_rows(rows)

        self.assertEqual([row["patient_id"] for row in rows], ["Patient-001", "Patient-002"])
        self.assertEqual(summary["run_count"], 2)
        self.assertEqual(summary["patients_with_positive_improvement"], 1)
        self.assertEqual(summary["patients_with_nonpositive_improvement"], 1)


if __name__ == "__main__":
    unittest.main()
