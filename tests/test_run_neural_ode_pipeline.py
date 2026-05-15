from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.run_neural_ode_pipeline import flatten_metric_row, write_metric_rows


class RunNeuralODEPipelineMetricsTest(unittest.TestCase):
    def test_flatten_metric_row_expands_modality_metrics(self) -> None:
        row = {
            "patient_id": "Patient-001",
            "history_weeks": [0, 4],
            "target_week": 8,
            "mse": 0.1,
            "mae": 0.2,
            "relative_flair_volume_diff": 0.3,
            "per_modality_mse": {"FLAIR": 0.11, "T1": 0.12, "T2": 0.13, "CT1": 0.14},
            "per_modality_mae": {"FLAIR": 0.21, "T1": 0.22, "T2": 0.23, "CT1": 0.24},
        }

        flat = flatten_metric_row(row)

        self.assertEqual(flat["history_weeks"], "0,4")
        self.assertEqual(flat["flair_mse"], 0.11)
        self.assertEqual(flat["ct1_mae"], 0.24)

    def test_write_metric_rows_writes_header_for_empty_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "metrics.csv"
            write_metric_rows([], output_path)

            with output_path.open(newline="") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(len(rows), 1)
        self.assertIn("patient_id", rows[0])
        self.assertIn("ct1_mae", rows[0])


if __name__ == "__main__":
    unittest.main()
