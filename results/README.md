# Results Notes

This folder contains small, checked-in result summaries and derived tables.
Large run directories, checkpoints, NIfTI files, and generated figures remain
outside version control.

Current tracked aggregate artifacts:

- `lumiere_full_v1_metrics.csv`: patient-level metric table.
- `lumiere_full_v1_metrics.json`: machine-readable aggregate summary and rows.
- `lumiere_full_v1_metrics.md`: compact human-readable result summary.

Regenerate these tables from local run summaries with:

```bash
python3 scripts/summarize_run_metrics.py \
  --runs-dir runs \
  --pattern 'lumiere_full_v1_Patient-*/run_summary.json' \
  --output-dir results \
  --prefix lumiere_full_v1_metrics
```

The current `lumiere_full_v1_metrics` tables were generated from local
`run_summary.json` files. The `metric_split` column records whether each row
uses holdout, all-sample, or train metrics. Use only a frozen, consistently
held-out rerun for final publication claims.
