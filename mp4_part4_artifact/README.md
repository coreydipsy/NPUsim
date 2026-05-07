MP4 Part IV artifact bundle.

Contents:

- `code/`: current source snapshots for the Part IV scripts and supporting KV-cache fix.
- `part4_changes.patch`: patch file for the current uncommitted code changes.
- `traces/`: simulation outputs grouped by task.

Notes:

- `run_pd_disagg.py` is included under `code/` as a source snapshot, even though it does not have a current uncommitted diff.
- The traces include the validation run, PD allocation sweep, KV-bandwidth sweep, and the co-located baseline run used for comparison.
