#!/usr/bin/env python3
"""Aggregate recorded trajectories against the frozen GCBench workflow manifest."""
import argparse
import collections
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory containing trajectories/<task_id>/trajectory.json")
    args = parser.parse_args()
    tasks = json.loads((ROOT / "task_generator/manifest.json").read_text())["tasks"]
    expected = {task["id"] for task in tasks}
    groups = collections.defaultdict(list)
    missing = []
    for task in tasks:
        file = args.run / "trajectories" / task["id"] / "trajectory.json"
        if not file.is_file():
            missing.append(task["id"])
            continue
        row = json.loads(file.read_text())
        total = row.get("checks_total", 0)
        full = total > 0 and row.get("checks_passed") == total
        groups[task["workflow"]].append((full, row.get("elapsed_seconds")))
        groups["Overall"].append((full, row.get("elapsed_seconds")))
    extras = sorted(file.parent.name for file in (args.run / "trajectories").glob("*/trajectory.json") if file.parent.name not in expected)
    report = {"benchmark": "GCBench", "expected_tasks": 340, "recorded_tasks": len(groups["Overall"]), "missing_tasks": missing, "extra_tasks": extras, "score_type": "recorded verifier outcome; not protocol-compliance certified", "groups": {}}
    for name, rows in sorted(groups.items()):
        times = [time for _, time in rows if isinstance(time, (float, int))]
        report["groups"][name] = {"tasks": len(rows), "full_pass": sum(full for full, _ in rows), "full_pass_rate": 100 * sum(full for full, _ in rows) / len(rows), "mean_time_seconds": statistics.mean(times) if times else None}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
