#!/usr/bin/env python3
"""Validate the frozen GCBench task inventory without running model evaluations."""
from pathlib import Path
import ast
import collections
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def validate():
    manifest = json.loads((ROOT / "task_generator/manifest.json").read_text())
    records = manifest["tasks"]
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids)) == manifest["task_count"] == 340
    actual = {p.parent.name for p in (ROOT / "task_generator/tasks").glob("*/task.json")}
    assert actual == set(ids), "Task directories differ from the frozen manifest"
    problems = []
    for record in records:
        directory = ROOT / "task_generator/tasks" / record["id"]
        task = json.loads((directory / "task.json").read_text())
        assert task["id"] == record["id"] and task["app"] == record["app"]
        assert task["task"].strip() and task["verification"]
        assert (ROOT / "verifiers" / task["app"] / (task["app"] + ".py")).is_file()
        for relative, expected in record["files"].items():
            file = ROOT / relative
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                problems.append("Missing or changed file: " + relative)
        actual_files = {str(p.relative_to(ROOT)) for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        if actual_files != set(record["files"]):
            problems.append("Asset inventory differs: " + record["id"])
        for entry in task.get("env", {}).get("files", []):
            local = directory / "env" / entry["filename"]
            # The existing runner can create missing seed files using this task-local generator.
            if not local.is_file() and not (directory / "env/create_env.py").is_file():
                problems.append("Missing seed asset: " + str(local.relative_to(ROOT)))
    for path in ROOT.rglob("*.py"):
        if ".git" not in path.parts:
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            except (SyntaxError, UnicodeError) as exc:
                problems.append(str(exc))
    skill_manifest = json.loads((ROOT / "skills/manifest.json").read_text())
    for condition in ["original", "repaired"]:
        skill_record = skill_manifest["conditions"][condition]
        assert skill_record["files"], f"Missing {condition} skill snapshot"
        for relative, expected in skill_record["files"].items():
            file = ROOT / relative
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                problems.append("Missing or changed skill source: " + relative)
    applications = collections.Counter(r["app"] for r in records)
    workflows = collections.Counter(r["workflow"] for r in records)
    assert len(applications) == manifest["application_count"] == 18
    assert len(workflows) == manifest["workflow_count"] == 12
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("GCBench: 340 tasks, 18 applications, 12 workflows; task definitions, seed hashes and both skill snapshots verified.")
    for workflow, count in sorted(workflows.items()):
        print(f"  {workflow}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate())
