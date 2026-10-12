#!/usr/bin/env python3
"""Install a frozen skill condition inside the benchmark desktop container."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=["original", "repaired"])
    parser.add_argument("--destination", type=Path, default=Path("/home/user/CLI-Anything"))
    args = parser.parse_args()
    source = ROOT / "skills" / args.condition
    harnesses = sorted(source.glob("*/agent-harness/setup.py"))
    if not harnesses:
        parser.error(f"Frozen {args.condition} skills are not available at {source}")
    if args.destination.exists():
        parser.error("Use a fresh container: the destination already exists; never mix skill conditions.")
    shutil.copytree(source, args.destination)
    for file in harnesses:
        harness = args.destination / file.relative_to(source).parent
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-e", str(harness)], check=True)
    # Make the frozen documentation discoverable through both CLI agents' standard skill paths.
    for file in sorted(args.destination.rglob("SKILL.md")):
        name = "cli-anything-" + file.relative_to(args.destination).parts[0]
        for relative in [".claude/skills", ".agents/skills"]:
            target = Path("/home/user") / relative / name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target / "SKILL.md")
    (Path("/home/user") / "gcbench-skill-condition.json").write_text(json.dumps({"benchmark": "GCBench", "condition": args.condition}, indent=2) + "\n")
    print(f"Installed {len(harnesses)} {args.condition} skill packages.")


if __name__ == "__main__":
    main()
