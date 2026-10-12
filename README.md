# GCBench

GCBench compares screen-only GUI agents and skill-mediated CLI agents on **340 desktop tasks across 18 applications and 12 workflow categories**. Both modalities receive the same task instruction, initial files, and final-state verifier. The action interfaces differ: GUI agents operate visible application controls; CLI agents must use CLI-Anything skills to change task results.

## Where are the tasks?

**The complete released task set is in [`task_generator/tasks/`](task_generator/tasks/).** Each subdirectory is one task:

```text
task_generator/tasks/<task_id>/
├── task.json          # Shared instruction, application, initial-state mapping, and verifier checks
├── env_manifest.json  # Initial file inventory
└── env/               # Initial files and any task-local setup/generation scripts
```

The frozen inventory and workflow assignments are in [`task_generator/manifest.json`](task_generator/manifest.json). It lists all 340 IDs and SHA-256 hashes of task definitions and seed assets. Application verifiers are in [`verifiers/`](verifiers/).

For example, [`shotcut_v2_uhd_4k_project/task.json`](task_generator/tasks/shotcut_v2_uhd_4k_project/task.json) is the rewriting example in the paper. [`drawio_kubernetes_cluster/task.json`](task_generator/tasks/drawio_kubernetes_cluster/task.json) is the structural diagram example.

| Workflow | Tasks |
|---|---:|
| Visual Design | 38 |
| Audio | 38 |
| Knowledge | 33 |
| CAD & 3D | 39 |
| Graphics Debugging | 31 |
| Documents | 33 |
| Video & Streaming | 29 |
| Spreadsheets | 30 |
| Presentations | 27 |
| Communication | 15 |
| Cross-app | 16 |
| Web | 11 |

This release contains the current 340-task GCBench set. Earlier task sets, grounded-prompt diagnostics, generation interfaces, experiment logs, and unrelated third-party source trees are omitted.

## Install and validate

Use Python 3.11 and a Linux x86-64 host with Docker. GUI evaluation also supports Apptainer. Hosted agents require your own API credentials; local GUI models require an OpenAI-compatible model server.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python scripts/validate_release.py
```

Validation checks the task inventory, workflow counts, seed-file availability, file hashes, verifier files, and Python syntax. It does not call models or run a desktop evaluation.

Build the desktop image containing the applications and verifier dependencies:

```bash
bash computer_env/provision/docker/build_image.sh gcbench-desktop:latest
```

The image targets `linux/amd64`. Container applications use `/home/user`; these are sandbox paths, not paths to create on your host.

## GUI evaluation

```bash
python evaluation/run_eval.py --list-apps
python evaluation/run_eval.py --list-models

# One task
python evaluation/run_eval.py \
  --env-backend docker --docker-image gcbench-desktop:latest \
  --app shotcut --task shotcut_v2_uhd_4k_project \
  --model gpt-5.5 --max-iterations 40 --sandbox-timeout 3600 \
  --output-dir gui-smoke

# All 340 tasks
python evaluation/run_eval.py \
  --env-backend docker --docker-image gcbench-desktop:latest \
  --model claude-opus-5-5 --max-iterations 40 --sandbox-timeout 3600 \
  --output-dir gui-opus55
```

The paper's GUI aliases are `gpt-5.5`, `gpt-5.6-sol`, `claude-sonnet-5`, `claude-opus-5-5`, `qwen3.8-27b`, `evocua-32b`, and `kimi-k3`. Use your model server/API endpoint and credentials for the selected agent. Preserve the requested model version instead of substituting a provider's moving alias.

## CLI evaluation: keep the two conditions separate

The paper's main CLI comparison uses **original CLI-Anything skills**. Its verifier-guided skill-repair experiment is a **separate diagnostic condition**, not the original baseline. Never install repaired skills into an image intended for the original condition.

Prepare a fresh desktop container for each condition. Install and authenticate Claude Code and/or Codex inside it; API configuration is user supplied. Copy this release into the container, then run the installer as the `user` account:

```bash
# Inside a fresh container, with the repository at /home/user/GCBench:
python3 /home/user/GCBench/scripts/install_skills.py --condition original
# In a DIFFERENT fresh container for the diagnostic condition:
python3 /home/user/GCBench/scripts/install_skills.py --condition repaired
```

Original skills are pinned to HKUDS/CLI-Anything commit [`1bb1010`](https://github.com/HKUDS/CLI-Anything/commit/1bb1010de4911f5c970091feccc5f8fb13301b2b), dated **2026-04-28**. The bundled repaired implementations are kept in a separate directory. Source evidence and file fingerprints are recorded in [`skills/provenance.json`](skills/provenance.json) and [`skills/manifest.json`](skills/manifest.json).

The installer creates `/home/user/CLI-Anything`, installs that condition's harnesses, exposes its skill documentation to both agents, and records the condition. Commit the configured containers as `gcbench-cli-original:latest` and `gcbench-cli-repaired:latest`. Detailed preparation, protocol settings, and reproduction boundaries are in [`docs/reproduction.md`](docs/reproduction.md).

```bash
# Main CLI condition: all 340 tasks
python evaluation/run_cli_eval.py \
  --provider claude --model opus-5.5 \
  --docker-image gcbench-cli-original:latest \
  --output-dir cli-original-opus55

# Verifier-guided repair diagnostic: all 340 tasks
python evaluation/run_cli_eval.py \
  --provider claude --model opus-5.5 \
  --docker-image gcbench-cli-repaired:latest \
  --output-dir cli-repaired-opus55
```

For the other original CLI baselines, use `--provider claude --model sonnet-5`, `--provider codex --model gpt-5.5`, or `--provider codex --model gpt-5.6-sol` with the original-skill image.

## Results and scoring

Trajectories are saved under `evaluation/runs/gui/<run_id>/trajectories/<task_id>/trajectory.json` or `evaluation/runs/cli/<run_id>/trajectories/<task_id>/trajectory.json`. They contain verifier outcomes and elapsed time. Summarize a run with:

```bash
python scripts/summarize_results.py evaluation/runs/gui/gui-opus55
python scripts/summarize_results.py evaluation/runs/cli/cli-original-opus55
```

Full pass requires every verifier check for a task to pass. Workflow and overall rates weight tasks equally. Missing tasks are reported explicitly; a partial run is not a complete 340-task result. Scores are recorded verifier outcomes, not certification that every trajectory obeyed its action policy. The paper separately reports a protocol-compliance audit and retains violating trajectories in its reported scores.

## Repository layout

```text
agents/                 GUI agents used in the paper
computer_env/           Desktop backends and image provisioning
evaluation/             GUI/CLI runners, app setup, prompts, and verification
task_generator/tasks/   The 340 released tasks and initial files
task_generator/manifest.json
verifiers/              Final-state checks for the 18 applications
skills/                 Frozen CLI skill conditions
scripts/                Validation, skill installation, and result aggregation
docs/reproduction.md    Protocol and environment preparation
```

## Third-party components

CLI-Anything skills derive from HKUDS/CLI-Anything. GUI action parsing includes adaptations from OSWorld. Desktop application and third-party component licenses remain applicable; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Model weights, hosted API access, and authenticated CLI-agent images are not bundled.
