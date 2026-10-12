# Reproducing GCBench

## Frozen task definition

The evaluated inventory is the 340 IDs in `task_generator/manifest.json`. Both runners load `task_generator/tasks/<id>/task.json["task"]`. Each task maps its seed files into the sandbox using `env.files`; app hooks initialize profiles and launch the application. The same `verification` entries are then evaluated against the final state in both modalities. CLI evaluation additionally reloads LibreOffice documents before UNO verification.

The release omits old feasibility annotations and generation ratings from task JSON. Task instructions, environment mappings, verification criteria, and initial assets are retained. An original skill's lack of a required operation is a CLI system limitation; it does not remove that task from GCBench.

## Runtime settings

These values describe the implementation, including settings not enumerated in the paper:

| Setting | GUI | CLI |
|---|---|---|
| Desktop | 1920 × 1080 | Same desktop applications and task initial files |
| Model screenshot | 1280 × 720 | CLI agent's native observations |
| History images | 5 per model call, including current image | Native CLI-agent context |
| Reasoning tier | `medium` where supported | `medium` |
| Output budget | 32,768 tokens per GUI model call | Native CLI-agent settings |
| Agent limit | 40 model interaction rounds | 200-second task-execution timeout |
| Task changes | Visible GUI operations only | CLI-Anything skill workflows only |

Forty GUI rounds do not mean forty atomic GUI actions: one model response can contain multiple actions. The CLI runner accepts `--max-iterations` for interface compatibility, but its execution limit is the 200-second timeout in `cli_agent_runner.py`; do not interpret it as a CLI action-count cap. Sandbox lifetime and task-execution timeout are different settings.

The GUI and CLI action policies are in `evaluation/runtime/agent_runner.py` and `evaluation/runtime/cli_agent_runner.py`. They are prompt constraints, not complete technical enforcement. Verifier code is uploaded by the harness before execution. Preserve these details when comparing to the paper; hardening the environment would define a different protocol.

## Prepare CLI images

Start a fresh desktop container and open its desktop if interactive authentication is needed:

```bash
docker run -d --name gcbench-original-setup \
  --shm-size 2g -p 6080:6080 gcbench-desktop:latest
docker cp . gcbench-original-setup:/home/user/GCBench
docker exec --user root gcbench-original-setup \
  chown -R user:user /home/user/GCBench
docker exec -it gcbench-original-setup bash
```

Inside the container, install the CLI agent(s) and authenticate them with your own account or API configuration. Verify `claude --version` and/or `codex --version`; record these versions alongside your run. The runtime executes these installed programs; it does not bundle their binaries or credentials.

As `user`, install exactly one frozen skill condition:

```bash
python3 /home/user/GCBench/scripts/install_skills.py --condition original
```

Then commit and remove only this preparation container:

```bash
docker commit gcbench-original-setup gcbench-cli-original:latest
docker rm -f gcbench-original-setup
```

Repeat in a separate fresh container using `--condition repaired`, a different setup-container name, and the image tag `gcbench-cli-repaired:latest`. Keep the two images separate. Do not install a current upstream skill checkout and label it as the frozen original condition.

The task image installs initial assets in `/home/user`, while skills are discoverable under `/home/user/CLI-Anything`. The environment requires the applications and their native dependencies; installing host-side Python requirements alone is insufficient.

## Reproduction boundaries

GCBench's tasks, initial files, checks, prompts, and evaluation code can be rerun. Exact success percentages are not deterministic guarantees: hosted model revisions, agent versions, application versions, API failures, and stochastic model outputs can affect outcomes. Record the Git commit, manifest version, desktop-image digest, skill condition, CLI-agent version, and resolved model IDs for each run.

The paper reports a verifier-guided repair diagnostic. It is not an out-of-the-box original CLI baseline or an isolated causal estimate of modality effects. Checkpoint-level coverage tests do not establish that every complete task is solvable by composing repaired skills.

Some task entries explicitly use an LLM judge in addition to executable application checks. These need the judge credentials in `.env.example`; the judge model and endpoint must be recorded. The paper's reported drawio scores include manual adjudication of judge failures. A fresh raw trajectory summary does not automatically reproduce those adjudications.

This release's `scripts/summarize_results.py` computes recorded full passes and mean recorded elapsed time. It does not reclassify protocol violations or apply undocumented manual corrections. The paper keeps violating trajectories in its scores and reports their counts separately.

## Skill version provenance

The original-condition source is pinned to upstream commit `1bb1010de4911f5c970091feccc5f8fb13301b2b` (2026-04-28). An original Sonnet 5 trajectory recorded this HEAD through `git log`; all original-condition run configurations reference the same image tag. Historical immutable image digests were not recorded. The repaired directory contains the verifier-guided implementations distributed with the study, frozen by file hashes in this release. The manifest identifies each condition independently; it does not claim that a historical execution image was byte-identical to a newly rebuilt image.
