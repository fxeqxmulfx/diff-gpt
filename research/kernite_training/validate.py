"""Run the complete shape/seed validation and real native checkpoint recovery."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from research.kernite_training.check import check
from research.kernite_training.export import export

NAMES = [
    f"cubic-{size}-{arm}-lr2"
    for size in ("depth", "width", "depth-width")
    for arm in ("original", "normalized")
]
STEPS = 64


def run(binary: Path, root: Path, *, recovery_only: bool = False) -> dict:
    """Retain complete positive and negative evidence in a new output directory."""
    root.mkdir(parents=True, exist_ok=False)
    results = []

    def execute(
        mode: str, bundle: Path, out: Path, updates: int, *, expect_success: bool = True
    ) -> None:
        command = [str(binary), mode, str(bundle), str(out), str(updates)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603 -- Explicit local executable and paths.
        (root / f"{out.name}-{mode}.log").write_text(result.stdout + result.stderr)
        if (result.returncode == 0) != expect_success:
            message = (
                f"Unexpected exit from {command}: {result.returncode}: {result.stderr}"
            )
            raise AssertionError(message)

    for name in [] if recovery_only else NAMES:
        for seed in (0, 1, 2):
            bundle = root / f"{name}-seed{seed}-input"
            out = root / f"{name}-seed{seed}-probe"
            export(bundle, name, seed, STEPS)
            execute("probe", bundle, out, STEPS)
            report = check(bundle, out, "cuda")
            results.append(
                {
                    "name": name,
                    "seed": seed,
                    "passed": report["passed"],
                    "report": str(out / "comparison-cuda.json"),
                    "checks": len(report["checks"]),
                }
            )
            (root / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            json.dumps(
                {"group": name, "passed": all(row["passed"] for row in results[-3:])}
            ),
            flush=True,
        )

    bundle = root / f"{NAMES[-1]}-seed2-input"
    if recovery_only:
        export(bundle, NAMES[-1], 2, STEPS)
    continuous, resumed = root / "continuous", root / "resumed"
    execute("train", bundle, continuous, STEPS)
    execute("train", bundle, resumed, 19)
    pending = root / "pending-restart"
    shutil.copytree(resumed, pending)
    journal = json.loads((pending / "execution.json").read_text())
    journal["possible_end"] = 38
    (pending / "execution.json").write_text(json.dumps(journal))
    execute("resume", bundle, pending, STEPS, expect_success=False)
    corrupted = root / "corrupt-restart"
    shutil.copytree(resumed, corrupted)
    checkpoint = bytearray((corrupted / "checkpoint.bin").read_bytes())
    checkpoint[-1] ^= 1
    (corrupted / "checkpoint.bin").write_bytes(checkpoint)
    execute("resume", bundle, corrupted, STEPS, expect_success=False)
    execute("resume", bundle, resumed, STEPS)
    exact = (continuous / "checkpoint.bin").read_bytes() == (
        resumed / "checkpoint.bin"
    ).read_bytes()
    histories = [
        [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
        for path in (continuous, resumed)
    ]
    same_history = (
        [r["step"] for r in histories[0]]
        == [r["step"] for r in histories[1]]
        == [0, 19, 38, 57, 64]
    )
    same_metrics = all(
        a["train"] == b["train"] and a["validation"] == b["validation"]
        for a, b in zip(*histories, strict=True)
    )
    report = {
        "scope": "recovery only"
        if recovery_only
        else "kernels, optimizer and recovery",
        "models": results,
        "checkpoint_bitwise_identical": exact,
        "observation_steps_identical": same_history,
        "metrics_identical": same_metrics,
        "corrupt_checkpoint_rejected": True,
        "uncertain_replay_rejected": True,
        "passed": all(row["passed"] for row in results)
        and exact
        and same_history
        and same_metrics,
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Validate the actual compiled executable, including its native resume command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recovery-only", action="store_true")
    args = parser.parse_args()
    result = run(args.binary.resolve(), args.output, recovery_only=args.recovery_only)
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
