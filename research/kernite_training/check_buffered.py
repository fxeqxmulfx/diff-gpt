"""Check real 500-epoch flush boundaries without polling a training process."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

FLUSH_UPDATES, TOTAL_UPDATES = 9500, 9519


def run(binary: Path, bundle: Path, output: Path) -> dict:
    """Wait for boundary events and verify every epoch observation survived."""
    command = [str(binary), "train", str(bundle), str(output), "9519"]
    with subprocess.Popen(  # noqa: S603 -- Explicit local native executable.
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        first = process.stdout.readline()
        initial = json.loads(first)
        intent = json.loads((output / "execution.json").read_text())
        if (
            initial["step"] != 0
            or intent["possible_end"] != FLUSH_UPDATES
            or intent["completed"] != 0
        ):
            message = "The initial durable intent does not cover exactly 500 epochs"
            raise AssertionError(message)
        remaining, stderr = process.communicate()
        (output / "stdout.jsonl").write_text(first + remaining)
        (output / "stderr.txt").write_text(stderr)
        if process.returncode:
            message = f"Native buffered trainer failed: {stderr}"
            raise RuntimeError(message)
    events = [json.loads(line) for line in (first + remaining).splitlines()]
    history = [
        json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()
    ]
    final = json.loads((output / "execution.json").read_text())
    report = {
        "stdout_steps": [row["step"] for row in events],
        "observations": len(history),
        "every_epoch_observed": [row["step"] for row in history]
        == list(range(0, 9520, 19)),
        "exact_normal_completion": final["completed"]
        == final["possible_end"]
        == TOTAL_UPDATES,
        "initial_intent": intent,
    }
    report["passed"] = (
        report["stdout_steps"] == [0, 9500, 9519]
        and report["every_epoch_observed"]
        and report["exact_normal_completion"]
    )
    (output / "buffer-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    """Validate flushing and observation cadence on an actual 501-epoch run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = run(args.binary.resolve(), args.bundle, args.output)
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
