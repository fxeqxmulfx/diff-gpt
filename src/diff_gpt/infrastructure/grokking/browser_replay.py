"""Replay grokking trajectories through the browser's ShortGrokking module, in Node."""

import json
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

BROWSER_REPLAY = """
import {pathToFileURL} from 'node:url';
import {readFileSync} from 'node:fs';
const {ShortGrokking} = await import(pathToFileURL(process.argv[1]).href);
const output = [];
for (const entry of JSON.parse(readFileSync(0, 'utf8'))) {
  const rule = new ShortGrokking();
  for (const row of entry.rows) rule.observe(row);
  output.push({name: entry.name, first99: rule.event?.epoch ?? null,
    first90: rule.first90, plateau_epochs: rule.plateau.epochs,
    rise_epochs: rule.rise, delayed: Boolean(rule.event?.delayed)});
}
console.log(JSON.stringify(output));
"""


def run_node(entries: list[dict[str, Any]], module: Path) -> list[dict[str, Any]]:
    """Return the browser module's verdicts on `entries`, computed by Node."""
    node = shutil.which("node")
    if node is None:
        msg = "Node is required for --browser-module"
        raise FileNotFoundError(msg)
    process = subprocess.run(  # noqa: S603 - fixed JS and argv; no shell interpolation.
        [node, "--input-type=module", "--eval", BROWSER_REPLAY, str(module.resolve())],
        input=json.dumps(entries),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(process.stdout)
