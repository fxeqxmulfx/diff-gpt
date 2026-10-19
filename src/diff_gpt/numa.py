# Copyright 2025 Mikhail Borisov
"""
NUMA topology helpers for pinning a process to one node's CPUs.

Reads /sys/devices/system/node/ (Linux) to discover the nodes and their CPU
lists, and keeps only the CPUs this process may use: those in its affinity
mask at import (a cpuset container or a SLURM job allows fewer than the
machine has). Nodes without such a CPU (memory-only CXL or HBM nodes, nodes
outside the cpuset) are skipped.

Without the topology (kernels without the NUMA subsystem, containers with the
path hidden) every allowed CPU forms one virtual node 0. Without the affinity
calls (macOS, Windows) pinning is a silent no-op.

`root` (the topology directory) and `allowed` (the usable CPUs, None when the
affinity is unknown) are parameters so that a fake topology can stand in.
"""

import os
from pathlib import Path

import torch

_NODE_ROOT = Path("/sys/devices/system/node")
# Read once: pinning narrows the affinity mask, after which the process could
# no longer be pinned to another node.
_ALLOWED: frozenset[int] | None = (
    frozenset(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
)


def _parse_cpulist(text: str) -> list[int]:
    out: list[int] = []
    for raw in text.split(","):
        chunk = raw.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def _topology(root: Path) -> dict[int, list[int]]:
    """Map every NUMA node under `root` to its CPUs; empty without the topology."""
    if not root.is_dir():
        return {}
    topology: dict[int, list[int]] = {}
    for entry in root.iterdir():
        name = entry.name
        if name.startswith("node") and name[4:].isdigit():
            cpulist = entry / "cpulist"
            text = cpulist.read_text() if cpulist.exists() else ""
            topology[int(name[4:])] = _parse_cpulist(text.strip())
    return topology


def _usable(cpus: list[int], allowed: frozenset[int] | None) -> list[int]:
    return cpus if allowed is None else [c for c in cpus if c in allowed]


def list_numa_nodes(
    *,
    root: Path = _NODE_ROOT,
    allowed: frozenset[int] | None = _ALLOWED,
) -> list[int]:
    """
    Return the ids of the NUMA nodes that have a CPU this process may use.

    [0], the virtual node, when the topology isn't exposed.
    """
    topology = _topology(root)
    return [n for n in sorted(topology) if _usable(topology[n], allowed)] or [0]


def cpus_on_node(
    node: int,
    *,
    root: Path = _NODE_ROOT,
    allowed: frozenset[int] | None = _ALLOWED,
) -> list[int]:
    """
    Return the CPUs of NUMA `node` that this process may use.

    Empty when the node has none or doesn't exist. Without the topology every
    node is the virtual node of all allowed CPUs (empty when the affinity is
    unknown too).
    """
    topology = _topology(root)
    if not topology:
        return sorted(allowed) if allowed is not None else []
    return _usable(topology.get(node, []), allowed)


def pin_to_node(
    node: int,
    *,
    root: Path = _NODE_ROOT,
    allowed: frozenset[int] | None = _ALLOWED,
) -> int:
    """
    Pin the current process to NUMA `node` and return the torch thread count.

    Sets the CPU affinity mask to the node's CPUs that the process may use,
    and the torch thread count and the OMP/MKL env vars to their number.
    Raises ValueError when the node has no such CPU. Without the affinity
    calls it changes nothing and returns the current thread count.

    Call this once, before the hot loop. `torch.set_num_threads` takes effect
    immediately; OMP_NUM_THREADS / MKL_NUM_THREADS are honored only by libs
    that read them at init, so for maximum effect also export them in the
    shell before launching the process.
    """
    if allowed is None:
        return torch.get_num_threads()
    cpus = cpus_on_node(node, root=root, allowed=allowed)
    if not cpus:
        msg = f"NUMA node {node} has no CPU this process may use"
        raise ValueError(msg)
    os.sched_setaffinity(0, cpus)
    n = len(cpus)
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    return n
