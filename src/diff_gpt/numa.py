"""
NUMA topology helpers for pinning a process to one node's CPUs.

Reads /sys/devices/system/node/ (Linux) to discover online nodes and their
CPU lists. On platforms that don't expose this (macOS, kernels without the
NUMA subsystem, containers with the path hidden), every function falls back
to a single virtual node covering the current affinity mask so callers get a
silent no-op instead of a crash.
"""
import os
from pathlib import Path

_NODE_ROOT = Path("/sys/devices/system/node")


def list_numa_nodes() -> list[int]:
    """Return ids of online NUMA nodes. [0] when topology isn't exposed."""
    if not _NODE_ROOT.exists():
        return [0]
    nodes: list[int] = []
    for entry in _NODE_ROOT.iterdir():
        name = entry.name
        if name.startswith("node") and name[4:].isdigit():
            nodes.append(int(name[4:]))
    return sorted(nodes) or [0]


def _parse_cpulist(text: str) -> list[int]:
    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def cpus_on_node(node: int) -> list[int]:
    """CPU ids belonging to NUMA `node`; falls back to current affinity."""
    cpulist = _NODE_ROOT / f"node{node}" / "cpulist"
    if not cpulist.exists():
        return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]
    cpus = _parse_cpulist(cpulist.read_text().strip())
    if cpus:
        return cpus
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [0]


def pin_to_node(node: int) -> int:
    """
    Pin the current process to NUMA `node` — CPU affinity mask, torch thread
    count, and OMP/MKL env vars. Returns the number of CPUs pinned.

    Call this once, before the hot loop. `torch.set_num_threads` takes effect
    immediately; OMP_NUM_THREADS / MKL_NUM_THREADS are honored only by libs
    that read them at init, so for maximum effect also export them in the
    shell before launching the process.
    """
    import torch

    cpus = cpus_on_node(node)
    if hasattr(os, "sched_setaffinity") and cpus:
        os.sched_setaffinity(0, set(cpus))
    n = max(1, len(cpus))
    torch.set_num_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    return n
