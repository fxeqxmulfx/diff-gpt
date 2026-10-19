# Copyright 2025 Mikhail Borisov
"""Tests for the NUMA pinning helpers and NUMA-aware DDP training."""

import os
import socket
from datetime import timedelta
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from diff_gpt.data_loader import DiffDataFrameDataLoader
from diff_gpt.encoder_decoder import get_domain_of_definition
from diff_gpt.model.gpt import GPT
from diff_gpt.numa import (
    _parse_cpulist,
    cpus_on_node,
    list_numa_nodes,
    pin_to_node,
)
from diff_gpt.train import train
from diff_gpt.train_ddp import _backend, train_numa

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

# Seconds that a spawned test worker may wait at a collective before failing.
COLLECTIVE_TIMEOUT_S = 30


@pytest.fixture
def _restore_numa_state() -> Iterator[None]:
    """
    Snapshot and restore the process-global state that NUMA pinning mutates.

    Pinning mutates process-global state (affinity, thread count, env
    vars) that persists across tests. Snapshot and restore so one test
    doesn't narrow the CPU set for the next.
    """
    threads = torch.get_num_threads()
    affinity = (
        set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    omp = os.environ.get("OMP_NUM_THREADS")
    mkl = os.environ.get("MKL_NUM_THREADS")
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        if affinity is not None:
            os.sched_setaffinity(0, affinity)
        for k, v in (("OMP_NUM_THREADS", omp), ("MKL_NUM_THREADS", mkl)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_parse_cpulist_ranges_and_singletons() -> None:
    """Check that _parse_cpulist expands ranges and singletons and maps '' to []."""
    assert _parse_cpulist("0-3") == [0, 1, 2, 3]
    assert _parse_cpulist("0-1,4,6-7") == [0, 1, 4, 6, 7]
    assert _parse_cpulist("5") == [5]
    assert _parse_cpulist("") == []


def test_list_numa_nodes_returns_at_least_one() -> None:
    """Check that list_numa_nodes() is non-empty and holds only non-negative ints."""
    nodes = list_numa_nodes()
    assert len(nodes) >= 1
    assert all(isinstance(n, int) and n >= 0 for n in nodes)


def test_cpus_on_node_nonempty_for_first_node() -> None:
    """Check that cpus_on_node gives the first node at least one CPU, all ints >= 0."""
    cpus = cpus_on_node(list_numa_nodes()[0])
    assert len(cpus) >= 1
    assert all(isinstance(c, int) and c >= 0 for c in cpus)


@pytest.mark.usefixtures("_restore_numa_state")
def test_pin_to_node_sets_thread_count_and_affinity() -> None:
    """Check that pin_to_node sets thread count, OMP/MKL env vars and CPU affinity."""
    n = pin_to_node(list_numa_nodes()[0])
    assert n >= 1
    assert torch.get_num_threads() == n
    assert os.environ["OMP_NUM_THREADS"] == str(n)
    assert os.environ["MKL_NUM_THREADS"] == str(n)
    if hasattr(os, "sched_getaffinity"):
        # Pinned affinity must be a subset of the requested node CPUs.
        assert set(os.sched_getaffinity(0)).issubset(
            set(cpus_on_node(list_numa_nodes()[0])),
        )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _build_cpu_model(rank: int) -> GPT:
    torch.manual_seed(0 + rank)
    return GPT(vocab_size=64, n_embd=16, block_size=32, n_head=2, n_layer=1)


def _make_model_factory() -> Callable[[int], GPT]:
    return _build_cpu_model


def _make_loader_factory(
    train_part: float = 0.8,
) -> Callable[[int, int], DiffDataFrameDataLoader]:
    def build(rank: int, _world: int) -> DiffDataFrameDataLoader:
        df = pd.DataFrame({"c0": np.arange(200, dtype=np.float64)})
        dom = get_domain_of_definition(
            df.to_numpy(dtype=np.float64),
            order_of_derivative=1,
            use_decimal=False,
        )
        rng = torch.Generator().manual_seed(42 + rank)
        return DiffDataFrameDataLoader(
            dfs=[df],
            block_size=32,
            batch_size=2,
            vocab_size=64,
            order_of_derivative=1,
            domain_of_definition=dom,
            use_decimal=False,
            device="cpu",
            train_part=train_part,
            rng=rng,
        )

    return build


def _build_loader(rank: int, world: int) -> DiffDataFrameDataLoader:
    """Module-level loader factory: spawned workers can unpickle it."""
    return _make_loader_factory()(rank, world)


@pytest.fixture
def _single_rank_gloo_pg() -> Iterator[None]:
    """
    Stand up a single-rank gloo process group on a free port, tear it down afterward.

    Lets tests exercise DDP's reducer (which enforces the unused-parameter
    check regardless of world size) without spawning child processes.
    """
    prev = {
        k: os.environ.get(k)
        for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(_free_port()),
        RANK="0",
        WORLD_SIZE="1",
    )
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.usefixtures("_single_rank_gloo_pg")
def test_plain_ddp_runs_gpt_training_steps() -> None:
    """
    Check that plain DDP runs two forward+backward cycles and every param gets a grad.

    train_ddp wraps the model in a plain `DDP(model)` (no
    find_unused_parameters / static_graph). DDP's reducer raises on the
    second backward if any parameter went without a gradient, so two
    forward+backward cycles must complete — with activation checkpointing
    on, as in training — and every parameter must hold a gradient.
    """
    torch.manual_seed(0)
    model = GPT(vocab_size=64, n_embd=16, block_size=32, n_head=2, n_layer=2)
    model.train()
    ddp = DDP(model)
    X = torch.randint(0, 64, (2, 16), dtype=torch.int64)
    Y = torch.randint(0, 64, (2, 16), dtype=torch.int64)
    for _ in range(2):
        _, loss = ddp(X, Y)
        loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name


@pytest.mark.usefixtures("_restore_numa_state")
def test_train_numa_single_node_fast_path(tmp_path: Path) -> None:
    """
    Check that single-node train_numa returns a finite loss and writes a checkpoint.

    nodes=[0] bypasses spawn and calls train() in-process — exercises
    the pin-and-train path that handles the common single-socket case.
    """
    ckpt = tmp_path / "best.pt"
    val_loss, secs = train_numa(
        model_factory=_make_model_factory(),
        loader_factory=_make_loader_factory(),
        train_kwargs={
            "max_iters": 5,
            "eval_interval": 5,
            "eval_iters": 2,
            "use_tqdm": False,
            "use_early_stopping": False,
        },
        nodes=[list_numa_nodes()[0]],
        checkpoint_path=str(ckpt),
    )
    assert np.isfinite(val_loss)
    assert secs >= 0
    assert ckpt.exists()
    assert ckpt.stat().st_size > 0


def _fake_topology(root: Path, cpulists: dict[int, str]) -> Path:
    """Write a sysfs-like NUMA tree under `root`: node<n>/cpulist per node."""
    for node, cpulist in cpulists.items():
        (root / f"node{node}").mkdir(parents=True)
        (root / f"node{node}" / "cpulist").write_text(cpulist + "\n")
    (root / "online").write_text("0\n")  # sysfs has non-node entries too
    return root


def test_nodes_and_cpus_are_limited_to_the_allowed_cpus(tmp_path: Path) -> None:
    """
    Check that nodes and their CPUs follow the CPUs the process may use.

    The CPU lists were the nodes' own, not the allowed ones (cpuset
    containers, SLURM), so a node without allowed CPUs failed in
    sched_setaffinity, and memory-only nodes (CXL, HBM) fell back to the full
    affinity, overlapping the other ranks.
    """
    root = _fake_topology(tmp_path, {0: "0-3", 1: "4-7", 2: "", 3: "8-11"})
    allowed = frozenset({1, 2, 5, 6, 7})
    assert list_numa_nodes(root=root, allowed=allowed) == [0, 1]
    assert cpus_on_node(0, root=root, allowed=allowed) == [1, 2]
    assert cpus_on_node(1, root=root, allowed=allowed) == [5, 6, 7]
    assert cpus_on_node(2, root=root, allowed=allowed) == []
    assert cpus_on_node(3, root=root, allowed=allowed) == []
    assert cpus_on_node(9, root=root, allowed=allowed) == []
    for node in (2, 3, 9):
        with pytest.raises(ValueError, match=f"node {node} has no CPU"):
            pin_to_node(node, root=root, allowed=allowed)


def test_without_a_topology_every_allowed_cpu_forms_node_0(tmp_path: Path) -> None:
    """Check the virtual node: one node 0 that holds every allowed CPU."""
    missing = tmp_path / "missing"
    allowed = frozenset({3, 1})
    assert list_numa_nodes(root=missing, allowed=allowed) == [0]
    assert cpus_on_node(0, root=missing, allowed=allowed) == [1, 3]


@pytest.mark.usefixtures("_restore_numa_state")
def test_pin_is_a_no_op_without_the_affinity_calls(tmp_path: Path) -> None:
    """
    Check that pinning changes nothing where the CPU affinity is unknown.

    Without os.sched_getaffinity (macOS, Windows) cpus_on_node returned [0]
    and pin_to_node cut torch to one thread, against the promised no-op.
    """
    threads = torch.get_num_threads()
    affinity = (
        set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    omp = os.environ.get("OMP_NUM_THREADS")
    root = _fake_topology(tmp_path, {0: "0"})
    assert pin_to_node(0, root=root, allowed=None) == threads
    assert torch.get_num_threads() == threads
    assert os.environ.get("OMP_NUM_THREADS") == omp
    if affinity is not None:
        assert set(os.sched_getaffinity(0)) == affinity
    assert cpus_on_node(0, root=tmp_path / "missing", allowed=None) == []


@pytest.mark.usefixtures("_restore_numa_state")
@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="no CPU affinity")
def test_pin_counts_threads_from_the_allowed_cpus_of_the_node(tmp_path: Path) -> None:
    """Check that the affinity and thread count leave out CPUs not allowed."""
    allowed = sorted(os.sched_getaffinity(0))
    cpu = allowed[0]
    root = _fake_topology(tmp_path, {0: f"{cpu},{max(allowed) + 1}"})
    assert pin_to_node(0, root=root, allowed=frozenset(allowed)) == 1
    assert set(os.sched_getaffinity(0)) == {cpu}
    assert torch.get_num_threads() == 1
    assert os.environ["OMP_NUM_THREADS"] == "1"


@pytest.mark.usefixtures("_restore_numa_state")
@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity") or len(os.sched_getaffinity(0)) < 2,  # noqa: PLR2004 - two nodes need two CPUs
    reason="needs two allowed CPUs",
)
def test_pin_moves_from_one_node_to_another(tmp_path: Path) -> None:
    """
    Check that a pinned process can be pinned to another node.

    The allowed CPUs are read at import; read after the first pin, they
    would hold only that node's CPUs.
    """
    a, b = sorted(os.sched_getaffinity(0))[:2]
    root = _fake_topology(tmp_path, {0: str(a), 1: str(b)})
    pin_to_node(0, root=root)
    assert set(os.sched_getaffinity(0)) == {a}
    pin_to_node(1, root=root)
    assert set(os.sched_getaffinity(0)) == {b}


def test_backend_follows_the_model_device() -> None:
    """
    Check that a CPU model gets Gloo and that NCCL needs a GPU per rank.

    NCCL was picked whenever CUDA was visible, so a CPU model failed ("No
    backend type associated with device type cpu"), and two ranks on one
    GPU failed with "Duplicate GPU detected".
    """
    model = _build_cpu_model(0)
    assert _backend(model, world_size=2) == "gloo"
    if torch.cuda.is_available():
        model = model.to("cuda")
        n_gpus = torch.cuda.device_count()
        assert _backend(model, world_size=n_gpus) == "nccl"
        with pytest.raises(ValueError, match="NCCL needs a GPU per rank"):
            _backend(model, world_size=n_gpus + 1)


@pytest.mark.usefixtures("_restore_numa_state")
def test_checkpoint_of_a_compiled_model_loads_into_a_plain_gpt(tmp_path: Path) -> None:
    """
    Check that the checkpoint of a torch.compile'd model has the bare keys.

    It carried `_orig_mod.*` keys, which a plain GPT cannot load.
    """

    def build_compiled(rank: int) -> GPT:
        return cast("GPT", torch.compile(_build_cpu_model(rank), backend="eager"))

    ckpt = tmp_path / "best.pt"
    train_numa(
        model_factory=build_compiled,
        loader_factory=_make_loader_factory(),
        train_kwargs={"max_iters": 1, "eval_iters": 1, "use_tqdm": False},
        nodes=[list_numa_nodes()[0]],
        checkpoint_path=str(ckpt),
    )
    _build_cpu_model(0).load_state_dict(torch.load(ckpt, weights_only=True))


@pytest.mark.slow
def test_train_numa_spawns_cpu_workers_on_gloo(tmp_path: Path) -> None:
    """
    Check that two workers train a CPU model, also where CUDA is visible.

    nodes=[n, n] takes the spawn path on one node. With CUDA visible the
    workers picked NCCL and failed on the CPU model.
    """
    ckpt = tmp_path / "best.pt"
    node = list_numa_nodes()[0]
    val_loss, _ = train_numa(
        model_factory=_build_cpu_model,
        loader_factory=_build_loader,
        train_kwargs={
            "max_iters": 4,
            "eval_interval": 2,
            "eval_iters": 2,
            "use_tqdm": False,
            "use_early_stopping": False,
        },
        nodes=[node, node],
        checkpoint_path=str(ckpt),
    )
    assert np.isfinite(val_loss)
    _build_cpu_model(0).load_state_dict(torch.load(ckpt, weights_only=True))


def _train_with_val_only_on_rank_0(
    rank: int,
    world_size: int,
    port: int,
    reports: mp.Queue,
) -> None:
    """Train in a Gloo group where only rank 0 has a val split; report errors."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=COLLECTIVE_TIMEOUT_S),
    )
    try:
        loader = _make_loader_factory(0.8 if rank == 0 else 1.0)(rank, world_size)
        train(
            _build_cpu_model(rank),
            loader,
            max_iters=2,
            eval_interval=1,
            eval_iters=1,
            use_tqdm=False,
        )
        reports.put((rank, "trained"))
    except ValueError as e:
        reports.put((rank, str(e)))
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_train_raises_on_every_rank_when_has_val_differs() -> None:
    """
    Check that ranks that disagree on has_val() all raise ValueError.

    A rank with a val split ran one more all-reduce per evaluation than a
    rank without, so Gloo failed on mismatched collectives (NCCL would
    hang until its timeout).
    """
    ctx = mp.get_context("spawn")
    reports = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_train_with_val_only_on_rank_0, args=(r, 2, port, reports))
        for r in range(2)
    ]
    for proc in procs:
        proc.start()
    try:
        got = dict(reports.get(timeout=4 * COLLECTIVE_TIMEOUT_S) for _ in procs)
    finally:
        for proc in procs:
            proc.join(timeout=COLLECTIVE_TIMEOUT_S)
    for rank in range(2):
        assert "differs across ranks" in got[rank], got
