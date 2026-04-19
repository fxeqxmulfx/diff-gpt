import os
import socket

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
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
from diff_gpt.train_ddp import train_numa


@pytest.fixture
def _restore_numa_state():
    """Pinning mutates process-global state (affinity, thread count, env
    vars) that persists across tests. Snapshot and restore so one test
    doesn't narrow the CPU set for the next."""
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


def test_parse_cpulist_ranges_and_singletons():
    assert _parse_cpulist("0-3") == [0, 1, 2, 3]
    assert _parse_cpulist("0-1,4,6-7") == [0, 1, 4, 6, 7]
    assert _parse_cpulist("5") == [5]
    assert _parse_cpulist("") == []


def test_list_numa_nodes_returns_at_least_one():
    nodes = list_numa_nodes()
    assert len(nodes) >= 1
    assert all(isinstance(n, int) and n >= 0 for n in nodes)


def test_cpus_on_node_nonempty_for_first_node():
    cpus = cpus_on_node(list_numa_nodes()[0])
    assert len(cpus) >= 1
    assert all(isinstance(c, int) and c >= 0 for c in cpus)


def test_pin_to_node_sets_thread_count_and_affinity(_restore_numa_state):
    n = pin_to_node(list_numa_nodes()[0])
    assert n >= 1
    assert torch.get_num_threads() == n
    assert os.environ["OMP_NUM_THREADS"] == str(n)
    assert os.environ["MKL_NUM_THREADS"] == str(n)
    if hasattr(os, "sched_getaffinity"):
        # Pinned affinity must be a subset of the requested node CPUs.
        assert set(os.sched_getaffinity(0)).issubset(
            set(cpus_on_node(list_numa_nodes()[0]))
        )


def _make_model_factory():
    def build(rank: int):
        torch.manual_seed(0 + rank)
        return GPT(vocab_size=64, n_embd=16, block_size=32, n_head=2, n_layer=1)

    return build


def _make_loader_factory():
    def build(rank: int, world: int):
        df = pd.DataFrame({"c0": np.arange(200, dtype=np.float64)})
        dom = get_domain_of_definition(
            df.to_numpy(dtype=np.float64), order_of_derivative=1, use_decimal=False
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
            train_part=0.8,
            rng=rng,
        )

    return build


@pytest.fixture
def _single_rank_gloo_pg():
    """
    Stand up a single-rank gloo process group on a free port, tear it
    down afterward. Lets tests exercise DDP's reducer (which enforces
    the unused-parameter check regardless of world size) without
    spawning child processes.
    """
    prev = {k: os.environ.get(k) for k in
            ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")}
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
        RANK="0", WORLD_SIZE="1",
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


def test_ddp_without_find_unused_raises_on_block_attn_res_shortcircuit(
    _single_rank_gloo_pg,
):
    """
    Regression proof: Block(layer_idx=0).attn_res_proj never receives a
    gradient because `block_attn_res` short-circuits when
    `partial_block is blocks[0]`. DDP's reducer reports this on the
    second backward — this test locks in the premise for the fix below.
    """
    torch.manual_seed(0)
    model = GPT(
        vocab_size=64, n_embd=16, block_size=32, n_head=2, n_layer=1,
        use_checkpoint=False,
    )
    ddp = DDP(model)  # find_unused_parameters=False by default
    X = torch.randint(0, 64, (2, 16), dtype=torch.int64)
    Y = torch.randint(0, 64, (2, 16), dtype=torch.int64)
    _, loss = ddp(X, Y)
    loss.backward()
    # The reducer flags the missing gradient on the next forward/backward.
    with pytest.raises(RuntimeError, match="unused|find_unused_parameters"):
        _, loss = ddp(X, Y)
        loss.backward()


def test_ddp_with_static_graph_runs_through_block_attn_res_shortcircuit(
    _single_rank_gloo_pg,
):
    """
    The fix in train_ddp.py: `static_graph=True` lets DDP record the
    deterministically-unused param on the first iter and stop checking
    afterward. Two forward+backward cycles must complete, and the
    short-circuited param must have either no grad or a zero grad —
    proving it really is the unused one (so the flag is load-bearing).
    """
    torch.manual_seed(0)
    model = GPT(
        vocab_size=64, n_embd=16, block_size=32, n_head=2, n_layer=1,
        use_checkpoint=False,
    )
    ddp = DDP(model, static_graph=True)
    X = torch.randint(0, 64, (2, 16), dtype=torch.int64)
    Y = torch.randint(0, 64, (2, 16), dtype=torch.int64)
    for _ in range(2):
        _, loss = ddp(X, Y)
        loss.backward()
    proj_grad = model.blocks[0].attn_res_proj.weight.grad
    if proj_grad is not None:
        assert torch.count_nonzero(proj_grad) == 0, (
            "block0.attn_res_proj.weight got a nonzero grad — the "
            "block_attn_res short-circuit no longer applies, so the "
            "static_graph=True flag may no longer be needed"
        )


def test_train_numa_single_node_fast_path(tmp_path, _restore_numa_state):
    """nodes=[0] bypasses spawn and calls train() in-process — exercises
    the pin-and-train path that handles the common single-socket case."""
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
    assert ckpt.exists() and ckpt.stat().st_size > 0
