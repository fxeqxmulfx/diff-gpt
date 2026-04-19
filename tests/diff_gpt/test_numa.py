import os

import numpy as np
import pandas as pd
import pytest
import torch

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
