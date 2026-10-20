"""
NUMA-aware multi-process training: one worker process per NUMA node.

Gradients are averaged across workers via `torch.distributed` DDP.

Each worker pins itself to its node's CPUs (see `numa.pin_to_node`),
builds its own model and loader from user-supplied factories, wraps the
model in `DistributedDataParallel`, and runs the standard `train()`
loop. `_estimate_loss` all-reduces eval losses so every rank makes the
same early-stop / save-best decision — without that, ranks desync and
DDP deadlocks at a collective.

The model's device picks the backend: Gloo for a CPU model, NCCL for a CUDA
model, which needs a GPU per rank.

Pattern:

    from diff_gpt.train_ddp import train_numa

    def build_model(rank: int) -> BaseGPT:
        return GPT(vocab_size=..., n_embd=..., ...)

    def build_loader(rank: int, world: int) -> DataLoader:
        # rank-dependent seed so each worker draws different batches
        rng = torch.Generator().manual_seed(42 + rank)
        return DiffDataFrameDataLoader(dfs=..., rng=rng, ...)

    val_loss, t = train_numa(
        model_factory=build_model,
        loader_factory=build_loader,
        train_kwargs={"max_iters": 5000},
        checkpoint_path="gpt.pt",
    )
"""

import os
import socket
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from diff_gpt.numa import list_numa_nodes, pin_to_node
from diff_gpt.train import train

if TYPE_CHECKING:
    from collections.abc import Callable

    from diff_gpt.model.gpt import BaseGPT
    from diff_gpt.train import DataLoader


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _backend(model: BaseGPT, world_size: int) -> str:
    """
    Return the process-group backend for `model`: Gloo on CPU, NCCL on CUDA.

    NCCL refuses two ranks on one GPU ("Duplicate GPU detected"), so a CUDA
    model with more ranks than GPUs raises ValueError.
    """
    if model.device.type != "cuda":
        return "gloo"
    n_gpus = torch.cuda.device_count()
    if world_size > n_gpus:
        msg = (
            f"NCCL needs a GPU per rank, but {world_size} ranks share "
            f"{n_gpus} GPU(s): build the model on the CPU or train on fewer nodes"
        )
        raise ValueError(msg)
    return "nccl"


def _unwrapped(model: torch.nn.Module) -> torch.nn.Module:
    """Return the module that `torch.compile` wrapped, else `model` itself."""
    return getattr(model, "_orig_mod", model)


def _worker(
    rank: int,
    world_size: int,
    node_ids: list[int],
    model_factory: Callable[[int], BaseGPT],
    loader_factory: Callable[[int, int], DataLoader],
    train_kwargs: dict[str, Any],
    master_addr: str,
    master_port: int,
    checkpoint_path: str | None,
    result_queue: mp.Queue,
) -> None:
    pin_to_node(node_ids[rank])

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # When CUDA is present, make each rank's own GPU the current device before
    # the model is built, so that `.to("cuda")` in the factory spreads the
    # ranks over the GPUs instead of stacking them on device 0.
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    model = model_factory(rank)
    dist.init_process_group(
        backend=_backend(model, world_size),
        rank=rank,
        world_size=world_size,
    )
    try:
        loader = loader_factory(rank, world_size)
        ddp = DDP(model)
        # DDP forwards every BaseGPT attribute / method used by train(), such
        # as parameters, state_dict, train/eval, __call__ and load_state_dict,
        # but isn't a BaseGPT subclass, hence the ignore.
        result = train(ddp, loader, **train_kwargs)  # pyright: ignore[reportArgumentType]
        if rank == 0:
            if checkpoint_path is not None:
                # Unwrap DDP and torch.compile, so the saved keys match a
                # standalone load of the user's bare model class.
                torch.save(_unwrapped(ddp.module).state_dict(), checkpoint_path)
            result_queue.put(result)
    finally:
        dist.destroy_process_group()


def train_numa(
    *,
    model_factory: Callable[[int], BaseGPT],
    loader_factory: Callable[[int, int], DataLoader],
    train_kwargs: dict[str, Any] | None = None,
    nodes: list[int] | None = None,
    checkpoint_path: str | None = None,
    master_addr: str = "127.0.0.1",
    master_port: int | None = None,
) -> tuple[float, int]:
    """
    Spawn a worker per entry in `nodes`, pin each to its node's CPUs and train via DDP.

    `nodes` defaults to every NUMA node with a CPU this process may use.

    `model_factory(rank) -> BaseGPT` constructs the model *inside* the
    worker. Using a factory (instead of a pre-built model) avoids
    cross-process pickling of CUDA state and guarantees each rank's
    allocations happen on its pinned node.

    `loader_factory(rank, world_size) -> DataLoader` builds the loader in
    the worker. Seed it by `rank` so workers draw different batches — if
    they see the same data, DDP averages identical gradients and you get
    no scaling benefit.

    `train_kwargs` is forwarded verbatim to `train()`. With more than one
    node, an `on_eval` in it runs in rank 0's worker process, so it must
    pickle (a module-level function) and keep what it records in files.

    `checkpoint_path`: rank 0 writes the unwrapped model `state_dict` here
    after training finishes, letting the launcher reload weights without
    sharing memory with the workers. The keys are those of the bare model
    class, also when the factory returns a `torch.compile`d model. The
    returned `(val_loss, secs)` is rank 0's.

    Workers use Gloo for a CPU model and NCCL for a CUDA model; with NCCL
    every rank needs a GPU of its own, so a CUDA model with more nodes than
    GPUs fails. Each rank's current CUDA device is GPU `rank`, so build the
    model with `.to("cuda")`.

    Single-node fast path: if `len(nodes) == 1`, no spawn and no DDP —
    just pin the current process and call `train()` directly.
    """
    if nodes is None:
        nodes = list_numa_nodes()
    if len(nodes) == 0:
        msg = "nodes must be non-empty"
        raise ValueError(msg)
    world = len(nodes)
    train_kwargs = dict(train_kwargs or {})

    if world == 1:
        pin_to_node(nodes[0])
        model = model_factory(0)
        loader = loader_factory(0, 1)
        result = train(model, loader, **train_kwargs)
        if checkpoint_path is not None:
            torch.save(_unwrapped(model).state_dict(), checkpoint_path)
        return result

    if master_port is None:
        master_port = _free_port()

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = []
    for r in range(world):
        p = ctx.Process(
            target=_worker,
            args=(
                r,
                world,
                nodes,
                model_factory,
                loader_factory,
                train_kwargs,
                master_addr,
                master_port,
                checkpoint_path,
                q,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    for p in procs:
        if p.exitcode != 0:
            msg = f"NUMA worker (pid {p.pid}) exited with code {p.exitcode}"
            raise RuntimeError(msg)
    return q.get()
