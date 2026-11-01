"""
Known-solution depth retrieval before integrating Attention Residuals.

The inputs are eight shuffled source vectors. One has a positive marker;
seven have a negative marker. Independent Gaussian payloads prevent a
constant answer. A fixed, explicit unit-RMS query defines the teacher's
finite-epsilon mixture. Only the student's query is trained. This isolates
score saturation and query conditioning; it does not test a full transformer.
All learning uses the repository's ScheduleFreeAdamW. Cases and success
criteria are written before training, with every outcome retained.
"""

import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from diff_gpt.domain.model.attention_residual import depth_attention, depth_scores
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

if TYPE_CHECKING:
    from torch import Tensor

WIDTH = 16
SOURCES = 8
EPS = 1e-6
UPDATES = 2000
BATCH = 64
SEEDS = (11, 17, 23)
MSE_TARGET = 1e-3
ACCURACY_TARGET = 0.99


@dataclass(frozen=True)
class ProbeCase:
    """A fixed scoring rule and initial query RMS, not a selected checkpoint."""

    mode: str
    scale: float
    seed: int

    @property
    def label(self) -> str:
        """Name the recorded case without depending on its outcome."""
        return f"{self.mode}-scale{self.scale:g}-s{self.seed}"


CASES = tuple(
    ProbeCase(mode, scale, seed)
    for mode, scales in (("key_only", (0.0, 1.0, 1e6)), ("qk", (0.0, 1e-8, 1.0, 1e6)))
    for scale in scales
    for seed in SEEDS
)


def dataset(seed: int, rows: int = 512) -> tuple[Tensor, Tensor, Tensor]:
    """Build the teacher directly from scalar normalization equations."""
    rng = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, SOURCES, WIDTH, generator=rng, dtype=torch.float64) / 2
    chosen = torch.randint(SOURCES, (rows,), generator=rng)
    values[:, :, 0] = -math.sqrt(WIDTH)
    values[torch.arange(rows), chosen, 0] = math.sqrt(WIDTH)
    # q*=sqrt(WIDTH)*e0 has RMS one; no student function defines the targets.
    scores = values[:, :, 0] / ((values.square().mean(-1) + EPS) * (1 + EPS)).sqrt()
    targets = (scores.softmax(-1).unsqueeze(-1) * values).sum(-2)
    return values.float(), targets.float(), chosen


class Probe(nn.Module):
    """Train a pseudo-query with the paper's key-only or corrected Q/K rule."""

    def __init__(self, case: ProbeCase) -> None:
        """Use the same wrong direction at each nonzero scale."""
        super().__init__()
        self.mode = case.mode
        query = torch.zeros(WIDTH)
        query[0], query[1] = -math.sqrt(WIDTH / 2), math.sqrt(WIDTH / 2)
        self.query = nn.Parameter(query * case.scale)

    def scores(self, values: Tensor) -> Tensor:
        """Compute key-only scores with unit gains, or normalized Q/K scores."""
        if self.mode == "qk":
            return depth_scores(values, self.query, EPS)
        return (F.rms_norm(values, (WIDTH,), eps=EPS) * self.query).sum(-1)

    def forward(self, values: Tensor) -> Tensor:
        """Aggregate original values under the chosen scoring rule."""
        if self.mode == "qk":
            return depth_attention(values, self.query, EPS)
        return (self.scores(values).softmax(-1).unsqueeze(-1) * values).sum(-2)


def observe(
    model: Probe,
    optimizer: ScheduleFreeAdamW,
    splits: dict[str, tuple[Tensor, Tensor, Tensor]],
    step: int,
) -> dict:
    """Measure average-weight errors and training-query gradients separately."""
    optimizer.eval()
    record = {"step": step, "splits": {}}
    with torch.no_grad():
        for name, (values, targets, chosen) in splits.items():
            scores = model.scores(values)
            weights = scores.softmax(-1)
            output = (weights.unsqueeze(-1) * values).sum(-2)
            record["splits"][name] = {
                "mse": float(F.mse_loss(output, targets)),
                "routing_accuracy": float((scores.argmax(-1) == chosen).float().mean()),
                "useful_mass": float(weights.gather(1, chosen[:, None]).mean()),
                "max_abs_score": float(scores.abs().max()),
                "entropy": float(
                    -(weights * weights.clamp_min(1e-30).log()).sum(-1).mean()
                ),
            }
    optimizer.train()
    values, targets, _ = splits["train"]
    (gradient,) = torch.autograd.grad(F.mse_loss(model(values), targets), model.query)
    record["query_rms"] = float(model.query.detach().square().mean().sqrt())
    record["query_gradient_l2"] = float(gradient.norm())
    record["finite_gradient"] = bool(torch.isfinite(gradient).all())
    return record


def run_case(case: ProbeCase, output: Path) -> dict:
    """Run all2000 updates; retain curves, complete optimizer state and failures."""
    started = time.perf_counter()
    torch.compiler.reset()
    torch.manual_seed(case.seed)
    model = Probe(case).cuda()
    splits = {
        name: tuple(x.cuda() for x in dataset(case.seed + offset))
        for name, offset in (("train", 1000), ("heldout", 2000))
    }
    hashes = {
        name: hashlib.sha256(values.cpu().numpy().tobytes()).hexdigest()
        for name, (values, _, _) in splits.items()
    }
    optimizer = ScheduleFreeAdamW(
        model.parameters(), lr=0.01, betas=(0.9, 0.98), weight_decay=0.0, warmup=10
    )
    compiled = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)
    rng = torch.Generator().manual_seed(case.seed + 3000)
    curve = [observe(model, optimizer, splits, 0)]
    values, targets, _ = splits["train"]
    for step in range(1, UPDATES + 1):
        indices = torch.randint(len(values), (BATCH,), generator=rng).cuda()
        torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(compiled(values[indices]), targets[indices])
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            curve.append(observe(model, optimizer, splits, step))
    torch.cuda.synchronize()
    final = curve[-1]["splits"]["heldout"]
    result = {
        "case": asdict(case),
        "label": case.label,
        "curves": curve,
        "input_sha256": hashes,
        "wall_seconds": time.perf_counter() - started,
        "success": final["mse"] <= MSE_TARGET
        and final["routing_accuracy"] >= ACCURACY_TARGET,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler": rng.get_state(),
            "step": UPDATES,
        },
        output / f"{case.label}.pt",
    )
    (output / f"{case.label}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(output: Path) -> dict:
    """Freeze the diagnostic cohort and run it once on the requested runtime."""
    if not platform.python_version().startswith(
        "3.15."
    ) or not torch.__version__.startswith("2.15."):
        msg = "This diagnostic requires Python3.15 and Torch2.15"
        raise RuntimeError(msg)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        msg = "Existing protocol: choose a new declared attempt directory"
        raise FileExistsError(msg)
    protocol = {
        "cases": [asdict(case) for case in CASES],
        "width": WIDTH,
        "sources": SOURCES,
        "eps": EPS,
        "updates": UPDATES,
        "batch": BATCH,
        "optimizer": {
            "name": "ScheduleFreeAdamW",
            "lr": 0.01,
            "betas": [0.9, 0.98],
            "weight_decay": 0.0,
            "warmup": 10,
        },
        "success": "final heldout MSE<=.001 and routing accuracy>=.99",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Query-only known-solution routing, not full-transformer learning.",
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    results = []
    for case in CASES:
        result = run_case(case, output)
        results.append(result)
        print(
            json.dumps(
                {
                    "label": case.label,
                    "success": result["success"],
                    "final": result["curves"][-1],
                }
            ),
            flush=True,
        )
    report = {"protocol": protocol, "results": results}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
