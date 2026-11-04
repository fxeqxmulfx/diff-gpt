"""Native AdamW and CUDA graphs; model arithmetic is unchanged."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from torch import nn

from diff_gpt.domain.model.gpt import GPT

if TYPE_CHECKING:
    from research.current_grokking.interfaces import Task
    from research.current_grokking.protocol import Case


class Logits(nn.Module):
    """Expose public GPT logits to the reused task evaluators."""

    def __init__(self, model: GPT) -> None:
        """Initialize the declared model and execution state."""
        super().__init__()
        if type(model) is not GPT:
            msg = "This cohort requires the current public GPT"
            raise TypeError(msg)
        self.model = model

    def forward(
        self, tokens: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return current-GPT logits at the requested positions."""
        return self.model(tokens, positions=positions)[0]


class Stepper:
    """Capture complete updates without consuming scientific warmup steps."""

    def __init__(
        self,
        model: Logits,
        task: Task,
        case: Case,
        device: str,
        *,
        compiled: bool = True,
    ) -> None:
        """Initialize the declared model and execution state."""
        self.model, self.task, self.case, self.device = model, task, case, device
        self.rate = torch.zeros((), device=device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.rate,
            betas=(0.9, 0.98),
            weight_decay=case.weight_decay,
            eps=1e-8,
            capturable=device == "cuda",
            fused=device == "cuda",
            foreach=False if device != "cuda" else None,
        )
        self.graphs, self.evaluations = {}, {}

        def objective(batch: torch.Tensor) -> torch.Tensor:
            output, targets = task.forward(model, batch)
            return task.loss(output, targets)

        self.objective = (
            torch.compile(
                objective, fullgraph=True, options={"triton.cudagraphs": False}
            )
            if compiled and device == "cuda"
            else objective
        )

    def update(self, index: torch.Tensor) -> torch.Tensor:
        """Issue the exact operations later replayed by a captured update."""
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.objective(self.task.inputs(index, static=True))
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def prepare(self, sizes: list[int]) -> None:
        """Warm and capture after restoring parameters, moments and RNG."""
        if self.device != "cuda":
            return
        values = [p.detach().clone() for p in self.model.parameters()]
        moments = copy.deepcopy(self.optimizer.state_dict())
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for size in sizes:
                for _ in range(3):
                    self.update(torch.arange(size, device=self.device))
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with torch.no_grad():
            for parameter, value in zip(self.model.parameters(), values, strict=True):
                parameter.copy_(value)
            if moments["state"]:
                self.optimizer.load_state_dict(moments)
                self.optimizer.param_groups[0]["lr"] = self.rate
            else:
                for state in self.optimizer.state.values():
                    for value in state.values():
                        value.zero_()
        self.optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        for size in sizes:
            index = torch.zeros(size, dtype=torch.long, device=self.device)
            graph = torch.cuda.CUDAGraph()
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph):
                loss = self.update(index)
            self.graphs[size] = (graph, index, loss)
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, indices: torch.Tensor, completed: int) -> None:
        """Apply one update at the declared warmup rate."""
        rate = (
            self.case.learning_rate * min(1.0, completed / self.case.warmup)
            if self.case.warmup
            else self.case.learning_rate
        )
        self.rate.fill_(rate)
        if self.device == "cuda":
            graph, index, _ = self.graphs[len(indices)]
            index.copy_(indices.to(self.device))
            graph.replay()
        else:
            self.model.train()
            self.update(indices)

    @torch.no_grad()
    def evaluate(self, name: str, batch: int = 256) -> dict:
        """Score every example of the named split."""
        self.model.eval()
        rows = self.task.splits[name]
        key = id(rows)
        if key not in self.evaluations:
            sums = self.task.accumulator()

            def calculate() -> None:
                sums.zero_()
                for start in range(0, len(rows), batch):
                    self.task.accumulate(
                        self.model, rows[start : start + batch], sums, static=True
                    )

            if self.device == "cuda":
                calculate()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    calculate()
            else:
                graph = None
            self.evaluations[key] = graph, sums
        graph, sums = self.evaluations[key]
        if graph is not None:
            graph.replay()
        else:
            sums.zero_()
            for start in range(0, len(rows), batch):
                self.task.accumulate(
                    self.model, rows[start : start + batch], sums, static=True
                )
        return self.task.metrics(sums.tolist(), rows)
