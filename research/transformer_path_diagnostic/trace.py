"""Observe the actual executed graph without replacing model equations."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch
from torch.nn import functional as F

from diff_gpt.domain.model import centered_mini
from research.early99.experiments import scalable_softmax
from research.early99_reporting.round2_audit import require

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self


class Trace:
    """Scoped module/function observers; return every original output unchanged."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Allocate observers without changing tensors, functions or random state."""
        self.model = model
        self.stack: list[str] = []
        self.live: dict[str, torch.Tensor] = {}
        self.values: dict[str, torch.Tensor] = {}
        self.gradients: dict[str, dict[str, torch.Tensor]] = {}
        self.objective: str | None = None
        self.guards: dict[str, torch.Tensor] = {}
        self.tokens: torch.Tensor | None = None
        self.handles = []
        self.context = ExitStack()
        self.rope_calls: dict[str, int] = {}
        self.modules: dict[str, torch.nn.Module] = {}

    def save(self, name: str, value: torch.Tensor) -> None:
        """Snapshot the activation now and register its gradient at this graph site."""
        require(name not in self.live, f"Repeated observed graph site: {name}")
        self.live[name] = value
        self.values[name] = value.detach().clone()
        if value.requires_grad:

            def receive(gradient: torch.Tensor) -> None:
                if self.objective is not None:
                    self.gradients.setdefault(self.objective, {})[name] = (
                        gradient.detach().clone()
                    )

            self.handles.append(value.register_hook(receive))

    def enter_module(self, name: str) -> Callable:
        """Record the first tensor input while retaining module nesting."""

        def enter(_module: torch.nn.Module, args: tuple, _kwargs: dict) -> None:
            self.stack.append(name)
            if args and isinstance(args[0], torch.Tensor):
                if args[0].is_floating_point():
                    self.save(name + ".input", args[0])
                elif name == "token_embedding_table":
                    self.tokens = args[0].detach().clone()

        return enter

    def leave_module(self, name: str) -> Callable:
        """Observe the actual output and prefix guard before returning it unchanged."""

        def leave(_module: torch.nn.Module, _args: tuple, output: object) -> None:
            if output is not None:
                value = output[0] if isinstance(output, tuple) else output
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    self.save(name + ".output", value)
                if (
                    isinstance(output, tuple)
                    and len(output) > 1
                    and isinstance(output[1], torch.Tensor)
                    and output[1].dtype == torch.bool
                ):
                    self.guards[name] = output[1].detach().clone()
            require(self.stack.pop() == name, "Unbalanced module observer")

        return leave

    def function(self, original: Callable, label: str) -> Callable:
        """Observe an existing functional operation exactly once, within its module."""

        def call(*args: object, **kwargs: object) -> torch.Tensor:
            active = bool(self.stack)
            name = self.stack[-1] + "." + label if active else ""
            if active and label == "rope":
                count = self.rope_calls.get(name, 0)
                self.rope_calls[name] = count + 1
                name += ".q" if count == 0 else ".k"
            if active:
                count = {"xsa": 3, "attention": 4}.get(label, 1)
                for index, value in enumerate(args[:count]):
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        self.save(name + f".input{index}", value)
            output = original(*args, **kwargs)
            if active:
                self.save(name + ".output", output)
            return output

        return call

    def __enter__(self) -> Self:
        """Install reversible observers only for this diagnostic execution."""
        for path, module in self.model.named_modules():
            name = path or "gpt"
            self.modules[name] = module
            self.handles.append(
                module.register_forward_pre_hook(
                    self.enter_module(name), with_kwargs=True
                )
            )
            self.handles.append(
                module.register_forward_hook(self.leave_module(name), always_call=True)
            )
            if hasattr(module, "attend_dense"):
                self.context.enter_context(
                    patch.object(
                        module,
                        "attend_dense",
                        self.function(module.attend_dense, "attention"),
                    )
                )
        for owner, name, label in [
            (F, "embedding", "lookup"),
            (F, "linear", "linear"),
            (F, "silu", "silu"),
            (torch.Tensor, "softmax", "softmax"),
            (centered_mini, "apply_rope", "rope"),
            (scalable_softmax, "apply_rope", "rope"),
            (centered_mini, "centered_projection", "xsa"),
            (scalable_softmax, "centered_projection", "xsa"),
        ]:
            self.context.enter_context(
                patch.object(owner, name, self.function(getattr(owner, name), label))
            )
        return self

    def __exit__(self, *exc: object) -> bool:
        """Restore functional entrypoints and remove all tensor/module hooks."""
        self.objective = None
        self.context.close()
        for handle in self.handles:
            handle.remove()
        return False
