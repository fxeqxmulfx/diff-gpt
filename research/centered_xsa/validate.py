"""Frozen compiled CUDA checks and paired div97 numerical regression.

Run one named case per process after the separate lab long phase has ended.
The original regression loader and library train() are reused directly.
Only forward/backward compilation and the legacy/guarded choice differ.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from diff_gpt.application.train import train
from diff_gpt.domain.model.gpt import GPT
from diff_gpt.domain.optimizer.schedule_free import ScheduleFreeAdamW

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EPS = torch.finfo(torch.float32).eps


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiled_forward(model):
    compiled = torch.compile(model.forward, mode="max-autotune", fullgraph=True,
                             dynamic=False)

    def forward(*args, **kwargs):
        torch.compiler.cudagraph_mark_step_begin()
        return compiled(*args, **kwargs)

    model.forward = forward


def numerics():
    torch.manual_seed(147)
    torch.set_num_threads(1)
    model = GPT(128, n_embd=128, n_head=4, n_layer=4, block_size=9,
                use_checkpoint=False).eval()
    for norm in model.modules():
        if isinstance(norm, nn.RMSNorm):
            norm.eps = EPS
    reference = copy.deepcopy(model).double()
    inputs = torch.tensor([[3]*9, [4,4,4,4,4,2,7,9,1],
                           [1,1,2,3,4,5,6,7,8], [9,2,3,4,5,6,7,8,1]])
    targets = torch.randint(128, inputs.shape)
    original_norm = F.rms_norm

    def fixed_norm(input, normalized_shape, weight=None, eps=None):
        return original_norm(input, normalized_shape, weight,
                             EPS if eps is None else eps)

    with patch.object(F, "rms_norm", fixed_norm):
        ref_logits, ref_loss = reference(inputs, targets)
        ref_grad = torch.autograd.grad(ref_loss, tuple(reference.parameters()))
    ref_logits = ref_logits.detach()
    ref_grad = torch.cat([g.detach().flatten() for g in ref_grad])
    model = model.cuda()
    inputs, targets = inputs.cuda(), targets.cuda()
    compiled_forward(model)
    results = []
    for attempt in range(3):
        model.zero_grad(set_to_none=True)
        logits, loss = model(inputs, targets)
        loss.backward()
        actual_logits = logits.detach().double().cpu().clone()
        actual_grad = torch.cat([p.grad.detach().flatten().double().cpu()
                                 for p in model.parameters()])
        delta = actual_grad - ref_grad
        row = {
            "attempt": attempt,
            "max_logit_error": float((actual_logits-ref_logits).abs().max()),
            "gradient_relative_rms_error": float(delta.norm()/ref_grad.norm()),
            "gradient_error_rms": float(delta.square().mean().sqrt()),
            "changed_predictions": int((actual_logits.argmax(-1) !=
                                        ref_logits.argmax(-1)).sum()),
            "finite": bool(torch.isfinite(actual_grad).all()),
        }
        row["pass"] = (row["finite"] and row["max_logit_error"] < 5e-4 and
                       row["gradient_relative_rms_error"] < 1e-4 and
                       row["changed_predictions"] == 0)
        results.append(row)
    return {"case": "numerics", "results": results, "pass": all(r["pass"] for r in results)}


def division(label, output):
    variant, seed = label.split("-")
    seed = int(seed)
    assert variant in ("legacy", "guarded") and seed in (11, 17, 23)
    path = ROOT / "tests/diff_gpt/application/test_div97.py"
    spec = importlib.util.spec_from_file_location("original_div97", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(seed)
    model = module._gpt().cuda()
    model.centered_xsa = variant == "guarded"
    for block in model.blocks:
        block.sa.centered_xsa = variant == "guarded"
    loader = module._DivisionLoader(seed, device="cuda")
    optimizer = ScheduleFreeAdamW(model.parameters(), lr=.01, betas=(.9,.98),
                                 weight_decay=1., eps=1e-8, warmup=10)
    compiled_forward(model)
    curve = []

    def on_eval(evaluation):
        with torch.no_grad():
            accuracy = module._accuracy(model, loader.rows["val"])
        row = {"step": evaluation.step, "heldout_accuracy": accuracy,
               "sampled_train_loss": evaluation.train_loss,
               "sampled_heldout_loss": evaluation.val_loss}
        curve.append(row)
        with (output / f"{label}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    started = time.monotonic()
    train(model, loader, max_iters=1000, eval_interval=10, eval_iters=1,
          use_tqdm=False, optimizer=optimizer, on_eval=on_eval)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "sampler": loader.rng.get_state(), "step": 1000},
               output / f"{label}.pt")
    crossings = [r["step"] for r in curve if r["heldout_accuracy"] >= .99]
    return {"case": label, "curve": curve, "first99": min(crossings) if crossings else None,
            "final_accuracy": curve[-1]["heldout_accuracy"],
            "wall_seconds_including_compile": time.monotonic() - started,
            "original_test_sha256": digest(path),
            "checkpoint_sha256": digest(output / f"{label}.pt"),
            "training_flops": model.estimate_flops()*512*2*1000,
            "pass": bool(crossings) and curve[-1]["heldout_accuracy"] >= .99}


def main():
    label, destination = sys.argv[1:]
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / f"{label}.json").exists(), "Do not overwrite a completed case"
    assert torch.cuda.is_available()
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    result = numerics() if label == "numerics" else division(label, output)
    result.update(python=sys.version, torch=torch.__version__,
                  device=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)),
                  protocol_sha256=digest(HERE / "validation_protocol.json"))
    (output / f"{label}.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "curve"}), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
