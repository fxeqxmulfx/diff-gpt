"""Replay and summarize every frozen compiled XSA regression outcome."""

import hashlib
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import sys
    runs = Path(sys.argv[1])
    torch.set_num_threads(1)
    protocol = json.loads((HERE / "validation_protocol.json").read_text())
    execution = json.loads((runs / "execution.json").read_text())
    assert execution["status"] == "completed_with_all_outcomes_retained"
    assert all(row["returncode"] == 0 for row in execution["cases"])
    path = ROOT / "tests/diff_gpt/application/test_div97.py"
    spec = importlib.util.spec_from_file_location("original_div97", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    numerics = json.loads((runs / "numerics.json").read_text())
    assert numerics["pass"]
    results = []
    for label in protocol["learning_cases"]:
        row = json.loads((runs / f"{label}.json").read_text())
        assert row["pass"] and row["python"].startswith("3.15.") and row["torch"].startswith("2.15.")
        assert row["protocol_sha256"] == digest(HERE / "validation_protocol.json")
        assert row["script_sha256"] == digest(HERE / "validate.py")
        assert row["original_test_sha256"] == digest(path)
        assert row["checkpoint_sha256"] == digest(runs / f"{label}.pt")
        checkpoint = torch.load(runs / f"{label}.pt", map_location="cpu", weights_only=True)
        group = checkpoint["optimizer"]["param_groups"][0]
        assert checkpoint["step"] == group["step"] == 1000 and group["training"] is False
        variant, seed = label.split("-")
        loader = original._DivisionLoader(int(seed), device="cpu")
        # Train() draws1000 training batches and two batches at each of101 evaluations.
        # Both halves have4656 rows, so every call has identical randint bounds.
        for _ in range(1000 + 2*101):
            torch.randint(original.N_TRAIN, (original.BATCH_SIZE,), generator=loader.rng)
        assert torch.equal(checkpoint["sampler"], loader.rng.get_state())
        model = original._gpt().eval()
        model.centered_xsa = variant == "guarded"
        for block in model.blocks:
            block.sa.centered_xsa = variant == "guarded"
        model.load_state_dict(checkpoint["model"])
        states = checkpoint["optimizer"]["state"]
        assert len(states) == len(tuple(model.parameters()))
        for parameter, key in zip(model.parameters(), group["params"], strict=True):
            state = states[key]
            assert set(state) == {"base", "second"}
            for value in state.values():
                assert value.shape == parameter.shape and value.dtype == parameter.dtype
                assert bool(torch.isfinite(value).all())
            assert bool((state["second"] >= 0).all())
        accuracy = original._accuracy(model, loader.rows["val"])
        assert accuracy == row["final_accuracy"] == 1.0
        row["replay"] = {"cpu_heldout_accuracy": accuracy,
                         "saved_average_mode": True, "sampler_exact": True,
                         "complete_optimizer_step": group["step"]}
        results.append(row)
    metrics = {}
    flops_per_update = original._gpt().estimate_flops()*original.BATCH_SIZE*original.BLOCK_SIZE
    assert abs(flops_per_update*1000/original.TRAIN_FLOPS_BASELINE-1) < .01
    for variant in ("legacy", "guarded"):
        crossings = [r["first99"] for r in results if r["case"].startswith(variant)]
        flops = flops_per_update*sum(crossings)/len(crossings)
        relative = abs(flops/original.FLOPS_TO_TARGET_BASELINE-1)
        metrics[variant] = {"crossings": crossings, "mean_updates": sum(crossings)/len(crossings),
                            "mean_flops_to99": flops, "historical_flops_relative_difference": relative,
                            "within_original_10percent_band": relative < original.FLOPS_TO_TARGET_TOLERANCE}
        assert metrics[variant]["within_original_10percent_band"]
    report = {
        "execution": execution, "protocol": protocol, "numerics": numerics,
        "learning": results, "historical_comparison": metrics,
        "protocol_correction": "The protocol's descriptive historical mean400 is rounded. Original ledger crossings400/430/360 have exact mean396.6667; the unchanged original FLOPs baseline5.20e11 and10% tolerance are used here.",
        "scope": "Compiled forward/backward adaptation of the existing loader and train loop; original foreach ScheduleFreeAdamW; no baseline update. Six final averages independently replayed, complete optimizer/sampler checked. No statistically established learning-speed improvement.",
    }
    (HERE / "validation_results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
