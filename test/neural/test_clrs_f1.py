# -*- coding: utf-8 -*-

"""
Part F1 of ``examples/CLRS_small`` -- DEAR's protocol on the length-free
arms.

Part F1 trains arm D and arm F-sized on DEAR's own data (Georgiev et
al., NeurIPS 2024): 10^5 trajectories at sizes 8..16, 100 validation at
``n = 16``, 100 test at ``n = 64``, lr 3e-4, batch 32, 100 epochs,
best-validation checkpoint selection sampled every epoch.  The cache is
**output-only** -- no hints, no per-sample step counts -- and the step
counts live in a sidecar file that only the one-time bound check may
open.  These tests pin the protocol: the tags, DEAR's hyperparameters,
the cache format, that the trajectory rule refuses the data, that a
training epoch shuffles across the sizes, that best-validation selection
keeps the best weights, and that both arms train and evaluate with the
sidecar file **deleted**.
"""

import importlib
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

EXAMPLE = Path(__file__).resolve().parents[2] / "docs" / "neural" \
    / "examples" / "CLRS_small"

SHARED = ("config", "dataset", "model", "train", "evaluate")


def _example(directory: Path) -> dict:
    """ The modules of one example, imported without leaving a trace. """
    saved = {name: sys.modules.pop(name, None) for name in SHARED}
    sys.path.insert(0, str(directory))
    try:
        return {name: importlib.import_module(name) for name in SHARED}
    finally:
        sys.path.remove(str(directory))
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module


if not (EXAMPLE / "data" / "bellman_ford-dear_val.npz").exists():
    pytest.skip("the DEAR cache is not built; run `python dataset.py "
                "--generate-dear` in examples/CLRS_small",
                allow_module_level=True)

CLRS = _example(EXAMPLE)
config, dataset = CLRS["config"], CLRS["dataset"]
zoo, training = CLRS["model"], CLRS["train"]

TINY = config.Widths(dim=4, state_dim=8, hidden=8, edge_dim=4, graph_dim=8)


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_the_f1_tags_are_their_own_and_every_earlier_tag_is_unchanged():
    """
    ``data`` defaults to ``"clrs30"``, so every tag of Parts A, B and D
    is byte-identical, and each F1 arm files under its own tag.
    """
    small = replace(config.H2_ARMS["O"], widths="small")
    earlier = {
        "O": small.tag,
        "F": replace(config.H2_ARMS["F"], widths="small").tag,
        "T-accum": replace(small, segment_steps=4,
                           segment_optim="per_batch").tag,
        "D": replace(small, segment_steps=4, segment_optim="per_batch",
                     depth_rule="sized").tag,
        "F-sized": replace(config.H2_ARMS["F"], widths="small",
                           depth_rule="sized").tag,
    }
    assert earlier["O"] == "p3-small-max-ptredge-probe"
    assert earlier["F"] == "p3-small-max-ptredge-probe-grounded"
    assert earlier["T-accum"] == "p3-small-max-ptredge-probe-seg4-acc"
    assert earlier["D"] == "p3-small-max-ptredge-probe-seg4-acc-szd"
    assert earlier["F-sized"] == "p3-small-max-ptredge-probe-szd-grounded"
    tags = {name: arm.tag for name, arm in config.F1_ARMS.items()}
    assert tags["D"] == "p3-max-ptredge-probe-seg4-acc-szd-dear-ev1"
    assert tags["F-sized"] == "p3-max-ptredge-probe-szd-dear-grounded-ev1"
    assert len(set(tags.values()) | set(earlier.values())) \
        == len(tags) + len(earlier)


def test_the_f1_protocol_is_dears():
    """
    DEAR's hyperparameters (``hyperparameters.py`` and ``configs/*.yaml``
    of their repository), pinned per arm: lr 3e-4, batch 32, 100 epochs,
    no weight decay, validation -- and so best-val checkpoint selection --
    sampled every epoch, on their data, under the sized rule.
    """
    for arm in config.F1_ARMS.values():
        assert arm.lr == 3e-4
        assert arm.batch_size == 32
        assert arm.epochs == 100
        assert arm.weight_decay == 0.0
        assert arm.eval_every == 1
        assert arm.data == "dear"
        assert arm.depth_rule == "sized"
        assert arm.probe
    assert config.F1_ARMS["D"].segment_steps == 4
    assert config.F1_ARMS["D"].segment_optim == "per_batch"
    assert config.F1_ARMS["D"].segment_detach
    assert config.F1_ARMS["F-sized"].solver == "grounded"
    assert config.DEAR["train"]["num_samples"] == 100_000
    assert config.DEAR["val"] == {"num_samples": 100, "num_nodes": 16,
                                  "seed": 96}
    assert config.DEAR["test"] == {"num_samples": 100, "num_nodes": 64,
                                   "seed": 241}


def test_a_dear_cache_is_output_only():
    """
    The files themselves hold inputs and outputs alone -- no hint
    arrays, no lengths -- and `read` fills the lengths field with the
    all-ones placeholder, which is literally one of the poisons of
    ``test_clrs_sized.py``'s bitwise gate.
    """
    for name in config.DEAR_SPLITS:
        stored = np.load(dataset.path_of("bellman_ford", name))
        assert "lengths" not in stored.files, name
        assert not [key for key in stored.files
                    if key.startswith("hint__")], name
        split = dataset.load("bellman_ford", name)
        assert not split.hints
        assert (split.lengths == 1).all()


def test_dear_data_refuses_the_trajectory_rule():
    budget = replace(config.F1_ARMS["D"], depth_rule="trajectory")
    with pytest.raises(ValueError, match="sized"):
        training.train_model("bellman_ford", budget, seed=0)


def _dear_batches(count: int = 8, size: int = 4):
    splits = dataset.load_for("bellman_ford", "dear")
    return zoo.Batches(splits["val"].subsample(count), size)


def _model(**kwargs):
    torch.manual_seed(0)
    np.random.seed(0)
    return zoo.build("bellman_ford", TINY, probe=True,
                     depth_rule="sized", **kwargs)


def test_both_arms_run_with_the_sidecar_deleted(tmp_path):
    """
    No training or evaluation path opens the sidecar: with the file
    moved away entirely, a D training step (the sized segmented epoch),
    an F-sized training step and both arms' evaluation passes all
    succeed on DEAR batches.
    """
    sidecar = dataset.dear_sidecar("bellman_ford")
    hidden = tmp_path / sidecar.name
    sidecar.rename(hidden)
    try:
        batches = _dear_batches()
        model = _model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        stats = training.train_epoch_segmented(
            model, batches, optimizer, 4, segment_optim="per_batch")
        assert stats["opt_steps"] == len(batches)
        assert model.predict(batches[0])
        fixed = _model(solver="grounded")
        loss, parts = fixed.loss(batches[0])
        assert torch.isfinite(loss)
        assert parts["hint"] == 0.0
        fixed.eval()
        assert fixed.predict(batches[0], factor=1.5)
    finally:
        hidden.rename(sidecar)


def test_an_epoch_shuffles_across_the_sizes():
    """
    ``Batches.over`` concatenates the per-size batches in size order and
    the epoch's shuffle is what mixes them: under a seeded order
    generator the visited sizes interleave rather than replaying the
    concatenation order.
    """
    splits = dataset.load_for("bellman_ford", "dear")
    batches = zoo.Batches.over(
        [splits["train8"].subsample(8), splits["train16"].subsample(8)], 4)
    visited = []

    class Recording:
        def __len__(self):
            return len(batches)

        def __getitem__(self, index):
            visited.append(batches[index].size)
            return batches[index]

    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training.train_epoch_segmented(
        model, Recording(), optimizer, 4, np.random.default_rng(0),
        segment_optim="per_batch")
    assert sorted(visited) == [8, 8, 16, 16]
    assert visited != sorted(visited)


def test_best_validation_selection_keeps_the_best_weights(tmp_path,
                                                          monkeypatch):
    """
    DEAR selects the checkpoint by validation score; so does
    :func:`train.train_model`, and the weights it returns are the best
    epoch's, bitwise, not the last one's.
    """
    monkeypatch.setattr(training, "ARTIFACTS", tmp_path)
    budget = replace(
        config.F1_ARMS["D"], epochs=2, n_train=90, n_wide=8,
        seeds=(0, ))
    model, record = training.train_model(
        "bellman_ford", budget, seed=0, widths=TINY, log=lambda *_: None)
    scored = [entry for entry in record["history"]
              if "val_selected" in entry]
    assert len(scored) == budget.epochs
    assert record["best"]["score"] == max(one["val_selected"]
                                          for one in scored)
    assert record["best"]["epoch"] == min(
        entry["epoch"] for entry in scored
        if entry["val_selected"] == record["best"]["score"])
    stored = {key: value for key, value in record["state_dict"].items()}
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), stored[key])
