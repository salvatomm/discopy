# -*- coding: utf-8 -*-

"""
The sized depth rule of ``examples/CLRS_small`` -- Part D's length-free
regime.

One claim carries the whole part: under ``depth_rule="sized"`` nothing a
model trains on or is scored by reads ``batch.lengths``, the ground-truth
per-sample step counts.  Every arm before Part D ran at the trajectory
rule's depth, which is information the deep-equilibrium literature says
its models do not need, and these tests are what "length-free" means
here: the lengths are overwritten with garbage and the losses and
predictions of both sized arms -- D (segmented, deep output supervision)
and F-sized (one-step gradient, final-checkpoint output loss) -- must be
**bitwise** what they were with the truth.
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


if not (EXAMPLE / "data" / "bellman_ford-val.npz").exists():
    pytest.skip("the CLRS-30 cache is not built; see examples/CLRS_small",
                allow_module_level=True)

CLRS = _example(EXAMPLE)
config, dataset = CLRS["config"], CLRS["dataset"]
zoo, training = CLRS["model"], CLRS["train"]

TINY = config.Widths(dim=4, state_dim=8, hidden=8, edge_dim=4, graph_dim=8)

#: The hint heads: the one module a *sized* training step owes no
#: gradient, since the hint loss is indexed by the trajectory clock and
#: a sized run may not read it (``model.Model.sized_loss``).
HINT_HEADS = {f"decoders.{name}"
              for name in dataset.probes("bellman_ford", "hint")}


def _is_hint_head(name: str) -> bool:
    return any(name.startswith(head + ".") for head in HINT_HEADS)

#: The garbage the lengths are overwritten with: too short and too long,
#: so a leak that clamps and one that extends are both caught.
POISONS = (lambda lengths, n: torch.ones_like(lengths),
           lambda lengths, n: torch.full_like(lengths, 10 * n))


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_the_sized_tags_are_their_own_and_every_earlier_tag_is_unchanged():
    """
    ``depth_rule`` defaults to the trajectory rule, so every tag of
    ``PART_A.md`` and ``PART_B.md`` is byte-identical, and each sized
    arm files under its own tag, distinct from the others and from every
    earlier arm's.
    """
    small = replace(config.H2_ARMS["O"], widths="small")
    tags = {
        "O": small.tag,
        "F": replace(config.H2_ARMS["F"], widths="small").tag,
        "T": replace(small, segment_steps=4).tag,
        "T-accum": replace(small, segment_steps=4,
                           segment_optim="per_batch").tag,
        "O-deep": replace(small, segment_steps=4, segment_optim="per_batch",
                          segment_detach=False).tag,
        "F-Anderson": replace(config.H2_ARMS["F"], widths="small",
                              solver="anderson").tag,
        "D": replace(small, segment_steps=4, segment_optim="per_batch",
                     depth_rule="sized").tag,
        "F-sized": replace(config.H2_ARMS["F"], widths="small",
                           depth_rule="sized").tag,
    }
    assert tags["O"] == "p3-small-max-ptredge-probe"
    assert tags["F"] == "p3-small-max-ptredge-probe-grounded"
    assert tags["T"] == "p3-small-max-ptredge-probe-seg4"
    assert tags["T-accum"] == "p3-small-max-ptredge-probe-seg4-acc"
    assert tags["O-deep"] == "p3-small-max-ptredge-probe-seg4-acc-nodetach"
    assert tags["F-Anderson"] == "p3-small-max-ptredge-probe-anderson"
    assert tags["D"] == "p3-small-max-ptredge-probe-seg4-acc-szd"
    assert tags["F-sized"] == "p3-small-max-ptredge-probe-szd-grounded"
    assert len(set(tags.values())) == len(tags)


def _batches(poison=None):
    found = zoo.Batches(dataset.load("bellman_ford", "val").subsample(4), 4)
    if poison is not None:
        for batch in found:
            batch.lengths = poison(batch.lengths, batch.size)
    return found


def _model(**kwargs):
    torch.manual_seed(0)
    np.random.seed(0)
    return zoo.build("bellman_ford", TINY, probe=True,
                     depth_rule="sized", **kwargs)


def _d_step(batches):
    """ One D training step: stats, and the parameters it leaves. """
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = training.train_epoch_segmented(
        model, batches, optimizer, 4, segment_optim="per_batch")
    return stats, {name: parameter.detach().clone()
                   for name, parameter in model.named_parameters()}


def test_a_sized_run_is_as_deep_as_the_size_says_and_never_the_lengths():
    model = _model()
    batch = _batches()[0]
    assert model.steps_of(batch) == zoo.SIZED * batch.size
    assert model.rounds_for(batch) == zoo.HOPS * zoo.SIZED * batch.size
    batch.lengths = torch.ones_like(batch.lengths)
    assert model.rounds_for(batch) == zoo.HOPS * zoo.SIZED * batch.size


@pytest.mark.parametrize("poison", POISONS)
def test_a_d_training_step_is_bitwise_blind_to_the_lengths(poison):
    """
    A D training step -- the sized segmented loop, accumulated to one
    optimizer step per batch -- returns bitwise the same loss, parts and
    updated parameters whatever ``batch.lengths`` says.
    """
    truth, updated = _d_step(_batches())
    poisoned, poisoned_updated = _d_step(_batches(poison))
    assert truth == poisoned
    assert all(torch.equal(updated[name], poisoned_updated[name])
               for name in updated)


@pytest.mark.parametrize("poison", POISONS)
def test_an_f_sized_training_loss_is_bitwise_blind_to_the_lengths(poison):
    model = _model(solver="grounded")
    loss, parts = model.loss(_batches()[0])
    poisoned_loss, poisoned_parts = model.loss(_batches(poison)[0])
    assert torch.equal(loss, poisoned_loss)
    assert parts == poisoned_parts
    assert parts["hint"] == 0.0


@pytest.mark.parametrize("solver", ("iterate", "grounded"))
@pytest.mark.parametrize("poison", POISONS)
def test_a_sized_evaluation_is_bitwise_blind_to_the_lengths(solver, poison):
    """
    The evaluation forward pass of both sized arms -- D executes as an
    ``Iterate``, F-sized as ``Grounded`` -- decodes bitwise the same
    predictions whatever ``batch.lengths`` says, at the sized depth and
    at a ladder factor of it.
    """
    model = _model(solver=solver)
    model.eval()
    for factor in (1.0, 1.5):
        truth = model.predict(_batches()[0], factor=factor)
        poisoned = model.predict(_batches(poison)[0], factor=factor)
        assert set(truth) == set(poisoned)
        for name in truth:
            assert np.array_equal(truth[name][0], poisoned[name][0])
            assert np.array_equal(truth[name][1], poisoned[name][1])


def test_a_d_step_trains_every_encoder():
    """
    The detach/ground interaction of the sized segmented loop must not
    silently freeze the encoders -- the lesion :class:`model.Grounded`
    documents: after one D step every parameter but the hint heads --
    which a sized loss owes nothing -- holds a gradient and every
    encoder a non-zero one, with exactly one optimizer step per batch.
    """
    model = _model()
    batches = _batches()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = training.train_epoch_segmented(
        model, batches, optimizer, 4, segment_optim="per_batch")
    assert stats["opt_steps"] == len(batches)
    for name, parameter in model.named_parameters():
        if _is_hint_head(name):
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
    for name, parameter in model.named_parameters():
        if name.startswith("encoders."):
            assert bool(parameter.grad.any()), name


def test_an_f_sized_step_trains_its_encoders():
    """
    After one F-sized training step every parameter but the hint heads
    holds a gradient and some encoder a non-zero one -- under the one-step gradient the edge
    -weight encoder's is exactly zero on this batch, arm F's
    pre-existing property (``PART_B.md``), so "some" is the honest
    assertion where D's is "every".
    """
    model = _model(solver="grounded")
    batches = _batches()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training.train_epoch(model, batches, optimizer)
    for name, parameter in model.named_parameters():
        if _is_hint_head(name):
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
    assert {name for name, parameter in model.named_parameters()
            if name.startswith("encoders.") and bool(parameter.grad.any())}
