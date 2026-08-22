# -*- coding: utf-8 -*-

"""
The segmented (TRM-style) training loop of ``examples/CLRS_small``.

One claim matters enough to pin on a real batch: a segment boundary
detaches the state, and the input families ride *in* the state on traced
loops, so without the carried re-attachment of ``train.grounder`` every
encoder would be severed from every segment after the first -- the same
lesion ``test_the_librarys_fixed_point_cannot_train_an_encoder`` pins for
the library's fixed point.  The gradients read here are the ones left by
the **last** optimizer step of a multi-segment batch, which exist only if
the re-attachment works.
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


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_the_segment_tag_is_its_own_and_default_tags_are_unchanged():
    budget = replace(config.H2_ARMS["O"], widths="small")
    assert budget.tag == "p3-small-max-ptredge-probe"
    assert replace(config.H2_ARMS["F"], widths="small").tag \
        == "p3-small-max-ptredge-probe-grounded"
    assert replace(budget, segment_steps=4).tag \
        == "p3-small-max-ptredge-probe-seg4"


def test_the_b1_tags_are_their_own_and_part_a_tags_are_unchanged():
    """
    The two Part B fields default to Part A's behaviour, so every tag of
    ``PART_A.md`` is byte-identical -- the assertions above already pin
    the three -- and each non-default value files under its own tag,
    distinct from the other and from Part A's.
    """
    budget = replace(config.H2_ARMS["O"], widths="small", segment_steps=4)
    accum = replace(budget, segment_optim="per_batch")
    nodetach = replace(accum, segment_detach=False)
    assert accum.tag == "p3-small-max-ptredge-probe-seg4-acc"
    assert nodetach.tag == "p3-small-max-ptredge-probe-seg4-acc-nodetach"
    assert len({budget.tag, accum.tag, nodetach.tag,
                replace(config.H2_ARMS["O"], widths="small").tag,
                replace(config.H2_ARMS["F"], widths="small").tag}) == 5
    # the fields are read by the segmented loop alone, so without
    # segmentation they change nothing and may not change the tag either
    assert replace(config.H2_ARMS["O"], segment_optim="per_batch").tag \
        == config.H2_ARMS["O"].tag


def _one_epoch(segment_optim, segment_detach):
    torch.manual_seed(0)
    np.random.seed(0)
    model = zoo.build("bellman_ford", TINY, probe=True)
    batches = zoo.Batches(
        dataset.load("bellman_ford", "val").subsample(4), 4)
    assert batches[0].steps > 2, "one segment alone proves nothing"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = training.train_epoch_segmented(
        model, batches, optimizer, 2, segment_optim=segment_optim,
        segment_detach=segment_detach)
    return model, stats, batches


@pytest.mark.parametrize("segment_detach", (True, False))
def test_a_per_batch_step_trains_every_encoder_once_per_batch(
        segment_detach):
    """
    Both Part B arms -- T-accum (``per_batch``, detached) and O-deep
    (``per_batch``, attached) -- leave a non-``None`` gradient on every
    parameter, a non-zero one on every encoder, and step the optimizer
    **exactly once per batch**.
    """
    model, stats, batches = _one_epoch("per_batch", segment_detach)
    assert stats["opt_steps"] == len(batches)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
    for name, parameter in model.named_parameters():
        if name.startswith("encoders."):
            assert bool(parameter.grad.any()), name


def test_anderson_with_no_memory_is_bitwise_grounded():
    """
    With ``memory=1`` there is never a second iterate to mix, so B4's
    :class:`model.Anderson` must reproduce :class:`model.Grounded`
    bitwise -- forward value and every checkpoint alike -- which pins
    the mixture as the only thing the new solver adds.
    """
    torch.manual_seed(0)
    model = zoo.build("bellman_ford", TINY, probe=True, solver="grounded")
    batch = zoo.Batches(
        dataset.load("bellman_ford", "val").subsample(4), 4)[0]
    interaction = model.map.compile(batch.diagram)
    carried = (("node", zoo.FEAT), ("edge", zoo.WEIGHT))
    state = model.initial(batch)
    rounds = model.rounds_for(batch)
    grounded = zoo.Grounded(carried=carried).run(
        interaction, state, deep=True, rounds=rounds)
    anderson = zoo.Anderson(carried=carried, memory=1).run(
        interaction, state, deep=True, rounds=rounds)
    assert torch.equal(grounded[0], anderson[0])
    assert len(grounded[1]) == len(anderson[1]) \
        and all(torch.equal(one, other)
                for one, other in zip(grounded[1], anderson[1]))


def test_an_anderson_arm_trains_its_encoders_and_files_apart():
    """
    F-Anderson differs from F in the no-grad forward trajectory alone:
    its tag is its own, and after one training step the set of encoder
    parameters holding a non-zero gradient is exactly arm F's on the
    same seed and batch -- the differentiated step is
    :class:`model.Grounded`'s, so whatever the one-step gradient
    reaches under ``grounded`` it reaches here, no more and no less.
    (Under the one-step gradient the edge-weight encoder's grad is
    exactly zero on this batch for *both* solvers -- one differentiated
    round cannot span edge to node to output -- which is arm F's
    pre-existing property, not the mixture's.)
    """
    assert replace(config.H2_ARMS["F"], widths="small",
                   solver="anderson").tag \
        == "p3-small-max-ptredge-probe-anderson"
    found = {}
    for solver in ("grounded", "anderson"):
        torch.manual_seed(0)
        np.random.seed(0)
        model = zoo.build("bellman_ford", TINY, probe=True, solver=solver)
        batches = zoo.Batches(
            dataset.load("bellman_ford", "val").subsample(4), 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        training.train_epoch(model, batches, optimizer)
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
        found[solver] = {name for name, parameter
                         in model.named_parameters()
                         if name.startswith("encoders.")
                         and bool(parameter.grad.any())}
    assert found["anderson"] == found["grounded"] != set()


def test_an_attached_boundary_needs_the_per_batch_optimizer():
    model = zoo.build("bellman_ford", TINY, probe=True)
    batches = zoo.Batches(
        dataset.load("bellman_ford", "val").subsample(4), 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="per_batch"):
        training.train_epoch_segmented(
            model, batches, optimizer, 2, segment_detach=False)


def test_a_segmented_step_trains_every_encoder():
    """
    After one epoch of the segmented loop on a real multi-segment batch,
    every parameter -- every encoder parameter above all -- holds the
    non-``None`` gradient of the *final* segment's backward pass.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    model = zoo.build("bellman_ford", TINY, probe=True)
    batches = zoo.Batches(
        dataset.load("bellman_ford", "val").subsample(4), 4)
    assert batches[0].steps > 2, "one segment alone proves nothing"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stats = training.train_epoch_segmented(model, batches, optimizer, 2)
    assert stats["opt_steps"] > len(batches)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
    for name, parameter in model.named_parameters():
        if name.startswith("encoders."):
            assert bool(parameter.grad.any()), name
