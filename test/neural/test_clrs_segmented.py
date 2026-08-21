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
    assert replace(budget, segment_steps=4).tag \
        == "p3-small-max-ptredge-probe-seg4"


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
