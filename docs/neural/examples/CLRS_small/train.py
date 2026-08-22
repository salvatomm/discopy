# -*- coding: utf-8 -*-

"""
Training one algorithm -- an ordinary PyTorch training loop.

    python train.py --quick                    # a few-second miniature
    python train.py --algorithms bfs --seed 0  # one recorded baseline
    python train.py                            # all three, all seeds

Nothing here is in :mod:`discopy.neural`, and that is the point: a
:class:`~model.Model` is a :class:`torch.nn.Module`, so it trains with an
ordinary optimizer and an ordinary loop.  What the file contains is the
*protocol* of the study:

* :func:`train_epoch` -- the supervision scheme.  One differentiated run of
  ``Iterate(deep=True)``, a hint loss on every algorithm step the trajectory
  defines and an output loss from the step it terminates on, one optimizer
  step per batch.  The run is as long as the *trajectory*
  (:func:`~model.rounds_of`), so the depth is a property of the sample
  rather than of the budget; ``--rounds`` pins Part 1's fixed depth
  instead, and :func:`depth_policy` is what an artefact records so that the
  two can never be read as one table.
* :class:`~model.Batches` -- what is different from ``examples/sudoku``,
  where one diagram served every puzzle.  Here a *batch* is a diagram -- the
  disjoint union of its members' incidence graphs -- so the batching is
  fixed rather than reshuffled and every diagram is compiled once and reused
  for the whole run.  With CLRS-30's 1000 training trajectories that is 32
  diagrams, which is why no pool machinery is needed.
* :func:`train_model` -- the registry: every run is cached under
  ``artifacts/`` as weights plus history plus metadata, so re-running a
  script re-loads instead of re-training.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace

import numpy as np
import torch

import dataset
import model as zoo
from config import (
    ALGORITHMS, ARTIFACTS, FULL, GRAD_CLIP, H2_ARMS, MIXED, QUICK, REGIME,
    SELECTION, SETTLE, WIDTHS, Budget, Widths)
from dataset import POS
from discopy.neural.cells import POOL
from model import Batches, POINTERS, SOLVERS


def seed_everything(seed: int) -> None:
    """ Fix every source of randomness we use. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def default_device() -> torch.device:
    """ The GPU if there is one. """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def single_threaded() -> None:
    """
    One intra-op thread, which is what these runs want on either device.

    A round of message passing is many small kernels, so a run is bound by
    the launches rather than by the arithmetic: on a GPU the extra threads
    only contend for the cores that issue them, and on a CPU handing a cell
    to a thread pool costs more than the cell -- which is also why
    ``docs/neural/NOTES.md`` records every golden single-threaded.
    ``NOTES.md`` beside this file measures what it is worth here.
    """
    torch.set_num_threads(1)


# --- one epoch -------------------------------------------------------------

def train_epoch(model, batches: Batches, optimizer, order=None) -> dict:
    """
    One pass over the batches: one differentiated run and one optimizer
    step each.

    The *order* of the batches is shuffled, their *contents* are not: a
    reshuffle across batches would draw new diagrams and pay for new
    compilations, which is the one place this example's arithmetic differs
    from a dense model's.

    Parameters:
        model : The model to train.
        batches : The training batches.
        optimizer : The optimizer to step.
        order : A ``numpy`` generator to shuffle the batch order with.

    Returns:
        The mean loss and its parts -- ``output``, ``hint`` and one
        ``probe/<name>`` per decoded probe -- and the optimizer steps
        taken.
    """
    model.train()
    indices = np.arange(len(batches))
    if order is not None:
        order.shuffle(indices)
    total, parts = 0.0, {}
    for index in indices:
        loss, found = model.loss(batches[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total += float(loss.detach())
        for key, value in found.items():
            parts[key] = parts.get(key, 0.0) + value
    steps = max(len(indices), 1)
    return {"loss": total / steps, "opt_steps": int(steps),
            **{key: value / steps for key, value in parts.items()}}


def grounder(model) -> zoo.Grounded:
    """
    The carried-family re-attachment of the segmented loop, exactly
    :class:`model.Grounded`'s: the input families ride on traced loops
    *inside* the state, so a bare detach at a segment boundary would
    sever the encoders from every segment's graph after the first.  The
    carried tuple is the one :func:`model.build` hands the grounded
    solver -- ``("node", FEAT)`` and, when the diagram has edge boxes,
    ``("edge", WEIGHT)``.
    """
    return zoo.Grounded(carried=(("node", zoo.FEAT), ) + (
        (("edge", zoo.WEIGHT), ) if zoo.has_edges(model.algorithm) else ()))


def segment_loss(model, batch, every, start):
    """
    The deep-supervision loss of one segment: the **output** probes
    decoded at the segment's final state and scored against the ground
    -truth outputs for **all** samples -- every segment end predicts the
    final answer, where :meth:`model.Model.loss` supervises the output
    from a sample's own termination onwards -- plus the hint loss on the
    steps the segment covers, under the model's own supervision regime
    (detached from the interaction when ``probe``).

    Parameters:
        model : The model being trained.
        batch : The batch being run.
        every : The segment's checkpoints, one per algorithm step.
        start : The zero-based algorithm step the segment starts at.

    Returns:
        The loss, and its parts as floats: ``output``, ``hint`` and one
        ``probe/<name>`` per decoded probe.
    """
    names = zoo.probes(model.algorithm, "output")
    output, hint, each = 0.0, 0.0, {}
    prediction = model.decode(batch, every[-1], names=names)
    for name in names:
        term = model.decoders[name].loss(
            prediction[name], batch.outputs[name])
        output, each[name] = output + term, each.get(name, 0.0) + term
    for step, found in enumerate(every, start=start):
        targets = model.hint_targets(batch, step)
        decoded = model.decode(
            batch, found.detach() if model.probe else found,
            names=list(targets)) if targets else {}
        for name, (truth, alive) in targets.items():
            term = model.decoders[name].loss(decoded[name][alive], truth)
            hint, each[name] = hint + term, each.get(name, 0.0) + term
    hint = hint / len(every)
    return output + model.hint_weight * hint, {
        "output": zoo._number(output), "hint": zoo._number(hint),
        **{f"probe/{name}": zoo._number(term)
           for name, term in each.items()}}


def train_epoch_segmented(model, batches: Batches, optimizer,
                          segment_steps: int, order=None,
                          segment_optim: str = "per_segment",
                          segment_detach: bool = True) -> dict:
    """
    One pass over the batches, TRM-style: the run is cut into segments
    of ``segment_steps`` algorithm steps -- the final one may be shorter
    -- and every segment is its own truncated-backprop training step.

    The total depth is :func:`train_epoch`'s, the trajectory rule's
    rounds for the batch, unchanged.  Each segment starts from the
    previous segment's final state, **detached**, with the carried input
    families re-attached by :func:`grounder` -- the encoders are re-run
    for every segment, since a backward pass consumes their forward
    graph and a detach would otherwise sever them.  The segment's own
    rounds are differentiated in full, its loss is
    :func:`segment_loss`, and the optimizer steps once per segment, so
    a batch takes ``ceil(steps / segment_steps)`` optimizer steps where
    :func:`train_epoch` takes one.

    Part B's two axes decompose that recipe (``PART_B.md``):

    * ``segment_optim="per_batch"`` accumulates the **mean** of the
      segment losses and steps the optimizer exactly **once per
      batch**, so the supervision placement is the segmented one and
      the step count is :func:`train_epoch`'s.  Under a detached
      boundary each segment's mean-scaled loss is backpropagated as it
      is produced -- the gradients accumulate, the graphs are freed --
      and with the boundary attached the accumulated loss is
      backpropagated once at the end of the batch.
    * ``segment_detach=False`` keeps the state attached at a segment
      boundary, so the backward pass flows through the whole run and
      the carried re-attachment is a no-op (nothing was severed).  It
      requires ``segment_optim="per_batch"``: stepping the optimizer
      inside an attached graph would differentiate stale parameters.

    Parameters:
        model : The model to train.
        batches : The training batches.
        optimizer : The optimizer to step.
        segment_steps : The algorithm steps one segment covers.
        order : A ``numpy`` generator to shuffle the batch order with.
        segment_optim : ``"per_segment"`` or ``"per_batch"``; see above.
        segment_detach : Whether a segment boundary detaches; see above.

    Returns:
        The mean loss and its parts over the segments, and the
        optimizer steps taken.
    """
    if segment_optim not in ("per_segment", "per_batch"):
        raise ValueError(f"segment_optim is {segment_optim!r}, not one "
                         f"of ('per_segment', 'per_batch')")
    if not segment_detach and segment_optim != "per_batch":
        raise ValueError("segment_detach=False needs "
                         "segment_optim='per_batch': an optimizer step "
                         "inside an attached graph would differentiate "
                         "stale parameters")
    per_batch = segment_optim == "per_batch"
    model.train()
    indices = np.arange(len(batches))
    if order is not None:
        order.shuffle(indices)
    total, parts, steps, segments = 0.0, {}, 0, 0
    for index in indices:
        batch = batches[index]
        interaction = model.map.compile(batch.diagram)
        count = -(-model.steps_of(batch) // segment_steps)
        if per_batch:
            optimizer.zero_grad(set_to_none=True)
        state, accumulated = None, 0.0
        for start in range(0, model.steps_of(batch), segment_steps):
            state = model.initial(batch) if state is None else (
                grounder(model).ground(
                    interaction, state, model.initial(batch))
                if segment_detach else state)
            stop = min(start + segment_steps, model.steps_of(batch))
            every = model.checkpoints(model.map(
                batch.diagram, state, deep=True,
                rounds=model.hops * (stop - start)))
            loss, found = segment_loss(model, batch, every, start)
            if not per_batch:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               GRAD_CLIP)
                optimizer.step()
                steps += 1
            elif segment_detach:
                (loss / count).backward()
            else:
                accumulated = accumulated + loss / count
            state = every[-1]
            total, segments = total + float(loss.detach()), segments + 1
            for key, value in found.items():
                parts[key] = parts.get(key, 0.0) + value
        if per_batch:
            if not segment_detach:
                accumulated.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            steps += 1
    segments = max(segments, 1)
    return {"loss": total / segments, "opt_steps": int(steps),
            **{key: value / segments for key, value in parts.items()}}


def per_probe(stats: dict) -> str:
    """ The ``probe/<name>`` terms of an epoch, as one line. """
    return "  ".join(f"{key.split('/', 1)[1]} {value:.4f}"
                     for key, value in stats.items()
                     if key.startswith("probe/"))


def steps_of(budget: Budget) -> int:
    """
    The algorithm steps a run covers, ``None`` under the trajectory rule.

    A budget states a *round* count when it pins Part 1's fixed depth, and
    a round is ``HOPS`` per step, so the conversion happens once and here.

    Example
    -------
    >>> steps_of(FULL), steps_of(replace(FULL, rounds=16))
    (None, 8)
    """
    return None if budget.rounds is None else budget.rounds // zoo.HOPS


def depth_policy(budget: Budget) -> str:
    """ How a run chose its depth, for the artefact to record. """
    return "trajectory" if budget.rounds is None else f"fixed:{budget.rounds}"


# --- a whole run -----------------------------------------------------------

def artifact_of(algorithm: str, budget: Budget, seed: int):
    """ Where a trained model is cached; see :attr:`~config.Budget.tag`. """
    return ARTIFACTS / f"{budget.tag}-{algorithm}-seed{seed}.pt"


def regime_of(algorithm: str, budget: Budget) -> Budget:
    """
    The budget a Part 3 row trains under, with its training-size regime
    read off :data:`config.REGIME` rather than defaulted.

    Mixed sizes were adopted in Part 2 as a blanket protocol and the
    ``minimum`` control killed that: at 200 trajectories of one size the
    row holds at 0.8281 and at 1000 trajectories of five sizes it
    collapses to 0.1719, so size mixing is destructive on at least one
    task and cannot be a default anyone forgets they chose.  The honest
    fallout is that every row owes a decision, made on the salvaged
    scoring of the mixed campaign, recorded in :data:`config.REGIME`
    before training and frozen after.

    An undeclared row raises rather than trains.  A default here would be
    a protocol choice made by whoever ran the script first, which is the
    same failure class as a stale checkpoint loaded by tag: it produces a
    number, and the number looks like everyone else's.

    Parameters:
        algorithm : The row.
        budget : The arm, one of :data:`config.H2_ARMS`.
    """
    found = REGIME.get(algorithm)
    if found not in ("mixed", "fixed"):
        raise ValueError(
            f"config.REGIME[{algorithm!r}] is {found!r}: declare it "
            f"'mixed' or 'fixed' from the salvaged scoring before "
            f"training, and freeze it after")
    return replace(budget, mixed=found == "mixed")


def train_model(algorithm: str, budget: Budget = FULL, seed: int = 0,
                widths: Widths = None, device=None, splits: dict = None,
                reuse: bool = True, log=print):
    """
    Train one model, or load it when it has already been trained.

    Validation runs every ``budget.eval_every`` epochs at the trained depth
    and the best-scoring weights are the ones kept: the out-of-distribution
    split is never looked at here, only in ``evaluate.py``.

    A reused artefact is checked against the budget's depth policy and
    refused when they differ.  The two regimes file under different tags,
    so this can only happen when an artefact predates a protocol -- which
    it did, once, and a table that mixes a fixed depth with a trajectory
    one is not a table.

    Parameters:
        algorithm : The algorithm to imitate.
        budget : What the run may spend.
        seed : The seed of the run.
        widths : The widths, the budget's by default.
        device : Where to train, the GPU by default.
        splits : The loaded splits, read from the cache by default.
        reuse : Whether to load an existing artifact instead of training.
        log : Where to print progress.

    Returns:
        The model and the record that was cached beside it.
    """
    widths = widths or WIDTHS[budget.widths]
    device = default_device() if device is None else device
    path = artifact_of(algorithm, budget, seed)
    seed_everything(seed)
    model = zoo.build(algorithm, widths, steps=steps_of(budget),
                      pool=budget.pool, edge_state=budget.edge_state,
                      hint_weight=budget.hint_weight,
                      settle=budget.settle, pointer=budget.pointer,
                      probe=budget.probe,
                      feedback=budget.feedback, forcing=budget.forcing,
                      trm=budget.trm,
                      solver=budget.solver, backward=budget.backward,
                      cache=4 + 2 * (1000 // max(budget.batch_size, 1)))
    if reuse and path.exists():
        stored = zoo.load_checkpoint(model, path)
        found = stored.get("depth")
        if found != depth_policy(budget):
            raise ValueError(
                f"{path.name} was trained under depth {found!r} and this "
                f"budget asks for {depth_policy(budget)!r}; a table cannot "
                f"mix the two, so move it aside or pass reuse=False")
        log(f"  {algorithm}/seed{seed}: loaded {path.name}")
        return model.to(device), stored

    splits = splits or dataset.load_all(algorithm)
    if budget.pos != "sampler":
        # `uniform` re-parameterizes an input without losing information,
        # so it applies to every split as the reference's randomised
        # position does; `shuffled` is an ablation of the training signal
        # and is confined to what the model learns from.
        splits = {name: one.repositioned(budget.pos, seed)
                  if budget.pos == "uniform" or name.startswith("train")
                  else one for name, one in splits.items()}
    train = Batches.over(
        [splits[f"train{size}"].subsample(
            budget.n_train and budget.n_train // len(MIXED))
         for size in MIXED], budget.batch_size, device) if budget.mixed \
        else Batches(splits["train"].subsample(budget.n_train),
                     budget.batch_size, device)
    valid = Batches(splits["val"], budget.eval_batch_size, device)
    zoo.fit_cache(model, train, valid)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=budget.lr,
                                  weight_decay=budget.weight_decay)
    order = np.random.default_rng(seed)

    calls = zoo.calls_per_round(model, train[0])
    zoo.check_layout(model, train[0])
    model.map.cache_stats(reset=True)
    history: list = []
    warm: dict = {}
    best = {"score": -1.0, "epoch": -1}
    weights = {key: value.detach().cpu().clone()
               for key, value in model.state_dict().items()}
    tick = time.perf_counter()
    for epoch in range(1, budget.epochs + 1):
        stats = train_epoch_segmented(
            model, train, optimizer, budget.segment_steps, order,
            budget.segment_optim, budget.segment_detach) \
            if budget.segment_steps \
            else train_epoch(model, train, optimizer, order)
        stats["epoch"] = epoch
        # the first epoch compiles, the rest must not: recorded from the
        # second, so that a cache one diagram too small cannot hide.
        found = model.map.cache_stats(reset=True)
        warm = found if epoch == 2 else warm
        if epoch % budget.eval_every == 0 or epoch == budget.epochs:
            stats.update({f"val_{key}": value for key, value in
                          zoo.evaluate_split(model, valid).items()})
            # "deep" selection scores a candidate by min(val, val at the
            # deepest sweep factor): a checkpoint that has reached the
            # answer *and stays there*, read off the n = 16 validation
            # split alone.  No out-of-distribution sample is touched here.
            chosen = stats["val_score"]
            if budget.selection == "deep":
                stats["val_deep_score"] = zoo.evaluate_split(
                    model, valid, factor=max(budget.sweep))["score"]
                chosen = min(chosen, stats["val_deep_score"])
            stats["val_selected"] = chosen
            if chosen > best["score"]:
                best = {"score": chosen, "epoch": epoch}
                weights = {key: value.detach().cpu().clone()
                           for key, value in model.state_dict().items()}
            log(f"  {algorithm}/seed{seed} epoch {epoch:4d}: "
                f"loss {stats['loss']:.4f} (out {stats['output']:.4f}, "
                f"hint {stats['hint']:.4f}) val {stats['val_score']:.4f}")
            log(f"       probes: {per_probe(stats)}")
        history.append(stats)
    elapsed = time.perf_counter() - tick

    model.load_state_dict({key: value.to(device)
                           for key, value in weights.items()})
    record = {
        "state_dict": weights, "history": history, "best": best,
        "algorithm": algorithm, "seed": seed, "budget": asdict(budget),
        "widths": widths.asdict(), "pool": budget.pool, "device": str(device),
        "torch": torch.__version__, "parameters": zoo.count_parameters(model),
        "calls_per_round": calls, "batches": len(train),
        "samples": train.samples, "seconds_per_epoch": elapsed / budget.epochs,
        "compile_cache": warm, "depth": depth_policy(budget),
        "rounds": [model.rounds_for(one) for one in train],
        "edge_state": budget.edge_state, "hint_weight": budget.hint_weight,
        "settle": budget.settle, "pointer": budget.pointer,
        "selection": budget.selection,
        "pos": budget.pos, "solver": budget.solver, "probe": budget.probe,
        "backward": budget.backward, "feedback": budget.feedback,
        "forcing": budget.forcing, "dense": budget.dense,
        "trm": budget.trm, "segment_steps": budget.segment_steps,
        "segment_optim": budget.segment_optim,
        "segment_detach": budget.segment_detach,
    }
    torch.save(record, path)
    log(f"  {algorithm}/seed{seed}: best val {best['score']:.4f} at epoch "
        f"{best['epoch']}, {elapsed / budget.epochs:.2f}s/epoch, "
        f"{calls} calls/round, {max(record['rounds'])} rounds "
        f"({record['depth']}), compile cache {warm.get('hits')}/"
        f"{warm.get('hits', 0) + warm.get('misses', 0)} hit on a warm "
        f"epoch -> {path.name}")
    return model, record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="*", default=ALGORITHMS)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None,
                        help="a fixed depth, instead of the trajectory rule")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval-every", dest="eval_every", type=int,
                        default=None,
                        help="epochs between validation passes, i.e. how "
                             "densely best-val selection samples the run; "
                             "it is part of the tag, so two cadences file "
                             "separately")
    parser.add_argument("--hint-weight", dest="hint_weight", type=float,
                        default=None, help="0 for the output-only ablation")
    parser.add_argument("--node-only", action="store_true",
                        help="H1's arm without an edge state")
    parser.add_argument("--pool", choices=sorted(POOL), default=None,
                        help="how every site reduces its message orbit")
    parser.add_argument("--widths", choices=sorted(WIDTHS), default=None)
    parser.add_argument("--pointer", choices=sorted(POINTERS),
                        default=None,
                        help="which node-pointer head to build")
    parser.add_argument("--n-train", dest="n_train", type=int,
                        default=None,
                        help="training trajectories, all of them by default")
    parser.add_argument("--pos", choices=POS, default=None,
                        help="what to do to the pos input")
    parser.add_argument("--mixed", action="store_true",
                        help="train on config.MIXED sizes")
    parser.add_argument("--settle", nargs="?", const="interior",
                        choices=[one for one in SETTLE if one], default=None,
                        help="hold a finished trajectory's last hint; bare "
                             "--settle is 'interior', which is what the "
                             "mixed campaign trained under and which cannot "
                             "reach the terminal checkpoint")
    parser.add_argument("--selection", choices=[one for one in SELECTION
                                               if one], default=None,
                        help="what best-val selection maximises; 'deep' is "
                             "min(val, val at 3x depth), still n = 16 only")
    parser.add_argument("--probe", action="store_true",
                        help="fit the hint heads on a detached state, i.e. "
                             "train the interaction on its output alone")
    parser.add_argument("--feedback", choices=("state", ), default=None,
                        help="re-encode hints into the state at every "
                             "checkpoint boundary: the floor's closed "
                             "loop, T-D's arm")
    parser.add_argument("--forcing", type=float, default=None,
                        help="the closed loop's teacher-forcing coin, "
                             "0.5 by default")
    parser.add_argument("--dense", action="store_true",
                        help="the complete-graph diagram whatever the "
                             "algorithm: the reference MPNN's wiring, "
                             "T-C's arm")
    parser.add_argument("--trm", action="store_true",
                        help="the TRM arm: artifacts/trm-design.md")
    parser.add_argument("--segment-steps", dest="segment_steps", type=int,
                        default=None,
                        help="cut the run into segments of this many "
                             "algorithm steps, each differentiated in "
                             "full from the previous one's detached end "
                             "and its own optimizer step: the TRM-style "
                             "segmented training loop, arm T of PART_A.md")
    parser.add_argument("--segment-optim", dest="segment_optim",
                        choices=("per_segment", "per_batch"), default=None,
                        help="when the segmented loop steps its optimizer: "
                             "'per_segment' is arm T, 'per_batch' "
                             "accumulates the mean of the segment losses "
                             "and steps once per batch (tag 'acc')")
    parser.add_argument("--no-segment-detach", dest="segment_detach",
                        action="store_const", const=False, default=None,
                        help="keep the state attached at segment "
                             "boundaries, so backprop flows through the "
                             "whole run and only the supervision placement "
                             "changes (tag 'nodetach'); needs "
                             "--segment-optim per_batch")
    parser.add_argument("--solver", choices=sorted(SOLVERS),
                        default=None, help="the execution policy")
    parser.add_argument("--backward", choices=("full", "last"), default=None,
                        help="a fixed point's differentiation policy")
    parser.add_argument("--arm", choices=sorted(H2_ARMS), default=None,
                        help="a Part 3 arm of config.H2_ARMS, whose size "
                             "regime is read off config.REGIME per row")
    parser.add_argument("--regime", choices=("fixed", "mixed"), default=None,
                        help="declare the size regime on the command line "
                             "instead of reading config.REGIME: this is what "
                             "the probe that decides config.REGIME runs "
                             "under, and nothing else may use it")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="retrain even when an artifact exists")
    arguments = parser.parse_args(argv)
    single_threaded()

    budget = H2_ARMS[arguments.arm] if arguments.arm \
        else QUICK if arguments.quick else FULL
    for key in ("epochs", "rounds", "lr", "pool", "widths",
                "hint_weight", "pointer", "pos", "settle",
                "solver", "backward", "n_train", "feedback", "forcing",
                "eval_every", "selection", "segment_steps",
                "segment_optim", "segment_detach"):
        if getattr(arguments, key) is not None:
            budget = replace(budget, **{key: getattr(arguments, key)})
    for key in ("mixed", "probe", "dense", "trm"):
        if getattr(arguments, key):
            budget = replace(budget, **{key: True})
    if arguments.node_only:
        budget = replace(budget, edge_state=False, widths="paired")
    seeds = tuple(arguments.seeds) if arguments.seeds else budget.seeds
    device = torch.device(arguments.device) if arguments.device \
        else default_device()

    summary: dict = {}
    for algorithm in arguments.algorithms:
        if budget.dense:
            dataset.densify(algorithm)
        splits = dataset.load_all(algorithm)
        # a Part 3 arm reads its size regime per row and refuses a row
        # that has not declared one; the probe that *decides* the regime
        # says so on the command line; anything else keeps the budget's.
        one = replace(budget, mixed=arguments.regime == "mixed") \
            if arguments.regime else \
            regime_of(algorithm, budget) if arguments.arm else budget
        for seed in seeds:
            _, record = train_model(algorithm, one, seed, device=device,
                                    splits=splits, reuse=not arguments.fresh)
            summary[f"{algorithm}/seed{seed}"] = record["best"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
