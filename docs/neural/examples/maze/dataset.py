# -*- coding: utf-8 -*-

"""
The Maze-Hard dataset: sapientinc/maze-30x30-hard-1k, encoded as the
5-channel lattice of the reference code (``maze_hard.py`` of
``github.com/lcrh/lattice-deduction-transformers``), plus the ports this
task instantiation needs and nothing else:

1. the loader -- CSV parsed **without stripping whitespace** (free cells
   are literal spaces; stripping silently truncates rows, the bug the
   reference loader documents), cached as one ``.pt`` per split;
2. :func:`verify`, the acceptance gate of Phase 1: every structural
   assertion run over a whole split;
3. :func:`sample_k_solutions`, the uniform shortest-path sampler over
   the all-shortest-paths DAG (ported line by line, bigint weights and
   all), feeding the alpha operator of the K > 1 trainer;
4. :func:`maze_classify` / the five evaluation buckets -- correctness on
   a maze is "any valid minimal path", not cell equality;
5. the synthetic generator of ``maze_synthetic.py`` (Searchformer's
   procedure, HRM's ``hard`` filter) for small-scale prototyping, and
   the straight-line diagnostic mode.

Channel order (wall, free, start, goal, path) is the reference's,
verbatim, so its solver and scorer port unchanged.  The lattice ``x``
gives walls/S/G as singletons and every free cell both ``free`` and
``path`` alive; ``y`` is the one-hot answer; the givens are the
singletons of ``x``.
"""

from __future__ import annotations

import csv
import heapq
import random
from pathlib import Path

import numpy as np
import torch

GRID = 30
N_CELLS = GRID * GRID

CH_WALL, CH_FREE, CH_START, CH_GOAL, CH_PATH = range(5)
N_CHANNELS = 5
CHANNELS = ("wall", "free", "start", "goal", "path")

#: Everything beside ``docs/neural``, like the sudoku example.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "maze_data"
ARTIFACTS = ROOT / "artifacts"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_QCHARS = {"#": CH_WALL, " ": CH_FREE, "S": CH_START, "G": CH_GOAL}
_ACHARS = {**_QCHARS, "o": CH_PATH}


# --- the loader -------------------------------------------------------------

def download(split: str = "train") -> Path:
    """
    Download and cache one split of ``sapientinc/maze-30x30-hard-1k`` as
    ``{q, a}`` channel-index arrays of shape ``(n, 900)`` uint8.
    """
    cache = DATA_DIR / f"maze_hard_{split}.pt"
    if cache.exists():
        return cache
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="sapientinc/maze-30x30-hard-1k",
                           filename=f"{split}.csv", repo_type="dataset")
    questions, answers = [], []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            # NEVER strip: free cells are literal spaces, and stripping
            # silently truncates rows (the reference loader's warning).
            q, a = row["question"], row["answer"]
            assert len(q) == N_CELLS, f"question length {len(q)}"
            assert len(a) == N_CELLS, f"answer length {len(a)}"
            questions.append(q)
            answers.append(a)
    lookup_q = np.full(256, 255, dtype=np.uint8)
    lookup_a = np.full(256, 255, dtype=np.uint8)
    for char, channel in _QCHARS.items():
        lookup_q[ord(char)] = channel
    for char, channel in _ACHARS.items():
        lookup_a[ord(char)] = channel
    q = lookup_q[np.frombuffer("".join(questions).encode("ascii"),
                               np.uint8).reshape(-1, N_CELLS)]
    a = lookup_a[np.frombuffer("".join(answers).encode("ascii"),
                               np.uint8).reshape(-1, N_CELLS)]
    assert not (q == 255).any(), "unknown character in a question"
    assert not (a == 255).any(), "unknown character in an answer"
    torch.save({"q": torch.from_numpy(q), "a": torch.from_numpy(a)}, cache)
    return cache


def encode(q, a=None):
    """
    The lattice ``x`` of channel-index questions ``(n, 900)`` -- and,
    given answers, the one-hot ``y``: ``x`` gives walls/S/G as
    singletons and free cells both ``free`` and ``path`` alive.
    """
    hot = np.eye(N_CHANNELS, dtype=np.float32)
    x = hot[q].copy()
    x[q == CH_FREE, CH_PATH] = 1.0
    if a is None:
        return x
    return x, hot[a].copy()


def load(split: str = "train"):
    """ One split as ``(x, y)`` float32 arrays, ``(n, 900, 5)`` each. """
    data = torch.load(download(split), map_location="cpu",
                      weights_only=True)
    return encode(data["q"].numpy(), data["a"].numpy())


def given_of(x):
    """ The protected cells: singletons of the lattice, ``(..., 900)``. """
    total = x.sum(-1) if isinstance(x, torch.Tensor) else x.sum(-1)
    return total == 1


# --- path checking ----------------------------------------------------------

def bfs_distances(walls, start):
    """
    4-connected BFS distance from ``start`` on the non-wall cells of a
    ``(H, W)`` bool grid; ``-1`` where unreachable or wall.
    """
    h, w = walls.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    dist[start] = 0
    queue, head = [start], 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        d = dist[r, c]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] \
                    and dist[nr, nc] < 0:
                dist[nr, nc] = d + 1
                queue.append((nr, nc))
    return dist


def endpoints(grid):
    """ The (S, G) positions of a channel-index ``(H, W)`` grid, or None. """
    s = np.argwhere(grid == CH_START)
    g = np.argwhere(grid == CH_GOAL)
    if len(s) != 1 or len(g) != 1:
        return None
    return (int(s[0, 0]), int(s[0, 1])), (int(g[0, 0]), int(g[0, 1]))


def classify_one(pred, gt):
    """
    One predicted channel-index grid against its ground truth:
    ``(valid, minimal, exact)``, the reference's ``maze_classify`` on a
    single puzzle.  ``valid`` = exactly one S and G, the traversable
    cells (S, G, path) one 4-connected component containing both, no
    detached islands; ``minimal`` = valid and the path-cell count equals
    the ground truth's (the GT is minimal by construction, so a length
    match is optimality); ``exact`` = minimal and cellwise equal.
    """
    ends = endpoints(pred)
    if ends is None:
        return False, False, False
    (sr, sc), (gr, gc) = ends
    traversable = np.isin(pred, (CH_START, CH_GOAL, CH_PATH))
    dist = bfs_distances(~traversable, (sr, sc))
    reached = dist[gr, gc] >= 0
    valid = bool(reached and (dist >= 0).sum() == traversable.sum())
    minimal = valid and int((pred == CH_PATH).sum()) \
        == int((gt == CH_PATH).sum())
    exact = minimal and bool(
        ((pred == CH_PATH) == (gt == CH_PATH)).all())
    return valid, minimal, exact


def check_minimal(question, answer):
    """
    The Phase-1 structural check of one puzzle, on channel-index grids:
    the answer's path is a single connected S-to-G component whose cell
    count equals the BFS shortest distance minus one -- i.e. the ground
    truth is a minimal path.  Raises with the reason on failure.
    """
    ends = endpoints(question)
    assert ends is not None, "not exactly one S and one G"
    s, g = ends
    assert endpoints(answer) == ends, "answer moved S or G"
    walls = question == CH_WALL
    assert (walls == (answer == CH_WALL)).all(), "answer moved a wall"
    dist = bfs_distances(walls, s)
    shortest = int(dist[g])
    assert shortest > 0, "G unreachable from S"
    valid, minimal, _ = classify_one(answer, answer)
    assert valid, "path disconnected or with islands"
    n_path = int((answer == CH_PATH).sum())
    assert n_path == shortest - 1, \
        f"path has {n_path} cells, shortest distance {shortest}"
    return shortest


# --- the K-solutions sampler ------------------------------------------------

def _count_paths_to_g(d_g, on_dag, g_pos):
    """ Shortest-path counts to G per on-DAG cell, exact Python ints. """
    h, w = d_g.shape
    paths = {g_pos: 1}
    order = sorted((int(d_g[r, c]), r, c)
                   for r in range(h) for c in range(w) if on_dag[r, c])
    for _, r, c in order:
        if (r, c) == g_pos:
            continue
        total = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and on_dag[nr, nc] \
                    and int(d_g[nr, nc]) == int(d_g[r, c]) - 1:
                total += paths.get((nr, nc), 0)
        paths[(r, c)] = total
    return paths


def _sample_one_path(d_g, on_dag, paths_to_g, s_pos, g_pos, rng):
    """ One uniform shortest path, as the set of its interior cells. """
    h, w = d_g.shape
    cur, visited = s_pos, []
    while cur != g_pos:
        cands, weights = [], []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < h and 0 <= nc < w and on_dag[nr, nc] \
                    and int(d_g[nr, nc]) == int(d_g[cur]) - 1:
                cands.append((nr, nc))
                weights.append(paths_to_g.get((nr, nc), 0))
        total = sum(weights)
        assert total > 0, "dead end on the shortest-path DAG"
        draw, acc = rng.randrange(total), 0
        for cand, weight in zip(cands, weights):
            acc += weight
            if draw < acc:
                cur = cand
                break
        if cur != g_pos:
            visited.append(cur)
    return set(visited)


def n_shortest_paths(x_grid) -> int:
    """ The exact number of minimal S-to-G paths of one lattice grid. """
    walls = x_grid[..., CH_WALL] > 0.5
    ends = endpoints(np.where(x_grid[..., CH_START] > 0.5, CH_START,
                     np.where(x_grid[..., CH_GOAL] > 0.5, CH_GOAL,
                              CH_FREE)))
    s_pos, g_pos = ends
    d_s = bfs_distances(walls, s_pos)
    d_g = bfs_distances(walls, g_pos)
    total = int(d_s[g_pos])
    on_dag = (d_s >= 0) & (d_g >= 0) \
        & (d_s.astype(np.int64) + d_g.astype(np.int64) == total)
    return _count_paths_to_g(d_g, on_dag, g_pos)[s_pos]


def sample_k_solutions(x_grid, y_grid, k: int, rng: random.Random):
    """
    ``k`` minimal solutions of one puzzle, ``(k, H*W, 5)`` float32:
    ``[0]`` is always the ground truth, the rest uniform samples from
    the all-shortest-paths DAG (suffix-count weighted walk; exact
    bigint arithmetic, since 30x30 path counts overflow int64).
    Duplicates are possible and harmless: alpha just collapses.
    """
    h, w, c = x_grid.shape
    out = np.zeros((k, h * w, c), dtype=np.float32)
    out[0] = y_grid.reshape(h * w, c)
    if k <= 1:
        return out
    walls = x_grid[..., CH_WALL] > 0.5
    s_mask = x_grid[..., CH_START] > 0.5
    g_mask = x_grid[..., CH_GOAL] > 0.5
    s_pos = tuple(int(v) for v in np.argwhere(s_mask)[0])
    g_pos = tuple(int(v) for v in np.argwhere(g_mask)[0])
    d_s = bfs_distances(walls, s_pos)
    d_g = bfs_distances(walls, g_pos)
    total = int(d_s[g_pos])
    assert total > 0, "G unreachable"
    on_dag = (d_s >= 0) & (d_g >= 0) \
        & (d_s.astype(np.int64) + d_g.astype(np.int64) == total)
    paths_to_g = _count_paths_to_g(d_g, on_dag, g_pos)
    base = np.zeros((h, w, c), dtype=np.float32)
    base[..., CH_WALL] = walls
    base[..., CH_START] = s_mask
    base[..., CH_GOAL] = g_mask
    base[..., CH_FREE] = ~walls & ~s_mask & ~g_mask
    for index in range(1, k):
        cells = _sample_one_path(d_g, on_dag, paths_to_g, s_pos, g_pos, rng)
        solution = base.copy()
        for r, col in cells:
            solution[r, col, CH_FREE] = 0.0
            solution[r, col, CH_PATH] = 1.0
        out[index] = solution.reshape(h * w, c)
    return out


# --- the synthetic generator ------------------------------------------------

def _astar(walls, s, g):
    """ A* on the 4-connected grid; the path from s to g or None. """
    h, w = walls.shape
    heur = lambda p: abs(p[0] - g[0]) + abs(p[1] - g[1])
    heap, counter = [(heur(s), 0, s)], 0
    came, gscore = {}, {s: 0}
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < h and 0 <= nxt[1] < w) \
                    or walls[nxt]:
                continue
            tentative = gscore[cur] + 1
            if tentative < gscore.get(nxt, 1 << 30):
                gscore[nxt] = tentative
                came[nxt] = cur
                counter += 1
                heapq.heappush(heap, (tentative + heur(nxt), counter, nxt))
    return None


def hard_min_path_len(h: int, w: int) -> int:
    """ HRM's difficulty rule: 12.2% of the cells (110 at 30x30). """
    return max(max(h, w), round(0.122 * h * w))


def generate_maze(h, w, rng, wall_lo=0.30, wall_hi=0.50,
                  min_path_len=None, max_attempts=200):
    """
    One synthetic maze by Searchformer's recipe: random walls, random
    distinct S/G, kept when A* finds a plan of ``min_path_len`` edges.
    Returns ``(x, y)`` of shape ``(h*w, 5)``, or None after
    ``max_attempts``.
    """
    n = h * w
    min_path_len = max(h, w) if min_path_len is None else min_path_len
    for _ in range(max_attempts):
        n_walls = int(round(float(rng.uniform(wall_lo, wall_hi)) * n))
        walls = np.zeros(n, dtype=bool)
        walls[rng.choice(n, size=n_walls, replace=False)] = True
        walls = walls.reshape(h, w)
        free = np.argwhere(~walls)
        if len(free) < 2:
            continue
        picked = rng.choice(len(free), size=2, replace=False)
        s = (int(free[picked[0], 0]), int(free[picked[0], 1]))
        g = (int(free[picked[1], 0]), int(free[picked[1], 1]))
        path = _astar(walls, s, g)
        if path is None or len(path) - 1 < min_path_len:
            continue
        grid = np.full((h, w), CH_FREE, dtype=np.uint8)
        grid[walls] = CH_WALL
        answer = grid.copy()
        for r, c in path[1:-1]:
            answer[r, c] = CH_PATH
        grid[s], grid[g] = CH_START, CH_GOAL
        answer[s], answer[g] = CH_START, CH_GOAL
        return encode(grid.reshape(1, n), answer.reshape(1, n))
    return None


def synthetic_pool(n: int, side: int, seed: int = 0, hard: bool = True,
                   wall_lo: float = 0.30, wall_hi: float = 0.50):
    """
    ``n`` synthetic mazes as ``(x, y)`` arrays of shape ``(n, side**2,
    5)``, under HRM's ``hard`` filter by default.
    """
    rng = np.random.default_rng(seed)
    min_len = hard_min_path_len(side, side) if hard else None
    xs, ys = [], []
    attempts = 0
    while len(xs) < n:
        attempts += 1
        assert attempts <= 500 * n, "generator cannot satisfy the filter"
        made = generate_maze(side, side, rng, wall_lo, wall_hi, min_len)
        if made is not None:
            xs.append(made[0][0])
            ys.append(made[1][0])
    return np.stack(xs), np.stack(ys)


def straight_line(x, y, side: int = GRID):
    """
    The reference's ``simplify_to_straight_line`` diagnostic: same S and
    G, all walls removed, the ground truth the rounded-linspace straight
    line between them.  A model that cannot learn this cannot see S from
    G at all.
    """
    x = x.reshape(side, side, N_CHANNELS).copy()
    y = y.reshape(side, side, N_CHANNELS).copy()
    s = tuple(np.argwhere(y[:, :, CH_START] > 0.5)[0])
    g = tuple(np.argwhere(y[:, :, CH_GOAL] > 0.5)[0])
    steps = max(abs(int(g[0]) - int(s[0])), abs(int(g[1]) - int(s[1]))) + 1
    rows = np.linspace(s[0], g[0], steps).round().astype(int)
    cols = np.linspace(s[1], g[1], steps).round().astype(int)
    cells = set(zip(rows.tolist(), cols.tolist())) - {tuple(map(int, s)),
                                                      tuple(map(int, g))}
    x[:], y[:] = 0.0, 0.0
    x[:, :, CH_FREE] = x[:, :, CH_PATH] = 1.0
    y[:, :, CH_FREE] = 1.0
    for r, c in cells:
        y[r, c] = 0.0
        y[r, c, CH_PATH] = 1.0
    for grid, (r, c), channel in ((x, s, CH_START), (x, g, CH_GOAL),
                                  (y, s, CH_START), (y, g, CH_GOAL)):
        grid[r, c] = 0.0
        grid[r, c, channel] = 1.0
    return (x.reshape(side * side, N_CHANNELS),
            y.reshape(side * side, N_CHANNELS))


# --- the Phase-1 gate -------------------------------------------------------

def verify(split: str = "train", k: int = 8, k_puzzles: int = 25,
           seed: int = 0, log=print) -> dict:
    """
    Every Phase-1 assertion over one split, and the report behind the
    gate: wall fraction, path length and free-cell distributions, and a
    K-sampler validity check on ``k_puzzles`` puzzles.
    """
    data = torch.load(download(split), map_location="cpu",
                      weights_only=True)
    q, a = data["q"].numpy(), data["a"].numpy()
    assert q.shape == a.shape == (1000, N_CELLS), q.shape
    x, y = encode(q, a)
    free = q == CH_FREE
    assert ((q == CH_START).sum(1) == 1).all(), "S not unique"
    assert ((q == CH_GOAL).sum(1) == 1).all(), "G not unique"
    # x: free cells exactly {free, path}; givens match y.
    assert (x[free].sum(-1) == 2).all()
    assert (x[free][:, [CH_FREE, CH_PATH]] == 1).all()
    given = ~free
    assert (x[given].sum(-1) == 1).all()
    assert (x[given] == y[given]).all(), "givens disagree with y"
    # y is one-hot and only rewrites free cells to path.
    assert (y.sum(-1) == 1).all()
    assert ((a == CH_PATH) <= free).all(), "path on a non-free cell"
    lengths = np.zeros(len(q), dtype=np.int64)
    for index in range(len(q)):
        lengths[index] = check_minimal(q[index].reshape(GRID, GRID),
                                       a[index].reshape(GRID, GRID))
    walls = (q == CH_WALL).mean(1)
    n_free = free.sum(1)
    n_path = (a == CH_PATH).sum(1)
    assert (n_path == lengths - 1).all()
    # HRM's ">= 110" counts the whole route, S and G included: the
    # minimum path-cell ('o') count of both splits is exactly 108.
    assert n_path.min() + 2 >= 110, f"route below 110: {n_path.min() + 2}"
    rng = random.Random(seed)
    log_paths = []
    for index in range(k_puzzles):
        grid_x = x[index].reshape(GRID, GRID, N_CHANNELS)
        sols = sample_k_solutions(grid_x, y[index].reshape(
            GRID, GRID, N_CHANNELS), k, rng)
        for sol in sols:
            pred = sol.argmax(-1).reshape(GRID, GRID)
            valid, minimal, _ = classify_one(
                pred, a[index].reshape(GRID, GRID))
            assert valid and minimal, f"bad sampled path, puzzle {index}"
        log_paths.append(n_shortest_paths(grid_x))
    report = {
        "split": split, "n": len(q),
        "wall_frac": (float(walls.min()), float(walls.mean()),
                      float(walls.max())),
        "path_cells": (int(n_path.min()), float(n_path.mean()),
                       int(n_path.max())),
        "free_cells": (int(n_free.min()), float(n_free.mean()),
                       int(n_free.max())),
        "n_shortest_paths_digits": (
            min(len(str(p)) for p in log_paths),
            max(len(str(p)) for p in log_paths)),
        "k_checked": (k_puzzles, k)}
    log(report)
    return report


if __name__ == "__main__":
    for split in ("train", "test"):
        verify(split)
    print("PHASE 1 GATE: all assertions passed on both splits")
