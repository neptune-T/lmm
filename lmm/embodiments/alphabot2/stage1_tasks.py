"""Stage-1 task distribution, held-out splits and instructions (numpy only).

Controlled mobile-manipulation distribution for AlphaBot 2 embodiment/task
competence -- deliberately NOT BEHAVIOR-scale diversity.

World layout (metres, radians; world +z up). Table A (the pick table) is fixed
at ``TABLE_A``; its near edge faces -x. Table B sits at one of three places
around the start area. Only relative poses matter (the floor is featureless).

Families::

    nav_pick_lift        navigate -> pick -> lift
    nav_pick_nav_place   navigate -> pick -> navigate -> place   (full task)
    short_pick_place     start near table A; pick -> place into a box on the SAME table
    reorient_pick_lift   start facing away (90-166 deg); turn -> approach -> pick -> lift

Factor cells (held-out by COMBINATION, per family)::

    base cell b (4)       (range, side): range A/B = nearer/farther start (for
                          reorient: smaller/larger yaw offset), side = left/right
                          lateral offset (reorient: turn direction ccw/cw)
    object cell o (3)     object on table A: left / centre / right
    receptacle cell r (3) table-B location: left / right / behind the start
                          (short_pick_place: box on table A, left or right of the
                          object, with table B as scenery)

Every level of every factor occurs in TRAIN; val/test hold out whole
(b, o, r) combinations, so evaluation needs modest compositional
generalization. Splits are decided by the cell tuple alone, never by frame.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

FAMILIES = ("nav_pick_lift", "nav_pick_nav_place", "short_pick_place", "reorient_pick_lift")
ARMS = ("left", "right")
OBJECT_NAMES = ("red cube", "green cylinder", "yellow block", "purple bar")
RECEPTACLE_NAME = "blue box"

TABLE_A = (1.60, 0.0, 0.0)
TABLE_A_DEPTH, TABLE_A_WIDTH = 0.50, 0.80
#: table-B placements, receptacle cell r -> (x, y, yaw); yaw = direction the robot faces when at it
TABLE_B = {0: (0.30, 1.85, math.pi / 2), 1: (0.30, -1.85, -math.pi / 2), 2: (-1.45, 0.0, math.pi)}
RECEPTACLE_LEVELS = ("table_b_left", "table_b_right", "table_b_behind")
OBJECT_LEVELS = ("left", "centre", "right")
BASE_LEVELS = ("A_left", "A_right", "B_left", "B_right")

#: nominal pad-midpoint point of each arm in the chassis frame (x fwd, y left)
ARM_NOMINAL = {"left": (0.45, 0.25), "right": (0.45, -0.25)}

#: held-out (b, o, r) combinations; identical for every family
HELDOUT = {
    "test": ((0, 2, 1), (3, 0, 2), (1, 1, 0), (2, 0, 1)),
    "val": ((2, 2, 2), (0, 1, 1)),
}
SPLITS = ("train", "val", "test")


def split_of(cells) -> str:
    c = tuple(int(v) for v in cells)
    for s in ("test", "val"):
        if c in HELDOUT[s]:
            return s
    return "train"


def all_cells():
    return [(b, o, r) for b in range(4) for o in range(3) for r in range(3)]


def cells_of(split: str):
    return [c for c in all_cells() if split_of(c) == split]


def instruction(family: str, arm: str, obj: str) -> str:
    if family in ("nav_pick_nav_place", "short_pick_place"):
        return f"Use the {arm} arm to pick up the {obj} and put it in the {RECEPTACLE_NAME}."
    return f"Use the {arm} arm to pick up the {obj} and lift it."


@dataclass
class TaskConfig:
    config_id: str
    split: str
    family: str
    arm: str
    object: str
    cells: tuple
    object_xy: tuple
    object_yaw: float
    table_a: tuple
    table_b: tuple
    box: tuple                      # (x, y, yaw, on_table)  on_table in {"a", "b"}
    base_xy: tuple
    base_yaw: float
    instruction: str
    seed: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cells"] = list(self.cells)
        return d


def _rot(y):
    c, s = math.cos(y), math.sin(y)
    return np.array([[c, -s], [s, c]])


def standoff(obj_xy, arm: str, approach_yaw: float) -> np.ndarray:
    """Chassis xy that puts ``obj_xy`` at the arm's nominal point when facing ``approach_yaw``."""
    return np.asarray(obj_xy, float) - _rot(approach_yaw) @ np.asarray(ARM_NOMINAL[arm])


def sample_config(rng: np.random.Generator, family: str, cells, arm: str, obj: str, config_id: str,
                  seed: int) -> TaskConfig:
    b, o, r = (int(v) for v in cells)
    ax, ay, ayaw = TABLE_A
    # object on table A, near edge region (x: edge + 0.05..0.12)
    edge = ax - TABLE_A_DEPTH / 2
    oy = {0: (0.10, 0.26), 1: (-0.08, 0.08), 2: (-0.26, -0.10)}[o]
    obj_xy = np.array([edge + rng.uniform(0.05, 0.12), rng.uniform(*oy)])
    obj_yaw = float(rng.uniform(-0.5, 0.5))
    tb = TABLE_B[r]
    # box on table B: centre 0.09-0.12 m behind the near edge (the chassis then stops 0.33-0.36 m
    # from the edge, as at table A; deeper boxes put the chassis footprint against table B)
    bl = np.array([-TABLE_A_DEPTH / 2 + rng.uniform(0.09, 0.12), rng.uniform(-0.18, 0.18)])
    bxy = np.array(tb[:2]) + _rot(tb[2]) @ bl
    box = (float(bxy[0]), float(bxy[1]), float(tb[2] + rng.uniform(-0.3, 0.3)), "b")
    rng_level, side = divmod(b, 2)          # rng_level 0 = A (nearer), 1 = B (farther); side 0 = left, 1 = right
    sgn = 1.0 if side == 0 else -1.0
    s = standoff(obj_xy, arm, ayaw)
    if family == "short_pick_place":
        # box on table A beside the object, on the side away from the other arm's workspace
        dy = rng.uniform(0.16, 0.20) * (1.0 if arm == "left" else -1.0)
        by = float(np.clip(obj_xy[1] + dy, -TABLE_A_WIDTH / 2 + 0.10, TABLE_A_WIDTH / 2 - 0.10))
        if abs(by - obj_xy[1]) < 0.15:           # clipped at the table end: go the other way
            by = float(obj_xy[1] - dy)
        box = (float(edge + rng.uniform(0.10, 0.13)), by, float(rng.uniform(-0.3, 0.3)), "a")
        # stand so that the midpoint of object and box sits at the arm's nominal point
        s = standoff((obj_xy + np.array(box[:2])) / 2, arm, ayaw)
        d = (0.05, 0.15) if rng_level == 0 else (0.15, 0.30)
        base = s + np.array([-rng.uniform(*d), sgn * rng.uniform(0.0, 0.15)])
        yaw = float(rng.uniform(-0.3, 0.3))
    elif family == "reorient_pick_lift":
        dist = rng.uniform(0.45, 0.85)
        base = s + np.array([-dist, sgn * rng.uniform(0.0, 0.25)])
        mag = rng.uniform(1.6, 2.3) if rng_level == 0 else rng.uniform(2.3, 2.9)
        yaw = float(sgn * mag)                   # left side -> facing ccw of the table (turn cw back) etc.
    else:
        d = (0.35, 0.70) if rng_level == 0 else (0.70, 1.20)
        base = s + np.array([-rng.uniform(*d), sgn * rng.uniform(0.15, 0.55)])
        yaw = float(rng.uniform(-0.5, 0.5))
    return TaskConfig(config_id=config_id, split=split_of(cells), family=family, arm=arm, object=obj,
                      cells=(b, o, r), object_xy=tuple(float(v) for v in obj_xy), object_yaw=obj_yaw,
                      table_a=TABLE_A, table_b=tuple(float(v) for v in tb), box=box,
                      base_xy=tuple(float(v) for v in base), base_yaw=yaw,
                      instruction=instruction(family, arm, obj), seed=int(seed))


def generate(split: str, per_family: dict, seed: int) -> list[TaskConfig]:
    """Balanced configs for one split: round-robin over the split's cells,
    alternating arms and cycling objects, independently per family."""
    cells = cells_of(split)
    out = []
    for fi, fam in enumerate(FAMILIES):
        n = int(per_family.get(fam, 0))
        rng = np.random.default_rng([seed, fi, SPLITS.index(split)])
        order = rng.permutation(len(cells))
        for i in range(n):
            c = cells[order[i % len(cells)]]
            arm = ARMS[i % 2]
            obj = OBJECT_NAMES[(i // 2) % len(OBJECT_NAMES)]
            cid = f"{split}_{fam}_{i:04d}"
            s = int(rng.integers(0, 2**31 - 1))
            out.append(sample_config(np.random.default_rng(s), fam, c, arm, obj, cid, s))
    return out
