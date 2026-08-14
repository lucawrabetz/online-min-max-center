"""Independently verify CCTA/BEA/NKCA solvers against a from-scratch transcription of
Algorithms 1/2/3 in paper.tex, over shipped test instances + randomized (incl. duplicate
points). Run from repo root: ./okc/bin/python scripts/check_pseudocode.py
Expected: all agree (objective AND facility set)."""

import glob, os, sys

sys.path.insert(0, os.getcwd())
import numpy as np
from log_config import setup_logging

setup_logging()
from allowed_types import FLInstanceType, _CCTA, _BEA, _NKCA
from problem import FLOfflineInstance, DPoint, euclidean_distance
from solvers import _SOLVER_FACTORY


def reference(points, Gamma, rule):
    """Direct transcription of Algorithms 1/2/3 over raw points. Returns (objective, F)."""
    T = len(points) - 1
    d = lambda a, b: float(np.linalg.norm(points[a] - points[b]))

    def v(t, F):
        return max(min(d(i, j) for j in F) for i in range(1, t + 1))

    F = {0}
    cum = 0.0
    total = 0.0
    for t in range(1, T + 1):
        vt_prev = v(t, F)
        build = (vt_prev >= Gamma) if rule == "nkca" else (cum + vt_prev >= Gamma)
        if build:
            ell = (
                t
                if rule == "bea"
                else max(range(1, t + 1), key=lambda i: min(d(i, j) for j in F))
            )
            F = F | {ell}
            cum = 0.0
            total += v(t, F)
        else:
            total += vt_prev
            cum += vt_prev
    return Gamma * (len(F) - 1) + total, sorted(F)


RULES = {_CCTA: "ccta", _BEA: "bea", _NKCA: "nkca"}
GAMMAS = [0.0, 0.5, 1.0, 1.4142135623730951, 2.0, 3.0, 10.0]


def run_solver(inst, sid):
    s = _SOLVER_FACTORY.solver(sid)
    s.configure_solver(inst)
    return s.solve(inst)


fails = checked = 0
for path in sorted(glob.glob("dat/test/*.csv")):
    name = os.path.basename(path)
    for Gamma in GAMMAS:
        for sid, rule in RULES.items():
            iid = FLInstanceType()
            iid.from_filename(name)
            inst = FLOfflineInstance(iid, distance=euclidean_distance)
            inst.read()
            inst.set_permutation_order("none", None)
            inst.set_gamma_run(Gamma)
            sol = run_solver(inst, sid)
            pts = [p.x for p in inst.points]
            eo, eF = reference(pts, Gamma, rule)
            gF = sorted({f for f in sol.facilities if f != -1} | {0})
            checked += 1
            if abs(sol.objective - eo) > 1e-9 or gF != eF:
                fails += 1
                print(
                    f"MISMATCH {name} G={Gamma} {sid.name}: got {sol.objective:.4f}/{gF} exp {eo:.4f}/{eF}"
                )
print(f"{checked - fails}/{checked} shipped-instance runs agree.")

rng = np.random.default_rng(0)
rf = rc = 0
for trial in range(400):
    T = int(rng.integers(1, 12))
    n = int(rng.integers(1, 4))
    if trial % 4 == 0:
        pool = rng.random((max(1, T // 3), n))
        pts = [pool[int(rng.integers(len(pool)))] for _ in range(T + 1)]
    else:
        pts = list(rng.random((T + 1, n)))
    Gamma = float(rng.choice([0.0, 0.25, 0.5, 1.0, 2.0, 5.0]))
    for sid, rule in RULES.items():
        inst = FLOfflineInstance(
            FLInstanceType("rand", n, T, 0), distance=euclidean_distance
        )
        inst.points = [DPoint(p) for p in pts]
        inst.set_distance_matrix()
        inst._is_set = True
        inst.set_permutation_order("none", None)
        inst.set_gamma_run(Gamma)
        sol = run_solver(inst, sid)
        eo, eF = reference(pts, Gamma, rule)
        gF = sorted({f for f in sol.facilities if f != -1} | {0})
        rc += 1
        if abs(sol.objective - eo) > 1e-9 or gF != eF:
            rf += 1
            if rf <= 5:
                print(
                    f"MISMATCH trial={trial} G={Gamma} {sid.name}: got {sol.objective:.4f}/{gF} exp {eo:.4f}/{eF}"
                )
print(f"{rc - rf}/{rc} randomized runs agree.")
