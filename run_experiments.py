import math
import logging
from argparse import ArgumentParser, Namespace
from typing import List, Optional

from experiments import FLExperiment
from allowed_types import _SOLVER_SETS, _SOLVERS

from log_config import setup_logging, _LOGGER
from problem import euclidean_distance, taxicab_distance

setup_logging()

DISTANCES = {"euclidean": euclidean_distance, "taxicab": taxicab_distance}
_SOLVERS_BY_NAME = {solver.name: solver for solver in _SOLVERS}
_PERM_CHOICES = ["none", "full", "nearest", "farthest"]

# Guard against accidentally launching a sweep of MILP solves. The MILPs are O(T^2)
# (fully offline) and O(T^3) (semi-offline) in size and run under a 1 hour time limit,
# so a Gamma sweep over a full instance set is a multi-day job. Pass --force to run one
# anyway.
_MILP_RUN_WARN_THRESHOLD = 8


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Run online facility location experiments over an instance set."
    )
    parser.add_argument("--set_name", type=str, default="test")
    parser.add_argument("--distance", type=str, default="euclidean", choices=DISTANCES)
    parser.add_argument("--write", action="store_true", default=True)
    parser.add_argument(
        "--no_write",
        dest="write",
        action="store_false",
        help="Run without appending anything to the results database.",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default="online",
        help=(
            "Which solvers to run: a named set (online = CCTA/BEA/NKCA, offline = the "
            "MILPs, all), or an explicit comma-separated list of solver names, e.g. "
            "'OMIP,CCTA,BEA,NKCA'. Valid names: " + ", ".join(_SOLVERS_BY_NAME) + "."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the guard on large sweeps that include the MILP solvers.",
    )

    gamma_group = parser.add_argument_group("fixed cost (Gamma)")
    gamma_group.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Single Gamma value. Omit (and omit the sweep) to use each instance's own Gamma.",
    )
    gamma_group.add_argument("--gamma_lo", type=float, default=None)
    gamma_group.add_argument("--gamma_hi", type=float, default=None)
    gamma_group.add_argument(
        "--gamma_step",
        type=float,
        default=1.0,
        help="Step for the --gamma_lo/--gamma_hi sweep. Default 1.0.",
    )

    parser.add_argument("--T", type=int, default=None)
    parser.add_argument(
        "--perm",
        type=str,
        nargs="+",
        default=["none"],
        choices=_PERM_CHOICES,
        help="One or more arrival-order permutations to run in a single sweep, e.g. "
        "--perm none nearest farthest.",
    )
    parser.add_argument(
        "--firstf",
        type=int,
        default=-1,
        help="Force the arrival order to start at this point. Requires --perm != none.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=-1,
        help=(
            "Number of permutation replications per Gamma, each forced to start at a "
            "different first facility (0, ..., reps-1). Requires --perm != none. "
            "This is independent of the Gamma sweep - replication over the instances "
            "in the set is automatic."
        ),
    )
    return parser.parse_args()


def resolve_solvers(spec: str):
    """
    Resolve the --solvers argument into a list of solver ids. Accepts a named set
    (online/offline/all) or an explicit comma-separated list of solver names.
    """
    spec = spec.strip()
    if spec in _SOLVER_SETS:
        return _SOLVER_SETS[spec]
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if not names:
        raise ValueError("--solvers is empty.")
    unknown = [nm for nm in names if nm not in _SOLVERS_BY_NAME]
    if unknown:
        raise ValueError(
            f"Unknown solver(s) {unknown}. Use a named set "
            f"({', '.join(sorted(_SOLVER_SETS))}) or names from "
            f"{', '.join(_SOLVERS_BY_NAME)}."
        )
    return [_SOLVERS_BY_NAME[nm] for nm in names]


def gamma_values(args: Namespace) -> List[Optional[float]]:
    """
    Resolve the Gamma values to run. Returns [None] to mean 'use each instance's own
    Gamma', which is what FLExperiment does when it is not given one.
    """
    sweeping = args.gamma_lo is not None or args.gamma_hi is not None
    if sweeping:
        if args.gamma_lo is None or args.gamma_hi is None:
            raise ValueError("--gamma_lo and --gamma_hi must be passed together.")
        if args.gamma is not None:
            raise ValueError("Pass either --gamma or the --gamma_lo/--gamma_hi sweep.")
        if args.gamma_step <= 0:
            raise ValueError("--gamma_step must be positive.")
        if args.gamma_hi < args.gamma_lo:
            raise ValueError("--gamma_hi must be at least --gamma_lo.")
        # Step by an integer count rather than accumulating, so that float error cannot
        # drop or duplicate the endpoint.
        num_steps = int(
            math.floor((args.gamma_hi - args.gamma_lo) / args.gamma_step + 1e-9)
        )
        return [args.gamma_lo + i * args.gamma_step for i in range(num_steps + 1)]
    if args.gamma is not None:
        return [args.gamma]
    return [None]


def first_facilities(args: Namespace) -> List[Optional[int]]:
    """
    Resolve the first facilities to force, one run each. Returns [None] for the usual
    case of not forcing one.
    """
    if args.reps > 0:
        if "none" in args.perm:
            raise ValueError(
                "--reps > 0 forces a first facility for each replication, which "
                "requires a permutation. Drop 'none' from --perm, or drop --reps "
                "(replication over the instances in the set is automatic)."
            )
        if args.firstf != -1:
            raise ValueError("Pass either --firstf or --reps, not both.")
        return list(range(args.reps))
    if args.firstf != -1:
        if "none" in args.perm:
            raise ValueError("--firstf cannot be combined with --perm none.")
        return [args.firstf]
    return [None]


def check_run_size(args: Namespace, solver_ids: List, num_runs: int) -> None:
    includes_milp = any(solver.name in {"OMIP", "SOMIP"} for solver in solver_ids)
    if includes_milp and num_runs > _MILP_RUN_WARN_THRESHOLD and not args.force:
        raise ValueError(
            f"This would launch {num_runs} experiment(s) including the MILP solvers "
            f"(--solvers {args.solvers}), each of which solves every instance in the "
            f"set under a 1 hour time limit. Narrow the sweep, drop the MILP solvers, "
            f"or pass --force if you really mean it."
        )


def main():
    args = parse_args()
    distance_function = DISTANCES[args.distance]
    solver_ids = resolve_solvers(args.solvers)
    gammas = gamma_values(args)
    firsts = first_facilities(args)
    perms = args.perm

    check_run_size(args, solver_ids, len(perms) * len(gammas) * len(firsts))
    _LOGGER.log_body(
        f"Sweeping {len(perms)} permutation(s) x {len(gammas)} Gamma value(s) x "
        f"{len(firsts)} first facility/ies over set {args.set_name} with solvers "
        f"[{', '.join(s.name for s in solver_ids)}]"
    )

    for perm in perms:
        for gamma in gammas:
            for first_facility in firsts:
                experiment = FLExperiment(
                    args.set_name, distance=distance_function, write=args.write
                )
                experiment.configure_experiment(
                    solver_ids=solver_ids,
                    gamma=gamma,
                    T=args.T,
                )
                experiment.run(permutation=perm, first_facility=first_facility)


if __name__ == "__main__":
    main()
