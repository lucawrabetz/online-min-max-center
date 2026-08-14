import os
import logging
import warnings
import numpy as np
from typing import List
from argparse import ArgumentParser
from allowed_types import FLSolverType, FLInstanceType
from util import _DAT
from problem import FLOfflineInstance, DPoint

from log_config import setup_logging, _LOGGER

setup_logging()

# Testbeds could be defined:
#  (a) "explicitly": a list of instance type objects
#  (b) "theoretically": a list of lists of values for instance parameters, leading to a list of instance type objects of all possible combinations

_RESERVED_SETNAMES = os.listdir(_DAT)


class TestbedGenerator:
    """
    Populate a test directory associated with a new set name with randomly generated instances.
    """

    def __init__(
        self,
        set_name: str,
        num_instances: int,
        instance_ids: List[FLInstanceType],
        fixed_zero: bool = False,
    ) -> None:
        """
        set_name: name of the testbed
        num_instances: number of instances to generate per instance type / shape / parameter combination
        fixed_zero: if True, override the initial facility x_0 (points[0]) to the origin.
            All other points (the arrivals x_1, ..., x_T) keep the usual uniform unit-square
            distribution. Used to pin the geometry of the do-nothing cost v_T(F_0).
        """
        if set_name in _RESERVED_SETNAMES:
            _LOGGER.log_warning(f"Set name {set_name} already exists in {_DAT}")
        self._set_name = set_name
        self._num_instances = num_instances
        self._instance_ids = instance_ids
        self._fixed_zero = fixed_zero
        self._final_ids: List[FLInstanceType] = []

    def generate_for_one_instanceid(self, instance_id: FLInstanceType) -> None:
        if instance_id.set_name != self._set_name:
            raise ValueError(
                f"Instance set name {instance_id.set_name} does not match testbed set name {self._set_name}."
            )
        if instance_id.id != -1:
            raise ValueError(
                f"Instance type {instance_id.set_name} already has id {instance_id.id}."
            )
        for i in range(self._num_instances):
            new_id = FLInstanceType(
                set_name=self._set_name, n=instance_id.n, T=instance_id.T
            )
            new_id.set_id(i)
            if os.path.exists(os.path.join(new_id.file_path)):
                _LOGGER.log_warning(
                    f"Instance {new_id.file_path} already exists, I will not generate and overwrite"
                )
                continue
            self._final_ids.append(new_id)
            instance = FLOfflineInstance(new_id)
            # for now, our random experiments always use the unit "square" and the interval [0, 1] for gamma, both uniform
            instance.set_random(low=0, high=1)
            if self._fixed_zero:
                # Pin x_0 to the origin, leaving the arrivals x_1, ..., x_T untouched,
                # then refresh the distance matrix to reflect the moved point.
                instance.points[0] = DPoint(np.zeros(new_id.n))
                instance.set_distance_matrix()
            instance.write_to_csv()
            _LOGGER.log_body(
                f"Generated instance {new_id.set_name}, n = {new_id.n}, T = {new_id.T}, number #{new_id.id}"
            )

    def write(self) -> List[FLInstanceType]:
        """
        Write instances to disk.
        Returns list of instance types with id's set.
        """
        if not os.path.exists(_DAT):
            os.makedirs(_DAT)
        if not os.path.exists(os.path.join(_DAT, self._set_name)):
            os.makedirs(os.path.join(_DAT, self._set_name))
        for instance_id in self._instance_ids:
            self.generate_for_one_instanceid(instance_id)
        return self._final_ids


def id_factory(set_name: str, n: List[int], T: List[int]) -> List[FLInstanceType]:
    """
    Generate instance types with all combinations of n and T.
    """
    ids = []
    for n_val in n:
        for T_val in T:
            ids.append(FLInstanceType(set_name=set_name, n=n_val, T=T_val))
    return ids


def main():
    parser = ArgumentParser()
    parser.add_argument("--set_name", type=str, default="test")
    parser.add_argument("--num_instances", type=int, default=30)
    parser.add_argument(
        "--fixed_zero",
        action="store_true",
        help="Fix the initial facility x_0 at the origin; arrivals stay uniform unit-square.",
    )
    parser.add_argument(
        "--T",
        type=int,
        nargs="+",
        default=[50],
        help="Number of demand points per instance. Pass several to generate one "
        "instance type per value. Runs can also take a prefix of a larger instance "
        "with run_experiments.py --T.",
    )
    args = parser.parse_args()
    set_name = args.set_name
    _LOGGER.log_header(f"Generating testbed {set_name}")
    num_instances = args.num_instances
    dimensions = [2]
    time_periods = args.T
    ids = id_factory(set_name, dimensions, time_periods)
    generator = TestbedGenerator(
        set_name, num_instances, ids, fixed_zero=args.fixed_zero
    )
    generator.write()


if __name__ == "__main__":
    main()
