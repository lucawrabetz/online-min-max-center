import os
import logging
from log_config import _LOGGER
from typing import List, Dict, Any
from util import _DAT

# opti-face


class FLSolverType:
    """
    Class to store all (id) information about a solver type:
    """

    def __init__(self, name: str, parameters: Dict[str, str] = {}) -> None:
        self._name: str = name
        self._parameters: Dict[str, str] = parameters

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> Dict[str, str]:
        return self._parameters


class FLInstanceType:
    """
    Class to store all (id) information about an instance type:
    """

    def __init__(
        self, set_name: str = "test", n: int = 2, T: int = 3, instance_id: int = None
    ) -> None:
        self.set_name: str = set_name
        self.n: int = n
        self.T: int = T
        # Unique identifier if all instance parameters already exist in another instance, only set for saved instances.
        self.id: int = -1 if instance_id is None else instance_id
        self._file_path: str = ""
        self.update_filepath()

    @property
    def file_path(self) -> str:
        self.update_filepath()
        if self._file_path == "":
            # TODO: add warning here
            return ""
        else:
            return self._file_path

    def update_filepath(self) -> None:
        if self.id >= -1:
            self._file_path = os.path.join(
                _DAT, self.set_name, f"{self.set_name}_{self.n}_{self.T}_{self.id}.csv"
            )
        else:
            # TODO: decouple/wrap instead of checking.
            self._file_path = ""

    def set_id(self, instance_id: int) -> None:
        self.id = instance_id
        self.update_filepath()

    def from_filename(self, filename: str) -> None:
        split = filename.split("_")
        if len(split) != 4:
            raise ValueError("Invalid filename")
        self.set_name = split[0]
        self.n = int(split[1])
        self.T = int(split[2])
        self.id = int(split[3].replace(".csv", ""))

    def print(self) -> None:
        _LOGGER.log_body(f"n: {self.n}, T: {self.T}")


# implementer

_TEST_SHAPE = FLInstanceType()

_OMIP = FLSolverType("OMIP")
_SOMIP = FLSolverType("SOMIP")
_CCTA = FLSolverType("CCTA")
_BEA = FLSolverType("BEA")
_NKCA = FLSolverType("NKCA")
_SOLVERS: List[Any] = [_OMIP, _SOMIP, _CCTA, _BEA, _NKCA]
# The online algorithms proceed iteration by iteration, so per-iteration timings are
# meaningful for them (and only for them).
_ONLINE_SOLVERS: List[Any] = [_CCTA, _BEA, _NKCA]
_ONLINE_SOLVER_NAMES: List[str] = [solver.name for solver in _ONLINE_SOLVERS]
# The MILP formulations. Note that these do not scale like the online algorithms do -
# see the note in run_experiments.py before running these over a large sweep.
_OFFLINE_SOLVERS: List[Any] = [_OMIP, _SOMIP]
_SOLVER_SETS: Dict[str, List[Any]] = {
    "all": _SOLVERS,
    "online": _ONLINE_SOLVERS,
    "offline": _OFFLINE_SOLVERS,
}

_BUILT_IN_TYPES: List[Any] = [str, int, float, bool]
_ALLOWED_TYPES: List[Any] = _SOLVERS.copy()
_ALLOWED_TYPES.extend(_BUILT_IN_TYPES)
