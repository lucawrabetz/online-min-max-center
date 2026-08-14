import csv, os, sys
import logging
import numpy as np
import pandas as pd
from typing import Callable, List, Tuple, Optional
from util import Data, _DAT, _FINALDB
from log_config import _LOGGER
from allowed_types import (
    FLInstanceType,
    _TEST_SHAPE,
    _CCTA,
    _OMIP,
    _ONLINE_SOLVER_NAMES,
)


class DPoint:
    """
    Class to hold a demand point in n-dimensional space. Doesn't have any setup methods, so doesn't inherit Data.
    """

    def __init__(self, x: np.ndarray) -> None:
        self.x: np.ndarray = np.asarray(x)
        self.n: int = len(x)

    def print(self) -> None:
        with np.printoptions(precision=4, suppress=True):
            _LOGGER.log_bodydebug(self.x)


class FLDistribution:
    """
    Class to hold "distributions" for generating demand points for FLOfflineInstances.
    """

    def __init__(self, instance_id: FLInstanceType) -> None:
        self.id: FLInstanceType = instance_id
        self.rng: np.random.Generator = np.random.default_rng()

    def gamma_interval(self, low: int, high: int) -> float:
        """
        Generate Gamma uniformly at random from [low, high].
        """
        return self.rng.uniform(low, high)

    def x_unit_square(self) -> List[DPoint]:
        """
        Generate points uniformly at random from the unit square of dimension n.
        """
        return [DPoint(self.rng.random((self.id.n,))) for _ in range(self.id.T + 1)]


### distance functions - metric for instances
### instance class has a callable attribute for this
def euclidean_distance(point_a: DPoint, point_b: DPoint) -> float:
    return np.linalg.norm(point_a.x - point_b.x)


def taxicab_distance(point_a: DPoint, point_b: DPoint) -> float:
    return np.linalg.norm(point_a.x - point_b.x, ord=1)


class FLOfflineInstance(Data):
    """
    Class to hold offline instance data for the online min max center problem.
    """

    def __init__(
        self,
        instance_id: FLInstanceType = _TEST_SHAPE,
        distance: Callable[[DPoint, DPoint], float] = euclidean_distance,
    ) -> None:
        super().__init__()
        self.id: FLInstanceType = instance_id
        self.points: List[DPoint] = [
            DPoint(np.zeros(self.id.n)) for _ in range(self.id.T + 1)
        ]
        # Note that the information encoded by these 3 variables could be encoded in less, but this is optimized for readability.
        # self.Gamma is the value currently active for solving this instance.
        # self.original_Gamma is the original value, i.e. the one written to the instance file.
        # self.set_Gamma is the value of gamma set manually for an experiment that is run with a different value than on file.
        # data functionality should look at set_Gamma when looking for the run parameter, and original_Gamma when looking for the instance parameter.
        # self.Gamma should always equal at least one of set and original - it is used in algorithm functionality
        self.Gamma: float = 0.0
        self.original_Gamma: float = 0.0
        self.set_Gamma: float = -1.0
        # same system as for gamma for T
        self.first_T(self.id.T)
        self.set_T: int = -1
        self.distances: np.ndarray = np.zeros((self.id.T + 1, self.id.T + 1))
        self.distance: Callable[[DPoint, DPoint], float] = distance
        self._permutation: str = "none"
        self._order: List[int]

    def get_offline_solution_facilities(self) -> Tuple[List[int], List[int]]:
        """
        Return the list of facilities set by the offline problem ran with the same gamma and null.
        """
        final_df = pd.read_csv(_FINALDB)
        offline_df = final_df[
            (final_df["set_name"] == self.id.set_name)
            & (final_df["n"] == self.id.n)
            & (final_df["T"] == self.id.T)
            & (final_df["T_run"] == self.set_T)
            & (final_df["Gamma_run"] == self.set_Gamma)
            & (final_df["id"] == self.id.id)
            & (final_df["perm"] == "none")
            & (final_df["solver"] == _OMIP.name)
            & (final_df["optimal"] == True)
        ]
        if offline_df.empty:
            raise ValueError("No offline solution found for this instance.")
        if offline_df["facilities_str"].nunique() != 1:
            raise ValueError(
                "Multiple facilities solutions found for this instance - check your database, something is up."
            )
        facilities_str = offline_df["facilities_str"].iat[0]
        facilities = [int(i) for i in facilities_str.split("-")]
        not_facilities = [i for i in range(self.id.T + 1) if i not in facilities]
        return facilities, not_facilities

    def min_min_distance(self, order: List[int], points: List[int]) -> int:
        # returns the index of the point with the minimum distance to its minimum distance already ordered point.
        min_distance = sys.float_info.max
        min_index = -1
        summary: str = ""
        for point in points:
            min_distance_point = sys.float_info.max
            for chosen in order:
                distance = self.distances[point][chosen]
                if distance < min_distance_point:
                    min_distance_point = distance
            summary += f"{point}, mind: {min_distance_point}; "
            if min_distance_point < min_distance:
                min_distance = min_distance_point
                min_index = point
        return min_index

    def max_min_distance(self, order: List[int], points: List[int]) -> int:
        # returns the index of the point with the maximum distance to its minimum distance already ordered point.
        max_distance = -1.0
        max_index = -1
        summary: str = ""
        for point in points:
            min_distance_point = sys.float_info.max
            for chosen in order:
                distance = self.distances[point][chosen]
                if distance < min_distance_point:
                    min_distance_point = distance
            summary += f"{point}, mind: {min_distance_point}; "
            if min_distance_point > max_distance:
                max_distance = min_distance_point
                max_index = point
        return max_index

    def nearest_facility_order(self, first_facility: Optional[int]) -> List[int]:
        first = first_facility if first_facility is not None else 0
        points: List[int] = [i for i in range(self.id.T + 1) if i != first]
        order: List[int] = [first]
        while len(points) > 0:
            new_index: int = self.min_min_distance(order, points)
            order.append(new_index)
            points.remove(new_index)
        return order

    def farthest_facility_order(self, first_facility: Optional[int]) -> List[int]:
        first = first_facility if first_facility is not None else 0
        points: List[int] = [i for i in range(self.id.T + 1) if i != first]
        order: List[int] = [first]
        while len(points) > 0:
            new_index: int = self.max_min_distance(order, points)
            order.append(new_index)
            points.remove(new_index)
        return order

    def set_permutation_order(
        self, permutation: str, first_facility: Optional[int]
    ) -> None:
        self._permutation = permutation
        # Hacky special case for *fixedzero instance sets: point 0 (x_0) is pinned to the
        # origin and must remain the initial facility F_0 = {x_0}, so it has to stay the
        # first point under every arrival order. Forcing first_facility = 0 keeps index 0
        # first while preserving each permutation's semantics (full pins it, nearest/
        # farthest build their greedy order starting from it). 'none' is already 0-first
        # and forbids a forced first facility, so we leave it untouched.
        if "fixedzero" in self.id.set_name and permutation != "none":
            if first_facility is not None and first_facility != 0:
                _LOGGER.log_warning(
                    f"fixedzero instance {self.id.set_name}: overriding requested first "
                    f"facility {first_facility} with 0 (x_0 must remain the origin facility)."
                )
            first_facility = 0
        none_order = [i for i in range(self.id.T + 1)]
        if permutation == "none":
            if first_facility is not None:
                raise ValueError(
                    "can't force first facility when passing none permutation"
                )
            self._order = none_order
            return
        elif permutation == "full":
            # uniform permutation of none_order
            if first_facility is not None:
                none_order.remove(first_facility)
                rest_of_facilities = list(np.random.permutation(none_order))
                self._order = [first_facility] + rest_of_facilities
            else:
                self._order = list(np.random.permutation(none_order))
            return
        elif permutation == "nearest":
            self._order = self.nearest_facility_order(first_facility)
            return
        elif permutation == "farthest":
            self._order = self.farthest_facility_order(first_facility)
            return
        # get the list of facilities set by the offline problem
        offline_facilities, offline_dpoints = self.get_offline_solution_facilities()
        if permutation == "start":
            # change
            self._order = list(np.random.permutation(offline_facilities)) + list(
                np.random.permutation(offline_dpoints)
            )
        elif permutation == "end":
            # change
            self._order = list(np.random.permutation(offline_dpoints)) + list(
                np.random.permutation(offline_facilities)
            )

    def set_distance_matrix(self) -> None:
        """
        Set distance matrix based on demand points.
        """
        for i in range(self.id.T):
            for j in range(i + 1, self.id.T + 1):
                if i == j:
                    self.distances[i][j] = 0.0
                else:
                    self.distances[i][j] = self.distance(self.points[i], self.points[j])
                    self.distances[j][i] = self.distances[i][j]

    def get_distance(self, i: int, j: int) -> float:
        """
        Return distance between points i and j.
        """
        return self.distances[self._order[i]][self._order[j]]

    def first_T(self, T: int) -> None:
        self.original_T = T
        self.T = T

    def new_active_T(self, T: int) -> None:
        """
        "slice" the points and distances to the new T, destroying the rest of the data (it is available in the file, and the assumption is that instances are constructed to be used for a single T, and we also prioritize the memory conditions during the experiment)
        """
        self.set_T = T
        self.T = T
        self.id.T = T
        self.points = self.points[: T + 1]
        self.distances = self.distances[: T + 1, : T + 1]

    def first_gamma(self, Gamma: float) -> None:
        self.original_Gamma = Gamma
        self.Gamma = Gamma

    def new_active_gamma(self, Gamma: float) -> None:
        self.set_Gamma = Gamma
        self.Gamma = Gamma

    def set_random(self, low: int = 5, high: int = 15, Gamma: float = None) -> None:
        """
        Set demand points by instantiating and calling on an FLDistribution object.
        Update distance matrix.
        Gamma is generated uniformly [0, 1] by default.
        To set it manually to a different value than its instance for a run, set_gamma_run.
        """
        dist = FLDistribution(self.id)
        if Gamma:
            self.first_gamma(Gamma)
        else:
            self.first_gamma(dist.gamma_interval(low, high))
        self.points = dist.x_unit_square()
        self.set_distance_matrix()
        self._is_set = True

    def set_gamma_run(self, Gamma: float):
        """
        Also changes self.gamma, so we don't need to change any functionality outside of it based on whether gamma was manually set. But we know it was if set_Gamma == Gamma, or by checking that self.set_Gamma >= 0.
        """
        if Gamma < 0:
            raise ValueError("Gamma must be non-negative.")
        self.new_active_gamma(Gamma)

    def set_T_run(self, T: int):
        """
        Also changes self.T, so we don't need to change any functionality outside of it based on whether gamma was manually set. But we know it was if set_T == T, or by checking that self.set_T >= 0.
        When we change T run, we also need to change the size of the following members, so that they are consistent with the new T and any solvers etc. constructed with this instance use data of the correct size:
            - self.points
            - self.distances
        do this in new_active_T
        """
        if T < 0:
            raise ValueError("T must be non-negative.")
        if T > self.original_T:
            raise ValueError("T must be less than or equal to the original T.")
        self.new_active_T(T)

    def read_pointscsv_file(self, file) -> List[np.ndarray]:
        data = np.genfromtxt(file, delimiter=",")
        arrays = [row for row in data]
        return arrays

    def read(self) -> None:
        """
        Read demand points from file if it exists in instance id.
        Check if the file is a points file or a distance matrix, read accordingly.
        Update distance matrix (if it was a points file).
        """
        if self.id.id == -1:
            raise ValueError("Instance has no id, cannot construct filename.")
        filepath = self.id.file_path
        if not os.path.exists(filepath):
            raise ValueError(f"File {filepath} does not exist.")
        if not filepath.endswith(".csv"):
            raise ValueError(f"File {filepath} is not a csv.")

        # check if the file is a distance matrix or a points file
        # read the first line of the file
        # if it has a single value (gamma) it is a points file
        # if it has multiple values (csv header for distance matrix) it is a distance matrix file
        with open(filepath, "r") as file:
            first_line = file.readline().strip().split(",")
            if len(first_line) == 1:
                self.Gamma = float(first_line[0])
                points: List[np.ndarray] = self.read_pointscsv_file(file)
                self.points = [DPoint(point) for point in points]
                self.set_distance_matrix()
                self._is_set = True
            else:
                # don't set skipheader to True, as it was already skipped by
                # first_line = file.readline()..... above
                self.distances = np.genfromtxt(file, delimiter=",")
                self._is_set = True
                pass

    def print(self) -> None:
        _LOGGER.log_bodydebug(f"n: {self.id.n}, T: {self.id.T}, Gamma: {self.Gamma}")
        for t in range(self.id.T + 1):
            self.points[t].print()

    def print_distance_matrix(self) -> None:
        _LOGGER.log_matrix(self.distances, "Distance")

    def write_to_csv(self) -> None:
        """
        Write instance to csv file.
        Line 1: Gamma
        Line 2: x_0^1, x_0^2, ..., x_0^n
        Line 3: x_1^1, x_1^2, ..., x_1^n
        ...
        Line T+1: x_T^1, x_T^2, ..., x_T^n
        """
        if self.id.id == -1:
            raise ValueError("Instance has no id, cannot construct filename.")
        filepath = self.id.file_path
        with open(filepath, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self.Gamma])
            for point in self.points:
                writer.writerow(point.x)


class CCTState(Data):
    """
    Class to hold state for the CCTCA algorithm.
    """

    def __init__(self) -> None:
        super().__init__()
        self.T: int = 0
        self.n: int = 0
        self.Gamma: float = 0.0
        self.objective: float = 0.0
        self.num_facilities: int = 0
        self.t_index: int = 0
        self.cum_var_cost: float = 0.0
        self.facilities: List[int] = []
        self.distance_to_closest_facility: List[float] = []
        self.service_costs: List[float] = []
        self.iteration_times_ms: List[float] = []
        self.average_iteration_time_ms: float = 0.0

    def configure_state(self, instance: FLOfflineInstance) -> None:
        """
        Configure algorithm global state.
        """
        if not instance.is_set:
            raise ValueError("Instance must be set before configuring state.")
        self.T = instance.id.T
        self.n = instance.id.n
        self.Gamma = instance.Gamma
        self.num_facilities = 1
        self.t_index = 1
        self.cum_var_cost = 0.0
        self.iteration_times_ms = [0.0]
        self.facilities = [
            -1 if t > 0 else 0 for t in range(self.T + 1)
        ]  # final decisions for each time period
        self.distance_to_closest_facility = [
            -1 if t > 0 else 0.0 for t in range(self.T + 1)
        ]  # current distances for existing points
        self.service_costs = [
            -1 if t > 0 else 0.0 for t in range(self.T + 1)
        ]  # final service costs for time periods
        self._is_set = True

    def add_iteration_time(self, time: float) -> None:
        self.iteration_times_ms.append(time * 1000)

    def update_distances_new_facility(
        self, instance: FLOfflineInstance, ell: int
    ) -> None:
        """
        Check if newly built facility ell is closer to any points than their current closest facility, update distances accordingly.
        """
        for i in range(1, self.t_index + 1):
            new_distance = instance.get_distance(i, ell)
            if new_distance < self.distance_to_closest_facility[i]:
                self.distance_to_closest_facility[i] = new_distance

    def update_distances_new_point(self, instance: FLOfflineInstance) -> None:
        """
        Compute closest facility to point, update distances accordingly (only at self.t_index).
        """
        min_distance = sys.float_info.max
        for j in self.facilities:
            if j == -1:
                continue
            new_distance = instance.get_distance(self.t_index, j)
            min_distance = min(min_distance, new_distance)
        self.distance_to_closest_facility[self.t_index] = min_distance

    def update(
        self, instance: FLOfflineInstance, ell: int = None, service_cost: float = None
    ) -> None:
        """
        Update algorithm state. This can be triggered by:
            - A new point self.offline_instance.points[self.t_index] arriving (ell = None, service_cost = None)
            - A new facility (ell) has been built
            - An iteration has ended and we have the final service cost for the time period (previous + new point on previous facility set)
        """
        if ell is not None and service_cost is not None:
            raise ValueError("ell and service_cost cannot be set at the same time.")
        elif ell is not None:
            _LOGGER.log_bodydebug(
                f"Updating state at time {self.t_index}, to add facility {ell}: {instance.points[ell].x}"
            )
            self.facilities[self.t_index] = ell
            self.num_facilities += 1
            self.update_distances_new_facility(instance, ell)
            # The service cost v_t(F_t) is charged to the objective for this time
            # period, but the cumulative variable cost counter is reset to 0 - it only
            # ever accumulates the variable costs incurred since the last facility was
            # built (Algorithm 1, Step 6).
            self.service_costs[self.t_index] = self.compute_service_cost(
                new_facility=True
            )
            self.cum_var_cost = 0.0
            self.t_index += 1
            logging.debug(
                f"Facilities: {self.facilities}, distances: {self.distance_to_closest_facility}."
            )
        elif service_cost is not None:
            logging.debug(
                f"Updating state at time {self.t_index}, for service cost {service_cost}."
            )
            self.facilities[self.t_index] = -1
            self.service_costs[self.t_index] = service_cost
            self.cum_var_cost += service_cost
            self.t_index += 1
            logging.debug(
                f"Facilities: {self.facilities}, distances: {self.distance_to_closest_facility}."
            )
        else:
            logging.debug(
                f"Updating state at time {self.t_index} to add demand point {self.t_index}: {instance.points[self.t_index].x}."
            )
            self.update_distances_new_point(instance)
            _LOGGER.log_bodydebug(
                f"Distances after update: {self.distance_to_closest_facility}"
            )

    def compute_service_cost(self, new_facility: bool = False) -> float:
        """
        Compute service cost for current time period.
        The updated self.distance_to_closest_facility[self.t_index] encodes the correct current facility set.
            - if no facility has been built since the last distance update, self.service_costs[self.t_index - 1] is a lower bound.
            - if a new facility has been built we have to check the whole distance list as the new facility may have better served some point that was the previous max min.
        """
        if new_facility:
            return max(self.distance_to_closest_facility[: self.t_index + 1])
        else:
            return max(
                self.distance_to_closest_facility[self.t_index],
                self.service_costs[self.t_index - 1],
            )

    def bare_final_state(
        self,
        instance: FLOfflineInstance,
        facilities_service_costs: Tuple[List[int], List[float]],
    ) -> None:
        """
        TODO: this stinks of decouple MILP model state from the CVTCA state.
        Set final state.
        Note: facilities_service_costs[0], the facility list, could be a list where:
            - the list is of size T+1, -1 indicates no facility was built at that time, i > 0 indicates facility i was built.
            - the list is just the facilities built, in no particular order (from a static problem)
        All logic in this function works for both options, as we just use the facility list to count the number of facilities built.
        Please keep this invariant if you change the function.
        """
        self.T = instance.id.T
        self.n = instance.id.n
        self.Gamma = instance.Gamma
        self.t_index = 1
        self.facilities = facilities_service_costs[0]
        self.service_costs = facilities_service_costs[1]
        self.distance_to_closest_facility = [
            -1 if i > 0 else 0.0 for i in range(self.T + 1)
        ]
        for x in self.facilities:
            if x != -1:
                self.num_facilities += 1
        while self.t_index < self.T + 1:
            self.update_distances_new_point(instance)
            self.t_index += 1
        self.t_index = self.T
        self._is_set = True

    def set_final_objective(self) -> None:
        """
        Set final objective value.
        """
        _LOGGER.log_debug("Computing final objective..")
        _LOGGER.log_debug("Final service costs", ":")
        # for t, service_cost in enumerate(self.service_costs):
        #    _LOGGER.log_debug(f"t: {t}, service cost: {service_cost}")
        _LOGGER.log_debug(f"Sum of service costs: {sum(self.service_costs)}")
        self.objective = self.Gamma * (self.num_facilities - 1) + sum(
            self.service_costs
        )
        _LOGGER.log_debug(f"Objective: {self.objective}")


class FLSolution(Data):
    """
    Class to hold a solution for the online min max center problem.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n: int = 0
        self.T: int = 0
        self.Gamma: float = 0.0
        self.objective: float = 0.0
        self.objective_horizon: List[float] = []
        self.running_time_s: float = 0.0
        self.running_time_ms: float = 0.0
        self.iteration_times_ms: List[float] = []
        self.optimal: bool = False
        self.unbounded: bool = False
        self.num_facilities: int = 0
        self.facilities: List[int] = []
        self.facilities_str: str = ""
        self.distance_to_closest_facility: List[float] = []
        self.service_costs: List[float] = []
        self.solver: str = ""

    def print(self, instance: FLOfflineInstance, name: str = "Final") -> None:
        _LOGGER.log_subheader(
            f" {name} Solution:  Objective: {self.objective:.2f}, Optimal: {self.optimal}, Running Time: {self.running_time_ms:.2f} ms"
        )
        if self.num_facilities == 1:
            _LOGGER.log_bodydebug("Built no additional facilities")
        elif self.num_facilities == 2:
            _LOGGER.log_bodydebug(
                f"Built 1 facility (in addition to x_0): {self.facilities}"
            )
        else:
            _LOGGER.log_bodydebug(
                f"Built {self.num_facilities - 1} facilities (in addition to x_0): {self.facilities}"
            )
        instance.print_distance_matrix()
        _LOGGER.log_bodydebug(
            f"Final service distances (closest distance to a facility, for all i): {self.distance_to_closest_facility}"
        )
        _LOGGER.log_bodydebug(
            f"Time horizon service costs (max min cost, for all t): {self.service_costs}"
        )

    def set_running_time(self, time: float) -> None:
        self.running_time_s = time
        self.running_time_ms = time * 1000

    def set_optimal(self, optimal: bool) -> None:
        self.optimal = optimal

    def set_unbounded(self, unbounded: bool) -> None:
        self.unbounded = unbounded

    def set_solver(self, solver: str) -> None:
        self.solver = solver

    def facility_to_str(self) -> None:
        """
        Works for both the full time horizon and the final facilities.
        """
        self.facilities_str = "0"
        for facility in self.facilities:
            if facility > 0:
                self.facilities_str += f"-{facility}"

    def from_cvtca(
        self, state: CCTState, time: float, optimal: bool, solver: str
    ) -> None:
        self.n = state.n
        self.T = state.T
        self.Gamma = state.Gamma
        self.objective = state.objective
        self.num_facilities = state.num_facilities
        self.facilities = state.facilities
        self.facility_to_str()
        self.distance_to_closest_facility = state.distance_to_closest_facility
        self.service_costs = state.service_costs
        self.iteration_times_ms = state.iteration_times_ms
        if solver in _ONLINE_SOLVER_NAMES:
            self.average_iteration_time_ms = sum(state.iteration_times_ms) / len(
                state.iteration_times_ms
            )
        else:
            self.average_iteration_time_ms = -1.0
        self.set_running_time(time)
        self.set_optimal(optimal)
        self.set_solver(solver)
        self._is_set = True


def main():
    instance = FLOfflineInstance(INSTANCE_SHAPES["test"])
    instance.set_random()


if __name__ == "__main__":
    main()
