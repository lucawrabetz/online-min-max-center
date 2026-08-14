import sys
import time
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Tuple, Dict
from log_config import gurobi_log_file, _LOGGER
from util import append_date, _EPSILON
from allowed_types import FLSolverType, _OMIP, _SOMIP, _CCTA, _BEA, _NKCA, _SOLVERS
from problem import FLOfflineInstance, FLSolution, CCTState
from gurobipy import Model, GRB, quicksum

GRBVar = Any
_GUROBITIMELIMIT_S = 3600


class IFLSolver(ABC):
    def __init__(self, solver_id: FLSolverType) -> None:
        self.T: int = 0
        self.objective: float = 0.0
        self.running_time_s: float = 0.0
        self.optimal: bool = False
        self.id: FLSolverType = solver_id
        pass

    @abstractmethod
    def configure_solver(self, instance: FLOfflineInstance) -> None:
        pass

    # TODO: add a clean timing decorator to this.
    @abstractmethod
    def solve(self, instance: FLOfflineInstance) -> FLSolution:
        pass


class IFLMIP(IFLSolver):
    def __init__(self, solver_id: FLSolverType) -> None:
        super().__init__(solver_id)

    def init_mip(self) -> None:
        self.model: Model = Model(self.id.name)
        self.model.setParam("LogToConsole", 0)
        self.model.setParam("LogFile", gurobi_log_file())
        self.model.setParam("TimeLimit", _GUROBITIMELIMIT_S)

    @abstractmethod
    def add_variables(self, instance: FLOfflineInstance) -> None:
        pass

    @abstractmethod
    def add_constraints(self, instance: FLOfflineInstance) -> None:
        pass

    @abstractmethod
    def add_objective_function(self, instance: FLOfflineInstance) -> None:
        pass

    @abstractmethod
    def final_facilities_service_costs(self) -> Tuple[List[int], List[float]]:
        pass

    def configure_solver(self, instance: FLOfflineInstance) -> None:
        if not instance.is_set:
            raise ValueError("Instance must be set before configuring solver.")
        self.init_mip()
        self.T = instance.id.T
        self.add_variables(instance)
        self.add_constraints(instance)
        self.add_objective_function(instance)

    def solve(self, instance: FLOfflineInstance) -> FLSolution:
        """Solve model."""
        start = time.time()
        self.model.optimize()
        if self.model.status == GRB.TIME_LIMIT:
            self.running_time_s = _GUROBITIMELIMIT_S
        else:
            self.running_time_s = time.time() - start
        self.optimal = self.model.status == GRB.OPTIMAL
        solution = FLSolution()
        state = CCTState()
        state.bare_final_state(instance, self.final_facilities_service_costs())
        state.set_final_objective()
        if state.objective - self.model.objVal > _EPSILON:
            raise ValueError(
                f"Objective mismatch: {state.objective} vs {self.model.objVal}"
            )
        solution.from_cvtca(state, self.running_time_s, self.optimal, self.id.name)
        return solution

    def write_model(self) -> None:
        """Write model to file."""
        filename = f"{append_date(self.id.name)} + '.lp'"
        self.model.write(filename)


class OfflineMIP(IFLMIP):
    def __init__(self) -> None:
        super().__init__(_OMIP)

    def add_variables(self, instance: FLOfflineInstance) -> None:
        self.y = [
            self.model.addVar(vtype=GRB.BINARY, name=f"y^{j}")
            for j in range(1, self.T + 1)
        ]
        self.z = [
            [
                self.model.addVar(vtype=GRB.BINARY, name=f"z^{i}_{j}")
                for j in range(0, self.T + 1)
            ]
            for i in range(1, self.T + 1)
        ]
        self.zeta = [
            self.model.addVar(vtype=GRB.CONTINUOUS, name=f"zeta^{t}")
            for t in range(1, self.T + 1)
        ]
        self.model.update()

    def add_zetalowerbound_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                self.model.addConstr(
                    self.zeta[t_index]
                    >= quicksum(
                        instance.get_distance(i, j) * self.z[i_index][j]
                        for j in range(self.T + 1)
                    ),
                    name=f"C_zeta_lb_{t}_{i}",
                )
        self.model.update()

    def add_linkzy_constraints(self, instance: FLOfflineInstance) -> None:
        for i in range(1, self.T + 1):
            i_index = i - 1
            for j in range(1, self.T + 1):
                # logging.debug(
                #     f"Adding linkzy constraint for i = {i}, j = {j}. i_index = {i_index}."
                # )
                # logging.debug(f"y: {self.y}, z: {self.z}")
                self.model.addConstr(
                    self.z[i_index][j] <= self.y[j - 1],
                    name=f"C_linkzy_{i}_{j}",
                )
        self.model.update()

    def add_oneassignment_constraints(self, instance: FLOfflineInstance) -> None:
        for i in range(1, self.T + 1):
            i_index = i - 1
            self.model.addConstr(
                quicksum(self.z[i_index][j] for j in range(self.T + 1)) == 1,
                name=f"C_oneassignment_{i}",
            )
        self.model.update()

    def add_constraints(self, instance: FLOfflineInstance) -> None:
        self.add_zetalowerbound_constraints(instance)
        self.add_linkzy_constraints(instance)
        self.add_oneassignment_constraints(instance)

    def add_objective_function(self, instance: FLOfflineInstance) -> None:
        self.model.setObjective(
            quicksum(self.y[j] for j in range(self.T)) * instance.Gamma
            + quicksum(self.zeta[t] for t in range(self.T)),
            GRB.MINIMIZE,
        )
        self.model.update()

    def final_facilities_service_costs(self) -> Tuple[List[int], List[float]]:
        built = [False for _ in range(self.T)]
        facilities = [0]  # static facility set, length depends on facilities built
        service_costs = [
            0.0
        ]  # from 0, ..., T, index t represents the service cost at time t
        for t in range(1, self.T + 1):
            t_index = t - 1
            service_costs.append(self.zeta[t_index].X)

        for j in range(1, t + 1):
            j_index = j - 1
            if self.y[j_index].X > 0.5:
                facilities.append(j)

        return (facilities, service_costs)


class SemiOfflineMIP(IFLMIP):
    def __init__(self) -> None:
        """Constructs blank gurobi model."""
        super().__init__(_SOMIP)

    def add_variables(self, instance: FLOfflineInstance) -> None:
        self.y = [
            [
                self.model.addVar(vtype=GRB.BINARY, name=f"y^{t}_{j}")
                for j in range(1, t + 1)
            ]
            for t in range(1, self.T + 1)
        ]
        self.z = [
            [
                [
                    self.model.addVar(vtype=GRB.BINARY, name=f"z^{t}_{i}_{j}")
                    for j in range(0, t + 1)
                ]
                for i in range(1, t + 1)
            ]
            for t in range(1, self.T + 1)
        ]
        self.zeta = [
            self.model.addVar(vtype=GRB.CONTINUOUS, name=f"zeta^{t}")
            for t in range(1, self.T + 1)
        ]
        self.model.update()

    def add_zetalowerbound_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                self.model.addConstr(
                    self.zeta[t_index]
                    >= quicksum(
                        instance.get_distance(i, j) * self.z[t_index][i_index][j]
                        for j in range(t + 1)
                    ),
                    name=f"C_zeta_lb_{t}_{i}",
                )
        self.model.update()

    def add_linkzy_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                for j in range(1, t + 1):
                    # logging.debug(f"Adding constraint for t = {t}, i = {i}, j = {j}")
                    self.model.addConstr(
                        self.z[t_index][i_index][j] <= self.y[t_index][j - 1],
                        name=f"C_linkzy_{t}_{i}_{j}",
                    )
        self.model.update()

    def add_assignment_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T + 1):
            t_index = t - 1
            for i in range(1, t + 1):
                i_index = i - 1
                self.model.addConstr(
                    quicksum(self.z[t_index][i_index][j] for j in range(t + 1)) == 1,
                    name=f"C_assignment_{t}_{i}",
                )
        self.model.update()

    def add_nondecreasingy_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T):
            t_index = t - 1
            for j in range(1, t + 1):
                j_index = j - 1
                self.model.addConstr(
                    self.y[t_index][j_index] <= self.y[t_index + 1][j_index],
                    name=f"C_nondecreasingy_{t}_{j}",
                )
        self.model.update()

    def add_atmostonefacility_constraints(self, instance: FLOfflineInstance) -> None:
        for t in range(1, self.T):
            t_index = t - 1
            self.model.addConstr(
                quicksum(self.y[t_index][j] for j in range(t)) + 1
                >= quicksum(self.y[t_index + 1][j] for j in range(t + 1)),
                name=f"C_atmostonefacility_{t}",
            )
        self.model.update()

    def add_constraints(self, instance: FLOfflineInstance) -> None:
        self.add_zetalowerbound_constraints(instance)
        self.add_linkzy_constraints(instance)
        self.add_assignment_constraints(instance)
        self.add_nondecreasingy_constraints(instance)
        # TODO: ask Jourdain about removing the following constraints.
        # self.add_atmostonefacility_constraints(instance)
        self.model.update()

    def add_objective_function(self, instance: FLOfflineInstance) -> None:
        self.model.setObjective(
            quicksum(self.y[self.T - 1][j] for j in range(self.T)) * instance.Gamma
            + quicksum(self.zeta[t] for t in range(self.T)),
            GRB.MINIMIZE,
        )
        self.model.update()

    def final_facilities_service_costs(self) -> Tuple[List[int], List[float]]:
        built = [False for _ in range(self.T)]
        facilities = [
            0
        ]  # from 0, ..., T, index t represents what we built at time t, -1 for nothing
        service_costs = [
            0.0
        ]  # from 0, ..., T, index t represents the service cost at time t
        for t in range(1, self.T + 1):
            facility = -1
            t_index = t - 1
            service_costs.append(self.zeta[t_index].X)
            for j in range(1, t + 1):
                j_index = j - 1
                if built[j_index]:
                    continue
                if self.y[t_index][j_index].X > 0.5:
                    built[j_index] = True
                    facility = j
                    break
            facilities.append(facility)
        return (facilities, service_costs)


class CCTAlgorithm(IFLSolver):
    def __init__(self, solver_id: FLSolverType = _CCTA) -> None:
        super().__init__(solver_id)

    def configure_solver(self, instance: FLOfflineInstance) -> None:
        if not instance.is_set:
            raise ValueError("Instance must be set before configuring solver.")
        self.state: CCTState = CCTState()
        self.offline_instance = instance
        self.T = instance.id.T
        self.Gamma = instance.Gamma
        self.state.configure_state(instance)

    def facility_selection(self) -> int:
        """
        Step 4 of Algorithm 1: ell in argmax_{i in [t]} min_{j in F_{t-1}} d(x_i, x_j).
        Distances updated with new point on previous facility set before calling.
        """
        ell = -1
        # Note: not sys.float_info.min, which is the smallest positive normal float -
        # all distances can be 0 (duplicate points), and ell must still be set.
        max_distance = -1.0
        for i in range(1, self.state.t_index + 1):
            if self.state.distance_to_closest_facility[i] > max_distance:
                ell = i
                max_distance = self.state.distance_to_closest_facility[i]
        return ell

    def build_condition(self, nobuild_service_cost: float) -> bool:
        """
        Step 3 of Algorithm 1: cumVarCost + v_t(F_{t-1}) >= Gamma.
        """
        return self.state.cum_var_cost + nobuild_service_cost >= self.Gamma

    def add_facility(self) -> None:
        ell = self.facility_selection()
        self.state.update(self.offline_instance, ell)

    def no_facility_update(self, service_cost: float) -> None:
        self.state.update(self.offline_instance, service_cost=service_cost)

    def single_iteration(self) -> None:
        # time.time() returns time in seconds since epoch
        start = time.time()
        _LOGGER.log_bodydebug(f"Starting iteration at time {self.state.t_index}")
        self.state.update(self.offline_instance)
        nobuild_service_cost = self.state.compute_service_cost()
        _LOGGER.log_bodydebug(
            f"(cumVarCost: {self.state.cum_var_cost}) + (no build service cost: {nobuild_service_cost}) = {self.state.cum_var_cost + nobuild_service_cost}, gamma: {self.Gamma}"
        )
        if self.build_condition(nobuild_service_cost):
            self.add_facility()
        else:
            self.no_facility_update(nobuild_service_cost)
        self.state.add_iteration_time(time.time() - start)

    def solve(self, instance: FLOfflineInstance) -> FLSolution:
        if not self.state.is_set:
            raise ValueError("Solver must be configured before solving.")
        # time.time() returns time in seconds since epoch
        start = time.time()
        while self.state.t_index <= self.T:
            it_start = time.time()
            self.single_iteration()
            _LOGGER.log_bodydebug(
                f"Iteration time (ms): {(time.time() - it_start) * 1000}"
            )
        self.running_time_s = time.time() - start
        self.state.set_final_objective()
        self.optimal = False
        solution = FLSolution()
        solution.from_cvtca(self.state, self.running_time_s, self.optimal, self.id.name)
        return solution


class BEAlgorithm(CCTAlgorithm):
    """
    Break-even algorithm (Algorithm 2). Identical to CCT except that the new facility is
    always located at the point that just arrived, rather than at the point farthest
    from its nearest facility.
    """

    def __init__(self) -> None:
        super().__init__(_BEA)

    def facility_selection(self) -> int:
        """Step 4 of Algorithm 2: ell <- t."""
        return self.state.t_index


class NKCAlgorithm(CCTAlgorithm):
    """
    Naive k-center algorithm (Algorithm 3). Identical to CCT except that the build
    decision ignores the cumulative variable cost counter, thresholding on the variable
    cost of the current time period alone.
    """

    def __init__(self) -> None:
        super().__init__(_NKCA)

    def build_condition(self, nobuild_service_cost: float) -> bool:
        """Step 2 of Algorithm 3: v_t(F_{t-1}) >= Gamma."""
        return nobuild_service_cost >= self.Gamma


class SolverTypeRegistry:
    def __init__(self, registry: Dict[FLSolverType, Any] = {}) -> None:
        self.solver_registry = registry

    def solver(self, solver_id: FLSolverType) -> IFLSolver:
        return self.solver_registry[solver_id]()


_SOLVER_FACTORY = SolverTypeRegistry(
    {
        _OMIP: OfflineMIP,
        _SOMIP: SemiOfflineMIP,
        _CCTA: CCTAlgorithm,
        _BEA: BEAlgorithm,
        _NKCA: NKCAlgorithm,
    }
)
