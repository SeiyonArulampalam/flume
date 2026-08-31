# Pin the underlying BLAS/LAPACK libraries to a single thread *before* NumPy is
# imported. This ensures each branch's linear solve runs single-threaded, so the
# observed speedup comes from Flume's DAG scheduler running independent branches
# concurrently (not from BLAS internally multi-threading a single solve). For the
# cleanest measurement, run this file on its own:
#     python -m unittest tests.test_parallel_speedup
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS Accelerate/vecLib
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import multiprocessing
import unittest
from typing import List

import numpy as np

from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
from flume.base_classes.system import System

# ---------------------------------------------------------------------------
# Demonstrator analysis classes.
#
# DAG structure (fan-out -> fan-in):
#
#     indeps ─▶ branch_0 ─┐
#        │  ─▶ branch_1 ─┤
#        │       ...      ├─▶ objective (sum)
#        │  ─▶ branch_N-1 ┘
#
# The N HeavyBranch nodes all depend only on `indeps`, so they are mutually
# independent and can execute concurrently. Each branch performs a large dense
# linear solve in both its forward and adjoint passes; np.linalg.solve calls into
# LAPACK, which releases the GIL, so the branches genuinely run in parallel. The
# objective (a cheap sum) fans them back in.
#
# Because the branches are independent in both the forward DAG and its transpose,
# BOTH the forward pass (System.execute) and the adjoint pass
# (System.execute_adjoint) parallelize across the branches.
# ---------------------------------------------------------------------------


class Independents(Analysis):
    """Source node that holds the design-variable vector x and passes it through."""

    def __init__(self, obj_name: str, sub_analyses=list, **kwargs):
        self.default_parameters = {"n": 10}
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        n = self.parameters["n"]
        self.variables = {
            "x": State(value=np.ones(n), desc="design variables", source=self)
        }
        return

    def _analyze(self):
        x = self.variables["x"].value
        self.outputs = {
            "x": State(
                value=np.array(x, dtype=float), desc="x passthrough", source=self
            )
        }
        return

    def _analyze_adjoint(self):
        # Pass-through: accumulate the output derivative into the variable derivative
        xb_out = self.outputs["x"].deriv
        xb = self.variables["x"].deriv
        self.variables["x"].set_deriv_value(xb + xb_out)
        return


class HeavyBranch(Analysis):
    """
    An independent, expensive branch. Computes the scalar

        y_i = x^T A_i^{-1} x

    via a dense LAPACK solve (GIL-releasing). Its adjoint performs a second dense
    solve, so both the forward and adjoint passes carry substantial parallelizable
    work.
    """

    def __init__(self, obj_name: str, sub_analyses=List[Independents], **kwargs):
        self.default_parameters = {"n": 10, "idx": 0}
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        n = self.parameters["n"]
        idx = self.parameters["idx"]

        # Fixed, well-conditioned matrix unique to this branch
        rng = np.random.default_rng(1000 + idx)
        self.A = rng.standard_normal((n, n)) + n * np.eye(n)

        # Output name is unique per branch so the objective can connect to each
        self._out_name = f"y{idx}"

        self.variables = {
            "x": State(value=np.ones(n), desc="branch input", source=self)
        }
        return

    def _analyze(self):
        x = self.variables["x"].value

        # Heavy, GIL-releasing dense solve
        z = np.linalg.solve(self.A, x)
        self._z = z

        # Scalar contribution y_i = x^T A_i^{-1} x
        out = float(x @ z)
        self.outputs = {
            self._out_name: State(value=out, desc="branch output", source=self)
        }
        return

    def _analyze_adjoint(self):
        yb = self.outputs[self._out_name].deriv  # scalar seed
        x = self.variables["x"].value

        # d(x^T A^{-1} x)/dx = (A^{-1} + A^{-T}) x = z + solve(A^T, x)
        grad = self._z + np.linalg.solve(self.A.T, x)  # second heavy solve

        xb = self.variables["x"].deriv
        self.variables["x"].set_deriv_value(xb + yb * grad)
        return


class SumObjective(Analysis):
    """Cheap fan-in node that sums the branch outputs into a single objective."""

    def __init__(self, obj_name: str, sub_analyses=list, **kwargs):
        self.default_parameters = {"n_branches": 1}
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        self._n = self.parameters["n_branches"]
        self.variables = {
            f"y{i}": State(value=1.0, desc=f"branch {i} value", source=self)
            for i in range(self._n)
        }
        return

    def _analyze(self):
        total = 0.0
        for i in range(self._n):
            total += self.variables[f"y{i}"].value
        self.outputs = {"J": State(value=float(total), desc="objective", source=self)}
        return

    def _analyze_adjoint(self):
        Jb = self.outputs["J"].deriv
        for i in range(self._n):
            yb = self.variables[f"y{i}"].deriv
            self.variables[f"y{i}"].set_deriv_value(yb + Jb)
        return


class TestParallelSpeedup(unittest.TestCase):
    """
    Verifies that parallel execution of the System's forward and adjoint passes is
    faster than serial execution for a DAG with independent, GIL-releasing branches
    (representative of a multidisciplinary problem with independent, expensive
    disciplines). Both the forward pass (execute) and the adjoint pass
    (execute_adjoint) are timed and required to be faster in parallel.
    """

    # Problem size and timing configuration
    N_BRANCHES = 8
    MAT_SIZE = 600
    N_REPS = 5

    def _build_system(self, parallel, tag):
        """Build an independent System (fresh analysis objects) in serial or parallel mode."""
        n = self.MAT_SIZE
        n_branches = self.N_BRANCHES

        indeps = Independents(obj_name="indeps", sub_analyses=[], n=n)

        branches = [
            HeavyBranch(obj_name=f"branch{i}", sub_analyses=[indeps], n=n, idx=i)
            for i in range(n_branches)
        ]

        objective = SumObjective(
            obj_name="obj", sub_analyses=branches, n_branches=n_branches
        )

        sys = System(
            sys_name="parallel_speedup",
            top_level_analysis_list=[objective],
            log_name="flume.log",
            log_prefix=f"tests/parallel_speedup_{tag}",
            parallel_execution=parallel,
            parallel_max_workers=max(2, self.N_BRANCHES),
        )

        sys.graph_network(filename="speedup_test_graph", output_directory="tests")

        sys.declare_design_vars(global_var_name={"indeps.x": {}})
        sys.declare_objective(global_obj_name="obj.J")

        return sys, indeps

    def _time_passes(self, sys, indeps, x0):
        """
        Time the forward and adjoint passes separately, averaged over N_REPS. A
        warm-up iteration (untimed) primes the DAG construction and caches.
        Returns (avg_forward_time, avg_adjoint_time, obj_value, design_derivs).
        """
        indeps.set_var_values(variables={"x": x0})

        # Warm-up (build DAG, prime caches) -- not timed
        sys.reset_analysis_flags()
        sys.execute()
        sys.execute_adjoint()

        fwd_total = 0.0
        adj_total = 0.0
        for _ in range(self.N_REPS):
            # Re-set the design variable each rep, mirroring how an optimizer drives the
            # System (a new design point per iteration via set_var_values)
            indeps.set_var_values(variables={"x": x0})
            sys.reset_analysis_flags()

            t0 = time.perf_counter()
            sys.execute()
            t1 = time.perf_counter()
            sys.execute_adjoint()
            t2 = time.perf_counter()

            fwd_total += t1 - t0
            adj_total += t2 - t1

        obj_val = sys.obj_analysis.outputs["J"].value
        return (
            fwd_total / self.N_REPS,
            adj_total / self.N_REPS,
            obj_val,
            sys.design_derivs,
        )

    def test_forward_and_adjoint_speedup(self):
        n_cores = multiprocessing.cpu_count()

        # Fixed design point (same for both systems)
        x0 = np.random.default_rng(0).uniform(0.5, 1.5, size=self.MAT_SIZE)

        # Build independent serial and parallel systems
        serial_sys, serial_indeps = self._build_system(parallel=False, tag="serial")
        parallel_sys, parallel_indeps = self._build_system(
            parallel=True, tag="parallel"
        )

        # Time both
        s_fwd, s_adj, s_obj, s_dd = self._time_passes(serial_sys, serial_indeps, x0)
        p_fwd, p_adj, p_obj, p_dd = self._time_passes(parallel_sys, parallel_indeps, x0)

        fwd_speedup = s_fwd / p_fwd
        adj_speedup = s_adj / p_adj

        # Report
        print(
            f"\n[parallel speedup] cores={n_cores}, branches={self.N_BRANCHES}, "
            f"mat_size={self.MAT_SIZE}, reps={self.N_REPS}"
        )
        print(
            f"  forward: serial={s_fwd*1e3:8.3f} ms  parallel={p_fwd*1e3:8.3f} ms  "
            f"speedup={fwd_speedup:5.2f}x"
        )
        print(
            f"  adjoint: serial={s_adj*1e3:8.3f} ms  parallel={p_adj*1e3:8.3f} ms  "
            f"speedup={adj_speedup:5.2f}x"
        )

        # --- Correctness: parallel must match serial exactly ---
        self.assertAlmostEqual(
            float(s_obj),
            float(p_obj),
            places=10,
            msg="Parallel objective value does not match serial.",
        )
        obj_sweep_id = (serial_sys.global_obj_name, 0)
        np.testing.assert_allclose(
            p_dd[obj_sweep_id]["indeps.x"],
            s_dd[obj_sweep_id]["indeps.x"],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Parallel objective gradient does not match serial.",
        )

        # --- Speed: parallel must be faster for both passes ---
        # A speedup assertion is only meaningful with multiple physical cores.
        if n_cores < 2:
            self.skipTest(
                f"Only {n_cores} core available; a parallel speedup cannot be demonstrated."
            )

        with self.subTest("forward pass faster in parallel"):
            self.assertGreater(
                fwd_speedup,
                1.1,
                msg=(
                    f"Forward pass was not faster in parallel "
                    f"(serial={s_fwd*1e3:.3f} ms, parallel={p_fwd*1e3:.3f} ms, "
                    f"speedup={fwd_speedup:.2f}x)."
                ),
            )

        with self.subTest("adjoint pass faster in parallel"):
            self.assertGreater(
                adj_speedup,
                1.1,
                msg=(
                    f"Adjoint pass was not faster in parallel "
                    f"(serial={s_adj*1e3:.3f} ms, parallel={p_adj*1e3:.3f} ms, "
                    f"speedup={adj_speedup:.2f}x)."
                ),
            )

        return


if __name__ == "__main__":
    unittest.main()
