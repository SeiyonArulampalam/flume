from flume.base_classes.system import System
from flume.interfaces.pyoptsparse_interface import FlumePyOptSparseInterface
from icecream import ic
import numpy as np
import unittest
from examples.rosenbrock.rosenbrock_problem_classes import (
    Rosenbrock,
    RosenbrockConstraint,
    RosenbrockDVs,
)
from examples.thomson_problem.thomson_problem_classes import (
    ParticlePositions,
    ParticleConstraints,
    PotentialEnergy,
)
from math import pi


class TestUnconstrainedRosenbrock(unittest.TestCase):
    """
    Tests the implementation of the optimization of the unconstrained Rosenbrock function using the FlumePyOptSparseInterface.
    """

    def setUp(self):
        # Construct the analysis object for the Rosenbrock function
        a = 1.0
        b = 100.0

        rosenbrock = Rosenbrock(obj_name="rosenbrock", sub_analyses=[], a=a, b=b)

        rosenbrock.set_var_values(variables={"x": 0.0, "y": 0.0})

        # Construct the system
        sys = System(
            sys_name="rosen_sys",
            top_level_analysis_list=[rosenbrock],
            log_name="flume.log",
            log_prefix="tests/rosenbrock_unconstrained_pyoptsparse",
        )

        # Declare the design variables for the system
        sys.declare_design_vars(
            global_var_name={
                "rosenbrock.x": {"lb": -2.0, "ub": 2.0},
                "rosenbrock.y": {"lb": -2.0, "ub": 2.0},
            }
        )

        sys.declare_objective(global_obj_name="rosenbrock.f")

        # Save the system as an attribute
        self.flume_sys = sys

        return

    def test_optimization_pyoptsparse_slsqp(self):
        """
        Tests the implementation of the SciPy interface for the unconstrained Rosenbrock problem.
        """

        # Construct the interface between Flume and pyOptSparse
        interface = FlumePyOptSparseInterface(flume_sys=self.flume_sys)

        # Set a random starting point
        x0 = np.random.uniform(low=-5.0, high=5.0, size=2)

        # Construct the initial design point dictionary
        x0dict = {"rosenbrock.x": x0[0].item(), "rosenbrock.y": x0[1].item()}

        # Optimize the problem with pyOptSparse
        sol = interface.optimize_system(
            x0dict=x0dict, opt_prob_name="Unconstrained_Rosenbrock", optimizer="SLSQP"
        )

        # Extract the final design point from the Solution object
        x = np.concatenate((sol.xStar["rosenbrock.x"], sol.xStar["rosenbrock.y"]))

        # Set the expected optimal values
        xstar = np.array([1.0, 1.0])

        # Check that the values match
        np.testing.assert_allclose(
            actual=x,
            desired=xstar,
            rtol=1e-3,
            err_msg="The computed optimal values do not match the expected solution for the constrained Rosenbrock function.",
            verbose=True,
        )

        return


class TestConstrainedRosenbrock(unittest.TestCase):
    """
    Tests the implementation of the optimization of the constrained Rosenbrock function using the FlumePyOptSparseInterface.
    """

    def setUp(self):

        # Construct the design variables object
        rosenbrock_dvs = RosenbrockDVs(obj_name="dvs", sub_analyses=[])

        # Construct the analysis object for the Rosenbrock function
        a = 1.0
        b = 100.0

        rosenbrock = Rosenbrock(
            obj_name="rosenbrock", sub_analyses=[rosenbrock_dvs], a=a, b=b
        )

        # Construct the analysis object for the constraint on the design variables
        rosenbrock_con = RosenbrockConstraint(
            obj_name="con", sub_analyses=[rosenbrock_dvs]
        )

        # Construct the system
        sys = System(
            sys_name="rosen_sys_con",
            top_level_analysis_list=[rosenbrock, rosenbrock_con],
            log_name="flume.log",
            log_prefix="tests/rosenbrock_constrained_pyoptsparse",
        )

        # Declare the design variables for the system
        sys.declare_design_vars(
            global_var_name={
                "dvs.x_dv": {"lb": -1.5, "ub": 1.5},
                "dvs.y_dv": {"lb": -1.5, "ub": 1.5},
            }
        )

        # Declare the objective
        sys.declare_objective(global_obj_name="rosenbrock.f")

        # Declare the constraint
        sys.declare_constraints(
            global_con_name={"con.g": {"direction": "geq", "rhs": -2.0}}
        )

        # Store the system as an attribute
        self.flume_sys = sys

        return

    def test_optimization_pyoptsparse_slsqp(self):
        """
        Tests that the solution for the constrained Rosenbrock optimization matches the expected solution.
        """

        # Construct the Scipy interface
        interface = FlumePyOptSparseInterface(flume_sys=self.flume_sys)

        # Set a random starting point
        x0 = np.random.uniform(low=-5.0, high=5.0, size=2)

        # Construct the initial design point dictionary
        x0dict = {"dvs.x_dv": x0[0].item(), "dvs.y_dv": x0[1].item()}

        # Optimize the problem with pyOptSparse
        sol = interface.optimize_system(
            x0dict=x0dict, opt_prob_name="Constrained_Rosenbrock", optimizer="SLSQP"
        )

        # Extract the final design point from the Solution object
        x = np.concatenate((sol.xStar["dvs.x_dv"], sol.xStar["dvs.y_dv"]))

        # Set the expected optimal values
        xstar = np.array([1.0, 1.0])

        # Check that the values match
        np.testing.assert_allclose(
            actual=x,
            desired=xstar,
            rtol=1e-3,
            err_msg="The computed optimal values do not match the expected solution for the constrained Rosenbrock function.",
            verbose=True,
        )

        return


class TestThomsonProblem(unittest.TestCase):
    """
    Tests the optimization of the Thomson problem using the FlumeSciPy interface. Here, the check is that the objective function value at the optimized point matches the value for the known, exact solutions within a relative error tolerance of 1e-3.
    """

    def construct_system(self, n_p):

        # Construct the analysis objects for the system
        positions = ParticlePositions(obj_name="positions", sub_analyses=[], n_p=n_p)

        energy = PotentialEnergy(obj_name="energy", sub_analyses=[positions], n_p=n_p)

        cons = ParticleConstraints(obj_name="cons", sub_analyses=[positions], n_p=n_p)

        # Construct the system
        sys = System(
            sys_name="thomson_problem",
            top_level_analysis_list=[energy, cons],
            log_name=f"flume_{n_p}.log",
            log_prefix="tests/thomson_problem_pyoptsparse",
        )
        self.sys = sys

        # Declare the design variables for the system
        sys.declare_design_vars(
            global_var_name={
                "positions.theta": {"lb": -pi, "ub": pi},
                "positions.phi": {"lb": 0.0, "ub": 2 * pi},
            }
        )

        # Declare the objective
        sys.declare_objective(global_obj_name="energy.f", obj_scale=1e-1)

        # Declare the constraints
        sys.declare_constraints(
            global_con_name={"cons.c": {"direction": "both", "rhs": 0.0}}
        )

        return sys

    def optimize_system(self, n_p: int):
        """
        Using the number of particles provided with n_p, optimizes the system using the FlumeScipyInterface and returns the objective function value.
        """

        flume_sys = self.construct_system(n_p=n_p)

        # Construct the Scipy interface
        interface = FlumePyOptSparseInterface(flume_sys=flume_sys)

        # Set random positions for x, y, z to start
        theta0 = np.random.uniform(size=n_p)
        phi0 = np.random.uniform(size=n_p)

        # Set the initial guess using theta0 and phi0
        x0dict = {"positions.theta": theta0, "positions.phi": phi0}

        # Optimize the system
        try:
            from pyoptsparse import SNOPT

            optimizer = "SNOPT"
        except:
            optimizer = "PSQP"

        sol = interface.optimize_system(
            x0dict=x0dict,
            opt_prob_name=f"ThomsonProblem_{n_p}_np",
            optimizer=optimizer,
            history_filename="history.hst",
        )

        # Check that the potential energy at the final point matches the expected value
        obj_val = sol.fStar / self.sys.obj_scale

        return obj_val

    def test_optimization_2np(self):
        """
        Tests that the solution for the thomson problem with N particles matches the expected solution.
        """

        # Construct the system
        n_p = 2

        # Optimize the system and get the objective value
        obj_val = self.optimize_system(n_p=n_p)
        obj_star = 0.5

        # Compute the relative error
        rel_error = abs(obj_val - obj_star) / obj_star

        # Perform the check
        self.assertLessEqual(
            a=rel_error,
            b=1e-3,
            msg=f"The optimal value of the objective function does not match the expected value for {n_p} particles within the relative error tolerance 1e-3.",
        )

        return

    def test_optimization_3np(self):
        """
        Tests that the solution for the thomson problem with N particles matches the expected solution.
        """

        # Construct the system
        n_p = 3

        # Optimize the system and get the objective value
        obj_val = self.optimize_system(n_p=n_p)
        obj_star = 1.732050808

        # Compute the relative error
        rel_error = abs(obj_val - obj_star) / obj_star

        # Perform the check
        self.assertLessEqual(
            a=rel_error,
            b=1e-3,
            msg=f"The optimal value of the objective function does not match the expected value for {n_p} particles within the relative error tolerance 1e-3.",
        )

        return

    def test_optimization_12np(self):
        """
        Tests that the solution for the thomson problem with N particles matches the expected solution.
        """

        # Construct the system
        n_p = 12

        # Optimize the system and get the objective value
        obj_val = self.optimize_system(n_p=n_p)
        obj_star = 49.165253058

        # Compute the relative error
        rel_error = abs(obj_val - obj_star) / obj_star

        # Perform the check
        self.assertLessEqual(
            a=rel_error,
            b=5e-3,
            msg=f"The optimal value of the objective function does not match the expected value for {n_p} particles within the relative error tolerance 5e-3.",
        )

        return
