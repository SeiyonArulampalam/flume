from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
import numpy as np
from typing import List
from .linear_solve import LinearStaticSolve


class Compliance(Analysis):
    def __init__(self, obj_name: str, sub_analyses=List[LinearStaticSolve], **kwargs):
        """
        Analysis class that computes the compliance for the beam by evaluating

        c = f.T @ u

        where u is the solution field (removes the Lagrange multipliers from the input d)

        Parameters
        ----------
        obj_name : str
            Name for the analysis object
        sub_analyses : list
            A list of sub-analyses for the object, which nominally contains an instance of the LinearStaticSolve class

        Keyword Arguments
        -----------------
        f : np.ndarray
            Array that specifies the force vector acting on the system. This should be a 1D array of size 2 * (nelems + 1) to account for the forces being applied to each node
        nelems : int
            Integer number that defines the number of elements for the beam model
        """

        # Set the default parameters
        self.default_parameters = {
            "f": np.ones(
                2
            ),  # Force vector acting on the system, which must be 2 * (nelems + 1) in shape
            "nelems": 10,  # number of elements in the system
        }

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default states for the variables
        d_var = State(
            value=np.ones(2 * (self.parameters["nelems"] + 1) + 2),
            desc="Displacement field for the beam, nominally determined from the LinearStaticSolve Analysis, which is augmented with the Lagrange multipliers.",
            source=self,
        )

        self.variables = {"d": d_var}

        return

    def _analyze(self):
        """
        Computes the compliance for the system
        """

        # Extract the variable values
        d = self.variables["d"].value

        # Extract the parameter values
        f = self.parameters["f"]

        # Remove the Lagrange multipliers from the solution field
        u = d[0:-2]

        # Compute the compliance
        c = np.dot(f, u)

        # Store the outputs
        self.outputs = {}

        self.outputs["c"] = State(value=c, desc="Compliance for the beam", source=self)

        return

    def _analyze_adjoint(self):
        """
        Perform the adjoint analysis to propagate the derivatives back to the solution field, d.
        """

        # Extract the derivative values
        cb = self.outputs["c"].deriv
        db = self.variables["d"].deriv

        # Extract the force array
        f = self.parameters["f"]

        # Compute the contributions to db
        db[0:-2] += cb * f

        # Assign the derivative values
        self.variables["d"].set_deriv_value(db)

        return


if __name__ == "__main__":

    # Set the number of elements and force data
    nelems = 2
    nnodes = 3
    f = np.random.uniform(size=2 * nnodes)

    # Construct the compliance object
    compliance = Compliance(obj_name="compliance", sub_analyses=[], f=f, nelems=nelems)

    # Test the isolated adjoint
    compliance.declare_design_vars(variables=["d"])
    compliance.test_isolated_adjoint(method="cs")
