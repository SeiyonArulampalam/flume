from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
import numpy as np
from .inertia import MomentofInertia
from typing import List
from icecream import ic
import matplotlib.pyplot as plt
from scipy.sparse.linalg import splu


class LinearStaticSolve(Analysis):
    def __init__(self, obj_name: str, sub_analyses=List[MomentofInertia], **kwargs):
        """
        Analysis class that computes the displacement field by solving the lienar system. Here, the stiffness matrix is augmented with Lagrange multipliers to account for boundary conditions at the clamped end.

        K @ d = f

        Parameters
        ----------
        obj_name : str
            Name for the analysis object
        sub_analyses : list
            A list of sub-analyses for the object, which nominally contains an instance of the MomentofInertia class

        Keyword Arguments
        -----------------
        nelems : int
            Integer number that defines the number of elements for the beam model
        E : float
            Young's modulus
        L : float
            Length of the beam
        f : np.ndarray
            Array that specifies the force vector acting on the system. This should be a 1D array of size 2 * (nelems + 1) to account for the forces being applied to each node
        """

        # Set the default parameters
        self.default_parameters = {
            "nelems": 10,  # number of elements in the model
            "E": 1.0,  # Young's modulus
            "L": 1.0,  # Length of the beam
            "f": np.ones(
                2
            ),  # Force vector acting on the system, which must be 2 * (nelems + 1) in shape
        }

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default states for the variables
        I_var = State(
            value=np.ones(self.parameters["nelems"]),
            desc="Moment of inertia values at each element in the beam",
            source=self,
        )

        self.variables = {"I": I_var}

        return

    def _analyze(self):
        """
        Analysis class that solves the linear system K @ u = f for the displacement field.
        """

        # Extract the variable values
        I = self.variables["I"].value

        # Extract the parameter values
        nelems = self.parameters["nelems"]
        E = self.parameters["E"]
        L = self.parameters["L"]
        f = self.parameters["f"]

        # Assemble the stiffness matrix for the system
        K = self._assemble_stiffness_matrix(I, E, L, nelems)
        self.K = K

        # Perform the LU decomposition for the stiffness matrix
        self.lu = splu(K)

        # Solve the linear system for the displacement field
        self.f = np.concatenate((f, np.zeros(2)))
        d = self.lu.solve(self.f)
        self.d = d

        # Store the outputs
        self.outputs = {}

        self.outputs["d"] = State(
            value=d,
            desc="Displacement field for the beam, augmented with the Lagrange multipliers that address the boundary conditions (final two entries)",
            source=self,
        )

        return

    def _assemble_stiffness_matrix(self, I, E, L, nelems):
        """
        Assembles the global stiffness matrix for the system using the values for I, E, L, and nelems.
        """

        # Compute the length of each element, assumed constant
        Le = L / nelems

        # Compute the local element stiffness matrix without the moment of inertia contribution
        K0 = np.array(
            [
                [12.0, -6.0, -12.0, -6.0],
                [-6.0, 4.0, 6.0, 2.0],
                [-12.0, 6.0, 12.0, 6.0],
                [-6.0, 2.0, 6.0, 4.0],
            ]
        )

        cxz = np.array([1.0, -Le, 1.0, -Le])
        Ke_by_I = E * (K0 / Le**3) * np.outer(cxz, cxz)

        # Store the local stiffness matrix without the contributions from the moment of inertia
        self.Ke_by_I = Ke_by_I

        # Loop through the elements and populate the entries in the global stiffness matrixn
        nnodes = nelems + 1
        size = 2 * nnodes + 2
        K = np.zeros((size, size), dtype=I.dtype)

        # Add the the first element contributions
        K[0:4, 0:4] = Ke_by_I * I[0]

        # Loop over remaining elements and add the contributions for the ith element into the global stiffness matrix
        for idx in range(1, nelems):
            K[2 * idx : 2 * idx + 4, 2 * idx : 2 * idx + 4] += Ke_by_I * I[idx]

        # Update the entries to account for the Lagrange multipliers
        K[0, -2] = 1.0
        K[1, -1] = 1.0
        K[-2, 0] = 1.0
        K[-1, 1] = 1.0

        # plt.matshow(K)
        # plt.show()

        return K

    def _analyze_adjoint(self):
        """
        Adjoint method for the linear static solve analysis.
        """
        # Extract the parameter values
        nelems = self.parameters["nelems"]

        # Extract the output value
        d = self.outputs["d"].value

        # Extract the derivatives of the outputs
        db = self.outputs["d"].deriv

        # Extract the derivatives of the variables
        Ib = self.variables["I"].deriv

        # Solve for the adjoint variables for the system
        psi = -self.lu.solve(db, trans="T")

        # Loop over each elements and add the contributions to the entries of Ib
        for idx in range(nelems):
            # Set the start/end indices
            start = 2 * idx
            end = start + 4

            # Extract the components of psi and d for the corresponding element
            psi_e = psi[start:end]
            d_e = d[start:end]

            # Compute the contributions for the current element
            Ib[idx] += psi_e.T @ (self.Ke_by_I @ d_e)

        # Assign the derivative values
        self.variables["I"].set_deriv_value(Ib)

        return


if __name__ == "__main__":

    # Set the parameter values
    nelems = 2
    nnodes = nelems + 1
    E = 1.0
    L = 1.0
    f = np.random.uniform(size=2 * nnodes)

    # Construct the object
    linear_solve = LinearStaticSolve(
        obj_name="linear_solve", sub_analyses=[], E=E, L=L, nelems=nelems, f=f
    )

    # Test the isolated adjoint
    linear_solve.declare_design_vars(variables=["I"])
    linear_solve.test_isolated_adjoint(method="cs", debug_print=False)
