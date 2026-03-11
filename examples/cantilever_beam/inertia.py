from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
import numpy as np
from typing import List
from .independents import Independents


class MomentofInertia(Analysis):
    def __init__(self, obj_name: str, sub_analyses=List[Independents], **kwargs):
        """
        Analysis class that computes the moment of inertia for each element in the cantilever beam model, assuming a square cross section.

        I[i] = 1/12 * b * h[i]**3

        Parameters
        ----------
        obj_name : str
            Name for the analysis object
        sub_analyses : list
            A list of sub-analyses for the object, which nominally contains an instance of the Independents class

        Keyword Arguments
        -----------------
        b : float
            Base value for the rectangular cross section
        nelems : int
            Integer number that defines the number of elements for the beam model
        """
        # Set the default parameters
        self.default_parameters = {
            "b": 1.0,  # base for the rectangular cross section
            "nelems": 10,  # number of elements for the beam
        }

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default states for the variables
        h_var = State(
            value=np.ones(self.parameters["nelems"]),
            desc="Height values for the beam, where the ith index corresponds to the height at ith element",
            source=self,
        )

        self.variables = {"h": h_var}

        return

    def _analyze(self):
        """
        Computes the moment of inertia of each element
        """

        # Extract the variable and parameter values
        b = self.parameters["b"]

        h = self.variables["h"].value

        # Compute the moment of inertia for each element (square cross section assumed)
        I = 1 / 12 * b * h**3

        # Store the outputs
        self.outputs = {}

        self.outputs["I"] = State(
            value=I,
            desc="Moment of inertia values at each element in the beam",
            source=self,
        )

        return

    def _analyze_adjoint(self):
        """
        Performs the adjoint analysis for the moment of inertia class
        """

        # Extract the derivative values for the variables/outputs
        hb = self.variables["h"].deriv
        Ib = self.outputs["I"].deriv

        # Extract the variable and parameter values
        h = self.variables["h"].value
        b = self.parameters["b"]

        # Compute the contribution to hb from Ib
        hb += Ib * 1 / 12 * b * 3 * h**2

        # Assign the derivative values
        self.variables["h"].set_deriv_value(deriv_val=hb)

        return


if __name__ == "__main__":

    # Construct the moment of inertia object
    nelems = 20
    inertia = MomentofInertia(obj_name="inertia_test", sub_analyses=[], nelems=nelems)

    # Test the isolated adjoint
    inertia.declare_design_vars(variables=["h"])
    inertia.test_isolated_adjoint(method="cs")
