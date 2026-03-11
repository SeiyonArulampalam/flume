from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
import numpy as np
from typing import List
from .independents import Independents


class BeamVolume(Analysis):
    def __init__(self, obj_name: str, sub_analyses=List[Independents], **kwargs):
        """
        Analysis class that computes the volume of the beam with the assumed rectangular cross section.

        Parameters
        ----------
        obj_name : str
            Name for the analysis object
        sub_analyses : list
            A list of sub-analyses for the object, which nominally contains an instance of the Independents class

        Keyword Arguments
        -----------------
        nelems : int
            Integer number that defines the number of elements for the beam model
        b : float
            Base value for the rectangular cross section
        L : float
            Length of the beam
        """
        # Set the default parameters
        self.default_parameters = {
            "nelems": 10,  # number of elements for the beam
            "b": 1.0,  # base for the rectangular cross section
            "L": 1.0,  # Length of the beam
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
        Compute the volume for the beam.
        """

        # Extract the variable value
        h = self.variables["h"].value

        # Extract the parameter values
        b = self.parameters["b"]
        L = self.parameters["L"]
        nelems = self.parameters["nelems"]

        # Compute the element length
        self.Le = L / nelems

        # Compute the volume for each element
        Ve = h * b * self.Le

        # Sum the element volumes to get the total volume
        V = np.sum(Ve)

        # Store the outputs
        self.outputs = {}

        self.outputs["V"] = State(value=V, desc="Volume for the beam", source=self)

        return

    def _analyze_adjoint(self):
        """
        Perform the adjoint analysis to propagate the derivatives back to h
        """

        # Extract the variable/output derivatives
        hb = self.variables["h"].deriv
        Vb = self.outputs["V"].deriv

        # Extract the parameter value for the base
        b = self.parameters["b"]

        # Compute the contributions to hb from Vb
        hb += Vb * b * self.Le

        # Assign the derivative values
        self.variables["h"].set_deriv_value(deriv_val=hb)

        return


if __name__ == "__main__":

    # Set the parameter values
    b = 1.0
    L = 1.0
    nelems = 10

    # Construct the volume object
    volume = BeamVolume(obj_name="volume", sub_analyses=[], b=b, L=L, nelems=nelems)

    # Test the isolated adjoint
    volume.declare_design_vars(variables=["h"])
    volume.test_isolated_adjoint(method="cs")
