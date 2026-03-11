from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
import numpy as np


class Independents(Analysis):
    def __init__(self, obj_name: str, sub_analyses=[], **kwargs):
        """
        Defines the independent variables for the cantilever beam example, which here is just the array of height values.

        Parameters
        ----------
        obj_name : str
            Name for the analysis object
        sub_analyses : list
            A list of sub-analyses for the object

        Keyword Arguments
        -----------------
        nelems : int
            Integer number that defines the number of elements for the beam model
        """

        # Set the default parameters
        self.default_parameters = {
            "nelems": 10,  # number of elements for the beam
        }

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default states for the variables
        h_var = State(
            value=np.ones(self.parameters["nelems"]),
            desc="Height design variable values for the beam, where the ith index corresponds to the height at ith element",
            source=self,
        )

        self.variables = {"h_dv": h_var}

        return

    def _analyze(self):
        """
        Set the value for h, which is distributed throughout the model, to h_dv
        """

        # Extract the variables
        h_dv = self.variables["h_dv"].value

        # Store the outputs
        self.outputs = {}

        self.outputs["h"] = State(
            value=h_dv,
            desc="Height values for the beam, where the ith index corresponds to the height at the ith element",
            source=self,
        )

        return

    def _analyze_adjoint(self):
        """
        Propagate the derivatives from h back to h_dv
        """

        # Extract the derivative values
        hb = self.outputs["h"].deriv
        h_dvb = self.variables["h_dv"].deriv

        # Update the derivatives for h_dv
        h_dvb += hb

        # Assign the derivative values
        self.variables["h_dv"].set_deriv_value(h_dvb)

        return
