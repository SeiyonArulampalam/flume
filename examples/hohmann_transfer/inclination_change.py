from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
from typing import List
from .independents import Independents


class TotalInclinationChange(Analysis):
    def __init__(self, obj_name: str, sub_analyses=List[Independents], **kwargs):
        """
        Analysis class that cmoputes the total inclination change for the Hohmann transfer, summing the two inclination changes for the two maneuvers.

        Parameters
        ----------
        obj_name : str
            Name for the analysis object
        sub_analyses : list
            A list of any sub-analyses for TransferOrbitAnalysis, which nominally contains an instance of the Independents class

        Keyword Arguments
        -----------------
        No input parameters for this object, so no **kwargs.
        """

        # Set the default parameters
        self.default_parameters = {}

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default states for the variables
        dinc1_var = State(
            value=0.0, desc="Inclination change for the first maneuver", source=self
        )
        dinc2_var = State(
            value=0.0, desc="Inclination change for the second maneuver", source=self
        )

        self.variables = {"dinc1": dinc1_var, "dinc2": dinc2_var}

        return

    def _analyze(self):
        """
        Compute the total delta V from the delta V values for the two maneuvers.
        """

        # Extract the variable values
        dinc1 = self.variables["dinc1"].value
        dinc2 = self.variables["dinc2"].value

        # Compute the total delta_v
        dinc_tot = dinc1 + dinc2

        # Store the outputs in the output dictionary
        self.outputs = {}

        self.outputs["dinc_tot"] = State(
            value=dinc_tot,
            desc="Total inclination change for the Hohmann transfer",
            source=self,
        )

        return

    def _analyze_adjoint(self):
        """
        Propagate the derivatives of the total delta V back to the individual delta V's.
        """

        # Extract the output/variable derivatives
        dinc_totb = self.outputs["dinc_tot"].deriv

        dinc1b = self.variables["dinc1"].deriv
        dinc2b = self.variables["dinc2"].deriv

        # Update the derivative values for the variables
        dinc1b += dinc_totb
        dinc2b += dinc_totb

        # Set the derivative values
        self.variables["dinc1"].set_deriv_value(deriv_val=dinc1b)
        self.variables["dinc2"].set_deriv_value(deriv_val=dinc2b)

        return
