from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
import numpy as np
from .transfer_orbit import TransferOrbitAnalysis
from .independents import Independents
from .vcirc import VCircAnalysis
from typing import List, Union


class DeltaVAnalysis(Analysis):
    """
    Analysis class that computes the change in velocity required to perform a maneuver given the magnitudes of the velocities before and after the maneuver and the inclination change between the orbits.
    """

    def __init__(
        self,
        obj_name: str,
        sub_analyses=List[Union[Independents, TransferOrbitAnalysis, VCircAnalysis]],
        variable_aliases=None,
        output_aliases=None,
        **kwargs,
    ):
        """
        Analysis class for computing change in velocity of two orbits.

        Parameters
        ----------
        obj_name : str
            Name for the analysis object
        sub_analyses : list
            A list of any sub-analyses for DeltaVAnalysis, which nominally contains instances of the Independents, TransferOrbitAnalysis, and VCircAnalysis classes
        variable_aliases : dict
            Optional argument that, when provided, renames the variable names from their default values. If provided, the user must provide key-value pairs for all variables, where the keys are the default names and the values are the updated variable names
        output_aliases : dict
            Optional argument that, when provided, renames the output names from their default values. If provided, the user must provide key-value pairs for all outputs, where the keys are the default names and the values are the updated output names

        Keyword Arguments
        -----------------
        No input parameters for this object, so no **kwargs.

        """

        # Set the default parameters for the object
        self.default_parameters = {}

        # Store the variable/output aliases
        self.var_aliases = variable_aliases
        self.output_aliases = output_aliases

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default States for the variables
        v1_var = State(value=1.0, desc="Initial velocity (km/s)", source=self)

        v2_var = State(value=1.0, desc="Final velocity (km/s)", source=self)

        dinc_var = State(value=1.0, desc="Plane change (rad)", source=self)

        # Construct the variables dictionary
        if self.var_aliases is not None:
            v1_name = self.var_aliases["v1"]
            v2_name = self.var_aliases["v2"]
            dinc_name = self.var_aliases["dinc"]

            self.variables = {v1_name: v1_var, v2_name: v2_var, dinc_name: dinc_var}

        else:
            self.variables = {"v1": v1_var, "v2": v2_var, "dinc": dinc_var}

        return

    def _analyze(self):
        """
        Private analysis method for the object that computes all of the outputs with the provided input variables and parameters to the object.
        """

        # Extract the variable values
        if self.var_aliases is not None:
            v1 = self.variables[self.var_aliases["v1"]].value
            v2 = self.variables[self.var_aliases["v2"]].value
            dinc = self.variables[self.var_aliases["dinc"]].value

        else:
            v1 = self.variables["v1"].value
            v2 = self.variables["v2"].value
            dinc = self.variables["dinc"].value

        # Compute the change in velocity
        delta_v = np.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * np.cos(dinc))

        # Store the output in the output dictionary
        self.outputs = {}

        if self.output_aliases is None:
            self.outputs["delta_v"] = State(
                value=delta_v, desc="Change in velocity (km/s)", source=self
            )
        else:
            self.outputs[self.output_aliases["delta_v"]] = State(
                value=delta_v, desc="Change in velocity (km/s)", source=self
            )

        return

    def _analyze_adjoint(self):
        """
        Private adjoint analysis method for the object that computes the derivative values for the input variables via the adjoint method.
        """

        # Extract the variable values
        if self.var_aliases is not None:
            v1 = self.variables[self.var_aliases["v1"]].value
            v2 = self.variables[self.var_aliases["v2"]].value
            dinc = self.variables[self.var_aliases["dinc"]].value

            v1b = self.variables[self.var_aliases["v1"]].deriv
            v2b = self.variables[self.var_aliases["v2"]].deriv
            dincb = self.variables[self.var_aliases["dinc"]].deriv

        else:
            v1 = self.variables["v1"].value
            v2 = self.variables["v2"].value
            dinc = self.variables["dinc"].value

            v1b = self.variables["v1"].deriv
            v2b = self.variables["v2"].deriv
            dincb = self.variables["dinc"].deriv

        # Extract the existing derivative values for the outputs and variables
        if self.output_aliases is not None:
            delta_vb = self.outputs[self.output_aliases["delta_v"]].deriv
            delta_v = self.outputs[self.output_aliases["delta_v"]].value
        else:
            delta_vb = self.outputs["delta_v"].deriv
            delta_v = self.outputs["delta_v"].value

        # Compute the derivative contribution to v1 from delta_v
        v1b += delta_vb * 0.5 / delta_v * (2 * v1 - 2 * v2 * np.cos(dinc))

        v2b += delta_vb * 0.5 / delta_v * (2 * v2 - 2 * v1 * np.cos(dinc))

        dincb += delta_vb * 0.5 / delta_v * (2 * v1 * v2 * np.sin(dinc))

        # Set the derivative values for the variables
        if self.var_aliases is not None:
            self.variables[self.var_aliases["v1"]].set_deriv_value(deriv_val=v1b)
            self.variables[self.var_aliases["v2"]].set_deriv_value(deriv_val=v2b)
            self.variables[self.var_aliases["dinc"]].set_deriv_value(deriv_val=dincb)
        else:
            self.variables["v1"].set_deriv_value(deriv_val=v1b)
            self.variables["v2"].set_deriv_value(deriv_val=v2b)
            self.variables["dinc"].set_deriv_value(deriv_val=dincb)

        return


if __name__ == "__main__":
    # Create a deltav analysis object
    deltav = DeltaVAnalysis(obj_name="test")

    # Declare design variables for the object
    deltav.declare_design_vars(variables=["v1", "v2", "dinc"])

    # Test the isolated adjoint implementation
    deltav.test_isolated_adjoint(debug_print=True)
