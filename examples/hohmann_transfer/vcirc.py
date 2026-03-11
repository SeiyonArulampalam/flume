from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
import numpy as np
from .independents import Independents
from typing import List


class VCircAnalysis(Analysis):
    """
    Analysis class that computes the velcity associated with the circular orbit defined by the input radius and graviational parameter.
    """

    def __init__(
        self,
        obj_name: str,
        sub_analyses=List[Independents],
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
            A list of any sub-analyses for the VCircAnalysis, which nominally contains an instance of the Independents class
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
        r_var = State(value=1.0, desc="Radius from central body (km)", source=self)

        mu_var = State(
            value=1.0,
            desc="Gravitational parameter of central body (km**3/s**2)",
            source=self,
        )

        # Construct the variables dictionary
        if self.var_aliases is not None:
            r_name = self.var_aliases["r"]
            mu_name = self.var_aliases["mu"]

            self.variables = {r_name: r_var, mu_name: mu_var}

        else:
            self.variables = {"r": r_var, "mu": mu_var}

    def _analyze(self):
        """
        Private analysis method for the object that computes all of the outputs with the provided input variables and parameters to the object.
        """

        if self.var_aliases is not None:
            r = self.variables[self.var_aliases["r"]].value
            mu = self.variables[self.var_aliases["mu"]].value
        else:
            r = self.variables["r"].value
            mu = self.variables["mu"].value

        # Compute the circular velocity for the orbit
        v_c = np.sqrt(mu / r)

        # Assign the outputs to the outputs dictionary
        self.outputs = {}

        if self.output_aliases is not None:
            self.outputs[self.output_aliases["v_c"]] = State(
                value=v_c,
                desc="Circular orbit velocity for the given radius and gravitational parameter (km/s)",
                source=self,
            )

        else:
            self.outputs["v_c"] = State(
                value=v_c,
                desc="Circular orbit velocity for the given radius and gravitational parameter (km/s)",
                source=self,
            )

        return

    def _analyze_adjoint(self):
        """
        Private adjoint analysis method for the object that computes the derivative values for the input variables via the adjoint method.
        """

        # Extract the derivative values for the outputs
        if self.output_aliases is not None:
            v_cb = self.outputs[self.output_aliases["v_c"]].deriv
        else:
            v_cb = self.outputs["v_c"].deriv

        # Extract the variable values/derivatives
        if self.var_aliases is not None:
            r = self.variables[self.var_aliases["r"]].value
            mu = self.variables[self.var_aliases["mu"]].value

            rb = self.variables[self.var_aliases["r"]].deriv
            mub = self.variables[self.var_aliases["mu"]].deriv

        else:
            r = self.variables["r"].value
            mu = self.variables["mu"].value

            rb = self.variables["r"].deriv
            mub = self.variables["mu"].deriv

        # Compute the derivative contribution to r from v_c
        rb += v_cb * 0.5 * (mu / r) ** (-1 / 2) * -mu / r**2

        # Compute the derivative contribution to mu from v_c
        mub += v_cb * 0.5 * (mu / r) ** (-1 / 2) / r

        # Set the derivative values for the variables
        if self.var_aliases is not None:
            self.variables[self.var_aliases["r"]].set_deriv_value(deriv_val=rb)
            self.variables[self.var_aliases["mu"]].set_deriv_value(deriv_val=mub)
        else:
            self.variables["r"].set_deriv_value(deriv_val=rb)
            self.variables["mu"].set_deriv_value(deriv_val=mub)

        return


if __name__ == "__main__":
    # Create a VCirc analysis object
    vcirc = VCircAnalysis(obj_name="test")

    # Declare design variables for the object
    vcirc.declare_design_vars(variables=["mu", "r"])

    # Test the isolated adjoint implementation
    vcirc.test_isolated_adjoint(debug_print=True)
