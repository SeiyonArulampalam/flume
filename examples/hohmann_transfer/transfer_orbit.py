from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
import numpy as np
from icecream import ic
from .independents import Independents
from typing import List


class TransferOrbitAnalysis(Analysis):
    """
    Analysis class that computes the velocity magnitudes at periapsis and apoapsis of an orbit for a given transfer orbit maneuver.
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
        Analysis clas for computing the velocities of the transfer orbit.

        Parameters
        ----------
        obj_name : str
            Name for the analysis object
        sub_analyses : list
            A list of any sub-analyses for TransferOrbitAnalysis, which nominally contains an instance of the Independents class
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
        ra_var = State(value=42164.0, desc="Apoapsis radius (km)", source=self)

        rp_var = State(value=7000.0, desc="Periapsis radius (km)", source=self)

        mu_var = State(
            value=398600.4418, desc="Graviational parameter (km**3/s**2)", source=self
        )

        # Construct the variables dictionary
        if self.var_aliases is not None:
            ra_name = self.var_aliases["ra"]
            rp_name = self.var_aliases["rp"]
            mu_name = self.var_aliases["mu"]

            self.variables = {ra_name: ra_var, rp_name: rp_var, mu_name: mu_var}
        else:
            self.variables = {"ra": ra_var, "rp": rp_var, "mu": mu_var}

        return

    def _analyze(self):
        """
        Private analysis method for the object that computes all of the outputs with the provided input variables and parameters to the object.
        """

        if self.var_aliases is not None:
            ra = self.variables[self.var_aliases["ra"]].value
            rp = self.variables[self.var_aliases["rp"]].value
            mu = self.variables[self.var_aliases["mu"]].value
        else:
            ra = self.variables["ra"].value
            rp = self.variables["rp"].value
            mu = self.variables["mu"].value

        # Compute the semi-major axis of the transfer orbit
        self.a = (ra + rp) / 2.0

        # Compute the eccentricity of the transfer orbit
        self.e = (self.a - rp) / self.a

        # Compute the semilatus rectrum of the orbit
        self.p = self.a * (1.0 - self.e**2)

        # Compute the specific angular momentum of the orbit
        self.h = np.sqrt(self.p * mu)

        # Compute the velocities at periapsis and apoapsis
        va = self.h / ra

        vp = self.h / rp

        # Store the outputs in the output dictionary
        self.outputs = {}

        if self.output_aliases is not None:
            self.outputs[self.output_aliases["va"]] = State(
                value=va, desc="Velocity at apoapsis (km/s)", source=self
            )

            self.outputs[self.output_aliases["vp"]] = State(
                value=vp, desc="Velocity at periapsis (km/s)", source=self
            )
        else:
            self.outputs["va"] = State(
                value=va, desc="Velocity at apoapsis (km/s)", source=self
            )

            self.outputs["vp"] = State(
                value=vp, desc="Velocity at periapsis (km/s)", source=self
            )

        return

    def _analyze_adjoint(self):
        """
        Private adjoint analysis method for the object that computes the derivative values for the input variables via the adjoint method.
        """

        # Extract the existing derivative values for the outputs and variables
        if self.output_aliases is not None:
            vab = self.outputs[self.output_aliases["va"]].deriv
            vpb = self.outputs[self.output_aliases["vp"]].deriv
        else:
            vab = self.outputs["va"].deriv
            vpb = self.outputs["vp"].deriv

        if self.var_aliases is not None:
            ra = self.variables[self.var_aliases["ra"]].value
            rp = self.variables[self.var_aliases["rp"]].value
            mu = self.variables[self.var_aliases["mu"]].value

            rab = self.variables[self.var_aliases["ra"]].deriv
            rpb = self.variables[self.var_aliases["rp"]].deriv
            mub = self.variables[self.var_aliases["mu"]].deriv
        else:
            ra = self.variables["ra"].value
            rp = self.variables["rp"].value
            mu = self.variables["mu"].value

            rab = self.variables["ra"].deriv
            rpb = self.variables["rp"].deriv
            mub = self.variables["mu"].deriv

        # Compute the derivative contribution to self.h from va and vp
        hb = vab / ra
        hb += vpb / rp

        # Compute the contributions to ra from va
        rab += vab * -self.h / ra**2

        # Compute the contributions to rp from vp
        rpb += vpb * -self.h / rp**2

        # Compute the contributions to mu from h
        mub = hb * 0.5 * (mu * self.p) ** (-1 / 2) * self.p

        # Compute the contributions to p from h
        pb = hb * 0.5 * (mu * self.p) ** (-1 / 2) * mu

        # Compute the contribution to a from p
        ab = pb * (1 - self.e**2)

        # Compute the contribution to e from p
        eb = pb * -2 * self.e * self.a

        # Compute the contribution to a from e
        ab += eb * rp / self.a**2

        # Compute the contribution to rp from e
        rpb += eb * -1 / self.a

        # Compute the contribution to ra from a
        rab += ab * 0.5

        # Compute the contribution to rp from a
        rpb += ab * 0.5

        # Set the derivative values for the variables
        if self.var_aliases is not None:
            self.variables[self.var_aliases["ra"]].set_deriv_value(deriv_val=rab)
            self.variables[self.var_aliases["rp"]].set_deriv_value(deriv_val=rpb)
            self.variables[self.var_aliases["mu"]].set_deriv_value(deriv_val=mub)

        else:
            self.variables["ra"].set_deriv_value(deriv_val=rab)
            self.variables["rp"].set_deriv_value(deriv_val=rpb)
            self.variables["mu"].set_deriv_value(deriv_val=mub)

        return


if __name__ == "__main__":
    # Create the transfer orbit analysis object
    transfer = TransferOrbitAnalysis(obj_name="test")

    # Declare design variables for the object
    transfer.declare_design_vars(variables=["ra", "rp", "mu"])

    # Test the isolated adjoint implementation
    transfer.test_isolated_adjoint(debug_print=True)
