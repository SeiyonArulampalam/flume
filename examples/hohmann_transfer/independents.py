from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State


class Independents(Analysis):
    def __init__(self, obj_name: str, sub_analyses=[], **kwargs):
        """
        Class that defines the independent variables for the Hohmann transfer example. Here, the independent varaibles are the radii for the LEO and GEO orbits, the graviational parameter, and the inclination changes for the first and second maneuvers.

        Parameters
        ----------
        obj_name : str
            Name for the analysis object
        sub_analyses : list
            A list of any sub-analyses, nominally empty

        Keyword Arguments
        -----------------
        No input parameters for this object, so no **kwargs
        """

        # Set the default parameters
        self.default_parameters = {}

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default states for the variables
        r1_var = State(value=6778.0, desc="Orbit radius for LEO", source=self)
        r2_var = State(value=42164.0, desc="Orbit radius for GEO", source=self)
        mu_var = State(value=398600.4418, desc="Graviational parameter", source=self)
        dinc1_var = State(
            value=0.0, desc="Inclination change for the first maneuver", source=self
        )
        dinc2_var = State(
            value=28.5, desc="Inclination change for the second maneuver", source=self
        )

        self.variables = {
            "r1_dv": r1_var,
            "r2_dv": r2_var,
            "mu_dv": mu_var,
            "dinc1_dv": dinc1_var,
            "dinc2_dv": dinc2_var,
        }

        return

    def _analyze(self):
        """
        Propagate the input variables to the outputs that are distributed throughout the model
        """

        # Extract the variable values
        r1 = self.variables["r1_dv"].value
        r2 = self.variables["r2_dv"].value
        mu = self.variables["mu_dv"].value
        dinc1 = self.variables["dinc1_dv"].value
        dinc2 = self.variables["dinc2_dv"].value

        # Store the outputs in the output dictionary
        self.outputs = {}

        self.outputs["r1"] = State(value=r1, desc="Orbit radius for LEO", source=self)
        self.outputs["r2"] = State(value=r2, desc="Orbit radius for GEO", source=self)
        self.outputs["mu"] = State(value=mu, desc="Graviational parameter", source=self)
        self.outputs["dinc1"] = State(
            value=dinc1, desc="Inclination change for the first maneuver", source=self
        )
        self.outputs["dinc2"] = State(
            value=dinc2, desc="Inclination change for the second maneuver", source=self
        )

        return

    def _analyze_adjoint(self):
        """
        Propagate the adjoint variables back to the design variables
        """

        # Extract the derivatives of the outputs
        r1b = self.outputs["r1"].deriv
        r2b = self.outputs["r2"].deriv
        mub = self.outputs["mu"].deriv
        dinc1b = self.outputs["dinc1"].deriv
        dinc2b = self.outputs["dinc2"].deriv

        # Assign the derivative values for the variables
        self.variables["r1_dv"].set_deriv_value(deriv_val=r1b)
        self.variables["r2_dv"].set_deriv_value(deriv_val=r2b)
        self.variables["mu_dv"].set_deriv_value(deriv_val=mub)
        self.variables["dinc1_dv"].set_deriv_value(deriv_val=dinc1b)
        self.variables["dinc2_dv"].set_deriv_value(deriv_val=dinc2b)

        return
