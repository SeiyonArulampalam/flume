from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
from icecream import ic
import numpy as np
from typing import List


class ParticlePositions(Analysis):
    def __init__(self, obj_name: str, sub_analyses=[], **kwargs):
        """
        Analysis object that computes the Cartesian coordinates for each particle in the system using the constant radius and angles for each particle. The angles are set as design variables for the optimization problem.

        More information about this problem can be found at the following link for the Benchmarking Optimization Software with COPS 3.0: Description in COPS manual found at https://www.mcs.anl.gov/~more/cops/.

        Parameters
        ----------
        obj_name : str
            Specifies the name for the object
        sub_analyses : list
            List that nominally is empty, as this is the object that contains the design variables for the System

        Keyword Arguments
        -----------------
        n_p : int
            Number of particles for the system, which defines the number of equality constraints
        r : float
            Radius value for the sphere
        """

        # Set the default parameters
        self.default_parameters = {
            "n_p": 3,  # number of electrons in the model
            "r": 1.0,  # constant radius for the sphere
        }

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default states for the variables, which corresponds to the angles that define the particles' positions in spherical coordinates
        theta_var = State(
            value=np.ones(self.parameters["n_p"]),
            desc="Array of polar angles (specifying the angle between the radial line and the polar axis), where the ith index corresponds to the angle for the ith particle.",
            source=self,
        )

        phi_var = State(
            value=np.ones(self.parameters["n_p"]),
            desc="Array of azimuthal angles (specifying the rotation of the radial line around the polar axis), where the ith index corresponds to the angle for the ith particle.",
            source=self,
        )

        self.variables = {"theta": theta_var, "phi": phi_var}

        return

    def _analyze(self):
        """
        Compute the Cartesian coordinates for each particle using each of the angles and the constant radius.
        """

        # Extract the radius
        r = self.parameters["r"]

        # Extract the variable values
        theta = self.variables["theta"].value
        phi = self.variables["phi"].value

        # Compute the x, y, and z coordinates for each particle
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        # Assign the outputs
        self.outputs = {}

        self.outputs["x"] = State(
            value=x,
            desc="Design variable array of x-position values, where the ith index corresponds to the x-coordinate for the ith particle.",
            source=self,
        )

        self.outputs["y"] = State(
            value=y,
            desc="Design variable array of y-position values, where the ith index corresponds to the y-coordinate for the ith particle.",
            source=self,
        )

        self.outputs["z"] = State(
            value=z,
            desc="Design variable array of z-position values, where the ith index corresponds to the z-coordinate for the ith particle.",
            source=self,
        )

        return

    def _analyze_adjoint(self):
        """
        Private adjoint analysis method to propagate the derivatives back to the design variables.
        """

        # Extract the radius
        r = self.parameters["r"]

        # Extract the variable derivatives and values
        thetab = self.variables["theta"].deriv
        phib = self.variables["phi"].deriv

        theta = self.variables["theta"].value
        phi = self.variables["phi"].value

        # Extract the output derivatives
        xb = self.outputs["x"].deriv
        yb = self.outputs["y"].deriv
        zb = self.outputs["z"].deriv

        # Update the derivative values
        thetab += (
            xb * (r * np.cos(theta) * np.cos(phi))
            + yb * (r * np.cos(theta) * np.sin(phi))
            + zb * (-r * np.sin(theta))
        )

        phib += xb * (r * np.sin(theta) * -np.sin(phi)) + yb * (
            r * np.sin(theta) * np.cos(phi)
        )

        self.variables["theta"].set_deriv_value(thetab)
        self.variables["phi"].set_deriv_value(phib)

        return


class PotentialEnergy(Analysis):
    def __init__(self, obj_name: str, sub_analyses=List[ParticlePositions], **kwargs):
        """
        Analysis class that computes the position constraint for each particle in the system, which corresponds to a constraint enforcing each particle to be on a sphere.

        More information about this problem can be found at the following link for the Benchmarking Optimization Software with COPS 3.0: Description in COPS manual found at https://www.mcs.anl.gov/~more/cops/.

        Parameters
        ----------
        obj_name : str
            Specifies the name for the object
        sub_analyses : list
            List that nominally contains an instance of the ParticlePositions class to provide the variables for x, y, z

        Keyword Arguments
        -----------------
        n_p : int
            Number of particles for the system, which defines the number of equality constraints
        epsilon : float
            Small numerical value that is used to fix the distance between particles if the squared distance between two particles falls below this threshold value. Prevents any particles from exactly converging to the same position.
        """

        # Set the default parameters
        self.default_parameters = {
            "n_p": 3,  # number of electrons in the model
            "epsilon": 1e-15,  # small quantity that prevents the particles from converging
        }

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default states for the variables
        x_var = State(
            value=np.ones(self.parameters["n_p"]),
            desc="Array of x-position values, where the ith index corresponds to the x-coordinate for the ith particle.",
            source=self,
        )

        y_var = State(
            value=np.ones(self.parameters["n_p"]),
            desc="Array of y-position values, where the ith index corresponds to the y-coordinate for the ith particle.",
            source=self,
        )

        z_var = State(
            value=np.ones(self.parameters["n_p"]),
            desc="Array of z-position values, where the ith index corresponds to the z-coordinate for the ith particle.",
            source=self,
        )

        self.variables = {"x": x_var, "y": y_var, "z": z_var}

        return

    def _analyze(self):
        """
        Computes the potential energy for the set of particles on a conducting sphere. This is the objective that is to be minimized.
        """
        # Extract the variable values
        x = self.variables["x"].value
        y = self.variables["y"].value
        z = self.variables["z"].value

        # Extract the number of particles
        n_p = self.parameters["n_p"]
        epsilon = self.parameters["epsilon"]

        # Initialize the output
        f = 0.0

        # Perform the loop and compute the contribution to the potential energy function for the current particle pair
        for i in range(n_p - 1):
            for j in range(i + 1, n_p):
                dist_sq = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2

                # Check that the squared distance is not below the threshold value, otherwise set it to epsilon
                if dist_sq < epsilon:
                    f += epsilon
                else:
                    f += dist_sq**-0.5

        # Assign the output value
        self.outputs = {}

        self.outputs["f"] = State(
            value=f,
            desc="Value of the potential energy for the configuration of particles.",
            source=self,
        )

        return

    def _analyze_adjoint(self):
        """
        Private adjoint analysis class that propagates the derivatives back to x, y, and z.
        """

        # Extract the variable values and derivatives
        xb = self.variables["x"].deriv
        yb = self.variables["y"].deriv
        zb = self.variables["z"].deriv

        x = self.variables["x"].value
        y = self.variables["y"].value
        z = self.variables["z"].value

        # Extract the number of particles
        n_p = self.parameters["n_p"]
        epsilon = self.parameters["epsilon"]

        # Extract the output derivative
        fb = self.outputs["f"].deriv

        # Loop over and compute the contributions to the derivatives from each particle pair
        for i in range(n_p - 1):
            for j in range(i + 1, n_p):
                dist_sq = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2

                # Check that the squared distance is not below the threshold value, otherwise set it to epsilon
                if dist_sq < epsilon:
                    dist_sq = epsilon

                xb[i] += -0.5 * dist_sq ** (-1.5) * (2 * (x[i] - x[j]))
                xb[j] += -0.5 * dist_sq ** (-1.5) * (-2 * (x[i] - x[j]))

                yb[i] += -0.5 * dist_sq**-1.5 * (2 * (y[i] - y[j]))
                yb[j] += -0.5 * dist_sq**-1.5 * (-2 * (y[i] - y[j]))

                zb[i] += -0.5 * dist_sq**-1.5 * (2 * (z[i] - z[j]))
                zb[j] += -0.5 * dist_sq**-1.5 * (-2 * (z[i] - z[j]))

        # Multiply by the adjoint variables
        xb *= fb
        yb *= fb
        zb *= fb

        self.variables["x"].set_deriv_value(deriv_val=xb)
        self.variables["y"].set_deriv_value(deriv_val=yb)
        self.variables["z"].set_deriv_value(deriv_val=zb)

        return


class ParticleConstraints(Analysis):
    def __init__(self, obj_name: str, sub_analyses=List[ParticlePositions], **kwargs):
        """
        Analysis class that computes the position constraint for each particle in the system, which corresponds to a constraint enforcing each particle to be on a sphere.

        More information about this problem can be found at the following link for the Benchmarking Optimization Software with COPS 3.0: Description in COPS manual found at https://www.mcs.anl.gov/~more/cops/.

        Parameters
        ----------
        obj_name : str
            Specifies the name for the object
        sub_analyses : list
            List that nominally contains an instance of the ParticlePositions class to provide the variables for x, y, z

        Keyword Arguments
        -----------------
        n_p : int
            Number of particles for the system, which defines the number of equality constraints
        """

        # Set the default parameters
        self.default_parameters = {
            "n_p": 3,  # number of electrons in the model
        }

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default states for the variables
        x_var = State(
            value=np.ones(self.parameters["n_p"]),
            desc="Array of x-position values, where the ith index corresponds to the x-coordinate for the ith particle.",
            source=self,
        )

        y_var = State(
            value=np.ones(self.parameters["n_p"]),
            desc="Array of y-position values, where the ith index corresponds to the y-coordinate for the ith particle.",
            source=self,
        )

        z_var = State(
            value=np.ones(self.parameters["n_p"]),
            desc="Array of z-position values, where the ith index corresponds to the z-coordinate for the ith particle.",
            source=self,
        )

        self.variables = {"x": x_var, "y": y_var, "z": z_var}

        return

    def _analyze(self):
        """
        Compute the constraint for each particle's position such that it must be on a sphere with radius = 1.
        """

        # Extract the variable values
        x = self.variables["x"].value
        y = self.variables["y"].value
        z = self.variables["z"].value

        # Compute the value for each constraint
        c = x[:] ** 2 + y[:] ** 2 + z[:] ** 2 - 1

        # Assign the outputs
        self.outputs = {}

        self.outputs["c"] = State(
            value=c,
            desc="Vector of values that defines the position constraint for each particle (constraint value of zero means the equality constraint is satisfied)",
            source=self,
        )

        return

    def _analyze_adjoint(self):
        """
        Private adjoint analysis that propagates the derivatives of the constraints back to the position vectors.
        """

        # Extract the output derivatives
        cb = self.outputs["c"].deriv

        # Extract the variable values and derivatives
        xb = self.variables["x"].deriv
        yb = self.variables["y"].deriv
        zb = self.variables["z"].deriv

        x = self.variables["x"].value
        y = self.variables["y"].value
        z = self.variables["z"].value

        # Compute the contributions to xb, yb, and zb from the constraint
        xb[:] += cb * 2 * x[:]
        yb[:] += cb * 2 * y[:]
        zb[:] += cb * 2 * z[:]

        self.variables["x"].set_deriv_value(deriv_val=xb)
        self.variables["y"].set_deriv_value(deriv_val=yb)
        self.variables["z"].set_deriv_value(deriv_val=zb)

        return


if __name__ == "__main__":

    # Set the number of particles
    n_p = 3

    # Create objects for the ParticlePositions, PotentialEnergy, and ParticleConstraints
    positions = ParticlePositions(obj_name="positions", sub_analyses=[], n_p=n_p)

    energy = PotentialEnergy(obj_name="energy", sub_analyses=[positions], n_p=n_p)

    cons = ParticleConstraints(obj_name="cons", sub_analyses=[positions], n_p=n_p)

    # Test the combined adjoints
    positions.declare_design_vars(variables=["theta", "phi"])

    energy.test_combined_adjoint(debug_print=False, method="cs")

    cons.test_combined_adjoint(debug_print=False, method="cs")
