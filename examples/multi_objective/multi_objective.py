# Description: this script defines an example that combines many different (albeit arbitrarily defined) Analysis classes into a System. The purpose of this example is to elaborate on how sub-analyses are defined between Analysis disciplines and illustrate the construction of a System. The system can be visualized using the graph_network method, as well. Optimization is not performed within this script, as this is more a demonstration of Analysis/System construction.

from flume.base_classes.analysis import Analysis
from flume.base_classes.state import State
from flume.base_classes.system import System
import numpy as np
from icecream import ic
from typing import List, Union
from math import exp


class FirstBlock(Analysis):

    def __init__(
        self,
        obj_name: str,
        sub_analyses=[],
        **kwargs,
    ):

        # Set default parameters
        self.default_parameters = {}

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default State for the variables
        xvar = State(value=1.0, desc="x state value", source=self)

        # Construct variables dictionary
        self.variables = {"x": xvar}

    def _analyze(self):
        # Extract the variable value
        x = self.variables["x"].value

        # Compute y
        y = x**2

        # Store the outputs
        self.outputs = {}

        self.outputs["y"] = State(value=y, desc="y value", source=self)

    def _analyze_adjoint(self):
        # Extract derivative of the output
        yb = self.outputs["y"].deriv

        # Extract the derivative value for x
        xb = self.variables["x"].deriv

        # Compute xb from y
        x = self.variables["x"].value
        xb += 2 * x * yb

        # Update the derivative values for the varialbes
        self.variables["x"].set_deriv_value(deriv_val=xb)

        return


class ObjTerm1(Analysis):

    def __init__(self, obj_name: str, sub_analyses=List[FirstBlock], **kwargs):
        # Set default parameters
        self.default_parameters = {}

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the defualt State for the variable
        y_var = State(value=1.0, desc="y value", source=self)

        # Construct vars dictionary
        self.variables = {"y": y_var}

        return

    def _analyze(self):
        # Extract variable values
        y = self.variables["y"].value

        # Compute f
        f = 2 * y

        # Store the outputs
        self.outputs = {}

        self.outputs["f"] = State(value=f, desc="f value", source=self)

        return

    def _analyze_adjoint(self):
        # Extract the output derivative
        fb = self.outputs["f"].deriv

        # Extract the variable derivative
        yb = self.variables["y"].deriv

        # Compute the contribution to yb from f
        yb += 2 * fb

        # Update the derivative values for the varialbes
        self.variables["y"].set_deriv_value(deriv_val=yb)

        return


class ObjTerm2(Analysis):

    def __init__(self, obj_name: str, sub_analyses=List[FirstBlock], **kwargs):
        # Set default parameters
        self.default_parameters = {}

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the defualt State for the variable
        y_var = State(value=1.0, desc="y value", source=self)

        # Construct vars dictionary
        self.variables = {"y": y_var}

        return

    def _analyze(self):
        # Extract variable values
        y = self.variables["y"].value

        # Compute g
        g = np.exp(y).item()

        # Store the outputs
        self.outputs = {}

        self.outputs["g"] = State(value=g, desc="g value", source=self)

        return

    def _analyze_adjoint(self):
        # Extract the output derivative
        gb = self.outputs["g"].deriv

        # Extract the variable derivative
        yb = self.variables["y"].deriv

        # Compute the contribution to yb from f
        y = self.variables["y"].value
        yb += exp(y) * gb

        # Update the derivative values for the varialbes
        self.variables["y"].set_deriv_value(deriv_val=yb)

        return


class MultiObjective(Analysis):

    def __init__(
        self, obj_name: str, sub_analyses=List[Union[ObjTerm1, ObjTerm2]], **kwargs
    ):

        # Set default parameters
        self.default_parameters = {"w": 0.5}

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default State for the variables
        f_var = State(value=1.0, desc="f state value", source=self)

        g_var = State(value=1.0, desc="g state value", source=self)

        # Construct variables dictionary
        self.variables = {"f": f_var, "g": g_var}

        return

    def _analyze(self):
        # Extract variable values
        f = self.variables["f"].value
        g = self.variables["g"].value
        w = self.parameters["w"]

        # Compute the multi objective value
        J = w * f + (1 - w) * g

        # Store the outputs
        self.outputs = {}

        self.outputs["J"] = State(
            value=J, desc="Multi-objective function value", source=self
        )

        return

    def _analyze_adjoint(self):
        # Extract output derivative
        Jb = self.outputs["J"].deriv

        # Extract the variable derivatives
        fb = self.variables["f"].deriv
        gb = self.variables["g"].deriv

        # Extract the weight value
        w = self.parameters["w"]

        # Compute the contribution to fb from J
        fb += Jb * w

        # Compute the contribution to gb from J
        gb += Jb * (1 - w)

        # Update the derivative values for the variables
        self.variables["f"].set_deriv_value(deriv_val=fb)
        self.variables["g"].set_deriv_value(deriv_val=gb)

        return


class Intermediate(Analysis):

    def __init__(self, obj_name: str, sub_analyses=List[ObjTerm1], **kwargs):

        # Set default parameters
        self.default_parameters = {}

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default State for the variables
        f_var = State(value=1.0, desc="f state value", source=self)

        # Construct variables dictionary
        self.variables = {"f": f_var}

        return

    def _analyze(self):

        # Extract the variable
        f = self.variables["f"].value

        # Compute the output
        h = 3 * f

        # Store the outputs
        self.outputs = {}

        self.outputs["h"] = State(
            value=h, desc="Intermediate constraint value, h", source=self
        )

        return

    def _analyze_adjoint(self):

        # Extract the output derivatives
        hb = self.outputs["h"].deriv

        # Extract the variable derivatives
        fb = self.variables["f"].deriv

        fb += hb * 3.0

        # Update the derivative values
        self.variables["f"].set_deriv_value(fb)

        return


class Intermediate2(Analysis):

    def __init__(self, obj_name: str, sub_analyses=List[ObjTerm2], **kwargs):

        # Set default parameters
        self.default_parameters = {}

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default State for the variables
        g_var = State(value=1.0, desc="g state value", source=self)

        # Construct variables dictionary
        self.variables = {"g": g_var}

        return

    def _analyze(self):

        # Extract the variable
        g = self.variables["g"].value

        # Compute the output
        i = 3 * g

        # Store the outputs
        self.outputs = {}

        self.outputs["i"] = State(value=i, desc="Intermediate value, i", source=self)

        return

    def _analyze_adjoint(self):

        # Extract the output derivatives
        ib = self.outputs["i"].deriv

        # Extract the variable derivatives
        gb = self.variables["g"].deriv

        gb += ib * 3.0

        # Update the derivative values
        self.variables["g"].set_deriv_value(gb)

        return


class Constraint(Analysis):

    def __init__(
        self,
        obj_name: str,
        sub_analyses=List[Union[Intermediate, Intermediate2]],
        **kwargs,
    ):

        # Set default parameters
        self.default_parameters = {}

        # Perform the base class object initialization
        super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

        # Set the default State for the variables
        i_var = State(value=1.0, desc="i state value", source=self)

        h_var = State(value=1.0, desc="h state value", source=self)

        # Construct variables dictionary
        self.variables = {"i": i_var, "h": h_var}

        return

    def _analyze(self):

        # Extract the variable values
        i = self.variables["i"].value

        h = self.variables["h"].value

        # Compute the output
        c = i * h

        # Store the outputs
        self.outputs = {}

        self.outputs["c"] = State(value=c, desc="Constraint value", source=self)

        return

    def _analyze_adjoint(self):

        # Extract the variable values
        i = self.variables["i"].value

        h = self.variables["h"].value

        # Extract the variable derivatives
        ib = self.variables["i"].deriv
        hb = self.variables["h"].deriv

        # Extract the output derivative
        cb = self.outputs["c"].deriv

        # Update xb and fb values
        ib += cb * h

        hb += cb * i

        # Update the derivative values
        self.variables["i"].set_deriv_value(ib)
        self.variables["h"].set_deriv_value(hb)

        return


if __name__ == "__main__":

    # Construct the four analysis blocks
    first = FirstBlock(obj_name="first")

    term1 = ObjTerm1(obj_name="term1", sub_analyses=[first])
    term2 = ObjTerm2(obj_name="term2", sub_analyses=[first])

    w = 0.5
    multi = MultiObjective(obj_name="multi", sub_analyses=[term1, term2], w=w)

    inter = Intermediate(obj_name="inter", sub_analyses=[term1])

    inter2 = Intermediate2(obj_name="inter2", sub_analyses=[term2])

    constraint = Constraint(obj_name="con", sub_analyses=[inter, inter2])

    # Test the isolated adjoints
    test_isolated_adjoints = False
    if test_isolated_adjoints:
        first.declare_design_vars(variables=["x"])
        first.test_isolated_adjoint(method="cs")

        term1.declare_design_vars(variables=["y"])
        term1.test_isolated_adjoint(method="cs")

        term2.declare_design_vars(variables=["y"])
        term2.test_isolated_adjoint(method="cs")

        inter.declare_design_vars(variables=["f"])
        inter.test_isolated_adjoint(method="cs")

        inter2.declare_design_vars(variables=["g"])
        inter2.test_isolated_adjoint(method="cs")

        multi.declare_design_vars(variables=["f", "g"])
        multi.test_isolated_adjoint(method="cs")

        constraint.declare_design_vars(variables=["h", "i"])
        constraint.test_isolated_adjoint(method="cs")

    # Test the combined adjoints
    test_combined_adjoints = False
    if test_combined_adjoints:
        first.declare_design_vars(variables=["x"])
        multi.test_combined_adjoint(method="fd", debug_print=False)

        constraint.test_combined_adjoint(method="fd")

    # Create the flume system
    sys = System(
        sys_name="multiobj_test",
        top_level_analysis_list=[multi, constraint],
        log_prefix="examples/multi_objective",
    )

    # Graph the system
    graph = sys.graph_network()
    graph.render("MultiObjective_SystemGraph", directory=sys.log_prefix, cleanup=True)

    # Declare the objective and constraints
    sys.declare_design_vars(global_var_name={"first.x": {"lb": -2.0, "ub": 2.0}})

    sys.declare_objective(global_obj_name="multi.J")

    sys.declare_constraints(
        global_con_name={"con.c": {"rhs": 10.0, "direction": "leq"}}
    )
