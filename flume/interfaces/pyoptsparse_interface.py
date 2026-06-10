import os
from flume.base_classes.system import System
import numpy as np
from icecream import ic
from pyoptsparse import Optimization, Optimizer, Solution


class FlumePyOptSparseInterface:

    def __init__(self, flume_sys: System, callback=None, update=None):
        """
        DOCS:
        """

        # Store the flume system as an attribute
        self.flume_sys = flume_sys

        # Store the callback function
        self.callback = callback

        # Store the update function
        self.update = update

        # Initialize the iteration counter
        self.it_counter = 0

    def _set_system_variables(self, xdict: dict, it_counter: int):
        """
        DOCS:
        """

        # Since system variables are being set, all analysis objects must be recomputed
        self.flume_sys.reset_analysis_flags(it_counter)

        # Loop through the design variables for the system and set the values for their components
        for var in self.flume_sys.design_vars_info:
            # Get the numeric values from the design variables dictionary
            x_val = xdict[var]

            # Extract the local name of the variable
            local_name = self.flume_sys.design_vars_info[var]["local_name"]

            # Declare the variable as a design variable for Flume
            self.flume_sys.design_vars_info[var]["instance"].declare_design_vars(
                variables=[local_name]
            )

            # Map any arays of size 1 to floats for Flume compatibility
            if isinstance(x_val, np.ndarray) and x_val.size == 1:
                x_val = float(x_val.item())

            # Set the variable value for the Flume analysis object
            self.flume_sys.design_vars_info[var]["instance"].set_var_values(
                variables={local_name: x_val}
            )

        return

    def _objconfun(self, xdict: dict):
        """
        DOCS:
        """

        # Call the update function, if it was provided
        if self.update is not None:
            self.update(it_num=self.it_counter)

        # If the Flume system does not have an FOI attribute, set it
        if not hasattr(self.flume_sys, "foi"):
            self.flume_sys.declare_foi(global_foi_name=[])

        # Check to make sure that the objective analysis info has been set, otherwise raise an error
        if not hasattr(self.flume_sys, "obj_analysis"):
            raise RuntimeError(
                f"The objective information for the system named '{self.flume_sys.sys_name}' has not yet been declared, so _objconfun can not be executed. Ensure that the function 'declare_objective' has been called."
            )

        # Set the variable values for the various analyses
        self._set_system_variables(xdict, self.it_counter)

        # Perform the analysis for the objective function
        self.flume_sys.obj_analysis.analyze(debug_print=False)

        # Extract the objective function output
        self.obj_name = self.flume_sys.obj_local_name
        obj = (
            self.flume_sys.obj_analysis.outputs[self.obj_name].value
            * self.flume_sys.obj_scale
        )

        # Store the objective function value in the funcs dictionary
        funcs = {}
        funcs[self.flume_sys.obj_local_name] = obj * self.flume_sys.obj_scale

        # Evaluate the constraint functions
        for con in self.flume_sys.con_info:
            # Perform the analysis for the current constraint function
            self.flume_sys.con_info[con]["instance"].analyze(debug_print=False)

            # Extract the output for the constraint
            con_name = self.flume_sys.con_info[con]["local_name"]

            con_val = self.flume_sys.con_info[con]["instance"].outputs[con_name].value

            # Store the constraint value in the funcs dictionary
            funcs[con_name] = con_val

        # Call the logger function
        self.flume_sys.log_information(iter_number=self.it_counter)

        # Set the failure flag
        fail = False

        # Update the iteration counter
        self.it_counter += 1

        return funcs, fail

    def _objconGradFun(self, xdict: dict):
        """
        DOCS:
        """

        return

    def _addVarGroups(self, x0dict: dict, optProb: Optimization):
        """
        DOCS:
        """

        # Add the design variables to the problem
        for var in self.flume_sys.design_vars_info:

            # Extract the local name for the current variable
            local_name = self.flume_sys.design_vars_info[var]["local_name"]

            # Extract the lower bound value, if it exists
            if "lb" in self.flume_sys.design_vars_info[var]:
                lb = self.flume_sys.design_vars_info[var]["lb"]
            else:
                lb = -1e30

            # Extract the upper bound value, if it exists
            if "ub" in self.flume_sys.design_vars_info[var]:
                ub = self.flume_sys.design_vars_info[var]["ub"]
            else:
                ub = 1e30

            # Extract the starting value
            x0 = x0dict[var]

            # Get the number of design variables in the group
            if isinstance(x0, np.ndarray):
                var_size = x0.size
            else:
                var_size = 1

            # Add the variable group to the pyOptSparse problem
            optProb.addVarGroup(
                name=var,
                nVars=var_size,
                varType="c",
                value=x0,
                lower=np.ones_like(x0) * lb,
                upper=np.ones_like(x0) * ub,
            )

        return

    def _addConGroups(self, optProb: Optimization):
        """
        DOCS:
        """

        # Loop through each constraint in the Flume System instance
        for con in self.flume_sys.con_info:
            # Extract the local name of the constraint
            con_name = self.flume_sys.con_info[con]["local_name"]

            # Perform the analysis to get the size for the constraint
            self.flume_sys.con_info[con]["instance"].analyze(debug_print=False)

            con_val = self.flume_sys.con_info[con]["instance"].outputs[con_name].value
            if isinstance(con_val, np.ndarray):
                con_size = con_val.size
            else:
                con_size = 1

            # Extract the direction of the constraint
            direction = self.flume_sys.con_info[con]["direction"]
            rhs_val = self.flume_sys.con_info[con]["rhs"]

            # Greater than inequality constraints
            if direction == "geq":
                # If rhs = 0.0...
                if rhs_val == 0.0:
                    optProb.addConGroup(
                        name=con_name, nCon=con_size, lower=0.0, upper=None
                    )
                # If rhs != 0.0...
                else:
                    optProb.addConGroup(
                        name=con_name,
                        nCon=con_size,
                        lower=1.0,
                        upper=None,
                        scale=rhs_val,
                    )
            # Less than inequality constraints
            elif direction == "leq":
                # If rhs = 0.0...
                if rhs_val == 0.0:
                    optProb.addConGroup(
                        name=con_name, nCon=con_size, lower=None, upper=0.0
                    )
                # If rhs != 0.0...
                else:
                    optProb.addConGroup(
                        name=con_name,
                        nCon=con_size,
                        lower=None,
                        upper=1.0,
                        scale=rhs_val,
                    )
            # Equality constraints
            elif direction == "both":
                # If rhs = 0.0...
                if rhs_val == 0.0:
                    optProb.addConGroup(
                        name=con_name, nCon=con_size, lower=0.0, upper=0.0
                    )
                # If rhs != 0.0...
                else:
                    optProb.addConGroup(
                        name=con_name,
                        nCon=con_size,
                        lower=1.0,
                        upper=1.0,
                        scale=rhs_val,
                    )

            else:
                raise RuntimeError(
                    "Constraint direction must be 'geq', 'leq', or 'both'."
                )

    def _construct_optimization_problem(
        self, opt_prob_name: str, x0dict: dict, optimizer: str, options: dict = None
    ) -> tuple[Optimizer, Optimization]:
        """
        DOCS:
        """

        # Construct the Optimization problem
        optProb = Optimization(
            name=opt_prob_name, objFun=self._objconfun, sens=self._objconGradFun
        )

        # Add the design variable groups to the problem
        self._addVarGroups(x0dict=x0dict, optProb=optProb)

        # Add the constraints to the problem
        self._addConGroups(optProb=optProb)

        # Add the objective to the problem
        obj_name = self.flume_sys.obj_local_name
        optProb.addObj(name=obj_name)

        # Set the optimizer and optimizer options
        if optimizer.lower() == "SLSQP".lower():
            from pyoptsparse import SLSQP

            opt = SLSQP(options=options)
        elif optimizer.lower() == "PSQP".lower():
            from pyoptsparse import PSQP

            opt = PSQP(options=options)
        elif optimizer.lower() == "NSGA2".lower():
            from pyoptsparse import NSGA2

            opt = NSGA2(options=options)
        elif optimizer.lower() == "SNOPT".lower():
            from pyoptsparse import SNOPT

            opt = SNOPT(options=options)
        else:
            raise NotImplementedError(
                f"Optimizer by the name of '{optimizer}' not yet implemented."
            )

        return opt, optProb

    def optimize_system(
        self,
        x0dict: dict,
        opt_prob_name: str,
        optimizer: str = "SLSQP",
        options: dict = None,
        output_dir: str = ".",
        # TODO: determine if an option to store History file should go here
    ) -> Solution:
        """
        DOCS:
        """

        # ic(optimizer)

        # Get default options if user did not provide them directly
        # if options is None:
        #     options = self._get_default_options(optimizer=optimizer)

        # Construct the optimization problem
        opt, optProb = self._construct_optimization_problem(
            opt_prob_name=opt_prob_name,
            x0dict=x0dict,
            optimizer=optimizer,
            options=options,
        )

        # Update the options with the output directory for the Flume system
        if options is None:
            self._update_options_output_directory(optimizer, opt)

        # Perform the optimization
        sol = opt(
            optProb, sens="FD"
        )  # TODO: need to set this to be the sensitivity function, not FD

        # Return the solution
        return sol

    def _update_options_output_directory(self, optimizer: str, opt: Optimizer):
        """
        DOCS:
        """

        # Update the output file names with the Flume output directory for each optimizer
        if optimizer.lower() == "slsqp":
            options = opt.defaultOptions
            options["IFILE"] = os.path.join(self.flume_sys.log_prefix, "SLSQP.out")
        elif optimizer.lower() == "psqp":
            options = opt.getOptions()
            options["IFILE"] = os.path.join(self.flume_sys.log_prefix, "PSQP.out")
        elif optimizer.lower() == "nsga2":
            options = opt.getOptions()
            options["PrintOut"] = (
                0  # note that there is no way to specify where the output files go besides the current directory, so turning them off entirely by default
            )
        elif optimizer.lower() == "snopt":
            options = opt.getOptions()
            options["Print file"] = os.path.join(
                self.flume_sys.log_prefix, "SNOPT_print.out"
            )
            options["Summary file"] = os.path.join(
                self.flume_sys.log_prefix, "SNOPT_summary.out"
            )
        else:
            raise NotImplementedError(
                f"Have not implemented default options for optimizer '{optimizer}' yet."
            )

        # Set the updated options
        opt.options = options

        return
