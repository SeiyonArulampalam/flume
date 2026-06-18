from flume.base_classes.analysis import Analysis
import graphviz as gv
from icecream import ic
from typing import List
from flume.interfaces.utils import Logger
import os
import numpy as np


class System:
    """
    This class is used to wrap multiple Analysis objects into a system, which can then be utilized to perform optimization using one of Flume's optimizer interfaces.
    """

    def __init__(
        self,
        sys_name: str,
        top_level_analysis_list: List[Analysis],
        log_name: str = "flume.log",
        log_prefix: str = ".",
    ):
        """
        Defines a class that wraps multiple analysis objects into a single system, which can then be utilized to perform optimization with one of Flume's optimizer interfaces after declaring design varaibles, an objective function, and (optionally) constraints.

        Parameters
        ----------
        sys_name : str
            Name that is assigned to the System
        top_level_analysis_list: list
            A list of instances of Analysis objects that will be utilized within the optimization problem. Here, the objects that should be provided are the analyses that define the objective function and any constraint functions so that these can be declared for the optimization formulation. Other Analysis classes, such as those used as sub-analyses, do not need to be provided here, as they should be provided when creating the objects in the top-level analysis list
        log_name : str
            String that defines the name to use for the log file, defaults to 'flume.log'
        log_prefix: str
            String that defines the output directory where the log file and other files are saved, defaults to the current directory '.'
        """

        # Store the name for the system
        self.sys_name = sys_name

        # Store the list of analyses
        self.top_level_analysis_list = top_level_analysis_list

        # Store the information for the logging
        self.log_name = log_name
        self.log_prefix = log_prefix

        if not os.path.isdir(self.log_prefix):
            os.mkdir(self.log_prefix)

        # Configure the file path for the log file
        self.outputs_log = Logger(log_path=self.log_prefix, log_name=self.log_name)
        self.profile_log = Logger(log_path=self.log_prefix, log_name="profile.log")

        # Assemble the list of all analysis objects
        self.full_analysis_list = self.assemble_full_analysis_list()

        # Initialize the constraints dictionary (default assumes no constraints)
        self.con_info = {}

        return

    def reset_analysis_flags(self):
        """
        Resets all of the analysis flags for all analyses within the system to be False. This is done when variable values are updated, as all systems must be analyzed again to propagate changes in design variable values.
        """

        # Loop through each top-level analysis, resetting each analysis in the stack
        for top_level in self.top_level_analysis_list:

            # If the stack does not exist, make it
            if not hasattr(top_level, "stack"):
                # Assemble the stack
                top_level.stack = top_level._make_stack()

            # Loop through each object within the current top-level's analysis stack and set the analyzed attribute to be False
            for analysis in top_level.stack:
                analysis.analyzed = False

        return

    def assemble_full_analysis_list(self):
        """
        Assembles the full list of analyses that comprise the overall system architecture.
        """

        # Initialize the analysis list
        full_analysis_list = []

        # Loop through each top-level analysis, assemble the stack, and append the analyses to the total analysis list
        for top_level in self.top_level_analysis_list:
            # Assemble the stack
            stack = top_level._make_stack()

            # Add the entries in the current stack to the list if they are not there already
            for analysis in stack:
                if analysis not in full_analysis_list:
                    full_analysis_list.append(analysis)
                else:
                    continue

        return full_analysis_list

    def graph_network(
        self,
        filename: str = None,
        output_directory: str = None,
        interactive: bool = False,
        format: str = "pdf",
    ):
        """
        Construct the visualization of the network associated with the Flume system using graphviz.

        Parameters
        ----------
        filename : str
            Name to use for the file that is created
        output_directory : str
            String that defines the directory where the file should be saved
        interactive : bool
            Boolean value that indicates whether the graph should be output in interactive mode. *This is an experimental feature at the moment*
        """

        # Create the graph, according to the interactive boolean argument
        if interactive:
            # Make the graph with graphviz in interactive mode
            graph = self._static_graph_network()

            # Render as an svg for interactive graph
            int_filename = filename + "_interactive"
            graph.render(
                filename=int_filename,
                directory=output_directory,
                format="svg",
                cleanup=True,
            )

            # Edit the SVG file to enable interactive features
            svg_filepath = output_directory + "/" + int_filename + ".svg"
            # ic(svg_filepath)

            # FIXME: Rewriting this adds ns0 to the output, need to fix
            self._enable_interactive_graph(svg_filepath=svg_filepath)

            # Embed the interactive svg into an HTML with the interactive capabilities
            self._create_interactive_html(
                output_directory=output_directory, svg_filepath=svg_filepath
            )

        else:
            # Make the graph with graphviz
            graph = self._static_graph_network()

            # Render the graph
            graph.render(
                filename=filename,
                directory=output_directory,
                cleanup=True,
                format=format,
            )

        return graph

    def _static_graph_network(self):
        """
        Private method that is used to greate the graphviz visual in static form, which is ultimately returned by this method.
        """

        # graph = nx.Graph()
        graph = gv.Digraph(
            name=f"{self.sys_name.upper()}",
            graph_attr={"rankdir": "LR", "ranksep": "0.7"},
            node_attr={"shape": "box", "fontname": "Helvetica"},
        )

        # Initialize an empty list to store the systems added to the graph as nodes
        self.nodes = []
        self.edges = {}

        # Loop through and add nodes to the graph (outer loop is top-level analyses, inner loop is individual sub-analyses)
        for analysis in self.top_level_analysis_list:
            # Check if the object has a stack already, otherwise assemble
            if hasattr(analysis, "stack"):
                stack = analysis.stack
                pass
            else:
                stack = analysis._make_stack()

            # Check if the object has been connected already, otherwise perform the analysis to establish connections map
            if analysis.connected:
                pass
            else:
                analysis.analyze()

            # Loop through sub analyses in the current stack and add the nodes if they do not already exist
            for i, sub in enumerate(stack):
                # Add node if it is not in the system network already
                if sub not in self.nodes:
                    self.nodes.append(sub)

                    # Set the color depending on whether the analysis is top-level
                    if sub in self.top_level_analysis_list:
                        # Extract the names of the outputs
                        out_labels = list(sub.outputs.keys())
                        out_str = ", ".join(out_labels)

                        # Add the node
                        graph.node(
                            sub.obj_name,
                            f"{sub.obj_name}\nOutputs: {out_str}",
                            color="red",
                        )
                    else:
                        graph.node(sub.obj_name, f"{sub.obj_name}")
                else:
                    pass

                # Add edge if not the first entry in the stack
                if i == 0:
                    pass
                else:
                    # Extract the States that are connected between the objects, if there are any connections
                    if hasattr(sub, "connects"):
                        connect_labels = list(sub.connects.keys())

                        # Loop through the keys in the connections dictionary
                        for out in connect_labels:

                            # Add the entry to the edges dictionary for the current output, if necessary
                            if out not in self.edges:
                                self.edges[out] = []

                            # Check if the edge already exists in the edges dictionary
                            if (sub.connects[out].obj_name, sub.obj_name) in self.edges[
                                out
                            ]:
                                pass
                            else:
                                # Add the edge label to the edges dictionary
                                self.edges[out].append(
                                    (sub.connects[out].obj_name, sub.obj_name)
                                )

                                # Add the edge to the graph and the connection label
                                graph.edge(
                                    sub.connects[out].obj_name,
                                    sub.obj_name,
                                    label=f"{out}",
                                )

        return graph

    def _enable_interactive_graph(self, svg_filepath):
        """
        Privat method that is used to enable interactive capabilities with the file provided with svg_filepath

        Parameters
        ----------
        svg_filepath : str
            String that provides a filepath to an SVG file, which will be modified to make it interactive
        """

        # Import xml
        import xml.etree.ElementTree as ET

        # Parse the graph
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.parse(svg_filepath)
        root = tree.getroot()

        # Set the id for the svg
        root.set("id", "my-svg")

        # Find node groups and edit
        for g in root.iter("{http://www.w3.org/2000/svg}g"):
            # Get the class type for the attribute
            attrib_class = g.attrib["class"]
            # print(attrib_class)

            # Replace the classes for graphs/nodes
            if attrib_class == "node":
                g.set("class", "graph-node")

            # title = g.find("{http://www.w3.org/2000/svg}title")
            # print(g.attrib)
            # g.get()
            # print(g)
            # print(title.text)

        # Rewrite the file
        tree.write(svg_filepath)

        return

    def _create_interactive_html(self, output_directory, svg_filepath):
        """
        Privat emethod that creates an interactive HTML using the file located at output_directory/svg_filepath

        Parameters
        ----------
        output_directory : str
            String that specifies the location of the output directory for the HTML
        svg_filepath : str
            String that specifies the location of the SVG file that is to be converted to interactive mode
        """

        from flume.base_classes.system_html import _write_html_file

        _write_html_file(output_directory=output_directory, svg_filepath=svg_filepath)

        return

    def _find_analysis_object(self, instance_name, var_name):
        """
        Searches through the full analysis list for the system and finds the instance of the analysis object associated with the argument for the instance_name.
        """

        # Loop through the full analysis list
        for analysis in self.full_analysis_list:
            # Check if the current object name matches the provided instance name and return if so
            if analysis.obj_name == instance_name:
                return analysis

        # Raise a RuntimeError if this point is reached, as the instance name did not find a match
        raise RuntimeError(
            f"No instance found for object named '{instance_name}'! Verify the definition for the State named '{instance_name}.{var_name}'"
        )

    def declare_objective(self, global_obj_name, obj_scale=1.0):
        """
        Sets the objective function for the optimization problem according to the provided global output name. This output should be associated with one of the top-level analyses for the system (i.e. included in the top_level_analysis_list).

        Parameters
        ----------
        global_obj_name : str
            A string that specifies the global name for the value that is to be used as the objective function (global name meaning 'object_name.local_output_name')
        obj_scale : float
            Float value that is *multiplied* by the value of the objective function, which should scale the objective function value to O(1). This is used by the optimizer interfaces internally to scale the objective.
        """

        # Using the provided objective name, store the associated analysis object and the local variable name
        obj_analysis_name, self.obj_local_name = global_obj_name.split(".")

        # Store the objective scale
        self.obj_scale = obj_scale

        # Find the instance for the objective analysis object
        self.obj_analysis = self._find_analysis_object(
            instance_name=obj_analysis_name, var_name=self.obj_local_name
        )

        return

    def declare_constraints(self, global_con_name: dict):
        """
        Sets the constraints for the optimization problem according to the provided global output names.

        Parameters
        ----------
        global_con_name_dict : dict
            This is a dictionary of dictionaries. The keys correspond to the global output names for the constraints. The inner dictionary specifies additional information about the structure of the constriant, including the following:

            * 'rhs' (float) - specifies the right hand side of the constraint equation. Internally, the optimizer interfaces will use this to convert the constraint to an equivalent form that normalizes it, If the 'rhs' value is 0.0, no scaling is applied
            * 'direction' (str) - string that is either 'geq' (>=), 'leq' (<=), or 'both' (=). Defaults to 'geq' in the event that a direction is not provided.

        Example
        -------

        Here, the constraints are defined as follows:
            x <= 1.0
            y >= 1.0
            z = 2.0

        Thus, the argument here is given as: global_con_name = {
                                                    "block1.x":{"direction":"leq", "rhs":1.0},
                                                    "block2.y":{"rhs":1.0}
                                                    "block3.z":{"direction": "both", "rhs": 2.0}
                                                    }
        """

        # Loop through the keys in the dictionary and add them to the constraints for the system
        for key in global_con_name.keys():
            # Add the key to the con_info dictionary
            self.con_info[key] = {}

            # Split the string
            con_analysis_name, con_local_name = key.split(".")

            # Find the analysis object for the current constraint
            con_analysis = self._find_analysis_object(
                con_analysis_name, var_name=con_local_name
            )

            # Add the analysis object and local constraint name for the con_info dictionary
            self.con_info[key]["instance"] = con_analysis
            self.con_info[key]["local_name"] = con_local_name
            self.con_info[key]["rhs"] = global_con_name[key]["rhs"]

            if "direction" not in global_con_name[key].keys():
                self.con_info[key]["direction"] = "geq"
            else:
                if global_con_name[key]["direction"] not in ["geq", "leq", "both"]:
                    raise RuntimeError(
                        f"The value for 'direction' for the constraint {key} must be 'geq', 'leq', or 'both' and not {self.con_info[key]['direction']}"
                    )
                else:
                    self.con_info[key]["direction"] = global_con_name[key]["direction"]

        return

    def declare_design_vars(self, global_var_name: dict):
        """
        Sets the design variables for the system according to the provided global variable names.

        Parameters
        ----------
        global_var_name : dict
            This is a dictionary of dictionaries. Keys for the first dictionary correspond to the global variable names that should be added to the set of design variables, and the values contain information about the variable bounds. If the inner dictionary is empty, then no bounds are provided. Otherwise, the values for the variables lower bound 'lb' and upper bound 'ub' will be added, if included.

        Example
        -------

        Here, the bounds are defined as follows:
            1.0 <= x <= 2.0
            y has no bounds
            0.0 <= z

        Thus, the argument here is: global_var_name = {"block1.x":{"lb":1.0, "lb":2.0}, "block2.y":{}, "block3.z":{"lb":0.0}}
        """

        # Initialize the design variables dictionary
        self.design_vars_info = {}

        # Loop through the keys in the dictionary and add them to the design variables for the system
        for key in global_var_name:
            # Add the key to the design_vars_info dictionary
            self.design_vars_info[key] = {}

            # Split the string for the current variable
            var_analysis_name, var_local_name = key.split(".")

            # Find the analysis object instance for the current design variable
            var_analysis = self._find_analysis_object(var_analysis_name, var_local_name)

            # Add the analysis object and local variable name to the dictionary
            self.design_vars_info[key]["instance"] = var_analysis
            self.design_vars_info[key]["local_name"] = var_local_name

            # Add the bounds, if they were specified
            if global_var_name[
                key
            ]:  # This is evaluated as True if the variable has at leaast one bound specified

                # Add the lower bound, if specified
                if "lb" in global_var_name[key].keys():
                    self.design_vars_info[key]["lb"] = global_var_name[key]["lb"]

                # Add the upper bound, if specified
                if "ub" in global_var_name[key].keys():
                    self.design_vars_info[key]["ub"] = global_var_name[key]["ub"]

                # Add the scale, if specified
                if "scale" in global_var_name[key].keys():
                    self.design_vars_info[key]["scale"] = global_var_name[key]["scale"]

            else:
                continue

        return

    def declare_foi(self, global_foi_name: list):
        """
        Sets the functions of interest that are to be tracked by the logger at each iteration. Here, the names must be provided according to their global names. If already declared, the objective and constraints are added by default, but other outputs can be included.
        """

        # Initialize the foi dictionary
        foi = {}

        # Add the objective if it has already been declared
        if hasattr(self, "obj_analysis"):
            foi["obj"] = {}

            foi["obj"]["instance"] = self.obj_analysis
            foi["obj"]["local_name"] = self.obj_local_name

        else:
            raise RuntimeError("No objective has been declared!")

        # Add the constraints if it has already been declared
        if hasattr(self, "con_info"):
            foi["cons"] = {}

            # For each constraint in self.con_info, add it to the foi dictionary
            for key in self.con_info:
                foi["cons"][key] = {}

                foi["cons"][key]["instance"] = self.con_info[key]["instance"]
                foi["cons"][key]["local_name"] = self.con_info[key]["local_name"]

        # Add the names in the global_foi_name input list to the foi dictionary
        foi["other"] = {}

        for name in global_foi_name:
            # Split the name for the global foi
            foi_analysis_name, foi_local_name = name.split(".")

            # Find the analysis object associated with the global_foi_name
            foi_analysis = self._find_analysis_object(
                instance_name=foi_analysis_name, var_name=foi_local_name
            )

            # Add the analysis object and local state name to the foi dictionary
            foi["other"][name] = {}
            foi["other"][name]["instance"] = foi_analysis
            foi["other"][name]["local_name"] = foi_local_name

        # Store the foi dictionary
        self.foi = foi

        return

    def log_information(self, iter_number):
        """
        Helper function that is used to log the values for the objective function, constraints, and other functions of interest at each iteration. Internally, this will update the log file for the System with this information at every iteration.

        Parameters
        ----------
        iter_number : int
            Current iteration number
        """

        # Check that the system has an FOI attribute, otherwise generate it (only needed if the user does not declare additional FOI to track)
        if not hasattr(self, "foi"):
            self.declare_foi(global_foi_name=[])

        # Log the header names if the current iter number is divisible by 10
        if iter_number % 10 == 0:
            # Log the header for the iter number and objective
            obj_header = f"obj: {self.obj_local_name}"
            self.outputs_log.log("\n%5s%20s" % ("iter", obj_header), end="")

            # Log the constraints
            for con in self.foi["cons"].keys():
                con_val = (
                    self.foi["cons"][con]["instance"]
                    .outputs[self.foi["cons"][con]["local_name"]]
                    .value
                )

                if isinstance(con_val, np.ndarray):
                    for i in range(con_val.size):
                        con_header = f"con: {self.foi['cons'][con]['local_name']}[{i}]"
                        self.outputs_log.log("%20s" % con_header, end="")

                else:
                    con_header = f"con: {self.foi['cons'][con]['local_name']}"
                    self.outputs_log.log("%20s" % con_header, end="")

            # Log the other functions of interest
            for other in self.foi["other"].keys():
                other_val = (
                    self.foi["other"][other]["instance"]
                    .outputs[self.foi["other"][other]["local_name"]]
                    .value
                )

                if isinstance(other_val, np.ndarray):
                    for i in range(other_val.size):
                        other_header = (
                            f"other: {self.foi['other'][other]['local_name']}[{i}]"
                        )
                        self.outputs_log.log("%20s" % other_header, end="")

                else:
                    other_header = f"other: {self.foi['other'][other]['local_name']}"
                self.outputs_log.log("%20s" % other_header, end="")

        # Log the values for the current iteration and objective value
        obj_val = (
            self.foi["obj"]["instance"].outputs[self.foi["obj"]["local_name"]].value
        )

        self.outputs_log.log("\n%5d%20.10e" % (iter_number, obj_val), end="")

        # Log the values for the constraints at the current iter
        for con in self.foi["cons"].keys():
            con_val = (
                self.foi["cons"][con]["instance"]
                .outputs[self.foi["cons"][con]["local_name"]]
                .value
            )

            if isinstance(con_val, np.ndarray):
                for i in range(con_val.size):
                    if not isinstance(con_val[i], str):
                        con_val_i = "%20.10e" % con_val[i]
                    self.outputs_log.log("%20s" % con_val_i, end="")

            else:
                if not isinstance(con_val, str):
                    con_val = "%20.10e" % con_val

                self.outputs_log.log("%20s" % con_val, end="")

        # Log the values for the other FOI
        for other in self.foi["other"].keys():
            other_val = (
                self.foi["other"][other]["instance"]
                .outputs[self.foi["other"][other]["local_name"]]
                .value
            )

            if isinstance(other_val, np.ndarray):
                for i in range(other_val.size):
                    if not isinstance(other_val[i], str):
                        other_val_i = "%20.10e" % other_val[i]
                    self.outputs_log.log("%20s" % other_val_i, end="")

            else:
                if not isinstance(other_val, str):
                    other_val = "%20.10e" % other_val

                self.outputs_log.log("%20s" % other_val, end="")

            # if not isinstance(other_val, str):
            #     other_val = "%20.10e" % other_val

            # self.outputs_log.log("%20s" % other_val, end="")

        return

    def profile_iteration(self, iter_number):
        """
        Helper function that is used to display the time taken for each analyze and analyze_adjoint method at the current iteration. Internally, this updates a profile log file that stores the timing information for the System at each iteration.

        Parameters
        ----------
        iter_number : int
            Current iteration number
        """

        # Log the analysis object names if the current iteration number is divisible by 10
        if iter_number % 10 == 0:
            # Log the header for the iter number and each analysis object name in the stack
            self.profile_log.log("\n%5s" % ("iter"), end="")

            for analysis in self.top_level_analysis_list:
                self.profile_log.log(
                    "%20s %20s"
                    % (analysis.obj_name + ": fwd", analysis.obj_name + ": adj"),
                    end="",
                )

        self.profile_log.log("\n%5d" % iter_number, end="")

        for analysis in self.top_level_analysis_list:
            self.profile_log.log(
                "%20.6f %20.6f" % (analysis.forward_total, analysis.adjoint_total),
                end="",
            )

        return
