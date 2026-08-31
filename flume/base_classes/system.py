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
        parallel_execution: bool = False,
        parallel_max_workers: int = 4,
        track_node_timing: bool = False,
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
        TODO: parallel exeuction
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

        # Store the attribute for whether parallel execution should be used
        self.parallel_execution = parallel_execution
        self.parallel_max_workers = parallel_max_workers

        # Store the attribute for whether per-node timing should be tracked. When
        # True, each node's individual analysis time is measured/stored (enabling
        # the summed-CPU and efficiency profile columns and the optional per-node
        # detail). When False, only the System-level wall-clock time is recorded.
        self.track_node_timing = track_node_timing

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

    def build_dag(self):
        """
        DOCS:
        """

        from collections import defaultdict

        # Initialize a set to store all of the unique Analysis objects in the DAG
        nodes = set()

        # Initialize a dictionary of lists, which will store all of the dependencies for each Analysis object
        dependents = defaultdict(list)

        # Initialize a dictionary, which will store information about how many dependencies remain for each node in the DAG. Used as a trigger for when an Analysis object can begin its execution (after all dependencies have been analyzed)
        remaining = {}

        # Define the local function to use to trace through the stack and build the DAG
        def visit(n):
            # Skip the node if it is already in the set of nodes
            if n in nodes:
                return

            # Add the node to the set
            nodes.add(n)

            # Set the number of dependencies for the current node in the 'remaining' dictionary
            remaining[n] = len(n.sub_analyses)

            # Loop through the sub-analyses for the current node, add them to the dependents dictionary, and visit each sub-analysis
            for sub in n.sub_analyses:
                dependents[sub].append(n)
                visit(sub)

        # Invoke the 'visit' function for all top-level Analysis objects (nodes) to construct the DAG
        for t in self.top_level_analysis_list:
            visit(t)

        # Store the information as attributes
        self.dag_nodes = nodes
        self.dag_dependents = dependents
        self.dag_remaining = remaining

        # Map each top-level analysis (i.e. a sink with no dependent Analyses) to its ancestor set (i.e. its dependencies )
        self.dag_sink_ancestors = {}
        for sink in self.top_level_analysis_list:
            # Define a set for the Analysis objects seen in the adjoint pass
            seen = set()

            # Append the sink (i.e. the top-level Analysis object) to the stack
            stack = [sink]

            while stack:
                # Pop the node off the stack
                n = stack.pop()

                # If it is already in the set of seen objects, continue
                if n in seen:
                    continue

                # Otherwise, add the node to the set
                seen.add(n)

                # Extend the stack by adding its sub-analyses into the stack
                stack.extend(n.sub_analyses)

            # Assign the ancestors for the current sink as all of the distinct Analysis objects in the "seen" set
            self.dag_sink_ancestors[sink] = seen

        return

    def execute(self, mode: str = "real", debug_print: bool = False):
        """
        Executes all top-level Analysis objects for the System. Operates in serial or in parallel, depending on the input parameter.
        """
        import time

        # Start the wall-clock timer for the entire forward pass (always measured)
        wall_start = time.perf_counter()

        # Get the boolean attribute for whether parallel execution should be used
        parallel_execution = self.parallel_execution

        # Whether to track per-node (and summed-CPU) timing
        track = self.track_node_timing

        # Summed per-node CPU time for the forward pass. Only meaningful when
        # per-node timing is tracked; otherwise left as NaN.
        self.dag_total_time = 0.0 if track else float("nan")

        # Serial path for the System, which sequentially executes the objective and all constraint Analysis objects
        if not parallel_execution:
            # Perform the analysis for the objective function
            self.obj_analysis.analyze(mode=mode, debug_print=debug_print)

            if track:
                self.dag_total_time += self.obj_analysis.forward_total

            # Perform the analysis for all constraint functions
            for con in self.con_info:
                con_instance = self.con_info[con]["instance"]
                con_instance.analyze(debug_print=debug_print)

                if track:
                    self.dag_total_time += con_instance.forward_total

        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Build the information that defines the DAG if it does not already exist
            if not hasattr(self, "dag_nodes"):
                self.build_dag()

            # Get the nodes and dependents attributes (static across calls)
            nodes = self.dag_nodes
            dependents = self.dag_dependents

            # Create a copy of the dag_remaining dictionary for use in the execution. The original is preserved so it can be accessed for subsequent executions (the copy is modified by decrementing the counters)
            remaining = self.dag_remaining.copy()

            # Get the maximum number of workers to use
            max_parallel_workers = self.parallel_max_workers

            # Define the local function that defines the procedures to execute when an Analysis object/node is to be analyzed
            def perform_node_analysis(analysis_obj: Analysis):

                # Construct/get the connections
                analysis_obj._connect()

                # Call the private analyze method, timing it per-node only if requested
                if track:
                    start = time.perf_counter()
                    analysis_obj._analyze()
                    analysis_obj.analyzed = True
                    end = time.perf_counter()

                    # Store the time it took to perform this Analysis
                    analysis_obj.analysis_time = end - start
                else:
                    start = end = None
                    analysis_obj._analyze()
                    analysis_obj.analyzed = True

                if debug_print:
                    import threading

                    tname = threading.current_thread().name
                    tid = threading.get_ident()

                    if track:
                        print(
                            f"[{tname} tid={tid}] {analysis_obj.obj_name}: "
                            f"start={start:.6f} end={end:.6f}"
                        )
                    else:
                        print(f"[{tname} tid={tid}] {analysis_obj.obj_name}")

            # Serially initialize all Analysis objects
            for n in nodes:
                n._initialize_analysis(mode=mode)

            # Construct the initial set of Analysis objects that are ready to be executed (i.e. anything that is a source for the DAG)
            initial_ready = [n for n in remaining if remaining[n] == 0]

            # Construct the ThreadPoolExecutor and execute all of the Analyses in the System
            with ThreadPoolExecutor(max_workers=max_parallel_workers) as pool:
                # Get the Future objects for the nodes that are ready to be evaluated (also starts the Analysis execution with the submit method)
                futures = {
                    pool.submit(perform_node_analysis, n): n for n in initial_ready
                }

                # Execute all Analysis objects until the entire DAG is complete
                while futures:
                    # Loop over the list of Future objects as they complete
                    for future in as_completed(list(futures)):
                        # Extract the node
                        node = futures.pop(future)

                        # Extract the data from the future (triggers the end of the perform_node_analysis)
                        future.result()

                        # Loop through the dependents for the current node
                        for dep in dependents[node]:
                            # Decrement the counter for the dependent
                            remaining[dep] -= 1

                            # Trigger the execution of the next Analysis object if all of its dependent nodes have finished executing
                            if remaining[dep] == 0:
                                futures[pool.submit(perform_node_analysis, dep)] = dep

                        break

            # Sum the per-node forward times for the DAG (only if tracking is on)
            if track:
                for n in nodes:
                    self.dag_total_time += n.analysis_time

        # Record the wall-clock time for the entire forward pass
        self.dag_forward_wall = time.perf_counter() - wall_start

        # Display the timing for the forward pass through the DAG, if requested
        if debug_print:
            print(
                f"DAG forward pass: wall={self.dag_forward_wall:.6f} s, "
                f"cpu-sum={self.dag_total_time} s"
            )

        return

    def _collect_sinks_info(self):
        """
        DOCS:
        """

        # Initialize the list that is returned. Each entry in the list is a tuple that contains (sweep_id, sink_analysis_obj, output_names, seed)
        sinks_info = []

        # Define the dictionary, which specifies the reverse mapping between the sweep ID and the sink analysis object
        sinks_of_info = {}

        # Get the sink information for the objective
        obj_sweep_id = (self.global_obj_name, 0)
        obj_sink_tuple = (
            obj_sweep_id,  # sweep ID (global_obj_name, index)
            self.obj_analysis,  # objective Analysis object
            self.obj_local_name,  # objective local name
            1.0,  # seed value
        )
        sinks_info.append(obj_sink_tuple)

        sinks_of_info[obj_sweep_id] = self.obj_analysis

        # Loop through each object in the top-level Analysis list, which corresponds to all sinks in the DAG
        for global_con_key in self.con_info:
            # Extract the information that goes into the tuple for the current sink
            instance = self.con_info[global_con_key]["instance"]
            out_local_name = self.con_info[global_con_key]["local_name"]

            # Get the output value and size, which is used for setting seed information
            con_val = instance.outputs[out_local_name].value

            # Set the seed info and sweep ID name depending on whether the constraint is a scalar or vector
            if isinstance(con_val, np.ndarray):
                # Loop over each entry in the constraint vector and populate info
                for i in range(con_val.size):
                    # Initialize a zero vector that is the shape of the con_val
                    seed = np.zeros(con_val.shape)

                    # Set the ith entry to 1
                    seed.flat[i] = 1.0

                    # Set the name
                    sweep_id_name = (global_con_key, i)

                    # Set the tuple for the current constraint
                    sinks_info.append((sweep_id_name, instance, out_local_name, seed))

                    # Add the entry to the dictionary which maps sweep ID to analysis object
                    sinks_of_info[sweep_id_name] = instance
            else:
                # Constraint is a scalar
                seed = 1.0
                sweep_id_name = (global_con_key, 0)

                # Set the tuple for the current constraint
                sinks_info.append((sweep_id_name, instance, out_local_name, seed))

                # Add the entry to the dictionary which maps sweep ID to analysis object
                sinks_of_info[sweep_id_name] = instance

        return sinks_info, sinks_of_info

    def _extract_dv_derivs(self, sweep_id=None):
        """
        Helper method which is used to extract the design derivatives for the current adjoint execution.
        """

        # Initialize the dictionary that will store the design derivatives for the current sweep ID value
        sweep_id_derivs = {}

        # Loop through the design variables, and extract the derivative value for the input sweep ID
        for var in self.design_vars_info:
            # Get the local name for the current variable
            local_name = self.design_vars_info[var]["local_name"]
            instance = self.design_vars_info[var]["instance"]

            # Extract the derivative value for the current variable
            deriv_val = instance.variables[local_name].get_deriv(sweep_id)

            # Store the derivative value in the dictionary object
            sweep_id_derivs[var] = deriv_val

        return sweep_id_derivs

    def execute_adjoint(self, debug_print: bool = False):
        """
        DOCS:
        """

        # Get the information for the sweeps that are required (i.e. the total number of quantities of interest, one objective and each constraint)
        sinks_info, sinks_of_info = self._collect_sinks_info()

        import time

        # Start the wall-clock timer for the entire adjoint pass (always measured)
        wall_start = time.perf_counter()

        # Whether to track summed-CPU timing
        track = self.track_node_timing

        # Summed adjoint CPU time (only populated where per-node timing is available)
        self.dag_adjoint_time = 0.0 if track else float("nan")

        # Initialize the dictionary that stores the derivatives of interest
        self.design_derivs = {}

        # Extract the parallel execution attribute
        parallel_execution = self.parallel_execution

        # Serial path for the adjoint analysis of the System, which sequentially executes the objective and all constraint Analysis objects (and serially traces through the stack)
        if not parallel_execution:
            # Loop through the sweeps, and execute all of the adjoint paths serially
            for sweep_id, sink_object, output_names, seed in sinks_info:
                # Here, _current_sweep is None, which triggers the original, serial execution
                sink_object._add_output_seed(outputs=[output_names], seed=seed)

                # Call the analzye_adjoint method
                sink_object.analyze_adjoint(debug_print=debug_print)

                # Accumulate the summed adjoint CPU time, if tracking is enabled
                if track:
                    self.dag_adjoint_time += sink_object.adjoint_total

                # Extract the derivatives and store them into the dictionary of design derivatives
                self.design_derivs[sweep_id] = self._extract_dv_derivs(sweep_id=None)

            # Record the wall-clock time for the adjoint pass
            self.dag_adjoint_wall = time.perf_counter() - wall_start

            return
        # Parallel path for the adjoint analysis (parallelizes over the top-level Analysis objects and within each sweep)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from flume.base_classes.state import _current_sweep, sweep_context
            import threading

            # Construct the DAG info if not done already
            if not hasattr(self, "dag_nodes"):
                self.build_dag()

            # Extract the DAG info
            nodes = self.dag_nodes
            dependents = self.dag_dependents
            sink_ancestors = self.dag_sink_ancestors

            # Perform the serial pre-pass to allocate zeroed derivatives for every (State, sweep_id) combo. Also, construct the state locks, which are used to prevent race conditions
            state_locks = {}
            for sweep_id, sink_object, output_names, seed in sinks_info:
                # Loop through each node in a sink's ancestors
                for node in sink_ancestors[sink_object]:
                    # Extract all states for the current node's variables and outputs, and ensure they have derivative info for the current sweep ID
                    for state in list(node.outputs.values()) + list(
                        node.variables.values()
                    ):
                        # Ensures that the State has a zeroed derivative value for the current sweep ID or creates it
                        state.ensure_sweep_slot(sweep_id)

                        # Set the information for the state lock
                        state_locks.setdefault(id(state), threading.Lock())

            # Set each sink's seed in its corresonding slot (done serially)
            for sweep_id, sink_object, output_names, seed in sinks_info:
                # Use the context manager with the current sweep ID to set the output seed for each sweep ID
                with sweep_context(sweep_id):
                    sink_object._add_output_seed(outputs=[output_names], seed=seed)

            # Construct the transpose of the "remaining" dictionary created for the forward DAG, which defines the nodes that need to be executed before information can be passed upstream in the adjoint pass
            remaining_adj = {}
            for sweep_id, sink_object, _, _ in sinks_info:
                # Get the ancestors for the current sink object/top-level Analysis object
                anc = sink_ancestors[sink_object]

                # For each sweep ID, constructs the dictionary of dependencies to trace through the DAG in reverse
                remaining_adj[sweep_id] = {
                    n: sum(1 for d in dependents[n] if d in anc) for n in anc
                }

            # Define the function used to perform the adjoint analysis for a given node
            def perform_node_adjoint(n, sweep_id):
                # Set the sweep ID
                token = _current_sweep.set(sweep_id)

                try:
                    # Lock the input States this node accumulates into (its variables), which alias upstream outputs shared with sibling consumers
                    locks = sorted({id(st) for st in n.variables.values()})
                    for lid in locks:
                        # Acquire the lock, which locks the State or blocks the thread until it is released
                        state_locks[lid].acquire()
                    try:
                        # Perform the private adjoint analysis for the node
                        n._analyze_adjoint()
                    finally:
                        # Reverse through and unlock any locked locks
                        for lid in reversed(locks):
                            state_locks[lid].release()
                finally:
                    # Reset the contextvar to the previous value, which is necessary for future use
                    _current_sweep.reset(token)

            # Setup the scheduler, which is responsible for scheduling all adjoint analyses (in parallel over each quantity of interest, and parralel within a given sweep)
            with ThreadPoolExecutor(max_workers=self.parallel_max_workers) as pool:
                futures = {}

                # Loop through all the sweeps
                for sweep_id, sink_object, _, _ in sinks_info:
                    # Loop through all nodes and values in the remaining adjoint dictionary
                    for n, r in remaining_adj[sweep_id].items():
                        # Execute the perform_node_adjoint function if the number of remaining nodes is zero
                        if r == 0:
                            futures[pool.submit(perform_node_adjoint, n, sweep_id)] = (
                                n,
                                sweep_id,
                            )

                # Execute all adjoint analyses until all derivatives computed across the entire DAG for each quantity of interest
                while futures:
                    # Loop over the list of Future objects as they complete
                    for fut in as_completed(list(futures)):
                        # Extract the node and sweep ID
                        n, sweep_id = futures.pop(fut)

                        # Extract the data from the future (triggers the end of perform_node_adjoint)
                        fut.result()

                        # Get the ancestors for the current top-level Analysis object/sink
                        anc = sink_ancestors[sinks_of_info[sweep_id]]

                        # Loop through the sub-analyses for the current node
                        for sub in n.sub_analyses:
                            # Continue if the sub analysis is not in the ancestors list
                            if sub not in anc:
                                continue

                            # Decrement the counter for the adjoint tracker
                            remaining_adj[sweep_id][sub] -= 1

                            # Trigger the execution of the node adjoint for the sub-analysis, if the remaining adjoint counter is zero
                            if remaining_adj[sweep_id][sub] == 0:
                                futures[
                                    pool.submit(perform_node_adjoint, sub, sweep_id)
                                ] = (sub, sweep_id)

                        break

            # Extract the design derivatives for each sweep
            for sweep_id, sink, _, _ in sinks_info:
                self.design_derivs[sweep_id] = self._extract_dv_derivs(
                    sweep_id=sweep_id
                )

            # Per-node adjoint CPU time is not tracked in the parallel path yet;
            # only the wall-clock time is reported for this pass.
            self.dag_adjoint_time = float("nan")

        # Record the wall-clock time for the adjoint pass
        self.dag_adjoint_wall = time.perf_counter() - wall_start

        return

    def graph_network(
        self,
        filename: str = None,
        output_directory: str = None,
        interactive: bool = False,
        format: str = "pdf",
        consolidated_graph: bool = False,
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

        elif consolidated_graph:
            # Make the consolidated version of the graph visualization
            graph = self._consolidate_static_graph_network(
                node_fillcolor="#8CD17D",
                top_level_fillcolor="#5FB7EA",
                opacity=0.5,
                penwidth=3,
            )

            # Render the graph
            graph.render(
                filename=filename,
                directory=output_directory,
                cleanup=True,
                format=format,
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

    def _consolidate_static_graph_network(
        self,
        node_fillcolor: str = None,
        top_level_fillcolor: str = None,
        opacity: float = 1.0,
        penwidth: float = 2.0,
    ):
        """
        Private method that builds a decluttered version of the static graphviz visual.

        Compared to ``_static_graph_network``, this version:
          * Groups all variables flowing between the same (source, target) pair
            of analyses into a single edge (no edge labels).
          * Variable names are stored as a tooltip (visible in SVG/HTML output).
          * Enables ``concentrate=True`` so graphviz merges shared edge segments.
          * Renders nodes with rounded corners.
          * Optionally fills nodes with a hex color code.

        Parameters
        ----------
        node_fillcolor : str, optional
            Hex color code (e.g. ``"#AED6F1"``) used to fill non-top-level
            analysis nodes. If ``None``, no fill is applied.
        top_level_fillcolor : str, optional
            Hex color code (e.g. ``"#F1948A"``) used to fill top-level analysis
            nodes. If ``None``, defaults to ``node_fillcolor`` when that is set,
            otherwise no fill is applied.
        opacity : float, optional
            Fill opacity from 0.0 (transparent) to 1.0 (fully opaque). Applied
            to both ``node_fillcolor`` and ``top_level_fillcolor`` by appending
            an alpha byte to the hex color. Defaults to 1.0.

        Returns
        -------
        graph : graphviz.Digraph
            The consolidated graphviz digraph.
        """

        # Helper: append alpha byte to a 6-digit hex color (#RRGGBB → #RRGGBBAA)
        def _apply_opacity(color):
            if color is None:
                return None
            alpha = format(round(opacity * 255), "02X")
            return (
                color.rstrip().rstrip(")") + alpha if color.startswith("#") else color
            )

        # Helper: return fully-opaque version of a hex color (strips any alpha byte)
        def _border_color(color):
            if color is None:
                return None
            return "#" + color.lstrip("#")[:6]

        node_fillcolor = _apply_opacity(node_fillcolor)
        top_level_fillcolor = _apply_opacity(top_level_fillcolor)

        # Determine base node style
        base_node_attrs = {
            "shape": "box",
            "style": "rounded",
            "fontname": "Helvetica",
            "penwidth": str(penwidth),
        }
        if node_fillcolor is not None:
            base_node_attrs["style"] = "rounded,filled"
            base_node_attrs["fillcolor"] = node_fillcolor
            base_node_attrs["color"] = _border_color(node_fillcolor)

        graph = gv.Digraph(
            name=f"{self.sys_name.upper()}",
            graph_attr={"rankdir": "LR", "ranksep": "0.7", "concentrate": "true"},
            node_attr=base_node_attrs,
        )

        self.nodes = []
        self.edges = {}

        # Resolve top-level fill: explicit arg > fallback to node_fillcolor > none
        _top_fill = (
            top_level_fillcolor if top_level_fillcolor is not None else node_fillcolor
        )

        for analysis in self.top_level_analysis_list:
            stack = (
                analysis.stack if hasattr(analysis, "stack") else analysis._make_stack()
            )

            if not analysis.connected:
                analysis.analyze()

            for i, sub in enumerate(stack):
                if sub not in self.nodes:
                    self.nodes.append(sub)

                    if sub in self.top_level_analysis_list:
                        out_str = ", ".join(sub.outputs.keys())
                        top_attrs = {}
                        if _top_fill is not None:
                            top_attrs["style"] = "rounded,filled"
                            top_attrs["fillcolor"] = _top_fill
                            top_attrs["color"] = _border_color(_top_fill)
                        label = f"<<B>{sub.obj_name}</B><BR/><I>Outputs: {out_str}</I>>"
                        graph.node(
                            sub.obj_name,
                            label,
                            **top_attrs,
                        )
                    else:
                        graph.node(sub.obj_name, f"{sub.obj_name}")

                if i == 0 or not hasattr(sub, "connects"):
                    continue

                # Group variables by (source, target) pair
                grouped = {}
                for out, source in sub.connects.items():
                    grouped.setdefault((source.obj_name, sub.obj_name), []).append(out)

                for edge_key, labels in grouped.items():
                    if edge_key in self.edges:
                        continue
                    self.edges[edge_key] = labels
                    tooltip = ", ".join(labels)
                    graph.edge(*edge_key, tooltip=tooltip, labeltooltip=tooltip)

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
        self.global_obj_name = global_obj_name
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

    def _compute_log_columns(self):
        """
        Computes column headers and widths for use in log_information. Each column
        width is set to the maximum of the header label length (plus 2 for padding)
        and 20 (to accommodate values formatted with %20.10e).

        Returns
        -------
        columns : list of tuple
            Each entry is (header_str, width, category, key, index) where category
            is 'obj', 'con', or 'other', key is the dictionary key, and index is
            the array index (or None for scalars).
        """

        columns = []

        # Objective column
        obj_header = f"obj: {self.obj_local_name}"
        columns.append(("obj", None, None, obj_header))

        # Constraint columns
        for con in self.foi["cons"].keys():
            con_val = (
                self.foi["cons"][con]["instance"]
                .outputs[self.foi["cons"][con]["local_name"]]
                .value
            )

            if isinstance(con_val, np.ndarray):
                for i in range(con_val.size):
                    con_header = f"con: {self.foi['cons'][con]['local_name']}[{i}]"
                    columns.append(("con", con, i, con_header))
            else:
                con_header = f"con: {self.foi['cons'][con]['local_name']}"
                columns.append(("con", con, None, con_header))

        # Other FOI columns
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
                    columns.append(("other", other, i, other_header))
            else:
                other_header = f"other: {self.foi['other'][other]['local_name']}"
                columns.append(("other", other, None, other_header))

        # Compute widths: at least 20 (for numeric formatting), or header length + 2 for padding
        col_info = []
        for category, key, index, header in columns:
            width = max(len(header) + 2, 20)
            col_info.append(
                {
                    "category": category,
                    "key": key,
                    "index": index,
                    "header": header,
                    "width": width,
                }
            )

        return col_info

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

        # Compute column layout (recompute every header cycle in case FOI structure changes)
        if iter_number % 10 == 0 or not hasattr(self, "_log_columns"):
            self._log_columns = self._compute_log_columns()

        # Log the header names if the current iter number is divisible by 10
        if iter_number % 10 == 0:
            self.outputs_log.log("\n%5s" % "iter", end="")
            for col in self._log_columns:
                fmt = f"%{col['width']}s"
                self.outputs_log.log(fmt % col["header"], end="")

        # Log the values for the current iteration
        self.outputs_log.log("\n%5d" % iter_number, end="")

        for col in self._log_columns:
            width = col["width"]
            category = col["category"]
            key = col["key"]
            index = col["index"]

            # Retrieve the value based on category
            if category == "obj":
                val = (
                    self.foi["obj"]["instance"]
                    .outputs[self.foi["obj"]["local_name"]]
                    .value
                )
            elif category == "con":
                val = (
                    self.foi["cons"][key]["instance"]
                    .outputs[self.foi["cons"][key]["local_name"]]
                    .value
                )
                if isinstance(val, np.ndarray):
                    val = val[index]
            else:  # "other"
                val = (
                    self.foi["other"][key]["instance"]
                    .outputs[self.foi["other"][key]["local_name"]]
                    .value
                )
                if isinstance(val, np.ndarray):
                    val = val[index]

            # Format the value
            if not isinstance(val, str):
                val_str = "%20.10e" % val
            else:
                val_str = val

            fmt = f"%{width}s"
            self.outputs_log.log(fmt % val_str, end="")

        return

    def profile_iteration(self, iter_number):
        """
        Helper function that is used to display the time taken for each analyze and analyze_adjoint method at the current iteration. Internally, this updates a profile log file that stores the timing information for the System at each iteration.

        Parameters
        ----------
        iter_number : int
            Current iteration number
        """

        # Gather the System-level timing metrics captured by execute()/execute_adjoint().
        # Missing attributes (e.g. a pass that has not run yet) are reported as NaN.
        fwd_wall = getattr(self, "dag_forward_wall", float("nan"))
        fwd_cpu = getattr(self, "dag_total_time", float("nan"))
        adj_wall = getattr(self, "dag_adjoint_wall", float("nan"))
        adj_cpu = getattr(self, "dag_adjoint_time", float("nan"))

        def _eff(cpu, wall):
            # Summed node CPU time / wall time ~ effective concurrency (1.0 == serial).
            # NaN when either value is unavailable (cpu-sum requires track_node_timing).
            if wall == wall and wall > 0 and cpu == cpu:
                return cpu / wall
            return float("nan")

        # Log the header and the execution mode every 10 iterations
        if iter_number % 10 == 0:
            mode = "parallel" if self.parallel_execution else "serial"
            self.profile_log.log(
                "\n# mode: %s, max_workers: %s, track_node_timing: %s"
                % (
                    mode,
                    getattr(self, "parallel_max_workers", "-"),
                    getattr(self, "track_node_timing", False),
                ),
                end="",
            )
            self.profile_log.log(
                "\n%5s%14s%14s%12s%14s%14s%12s"
                % (
                    "iter",
                    "fwd_wall",
                    "fwd_cpu",
                    "fwd_eff",
                    "adj_wall",
                    "adj_cpu",
                    "adj_eff",
                ),
                end="",
            )

        # Log the System-level timing row for the current iteration
        self.profile_log.log(
            "\n%5d%14.6f%14.6f%12.2f%14.6f%14.6f%12.2f"
            % (
                iter_number,
                fwd_wall,
                fwd_cpu,
                _eff(fwd_cpu, fwd_wall),
                adj_wall,
                adj_cpu,
                _eff(adj_cpu, adj_wall),
            ),
            end="",
        )

        # Optionally log a per-node forward-time breakdown when per-node timing is
        # tracked. Uses the unique DAG nodes so shared sub-analyses are not double
        # counted (unlike a per-top-level breakdown).
        if getattr(self, "track_node_timing", False) and hasattr(self, "dag_nodes"):
            for n in sorted(
                self.dag_nodes,
                key=lambda a: getattr(a, "analysis_time", 0.0),
                reverse=True,
            ):
                self.profile_log.log(
                    "\n    %-28s fwd=%12.6f"
                    % (n.obj_name, getattr(n, "analysis_time", float("nan"))),
                    end="",
                )

        return
