import numpy as np
import warnings
from flume.base_classes.state import State
import time


# Define the default warning format
def custom_formatwarning(msg, *args, **kwargs):
    return str(msg) + "\n"


# Set the default warning format
warnings.formatwarning = custom_formatwarning


class Analysis:
    """
    This is a base class for all derived analysis classes that establishes a set of common methods. For the variables and outputs that are stored as attributes of the class, data is stored in dictionaries with key-value pairs structured as "variable/output name" : State object that describes the variable/output.
    """

    def __init__(self, obj_name: str, sub_analyses: list = [], **kwargs):
        """
        Base class construction for an instance of an Analysis object within the Flume framework.

        Parameters
        ----------
        obj_name : str
            Name for the analysis object, which should be unique among all Analysis objects that are included for a System
        sub_analyses : list
            A list of sub-analyses for the object, which will either be empty or contain other instances of objects that inherit from the Analysis base class

        Keyword Arguments
        -----------------
        Here, **kwargs are keyword arguments that correspond to parameters that should be set for Analysis classes. There are no keyword arguments/parameters by default, but the user can alter this by creating the self.default_parameters dictionary in their class that inherits from the Analysis base class
        """

        # Store the name for the property object
        self.obj_name = obj_name

        # Store the list of sub-analyses for the object
        self.sub_analyses = sub_analyses

        # Record arguments from the initialization
        self._log = {}
        self._log["init_args"] = {}
        for arg in kwargs:
            self._log["init_args"][arg] = kwargs[arg]

        # Check to make sure that input kwargs are valid parameters
        for key in kwargs:
            if key not in self.default_parameters:
                raise KeyError(
                    f"Kwarg '{key}' is not a valid option. Valid parameter kwargs are '{[key for key in self.default_parameters]}'."
                )

        # Set the parameters for the object using the kwargs or default parameters
        self.parameters = {}
        for key in self.default_parameters:
            if key in kwargs:
                if type(kwargs[key]) == type(self.default_parameters[key]):
                    self.parameters[key] = kwargs[key]
                else:
                    raise TypeError(
                        f"Type for parameter '{key}' does not match the expected type. Input type for {key} was {type(kwargs[key])} but expected is {type(self.default_parameters[key])}."
                    )
            else:
                self.parameters[key] = self.default_parameters[key]

        # Set the arritube that determines whether the object's forward analysis has been initialized
        self.initialized = False

        # Set the attribute that determines whether the adjoint has been initialized
        self.adjoint_initialized = False

        # Set the attribute that determines whether the sub-analyses have been connected to the primary analysis
        self.connected = False

        # Initialize the list that stores the variable names for the variables that are declared as design variables; this is empty by default
        self.design_vars_list = []

        # Call the method that checks whether the current object has all required user-defined methods
        self.has_methods = self._check_methods()

    def _check_methods(self) -> bool:
        """
        Private method that is always called for all instances of an AnalysisBase class. This ensures that any derived classes contain an _analyze method, which is unique for each derived class.

        Parameters
        ----------
        None

        Returns
        -------
        has_methods : bool
            A boolean that states whether or not the derived class has an _analyze method.
        """

        # Check for _analyze method
        if not (
            hasattr(self.__class__, "_analyze")
            and callable(getattr(self.__class__, "_analyze"))
        ):
            raise AttributeError(
                f"The class {self.__class__.__name__} does not have a method named '_analyze', which is a required method. \n"
            )

        # Check for the _analyze_adjoint method, raise warning
        if not (
            hasattr(self.__class__, "_analyze_adjoint")
            and callable(getattr(self.__class__, "_analyze_adjoint"))
        ):
            warnings.warn(
                f"WARNING: The class {self.__class__.__name__} does not have a method named '_analyze_adjoint', which is a required method to obtain derivative information. This is only raised as a warning so that other methods can be performed, but optimization cannot be performed without this method. \n"
            )

        has_methods = True

        return has_methods

    def _make_stack(self) -> list:
        """
        Assembles a stack of analysis objects in a list, which is used during the analysis and adjoint analysis procedures.

        Parameters
        ----------
        None

        Returns
        -------
        stack : list
            A list of analysis objects that defines the order for which the analysis (and adjoint analysis) are to be performed. When forming the stack, any sub-analyses for the primary analysis are also parsed.
        """
        # Initialize an empty list for the stack at this level
        stack = []

        # Add sub-analyses to the stack
        for sub_analysis in self.sub_analyses:
            if sub_analysis not in stack:
                stack.extend(sub_analysis._make_stack())

        stack.append(self)

        return stack

    def set_var_values(self, variables: dict):
        """
        Sets the values for the varaibles based on the entries contained in the variables dictionary

        Parameters
        ----------
        variables: dict
            A dictionary that is structued with key-value pairs, where the key is the local name of the variable and the value is the numerical value to set for the variable.

        Returns
        -------
        None, but internally State objects for the specified variables are updated with the provided values.
        """

        # Initialize an attribute that will store information about which variables were set by user
        self.defined_vars = []

        # Set the variable values that are specified in the input dictionary
        for var in variables:
            # Check to see if the specified variable is a valid option
            if var in self.variables:
                # Check the type of the input variable against the default value

                if hasattr(variables[var], "shape"):
                    complex_check = np.iscomplexobj(variables[var])
                    # complex_check = isinstance(variables[var].dtype, complex)
                else:
                    complex_check = isinstance(variables[var], complex)

                if (
                    isinstance(variables[var], self.variables[var].data_type)
                    or complex_check
                ):

                    # If the variable is a numpy array and has a defined shape, check to make sure the shape of the input variable matches the expected shape; if the shapes do not match, an error is raised
                    if (
                        hasattr(self.variables[var], "shape")
                        and np.shape(variables[var]) != self.variables[var].shape
                    ):

                        raise ValueError(
                            f"Provided shape of {np.shape(variables[var])} for variable '{var}' does not match expected shape of {self.variables[var].shape}."
                        )

                    # Recreate the State object for the current variable
                    self.variables[var] = State(
                        value=variables[var], desc=self.variables[var].desc, source=self
                    )
                # Return an error if the types do not match
                else:
                    raise TypeError(
                        f"Input for '{var}' must be of type {self.variables[var].data_type.__name__}."
                    )
            # Raise an error if the
            else:
                raise KeyError(
                    f"Variable '{var}' is not a valid variable for {self.__class__.__name__} object named '{self.obj_name}'."
                )

            # Add the variable to an attribute that stores whether or not
            self.defined_vars.append(var)

        # Update the analyzed attribute, since after setting values for the design variables the analysis procedure would need to be performed again
        self.initialized = False
        self.analyzed = False

        return

    def get_var_values(self, variables=["all"], display_data=False) -> dict:
        """
        Gets the values for the variables specified in the variables input list.

        Parameters
        ----------
        variables : list
            A list of strings that specifies the local variable names that are to be returned. Default is "all", which returns all of the variables for the object.

        Returns
        -------
        var_values : dict
            A dictionary that contains key-value pairs for the desired variables based on the input to the function.
        display_data : bool
            Boolean value that dictates whether variable values should be displayed when this method is called. Default is False.
        """

        # Initialize the dictionary that stores the key-value pairs for the variables
        var_values = {}

        # Determine which variables to return
        if variables == ["all"]:
            for var in self.variables:
                var_values[var] = self.variables[var].value
        else:
            for var in variables:

                # Check to make sure that the provided variable name is a valid name
                if var in self.variables:
                    var_values[var] = self.variables[var].value

                # Otherwise, raise an error
                else:
                    var_keys = self.variables.keys()
                    raise ValueError(
                        f"Variable '{var}' is not a valid variable for {self.__class__.__name__} object named '{self.obj_name}'. Valid variables are {[key for key in var_keys]}."
                    )

        if display_data:
            data_str = f"\n Variable values for object named '{self.obj_name}':"
            print(data_str)
            print("-" * len(data_str))
            for var in var_values:
                print(f"    {var}  = {var_values[var]}")

        return var_values

    def declare_design_vars(self, variables: list):
        """
        Declare design variables for the analysis object. This could be all of the variables for the object or a subset based on the user input to the method.

        Parameters
        ----------
        variables : list
            A list of strings which correspond to the variables that are to be included as design variables for the analysis object. This list is stored internally as an attribute and specifies which variables are included in adjoint tests.
        """

        # Loop through the variables provided in the input and add to the design_vars list (also checking to make sure no extra/incorrect vars are provided)
        for var in variables:

            if var not in self.variables:
                var_keys = self.variables.keys()

                raise ValueError(
                    f"Variable {var} is not a valid variable for object {self.__class__.__name__} named '{self.obj_name}'. Valid variables are {[key for key in var_keys]}."
                )

            else:
                self.design_vars_list.append(var)

        return

    def get_var_derivs(self, variables=["all"]) -> dict:
        """
        Gets the derivative values for the variables specified in the variables input list.

        Parameters
        ----------
        variables : list
            A list of strings that specifies the local variable names that are to be returned. Default is "all", which returns all of the variables for the object.

        Returns
        -------
        var_derivs : dict
            A dictionary that contains key-value pairs for the desired variable derivatives based on the input to the function.
        """

        # Initialize the dictionary that stores the key-value pairs for the variable derivatives
        var_derivs = {}

        # Determine which derivatives to return
        if variables == ["all"]:
            for var in self.variables:
                var_derivs[var] = self.variables[var].deriv
        else:
            for var in variables:

                # Check to make sure that the provided variable name is a valid name
                if var in self.variables:
                    var_derivs[var] = self.variables[var].deriv

                # Otherwise, raise an error
                else:
                    var_keys = self.variables.keys()
                    raise ValueError(
                        f"Variable '{var}' is not a valid variable for {self.__class__.__name__} object named '{self.obj_name}'. Valid variables are {[key for key in var_keys]}."
                    )

        return var_derivs

    def get_var_metadata(self, variables: list = ["all"], display_info=False) -> dict:
        """
        Gets the metadata associated with the variables specified in the variables input list.

        Parameters
        ----------
        variables : list
            A list of strings that specifies the local variable names that are to be accessed. Default is "all", which returns the metadata for all variables for the object.

        Returns
        -------
        var_meta : dict
            A dictionary of dictionaries for the specified variables in the input. Each sub-dictionary contains key-value pairs for the following information: variable type, shape (if applicable), and description.
        display_info : bool
            Boolean value that dictates whether variable metadata should be displayed when this method is called. Default is False.
        """

        # Initialize the dictionary for the metadata for the variables
        var_meta = {}

        # Determine which variables to get the metadata for
        if variables == ["all"]:
            for var in self.variables:
                # Create the sub-dictionary for the specific variable
                var_meta[var] = {}

                # Store the type information
                var_meta[var]["type"] = self.variables[var].data_type.__name__

                # If necessary, store the shape information
                if hasattr(self.variables[var], "shape"):
                    var_meta[var]["shape"] = self.variables[var].shape

                # Store the description
                var_meta[var]["descript"] = self.variables[var].desc

        else:
            for var in variables:
                # Create the sub-dictionary for the specific variable
                var_meta[var] = {}

                # Check to make sure that the variable is contained within the variables dictionary
                if var in self.variables:

                    # Store the type information
                    var_meta[var]["type"] = self.variables[var].data_type.__name__

                    # If necessary, store the shape information
                    if hasattr(self.variables[var], "shape"):
                        var_meta[var]["shape"] = self.variables[var].shape

                    # Store the description
                    var_meta[var]["descript"] = self.variables[var].desc

                # Raise an error if the specified variable is not a valid option
                else:
                    var_keys = self.variables.keys()
                    raise KeyError(
                        f"Variable '{var}' is not a valid variable for {self.__class__.__name__} object named '{self.obj_name}'. Valid variables are {[key for key in var_keys]}."
                    )

        if display_info:
            for var in var_meta:
                var_str = f"\n\n Variable '{var}' Information:"
                print(var_str)
                print("-" * len(var_str))
                for key in var_meta[var]:
                    print(f"    {key} : {var_meta[var][key]}")

        return var_meta

    def get_parameter_values(self, params: list = ["all"], display_data=False) -> dict:
        """
        Gets the values for the parameters specified in the parameters input list.

        Parameters
        ----------
        params : list
            A list of strings that specifies the local parameter names that are to be returned. Default is "all", which returns all of the parameters for the object.

        Returns
        -------
        param_values : dict
            A dictionary that contains key-value pairs for the desired parameters based on the input to the function.
        display_data : bool
            Boolean value that dictates whether parameter values should be displayed when this method is called. Default is False.
        """

        # Initialize the dictionary that stores the key-value pairs for the parameters
        param_values = {}

        # Determine which parameters to return
        if params == ["all"]:
            param_values = self.parameters
        else:
            for param in params:
                if param in self.parameters:
                    param_values[param] = self.parameters[param]
                else:
                    param_keys = self.parameters.keys()
                    raise ValueError(
                        f"Parameter '{param}' is not a valid parameter for {self.__class__.__name__} object named '{self.obj_name}'. Valid parameters are {[param for param in param_keys]}."
                    )

        if display_data:
            data_str = f"\n Parameter values for object named '{self.obj_name}':"
            print(data_str)
            print("-" * len(data_str))
            for param in param_values:
                print(f"    {param}  = {param_values[param]}")

        return param_values

    def _set_data_type(self, mode):
        """
        Declares the mode that should be used when performing the analysis procedure, i.e. real or complex.

        Parameters
        ----------
        mode : str
            Either 'real' or 'complex', which then sets an attribute for dtype to be float or complex. This is needed when setting data type for numpy arrays during analysis procedures.
        """

        if mode == "real":
            self.dtype = float
        elif mode == "complex":
            self.dtype = complex
        else:
            raise ValueError("Analysis mode must be either 'real' or 'complex'.")

    def _initialize_analysis(self, mode: str = "real", isolated_test: bool = False):
        """
        Initializes the internal data for the analysis procedure contained within the object. Nominally, this simply sets an attribute called 'analyzed', which is a boolean that stores whether the analysis procedure has been performed. If other procedures are required for this function, the derived class should contain an instance of this method and then call super()._initialize_analysis() immediately before the return statement.

        Parameters
        ----------
        mode : str
            Mode which specifies whether the analysis should be performed with real or complex numbers/data types
        isolated_test : bool
            Defaults to False. This is only used internally for the isolated adjoint test to avoid the need to check for sub-analyses.
        """

        # Set the data type based on the analysis mode
        self._set_data_type(mode=mode)

        if not isolated_test:
            # Modify the analyzed attribute for the current object, if necessary
            if not hasattr(self, "analyzed"):
                # If the object does not have an analyzed attribute, set it to False
                self.analyzed = False

            # If the object has any sub-analyses...
            elif any(self.sub_analyses):

                # Loop through each sub analysis and extract its analyzed attribute
                subs_analyzed = []
                for sub in self.sub_analyses:
                    subs_analyzed.append(sub.analyzed)

                # If not all sub-analyses have been analyzed, set the current object's analyzed attribute to False (values will change from sub-analyses, so need to recompute this object)
                if not all(subs_analyzed):
                    self.analyzed = False

                # If all sub-analyses have been analyzed and the current object is already analyzed, keep analyzed the same (as True)
                elif all(subs_analyzed) and self.analyzed:
                    self.analyzed = True

                # If all sub-analyses have been analyzed but this object is not analyzed, keep analyzed the same (as False); this object needs to be analyzed
                else:
                    self.analyzed = False

            # If the object does not have any sub-analyses...
            elif not any(self.sub_analyses):

                # If it has been analyzed, keep the same (as True)
                if self.analyzed:
                    self.analyzed = True
                # If it has not been analyzed, keep the same (as False)
                else:
                    self.analyzed = False

            # Raise an error if this situation occurs, as there is a logic detection issue
            else:
                raise RuntimeError(
                    f"Unknown logic situation encountered for object named {self.obj_name} during analysis initialization!"
                )

        return

    def analyze(self, mode="real", debug_print=False):
        """
        Performs the analysis procedure for the object after combining the objects into a list that contains the stack. For each analysis in the stack, connections are established between sub-analyses and primary-analyses and then the private _analyze method is called.
        """

        # Assemble the stack
        self.stack = self._make_stack()

        # Initialize the analyses for the items in the stack
        init_seen = set()
        for analysis in self.stack:
            if analysis not in init_seen:
                initialize_start = time.time()
                analysis._initialize_analysis(mode=mode)
                initialize_end = time.time()

                if debug_print:
                    print(
                        f"Initialized analysis for object named '{analysis.obj_name} in {(initialize_end - initialize_start):.4f}'."
                    )

                init_seen.add(analysis)

        # Perform the analysis for each member in the stack
        self.forward_total = 0.0
        self.forward_stack = []
        seen = set()
        for analysis in self.stack:
            # Perform the connections for each object in the stack
            analysis._connect()

            # Add the analysis to the seen list and the forward stack, if it has not already been seen
            if analysis not in seen:
                seen.add(analysis)
                self.forward_stack.append(analysis)

            # Perform the analysis for each object in the stack
            if not analysis.analyzed:
                start = time.time()
                analysis._analyze()
                analysis.analyzed = True
                end = time.time()

                analysis_time = end - start
                analysis.forward_profile = analysis_time

                # Add the total time for the current analyze method
                self.forward_total += analysis_time
                if debug_print:
                    print(
                        f"Analysis performed for '{analysis.obj_name}' in {analysis_time} seconds."
                    )

        return

    def _initialize_adjoint(self):
        """
        Initializes the internal data for the underlying adjoint analysis. Performs checks that the adjoint analysis has been initialized and that the primary analysis has been performed for the given object.
        """

        # Check to make sure that the object's analysis has been initialized
        assert hasattr(
            self, "analyzed"
        ), f"The analysis for '{self.__class__.__name__}' named '{self.obj_name}' has not been initialized yet, so the adjoint cannot be initialized."

        # Check to make sure that the object has been analyzed, otherwise raise an error
        assert (
            self.analyzed
        ), f"'{self.__class__.__name__}' named '{self.obj_name}' has not been analyzed yet, so the adjoint cannot be initialized."

        # Initialize the adjoint variables
        for var in self.variables:
            if hasattr(self.variables[var], "shape"):
                self.variables[var].deriv = np.zeros(self.variables[var].shape)
            else:
                self.variables[var].deriv = 0.0

        # Update the attribute that flags whether the adjoint has been initialized
        self.adjoint_initialized = True
        self.adjoint_analyzed = False

        return

    def analyze_adjoint(self, debug_print=False):
        """
        Performs the adjoint analysis for the object and propagates any computations back through other sub-analyses included in the stack. Internally, this is done by reversing through the stack performing the adjoint analysis for each individual object.
        """

        if debug_print:
            print("\n STACK:")
            for i, analysis in enumerate(self.stack):
                print(f"analysis {i} = {analysis.obj_name}")

        # Make the stack for the object if it does not already exist (this may occur if the analysis class is not called as the top-level analysis)
        if not hasattr(self, "stack"):
            self.stack = self._make_stack()

        for analysis in self.forward_stack[::-1]:
            analysis._initialize_adjoint()

            if debug_print:
                print(
                    f"\n Initialized adjoint for class {analysis.__class__.__name__} named '{analysis.obj_name}'."
                )

        self.adjoint_total = 0.0
        for analysis in self.forward_stack[::-1]:

            start = time.time()
            analysis._analyze_adjoint()
            analysis.adjoint_analysed = True
            end = time.time()

            adjoint_time = end - start
            analysis.adjoint_profile = adjoint_time

            self.adjoint_total += adjoint_time

            if debug_print:
                print(
                    f"\n Performed adjoint analysis for class {analysis.__class__.__name__} named '{analysis.obj_name}'."
                )

        return

    def _add_output_seed(self, outputs=["all"], seed=1.0, random_seeds=False):
        """
        Sets a seed value for the adjoint method for a given set of outputs, where the default set is all of the outputs.

        Parameters
        ----------
        outputs : list
            A list of strings that specifies the local output names for which the seeds are to be set. Default is "all", so all seeds are set if not specified otherwise.

        Returns
        -------
        None, but internally sets the seed values for the adjoint analysis.
        """

        # Initialize the seeds for the adjoint method
        self.seeds = {}

        # Zero out the output derivative values for all objects in the analysis stack, which may exist from previous adjoint analyses
        if hasattr(self, "stack"):
            for analysis in self.stack:
                for out in analysis.outputs:
                    if hasattr(analysis.outputs[out], "shape"):
                        analysis.outputs[out].deriv = np.zeros(
                            analysis.outputs[out].shape
                        )
                    else:
                        analysis.outputs[out].deriv = 0.0

        # Determine which seeds to set
        if outputs == ["all"]:
            # Set the seeds for all outputs, adhering to the shape of arrays when necessary
            for out in self.outputs:
                # Check to see if the State object has a defined shape for numpy arrays
                if hasattr(self.outputs[out], "shape"):
                    if random_seeds:
                        val = np.random.uniform(size=self.outputs[out].shape)
                    else:
                        if not hasattr(seed, "shape"):
                            raise ValueError(
                                f"Provided seed for output '{out}' is a scalar, but it should have the shape '{self.outputs[out].shape}'."
                            )
                        elif seed.shape != self.outputs[out].shape:
                            raise ValueError(
                                f"The output seed shape for output '{out}' of the object named {self.obj_name}' does not match the expected shape of {self.outputs[out].shape}'."
                            )
                        else:
                            val = seed

                        # val = np.ones(self.outputs[out].shape) * seed
                # If there is not a shape, set a scalar value
                else:
                    if random_seeds:
                        val = np.random.uniform()
                    else:
                        val = seed

                # Assign the seed value to the State's derivative attribute and copy the data to the seeds attribute
                self.seeds[out + "b"] = np.copy(val)
                self.outputs[out].deriv = val

        else:
            for out in outputs:
                if out not in self.outputs:
                    out_keys = self.outputs.keys()
                    raise ValueError(
                        f"Output '{out}' is not a valid output for a {self.__class__.__name__} object named '{self.obj_name}' object. Valid outputs are {[out for out in out_keys]}."
                    )

            for out in self.outputs:
                if out in outputs:
                    # Check to see if the State object has a defined shape for numpy arrays
                    if hasattr(self.outputs[out], "shape"):
                        if random_seeds:
                            val = np.random.uniform(size=self.outputs[out].shape)
                        else:
                            if not hasattr(seed, "shape"):
                                raise ValueError(
                                    f"Provided seed for output '{out}' is a scalar, but it should have the shape '{self.outputs[out].shape}'."
                                )
                            elif seed.shape != self.outputs[out].shape:
                                raise ValueError(
                                    f"The output seed shape for output '{out}' of the object named {self.obj_name}' does not match the expected shape of {self.outputs[out].shape}'."
                                )
                            else:
                                val = seed
                            # val = np.ones(self.outputs[out].shape) * seed

                    # If there is not a shape, set a scalar value
                    else:
                        if random_seeds:
                            val = np.random.uniform()
                        else:
                            val = seed

                    # Assign the seed value to the State's derivative attribute and copy the data to the seeds attribute
                    self.seeds[out + "b"] = np.copy(val)
                    self.outputs[out].deriv = val
                else:
                    # Check to see if the State object has a defined shape for numpy arrays
                    if hasattr(self.outputs[out], "shape"):
                        val = np.zeros(shape=self.outputs[out].shape)

                    # If there is not a shape, set a scalar value
                    else:
                        val = 0.0

                    # Assign the seed value to the State's derivative attribute and copy the data to the seeds attribute
                    self.seeds[out + "b"] = np.copy(val)
                    self.outputs[out].deriv = val

        orig_seeds = self.seeds

        return orig_seeds

    def _connect(self):
        """
        Establishes any connections between the outputs of the sub-analyses for the object and the object's variables. The first time that this method is called, a dictionary is created that stores a connection map for the appropriate variables of the primary analysis. This connections dictionary is structured such that they keys refer to the local name of the variable and the values point to the object in the sub-analysis list that is the source for the variable.
        """

        # Checks to see if sub_analyses is an empty list, and returns if so
        if not self.sub_analyses:
            return

        # Use the connections map to connect sub-analyses to main analysis or create the connections map if it does not exist
        if self.connected:
            for name in self.connects:
                self.variables[name] = self.connects[name].outputs[name]
        else:
            # Create an empty dictionary for the connections
            self.connects = {}

            # Loop through each sub-analysis for the object
            for sub in self.sub_analyses:
                # Loop through each output in the sub-analysis
                for out in sub.outputs:
                    # If the output of the sub-analysis is an input variable to the current analysis, set it with the value

                    if out in self.variables:
                        # Store the connection in the connections map
                        self.connects[out] = sub

                        # Set the analysis variable with the sub-analysis value
                        self.variables[out] = sub.outputs[out]

            # Update the attribute that states whether or not the connections map has been made
            self.connected = True

    def get_output_values(self, outputs: list = ["all"], display_data=False) -> dict:
        """
        Gets the values for the specified outputs.

        Parameters
        ----------
        outputs : list
            A list of strings that specifies the local output names that are to be returned. Default is "all", which returns all of the outputs for the object.

        Returns
        -------
        output_values : dict
            A dictionary that contains key-value pairs for the desired outputs based on the input to the function.
        display_data : bool
            Boolean value that dictates whether output values should be displayed when this method is called. Default is False.
        """

        # Verify that the analysis has been performed
        if not hasattr(self, "analyzed") or not self.analyzed:
            raise ValueError(
                f"The {self.__class__.__name__} object named '{self.obj_name}' has not been analyzed yet, so there are no outputs to return. This method should not be called until 'analyze' is called."
            )

        # Initialize the dictionary that stores the key-value pairs for the outputs
        output_values = {}

        # Determine which outputs to return
        if outputs == ["all"]:
            for out in self.outputs:
                output_values[out] = self.outputs[out].value
        else:
            for out in outputs:
                if out in self.outputs:
                    output_values[out] = self.outputs[out].value
                else:
                    out_keys = self.outputs.keys()
                    raise ValueError(
                        f"Output '{out}' is not a valid output for a {self.__class__.__name__} object named '{self.obj_name}' object. Valid outputs are {[out for out in out_keys]}."
                    )

        if display_data:
            data_str = f"\n Output values for object named '{self.obj_name}':"
            print(data_str)
            print("-" * len(data_str))
            for out in output_values:
                print(f"    {out}  = {output_values[out]}")

        return output_values

    def get_output_metadata(self, outputs: list = ["all"], display_info=False) -> dict:
        """
        Gets the metadata associated with the outputs specified in the outputs list.

        Parameters
        ----------
        outputs : list
            A list of strings that specifies the local output names that are to be accessed. Default is "all", which returns the metadata for all outputs for the object.

        Returns
        -------
        out_meta : dict
            A dictionary of dictionaries for the specified outputs. Each sub-dictionary contains key-value pairs for the following information: variable type, shape (if applicable), and description.
        display_info : bool
            Boolean value that dictates whether variable metadata should be displayed when this method is called. Default is False.
        """

        # Verify that the analysis has been performed
        if not hasattr(self, "analyzed") or not self.analyzed:
            raise ValueError(
                f"The {self.__class__.__name__} object named '{self.obj_name}' has not been analyzed yet, so there are no outputs to return. This method should not be called until 'analyze' is called."
            )

        # Initialize a dictionary for the metadata of the outputs
        out_meta = {}

        # Determine which outputs to get the metadata for
        if outputs == ["all"]:
            for out in self.outputs:
                # Create the sub-dictionary for
                out_meta[out] = {}

                # Store the type information
                out_meta[out]["type"] = self.outputs[out].data_type.__name__

                # If necessary, store the shape information
                if hasattr(self.outputs[out], "shape"):
                    out_meta[out]["shape"] = self.outputs[out].shape

                # Store the description
                out_meta[out]["descript"] = self.outputs[out].desc

        else:
            for out in outputs:
                # Create the sub-dictionary for the specific outputs
                out_meta[out] = {}

                # Check to make sure that the specified output is contained within the outputs dictionary
                if out in self.outputs:
                    # Store the type information
                    out_meta[out]["type"] = self.outputs[out].data_type.__name__

                    # If necessary, store the shape information
                    if hasattr(self.outputs[out], "shape"):
                        out_meta[out]["shape"] = self.outputs[out].shape

                    # Store the description
                    out_meta[out]["descript"] = self.outputs[out].desc
                # Raise an error if the specified output is not a valid option
                else:
                    out_keys = self.outputs.keys()
                    raise KeyError(
                        f"Output {out} is not a valid output for {self.__class__.__name__} object named '{self.obj_name}'. Valid outputs are {[key for key in out_keys]}."
                    )

        if display_info:
            for out in out_meta:
                out_str = f"\n\n Output '{out}' Information:"
                print(out_str)
                print("-" * len(out_str))
                for key in out_meta[out]:
                    print(f"    {key} : {out_meta[out][key]}")

        return out_meta

    def get_global_name(self, local_name: str) -> str:
        """
        Gets the global name for an input local name. This function works for all parameters, variables, and outputs.

        Parameters
        ----------
        local_name : str
            A string that is the local name for a given parameter, variable, or output of the object.

        Returns
        -------
        global_name : str
            A string that is the global name for a given parameter, variable, or output of the object.
        """

        # This check allows for variable/parameter global names to be accessed before the analyze method is called, but it will prevent the user from accessing any (potential) output names
        if not self.analyzed and not (
            local_name in self.parameters or local_name in self.variables
        ):
            raise ValueError(
                f"The {self.__class__.__name__} object named '{self.obj_name}' has not been analyzed yet, so the global names of outputs cannot be accessed. If you want to get the global name of an output, make sure 'analyze' has been called first."
            )

        # Check to make sure that the provided local name is an output, parameter, or variable
        if (
            local_name not in self.parameters
            and local_name not in self.variables
            and local_name not in self.outputs
        ):
            raise ValueError(
                f"Local name '{local_name}' is not within the parameter, variable, or output sets."
            )
        else:
            global_name = f"{self.obj_name}.{local_name}"

        return global_name

    def test_isolated_adjoint(
        self,
        print_res=True,
        debug_print=False,
        method="cs",
        defined_vars={},
        dh_fd=1e-6,
        dh_cs=1e-30,
    ):
        """
        Tests the adjoint implementation for an analysis object (ignores any sub-analyses for the isolated case). Currently, only the complex-step method is implemented for this test, so calculations must be complex-safe.

        Parameters
        ----------

        defined_vars : dictionary
            Dictionary that specifies whether any variables that have defined values for the adjoint test. This is nominally empty, but is sometimes necessary for variables that have strict bounds.

        Returns
        -------
        None, but displays results of adjoint analysis test
        """

        # Set random variable values for all variables
        def_var_values = {}

        if debug_print:
            print("\n ORIGINAL VARIABLE VALUES:")
        for var in self.variables:

            # Check to see that the variable is in the design variables list
            if var in self.design_vars_list:
                # Check to see if the variable value is defined (used in some cases where values take on restricted bounds)
                if var in defined_vars:
                    def_var_values[var] = defined_vars[var]
                else:
                    # Check to see if the variable has a shape, which is necessary for setting variables to be the right size
                    if hasattr(self.variables[var], "shape"):
                        def_var_values[var] = np.random.uniform(
                            size=self.variables[var].shape
                        )
                    else:
                        def_var_values[var] = np.random.uniform()

                if debug_print:
                    print(f"    \n {var} value:", def_var_values[var])
            else:
                if debug_print:
                    print(f"{var} not declared as a design variable.")
                continue

        # Call the set variable values method
        self.set_var_values(def_var_values)

        # Initialize the analysis (use isolated test mode to avoid check for sub-analyses)
        self._initialize_analysis(isolated_test=True)

        # Check to make sure that the design variables list is not empty
        num_dvs = len(self.design_vars_list)
        if num_dvs == 0:
            raise RuntimeError(
                "The number of declared design variables is zero, which is invalid for performing the adjoint test."
            )

        # Analyze the system
        self._analyze()
        self.analyzed = True
        outs0 = self.get_output_values()

        if debug_print:
            print("ORIGINAL OUTPUT VALUES: \n")
            for out in outs0:
                print(out, ":", outs0[out])

        # Initialize the adjoint
        self._initialize_adjoint()

        # Set the seeds for the outputs
        orig_seeds = self._add_output_seed(outputs=["all"], random_seeds=True)

        # Extract the seed values before they are overwritten
        out_seeds = self.seeds

        # Perform the adjoint analysis
        self._analyze_adjoint()

        # Extract the derivative values from the outputs after the first analysis
        out_derivs = {}
        for out in self.outputs:
            out_derivs[out] = self.outputs[out].deriv

        # Extract the derivative values
        deriv_vals = self.get_var_derivs(variables=self.design_vars_list)

        # Set perturbation values for the variables
        perts = {}
        for var in deriv_vals:

            perts["pert_" + var] = np.random.uniform(size=np.size(deriv_vals[var]))

        # Compute the exact derivatives with the perturbation values
        ans_tot = 0.0

        if debug_print:
            print("\n ADDING CONTRIBUTIONS TO EXACT DERIVATIVE")

        for var in deriv_vals:

            # Compute the contribution to the exact derivative for the current variable
            if isinstance(def_var_values[var], np.ndarray):
                if deriv_vals[var].ndim == 2:
                    contr = np.dot(perts["pert_" + var], deriv_vals[var].flatten())
                else:
                    contr = np.dot(perts["pert_" + var], deriv_vals[var])
            else:
                contr = np.dot(perts["pert_" + var], deriv_vals[var])

            # Add the contribution for the current variable to the total value
            ans_tot += contr

            if debug_print:
                print(
                    f"\n    Added contribution for variable {var} from object {self.__class__.__name__} to ans_tot with a contribution of {contr}."
                )

        if debug_print:
            print("\n ORIGNAL SEED VALUES:")
            print(orig_seeds)

            print("\n OUTPUT SEED VALUES (AFTER ADJOINT ANALYSIS):")
            print(out_seeds)

        # Set the step size based on the method used for the derivative check
        if method == "cs":
            dh = dh_cs
            mode = "complex"
        elif method == "fd":
            dh = dh_fd
            mode = "real"
        else:
            raise ValueError("Method must be 'fd' or 'cs'.")

        # Set the perturbed design variable values
        pert_var_values = {}
        if debug_print:
            print("\n PERTURBED VARIABLE VALUES:")
        for var in self.variables:

            if var in self.design_vars_list:

                if method == "cs":
                    # Perturb the design variable by a small amount in the imaginary plane in the previously determined random direction

                    if isinstance(def_var_values[var], np.ndarray):
                        if def_var_values[var].ndim == 2:
                            pert_var_values[var] = def_var_values[
                                var
                            ] + 1.0j * dh * np.reshape(
                                perts["pert_" + var], def_var_values[var].shape
                            )
                        else:
                            pert_var_values[var] = (
                                def_var_values[var] + 1.0j * dh * perts["pert_" + var]
                            )
                    else:
                        pert_var_values[var] = (
                            def_var_values[var] + 1.0j * dh * perts["pert_" + var]
                        )

                elif method == "fd":

                    # This logic corrects the required type for float inputs using the item() method (otherwise they are converted to numpy arrays when the perturation is added)
                    if isinstance(def_var_values[var], float):
                        pert_var_values[var] = (
                            def_var_values[var] + dh * perts["pert_" + var].item()
                        )
                    else:
                        if def_var_values[var].ndim == 2:
                            pert_var_values[var] = def_var_values[
                                var
                            ] + dh * np.reshape(
                                perts["pert_" + var], def_var_values[var].shape
                            )
                        else:
                            pert_var_values[var] = (
                                def_var_values[var] + dh * perts["pert_" + var]
                            )

                if debug_print:
                    print(f"    \n {var} value: ", pert_var_values[var])
            else:
                continue

        self.set_var_values(pert_var_values)

        # Compute the outputs using the perturbed values (initialize the analysis with isolated test set to true to avoid checks for any sub-analyses)
        self._initialize_analysis(mode=mode, isolated_test=True)

        self._analyze()
        self.analyzed = True

        # Extract the perturbed output values
        outs = self.get_output_values()

        # Reset the analyzed flag
        self.analyzed = False

        if debug_print:
            print("\n PERTURBED OUTPUT VALUES:")
            for out in outs:
                print(out, ":", outs[out])

        # Compute the derivative numerically
        cs = 0.0
        fd = 0.0

        for out in outs:

            if method == "cs":
                # Compute the complex-step contribution value for the current output
                cs_contr = np.sum(out_derivs[out] * np.imag(outs[out]) / dh)

                # Add the contribution to the total derivative check using complex-step
                cs += cs_contr

                if debug_print:
                    print(
                        f"\n    Added contribution for output {out} to cs with a contribution of {cs_contr}."
                    )
            elif method == "fd":
                # Compute the finite-difference contribution value for the current output
                fd_contr = np.sum(out_derivs[out] * (outs[out] - outs0[out]) / dh)

                # Add the contribution to the total derivative check using complex-step
                fd += fd_contr

                if debug_print:
                    print(
                        f"\n    Added contribution for output {out} to fd with a contribution of {fd_contr}."
                    )

        # Compute the relative error and display test results
        if method == "cs":
            err = (ans_tot - cs) / cs
        elif method == "fd":
            err = (ans_tot - fd) / fd

        if print_res:
            test_case = f"ISOLATED adjoint test results for {self.__class__.__name__} object named '{self.obj_name}':"

            print("\n", "-" * len(test_case))
            print(test_case)
            print("-" * len(test_case))

            if method == "cs":
                print("\n %25s  %25s  %25s" % ("Answer", "Complex-step", "Rel Error"))
                print("%25.15e  %25.15e  %25.15e" % (ans_tot, cs, err))

                ratio = ans_tot / cs
                print("\n ratio (ans/cs): ", ratio)
                print("\n 1/ratio (cs/ans): ", 1 / ratio)
                print("-" * len(test_case))
            elif method == "fd":
                print(
                    "\n %25s  %25s  %25s" % ("Answer", "Finite-Difference", "Rel Error")
                )
                print("%25.15e  %25.15e  %25.15e" % (ans_tot, fd, err))

                ratio = ans_tot / fd
                print("\n ratio (ans/fd): ", ratio)
                print("\n 1/ratio (fd/ans): ", 1 / ratio)
                print("-" * len(test_case))

        return err

    def test_combined_adjoint(
        self,
        print_res=True,
        debug_print=False,
        method="cs",
        defined_vars={},
        dh_cs=1e-30,
        dh_fd=1e-6,
    ):
        """
        Tests the adjoint implementation for an analysis object and accounts for any sub-analyses within the stack.

        Parameters
        ----------
        debug_print : bool
            Boolean value that specifies whether debug print statements should be displayed
        method : str
            Either 'cs' or 'fd', which specifies that the method to check against the adjoint implementation is complex-step or finite-difference, respectively
        defined_vars : dictionary
            A dictionary where the keys correspond to the global variable names and the values are the numeric values that the variables should have. For example, if the user wants to set a variable 'x' for an analysis object 'analysis' to be value 1.0, defined_vars should be given as defined_vars = {'analysis.x': 1.0}.
        """

        # Perform a preliminary analysis to assemble the stack and form the connection maps
        self.analyze()

        # Check to make sure that the design variables list is not empty for all of the objects
        num_dvs = 0
        for obj in self.stack:
            num_dvs += len(obj.design_vars_list)

        if num_dvs == 0:
            raise RuntimeError(
                "The number of declared design variables is zero, which is invalid for performing the adjoint test."
            )

        # If necessary, display the connections map information for the objects in the system
        if debug_print:
            for obj in self.stack:
                print(f"\n Object: {obj.__class__.__name__}")
                if hasattr(obj, "connects"):
                    print(f"    Connections map : {obj.connects}")
                else:
                    print(f"    No connections map")

        def_var_values = {}

        # For each object in the stack, determine whether a variable is self-sourced or inherited from a sub-analysis and store that information
        for obj in self.stack:

            def_var_values[obj] = {}

            if debug_print:
                print(f"\n VARIABLES DETAILS FOR {obj.__class__.__name__}:")

            for var in obj.variables:

                if var in obj.design_vars_list:
                    if obj.variables[var].source == obj:
                        def_var_values[obj][var] = None
                        if debug_print:
                            print(f"{var} is self-sourced")
                    else:
                        if debug_print:
                            print(
                                f"{var} inherited from {obj.variables[var].source.__class__.__name__}"
                            )

                        pass
                else:
                    if debug_print:
                        print(f"{var} not declared as a design variable.")
                    continue

        # For the objects that have randomly defined variable values, set a random variable value for the test
        for obj in def_var_values:

            # Loop through the variables for the current object in the stack
            for var in def_var_values[obj]:

                # Check to see if the variable value is defined (used in some cases where values take on restricted bounds)
                if f"{obj.obj_name}.{var}" in defined_vars:
                    def_var_values[obj][var] = defined_vars[f"{obj.obj_name}.{var}"]
                else:
                    # If the variable has a shape, create a numpy array of that shape
                    if hasattr(obj.variables[var], "shape"):
                        def_var_values[obj][var] = np.random.uniform(
                            size=obj.variables[var].shape
                        )

                    # Otherwise, set a random scalar value
                    else:
                        def_var_values[obj][var] = np.random.uniform()

            # Set the variable values for the current object
            obj.set_var_values(def_var_values[obj])

            if debug_print:
                print(f"\n Variables for {obj.__class__.__name__}:")
                for var in obj.variables:
                    print(f"    \n {var} value : ", obj.variables[var].value)

        if debug_print:
            print("\n", def_var_values)

        # Initialize and perform the analysis
        self.analyze(mode="real")

        # Extract the original output values
        outs_orig = self.get_output_values()

        if debug_print:
            print("\n OUTPUTS FOR ORIGINAL VARIABLE VALUES:")
            for out in outs_orig:
                print(out, ":", outs_orig[out])

        # Set the seeds for the outputs on the top-level analysis
        self._add_output_seed(outputs=["all"], random_seeds=True)

        # Perform the adjoint analysis
        self.analyze_adjoint(debug_print=False)

        # Extract the derivative values from the outputs after the first analysis
        out_derivs = {}

        for out in self.outputs:
            out_derivs[out] = self.outputs[out].deriv

        # Extract the derivative values for all of the independent variables
        var_deriv_vals = {}
        pert_vals = {}

        if debug_print:
            print("\n PERTURBED VARIABLES:")

        for obj in def_var_values:

            # Initialize an empty list that will store the keys for the independent variables for the current object
            var_list = []

            # Initialize a sub-dictionary that will store the perturbation value for the independent variables for the current object
            pert_vals[obj] = {}

            # Loop through the independent variables
            for var in def_var_values[obj]:
                # Append the variable name to the list
                var_list.append(var)

                # Compute a random perturbation direction/value for the current variable
                if hasattr(def_var_values[obj][var], "size"):
                    pert_vals[obj][var] = np.random.uniform(
                        size=np.size(def_var_values[obj][var])
                    )
                else:
                    pert_vals[obj][var] = np.random.uniform()

            if debug_print:
                print(f"Variable list for extracting derivatives : {var_list}")

            # Extract the derivatives for the independent variables for the current object
            var_deriv_vals[obj] = obj.get_var_derivs(variables=var_list)

        # Set perturbation values for the independent variables
        if debug_print:
            print("\n Perturbation values: \n", pert_vals)

        # Compute the exact derivative with the perturbation values
        ans_tot = 0.0

        if debug_print:
            print("\n ADDING CONTRIBUTIONS TO EXACT DERIVATIVE")
        # Loop through the objects in the stack
        for obj in var_deriv_vals:

            # Loop through the variables for the current object
            for var in var_deriv_vals[obj]:

                # Compute the contribution to the exact derivative for the current variable
                contr = np.dot(pert_vals[obj][var], var_deriv_vals[obj][var])

                # Add the contribution for the current variable to the total value
                ans_tot += contr

                if debug_print:
                    print(
                        f"\n    Added contribution for variable {var} from object {obj.__class__.__name__} to ans_tot with a contribution of {contr}."
                    )

        if debug_print:
            print("\n ans_tot = ", ans_tot)

        # Set the perturbed design variable values
        pert_var_values = {}

        # Loop through the objects in the stack
        for obj in pert_vals:

            pert_var_values[obj] = {}

            # Perturb each independent variable for the current object
            for var in pert_vals[obj]:

                # Perturb the design variable by a small amount in the imaginary plane in the previously determined random direction
                if method == "cs":
                    dh = dh_cs
                    pert_var_values[obj][var] = (
                        def_var_values[obj][var] + 1.0j * dh * pert_vals[obj][var]
                    )
                elif method == "fd":
                    dh = dh_fd
                    pert_var_values[obj][var] = (
                        def_var_values[obj][var] + dh * pert_vals[obj][var]
                    )
                else:
                    raise ValueError("Method must be 'fd' or 'cs'.")

            # Set the perturbed variable values
            obj.set_var_values(pert_var_values[obj])

        if debug_print:
            print("\n Perturbed variable values: \n", pert_var_values)

        # Perform the analysis using the perturbed values
        if method == "cs":
            self.analyze(mode="complex")
        elif method == "fd":
            self.analyze(mode="real")

        # Extract the perturbed output values
        outs = self.get_output_values()

        if debug_print:
            print("\n OUTPUTS FOR PERTURBED VARIABLE VALUES:")
            for out in outs:
                print(out, ":", outs[out])

        # Compute the derivative numerically
        cs = 0.0
        fd = 0.0
        if debug_print:
            print("\n ADDING CONTRIBUTIONS TO NUMERICAL DERIVATIVE")
        for out in outs:

            if method == "cs":
                # Compute the complex-step contribution value for the current output
                cs_contr = np.sum(out_derivs[out] * np.imag(outs[out]) / dh)

                # Add the contribution to the total derivative check using complex-step
                cs += cs_contr

                if debug_print:
                    print(
                        f"\n    Added contribution for output {out} to cs with a contribution of {cs_contr}."
                    )
            elif method == "fd":
                # Compute the finite-difference contribution value for the current output
                fd_contr = np.sum(out_derivs[out] * (outs[out] - outs_orig[out]) / dh)

                # Add the contribution to the total derivative check using complex-step
                fd += fd_contr

                if debug_print:
                    print(
                        f"\n    Added contribution for output {out} to fd with a contribution of {fd_contr}."
                    )

        # Compute the relative error and display test results
        if method == "cs":
            err = (ans_tot - cs) / cs
        elif method == "fd":
            err = (ans_tot - fd) / fd

        test_case = f"\n COMBINED adjoint test results for {self.__class__.__name__} object named {self.obj_name}:"

        if print_res:
            print("\n", "-" * len(test_case))
            print(test_case)
            print("-" * len(test_case))

            if method == "cs":
                print("\n %25s  %25s  %25s" % ("Answer", "Complex-step", "Rel Error"))
                print("%25.15e  %25.15e  %25.15e" % (ans_tot, cs, err))

                ratio = ans_tot / cs
                print("\n ratio (ans/cs): ", ratio)
                print("\n 1/ratio (cs/ans): ", 1 / ratio)
                print("-" * len(test_case))
            elif method == "fd":
                print(
                    "\n %25s  %25s  %25s" % ("Answer", "Finite-Difference", "Rel Error")
                )
                print("%25.15e  %25.15e  %25.15e" % (ans_tot, fd, err))

                ratio = ans_tot / fd
                print("\n ratio (ans/fd): ", ratio)
                print("\n 1/ratio (fd/ans): ", 1 / ratio)
                print("-" * len(test_case))

        return err
