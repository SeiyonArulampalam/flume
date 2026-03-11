---
title: "Flume: A Lightweight Framework for Engineering Design Optimization based on Directed Acyclic Graphs"
tags:
  - Multidisciplinary design optimization
  - Optimization framework
  - Python
authors:
  - name: Cameron S. Smith
    orcid: 0000-0002-3397-2223
    affiliation: 1
    corresponding: true
  - name: Graeme J. Kennedy
    orcid: 0000-0002-4223-3043
    affiliation: 1
affiliations:
  - name: Georgia Institute of Technology, United States
    index: 1
    ror: 01zkghx44
date: 11 March 2026
bibliography: paper.bib
---

# Summary

Engineering design problems often involve a sequence of coupled analyses that are used to compute quantities of interest within an optimization problem.
Derivative-based optimization methods provide a scalable way to solve these design problems if the derivatives can be computed efficiently.
To achieve scalability and efficiency, the adjoint method is used to calculate the derivatives of the quantities of interest from the analysis sequence that constitute a system.

To more effectively organize and solve these design problems, we have developed a framework to facilitate the system-level construction and execution for both the forward analysis and adjoint-based derivative evaluation.
This framework, entitled _Flume_, is designed around systems that can be described by a directed acyclic graph (DAG).
While this specific architecture excludes systems that have implicitly coupled relationships, it is applicable for a variety of problems.
Following the DAG structure, each node represents an individual analysis that needs to be performed for the optimization problem, and the edges represent connections between analyses and denote the flow of information.
By describing a system in this manner, the DAG structure can be constructed programmatically, motivating the development of a framework to address this task.

To ensure that the framework is extensible, lightweight, and minimalistic, three primary classes have been implemented in Python to capture the necessary functionality: _State_, _Analysis_, and _System_.
The first class, _State_, provides an object that stores numerical data along with some additional metadata, including type, shape, and object source information.
_Analysis_ is the foundation of Flume, and its primary tasks are to execute the forward and adjoint procedures to obtain the outputs and derivatives needed for optimization.
Finally, _System_ provides a set of methods to declare the objective function, constraints, and design variables that will be used within the optimization problem, as well as a means to visualize the DAG.
To utilize Flume, a user's primary responsibility is to construct the individual analyses, inherited from the _Analysis_ base class, that are needed for their _System_.
By adhering to a set of naming and architectural requirements when scripting these analyses, the framework's backend will automatically connect outputs to variables that share the same name.
This provides the user with a streamlined workflow, enabling them to focus on implementing new features and procedures instead of managing the integration.

The structure of _Flume_ is visualized in \autoref{fig:abstractsystem}, which depicts an abstracted _System_ that encapsulates four distinct _Analysis_ objects.
Arrows that link _Analysis_ objects denote _State_ objects that connect outputs of one discipline to variables another.
_Analysis_ objects that are outlined in red and are labeled with "Top-level **_Analysis_** Object" are those that define output _States_ that are utilized for optimization.
Thus, the arrows that extend beyond the _System_ boundary are _States_ that define design variables, the objective function, or constraint functions for an optimization problem that is wrapped within the framework.

![Abstracted *System* that illustrates the structure of the framework. Here, *State*, *Analysis*, and *System* are emphasized to denote the use of the primary classes provided within the library. Arrows extending beyond the boundary of the *System* denote quantities that are utilized for numerical optimization. \label{fig:abstractsystem}](Images/Flume_DAG_Diagram.svg){width=100%}

# Statement of need

Frameworks for organizing and solving engineering design optimization problems have a common set of features and requirements, including modularity, intuitive user interfaces, object-oriented principles, and minimal overhead [@salas1998framework; @padula2006multidisciplinary].
For optimization problems with many design variables and few constraint functions, the adjoint method becomes the most computationally efficient and scalable way to compute the required derivatives [@mdobook].
Several commercial software tools are available for solving engineering optimization problems, such as the _Adaptive Modeling Language suite_ [@AMLtechnosoft], _HEEDS_ [@heeds], _ISight_ [@isight], _ModelCenter_ [@modelcenter], and _modeFRONTIER_ [@modeFRONTIER], but they are not open-source.
Some open-source frameworks, such as _MACH_ [@Kenway2014MACH] and _FUNtoFEM_ [@Jacobson2018FUNtoFEM], are successful in solving coupled aerostructural design optimization problems, but their use for this specific type of problem makes their extension to other disciplines more challenging.
_pyOptSparse_ [@Wu2020pyoptsparse] is a more general framework for solving constrained nonlinear optimization problems.
However, _pyOptSparse_'s object-oriented approach does not easily define a hierarchical approach for constructing analysis sequences, which introduces implementation complexity when the adjoint method is required.
_OpenMDAO_ [@gray2019openmdao] addresses this organizational challenge by utilizing a hierarchical approach, and _MPhys_ [@Yildirim2025mphys] provides functionality to integrate simulation software within the _OpenMDAO_ framework.
Derivative evaluation is facilitated using the modular analysis and unified derivatives (MAUD) architecture [@Hwang2018maud], but the _OpenMDAO_ framework has many requirements and base classes that create implementation challenges if extending its functionality.

While it is evident that users are presented with several viable options to perform numerical optimization, _Flume_ was created to address the challenges identified above with the existing frameworks.
First, _Flume_ is particularly architected to facilitate the assembly of total derivatives using the adjoint method to perform gradient-based optimization.
This ensures that the framework can be applied to a variety of problems with many design variables and few constraints while preserving scalability and computational efficiency.
Object-oriented principles and inheritance are used, but the framework is also minimalistic with only three primary classes to ensure it is easily extensible.
A hierarchical approach also facilitates the derivative evaluation, as each individual analysis discipline must only consider its respective variable-output combination.
Then, based on the adjoint method, the framework is responsible for propagating information through the model to compute the total derivative information that is used for optimization.
This provides users with a streamlined approach, where they are primarily responsible for implementing the forward analysis and derivative evaluation techniques, while _Flume_ orchestrates the organization and execution of the model.

# Software design

During the construction of the _Flume_ framework, several design decisions were made to prioritize flexibility and simplicity.
The framework is written entirely in Python, and the only core numerical requirement is NumPy to support array operations.
Within the framework, an object-oriented approach is employed with three primary classes:

- _State_: an object that defines the numerical value and its derivative for variables and outputs in the framework. It also stores additional metadata, such as its type, shape, description, and object source information.
- _Analysis_: an abstract base class that users inherit from to define two member functions. The first function uses variables, instances of _State_, to compute outputs, which are also _State_ instances. The second function calculates the derivatives using the adjoint method, propagating derivatives from the output instances to accumulate contributions to the input instances.
- _System_: a container for _Analysis_ objects that defines an analysis sequence. Also, it provides a set of methods to declare the objective function, constraints, and design variables that characterize an optimization problem.

When defining analysis disciplines, users inherit from the _Analysis_ base class, while _State_ and _System_ objects are instantiated directly.
Relying on these three classes provides a minimal application programming interface (API), but their generality ensures that the framework is modular and can be supplemented with additional features for a user's specific needs.
For example, C++ can be integrated with pybind11 to leverage the benefits of a compiled language for computationally-intensive analyses.
This also enables the use of external tools that implement adjoint-based methods to compute derivatives for gradient-based optimization.

The core components of _Flume_ enable the forward analysis and adjoint-based derivative evaluation.
To perform optimization, two optimizer interfaces are currently supported with SciPy [@2020SciPy-NMeth] and ParOpt [@Chin2019paropt].
These interfaces provide the means of connecting an instance of _System_ to an optimizer that will perform the numerical optimization.
This highlights the practicality of the framework, as it contains the necessary functionality to quickly translate an analysis procedure into a design optimization problem.

# Research impact statement

To date, the framework has been implemented for two applications beyond the simpler examples provided within the repository, which detail the construction of _Analysis_ disciplines and _System_ architectures.
_Flume_ has been used for topology optimization, with specific applications for initial post-buckling behavior [@smith2026postbuckling] and inverse design problems.
The network complexity of these systems, such as in \autoref{fig:inversedesign}a and in \autoref{fig:postbuckling}a, emphasizes the importance of a tool like _Flume_.
The representative _System_ diagrams and examples of the topology optimization formulations applied to a sample domain are given for the inverse design and post-buckling problems in \autoref{fig:inversedesign} and \autoref{fig:postbuckling}, respectively.
![Demonstration of *Flume* applied to topology optimization for inverse design with natural frequency applications. \label{fig:inversedesign}](Images/Inverse_Design_Sample.svg){width=90%}

![Demonstration of *Flume* applied to topology optimization for buckling load factor maximization with considerations for initial post-buckling behavior. \label{fig:postbuckling}](Images/Post_Buckling_Sample.svg){width=95%}

# _Flume_ by Example: Constrained Rosenbrock

To demonstrate the application of _Flume_ and how a user interfaces with the _Analysis_ base class and _System_ class, a constrained Rosenbrock problem is considered in this section.
The optimization statement for this example is given by

$$
\begin{aligned}
\min_{x, y} &\quad f(x,y)=(a-x)^2 + b(y - x^2)^2 \\
\textrm{s.t.} &\quad g(x, y) = x^2 + y^2 \leq 1
\end{aligned}
$$

Within _Flume_, this is implemented by constructing three distinct _Analysis_ objects: one to define the design variables, another to compute the objective function, and the last to compute the inequality constraint.
As a demonstration, the code for the objective function _Analysis_ class is included below.
Here, it is worth discussing a few key features regarding the structure of the code.

- The `__init__` method is responsible for defining the default parameter and variable values for the class. This specifies the full set of inputs that are required to compute the outputs, where, generally, parameters are fixed inputs and variables will nominally change throughout the process of an optimization. The variables and parameters are dictionaries that are stored as attributes of the associated class, and the variables are instances of the _State_ class.
- The `_analyze` method defines the forward analysis for evaluating the value of the Rosenbrock function. This utilizes the parameter and variable values stored within the class, and then the output _State_ is assigned into an `outputs` dictionary.
- The `_analyze_adjoint` method performs the adjoint analysis for the computations associated with the current class. Here, the variables are treated as independent, and the contributions from the adjoint variables associated with the outputs are accumulated into any previously computed derivatives for the variables.

When creating subclasses that inherit from _Analysis_, the underscore is used to indicate that `_analyze` and `_analyze_adjoint` are private, helper methods.
This follows the _Template Method_ behavioral design pattern [@gamma1994design], where the user must provide the internal hooks to perform the overarching forward analysis and derivative evaluation procedures.

\small

```python
class Rosenbrock(Analysis):

  def __init__(self, obj_name: str, sub_analyses=[], **kwargs):

      # Set the default parameters
      self.default_parameters = {"a": 1.0, "b": 100.0}

      # Perform the base class object initialization
      super().__init__(obj_name=obj_name, sub_analyses=sub_analyses, **kwargs)

      # Set the default State for the variables
      xvar = State(value=0.0, desc="x state value", source=self)
      yvar = State(value=0.0, desc="y state value", source=self)

      # Construct variables dictionary
      self.variables = {"x": xvar, "y": yvar}

      return

  def _analyze(self):
      # Extract the variable values
      x = self.variables["x"].value
      y = self.variables["y"].value

      # Extract the parameter values
      a = self.parameters["a"]
      b = self.parameters["b"]

      # Compute the value of the Rosenbrock function
      f = (a - x) ** 2 + b * (y - x**2) ** 2

      # Store the outputs
      self.outputs = {}

      self.outputs["f"] = State(
          value=f, desc="Rosenbrock function value", source=self
      )

      return

  def _analyze_adjoint(self):
      # Extract the derivatives of the outputs
      fb = self.outputs["f"].deriv

      # Extract the variable values
      x = self.variables["x"].value
      y = self.variables["y"].value

      # Extract the variable derivatives
      xb = self.variables["x"].deriv
      yb = self.variables["y"].deriv

      # Extract the parameter values
      a = self.parameters["a"]
      b = self.parameters["b"]

      # Compute xb
      xb += (2 * (a - x) * -1 + 2.0 * b * (y - x**2) * -2 * x) * fb

      # Compute yb
      yb += (2 * b * (y - x**2)) * fb

      # Assign the derivative values
      self.variables["x"].set_deriv_value(deriv_val=xb)
      self.variables["y"].set_deriv_value(deriv_val=yb)

      return
```

\normalsize

The _Analysis_ classes that define the design variables and compute the constraint function are similar in structure to the one above.
Next, the section below outlines how the user sets up a _System_ and optimizes with the _FlumeScipyInterface_.
Again, a few salient points are discussed.

- Instances for the _RosenbrockDVs_, _Rosenbrock_ and _RosenbrockConstraint_ objects are each constructed. Here, the instance for _RosenbrockDVs_ is passed as a sub-analysis to the _Rosenbrock_ and _RosenbrockConstraint_ objects during construction, which establishes a connection between the _State_ objects for _x_ and _y_. This ensures that the same values are used when computing the objective or constraint function and provides paths that trace back to the same set of design variables.
- The _System_ is constructed, where the top-level analyses are provided as a list. Effectively, this list defines the _Analysis_ objects that are responsible for computing the objective and constraints for the optimization problem. Any sub-analyses are not required to be provided here, as this information is encoded within the object construction for the top-level analyses.
- The design variables, objective, and constraints are all declared for the _System_, which are stored and accessed when defining the optimization problem. Here, these quantities are declared by using the global names, which are given by `obj_name.local_name`. The user can also provide information for design variable bounds and constraint direction and right-hand side values.
- Finally, in this example, the _FlumeScipyInterface_ is utilized to formulate an optimization problem using the design variable, objective, and constraints declared for the _System_. This interface internally wraps SciPy optimize's `minimize` function, and the method and options can be controlled by the user.

\small

```python
# Construct the design variables object
rosenbrock_dvs = RosenbrockDVs(obj_name="dvs", sub_analyses=[])

# Construct the analysis object for the Rosenbrock function
a = 1.0
b = 100.0

rosenbrock = Rosenbrock(
    obj_name="rosenbrock", sub_analyses=[rosenbrock_dvs], a=a, b=b
)

# Construct the analysis object for the constraint on the design variables
rosenbrock_con = RosenbrockConstraint(
    obj_name="con", sub_analyses=[rosenbrock_dvs]
)

# Construct the system
sys = System(
    sys_name="rosen_sys_con",
    top_level_analysis_list=[rosenbrock, rosenbrock_con],
    log_name="flume.log",
    log_prefix="tests/rosenbrock_constrained",
)

# Declare the design variables for the system
sys.declare_design_vars(
    global_var_name={
        "dvs.x_dv": {"lb": -1.5, "ub": 1.5},
        "dvs.y_dv": {"lb": -1.5, "ub": 1.5},
    }
)

# Declare the objective
sys.declare_objective(global_obj_name="rosenbrock.f")

# Declare the constraint
sys.declare_constraints(
    global_con_name={"con.g": {"direction": "leq", "rhs": 1.0}}
)

# Construct the Scipy interface
interface = FlumeScipyInterface(flume_sys=self.flume_sys)

# Set a random starting point
x0 = np.random.uniform(low=-5.0, high=5.0, size=2)

# Optimize the problem with SciPy minimize
x, res = interface.optimize_system(x0=x0, method="SLSQP")

```

\normalsize

# AI usage disclosure

Regarding code development for _Flume_, generative AI tools, specifically ChatGPT, were minimally used to assist with interpreting Python error statements that were encountered during the framework's construction. Beyond this use case, generative AI was not utilized to write any software, sections of the manuscript, or the documentation.
