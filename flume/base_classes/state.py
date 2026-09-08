import numpy as np
import contextvars
from contextlib import contextmanager

# Define the active sweep for the current thread/task, which really manages which function of interest is being considered for the derivative evaluations. Deaults to None, which is serial version
_current_sweep = contextvars.ContextVar("current_sweep", default=None)

# Define the active "writer" for the current thread/task, which is the Analysis node currently executing its _analyze_adjoint. When set (parallel adjoint), a node's derivative reads/writes are routed to a private per-writer buffer keyed by (sweep_id, writer_id), so independent nodes that accumulate into the same shared State do so lock-free (their contributions are summed later in a reduction step).
_current_writer = contextvars.ContextVar("current_writer", default=None)


class State:
    """
    This is a class to use for establishing a structure for variables and outputs of analysis classes.
    """

    def __init__(self, value, desc: str, deriv=None, source=None):
        """
        Base class that is used to wrap variables and outputs within the Flume framework.

        Parameters
        ----------
        value : float or np.ndarray
            Numeric value for the State
        desc : str
            String that describes the State
        deriv : float or np.ndarray
            Value for the State's derivative. This is None by default, and the user does not need to provide a numeric value when constructing the object
        source : class instance
            Object where the State data is sourced from. When creating states, this should be set to "self", otherwise the framework will raise an error

        Attributes
        ----------
        data_type : type
            The data type for the value of the State
        shape : tuple
            This is only created if the value for the State is a NumPy array, and it corresponds to the shape for the value
        """
        # Set the variable value
        self.value = value

        # Set the variable data type
        self.data_type = type(self.value)

        # Set the variable shape if the value is a numpy array
        if isinstance(self.value, np.ndarray):
            self.shape = np.shape(self.value)

        # Set the description for the variable
        self.desc = desc

        # Initialize the dictionary that will store all of the derivative values (structured such that derivatives will get accumulated from each quantity of interest in parallel without race conditions)
        self._deriv_store = {}
        if deriv is not None:
            # Original, serial path where derivative gets set directly
            self._deriv_store[None] = deriv
        else:
            self._deriv_store[None] = self._zero_deriv()

        # # Set the derivative value
        # # self.deriv = deriv
        # if deriv is not None:
        #     self.deriv = deriv
        # else:
        #     self.deriv = np.zeros_like(self.value)

        # Set the source for the variable
        if source is not None:
            self.source = source
        else:
            raise RuntimeError(
                "Argument 'source' was not set! Make sure that this is set to 'self' during default State construction within the __init__ method."
            )

    def _zero_deriv(self):
        """
        Zeros the derivative value in the event that the derivative is not provided during object creation
        """
        # Return a zero NumPy array if the value is a NumPy array
        if isinstance(self.value, np.ndarray):
            return np.zeros(self.shape)
        # Otherwise, return a scalar
        else:
            return 0.0

    def _deriv_key(self):
        """
        Resolve the derivative-storage key for the current execution context.

        During a parallel adjoint pass a writer is active, so reads/writes are routedto that writer's private contribution buffer keyed by (sweep_id, writer_id). Otherwise (fully serial adjoint, seeding, or post-reduction extraction) the key is just the sweep_id (None in the fully serial case), i.e. the canonical slot.
        """
        # Get the current sweep (i.e. which output of interest is being considered)

        # Get the current writer (i.e. which Analysis node is executing the _analyze_adjoint call)
        sweep = _current_sweep.get()
        writer = _current_writer.get()

        # If the writer is None, then only the sweep is returned
        if writer is None:
            return sweep
        return (sweep, writer)

    @property
    def deriv(self):
        """
        Defines the derivative property, accessing a specific key in the dictionary for the current context (sweep and, if set, writer).
        """

        # Access and return the derivative value for the current context key
        return self._deriv_store[self._deriv_key()]

    @deriv.setter
    def deriv(self, deriv_val):
        """
        Setter method for the deriv property, which assigns the numeric value for the input `deriv_val` in the private derivative storage dictionary
        """

        # Assign the deriv val for the current context key
        self._deriv_store[self._deriv_key()] = deriv_val

        return

    def set_deriv_value(self, deriv_val):
        """
        Method to set the derivative value in the derivative object, accounting for the ContextVar, if necessary. Utilizes the setter method for the `deriv` property.
        """

        # Set the derivative value
        self.deriv = deriv_val

        return

    def get_deriv(self, sweep_id=None):
        """
        Helper method that gets the derivative value for the given sweep ID (which is used to account for which quantity of interest is being tracked)
        """

        # Return the derivative value
        return self._deriv_store[sweep_id]

    def ensure_sweep_slot(self, sweep_id):
        """
        Method used to serially ensure that all keys exist for all of the necessary parallel paths, as the dictionary key insertion during execution is not thread-safe.
        """

        # if sweep_id not in self._deriv_store:
        self._deriv_store[sweep_id] = self._zero_deriv()

        return

    def prepare_writer_slot(self, sweep_id, writer_id):
        """
        Serially create (and zero) the private per-writer contribution buffer for the (sweep_id, writer_id) pair, prior to the parallel adjoint. Pre-creating these keys is required because dictionary key insertion during parallel execution is not thread-safe; during the parallel phase only the values of existing keys are mutated.
        """

        self._deriv_store[(sweep_id, writer_id)] = self._zero_deriv()

        return

    def reduce_seed(self, sweep_id, reader_id):
        """
        Compute the incoming adjoint (seed) for this State in the given sweep and placeit in the reader's private buffer so that the reader's _analyze_adjoint observesthe correct value.

        The seed is the canonical seed for this sweep (set by _add_output_seed for sink outputs, zero otherwise) plus the sum of every consumer's contribution buffer for this sweep. This is called from the producer of the State, which (by the transpose-graph readiness) runs only after all consumers have finished, so every consumer buffer is final.

        Parameters
        ----------
        sweep_id
            The sweep (quantity of interest) currently being processed.
        reader_id
            The id() of the node that will read this seed (the State's producer).
        """

        # Start from the canonical seed for this sweep (0 for non-sink outputs)
        total = self._deriv_store.get(sweep_id)
        if total is None:
            total = self._zero_deriv()
        elif isinstance(total, np.ndarray):
            # Copy so the canonical slot is not mutated in place
            total = total.copy()

        # Add the contributions from all other writers (the consumers) for this sweep
        for key, val in list(self._deriv_store.items()):
            if isinstance(key, tuple) and key[0] == sweep_id and key[1] != reader_id:
                total = total + val

        # Store the reduced seed into the reader's private buffer
        self._deriv_store[(sweep_id, reader_id)] = total

        return

    def reduce_to_canonical(self, sweep_id):
        """
        Sum all per-writer contribution buffers for the given sweep into the canonical sweep_id slot. Used after the parallel adjoint completes so that get_deriv (which reads the canonical slot) returns the total accumulated derivative -- e.g. for design-variable gradient extraction.
        """

        # Initialize the total derivative value
        total = self._zero_deriv()

        # Loop through all of the stored derivative contributions and accumulate into the total design derivative
        for key, val in list(self._deriv_store.items()):
            if isinstance(key, tuple) and key[0] == sweep_id:
                total = total + val

        # Store the total design derivative value
        self._deriv_store[sweep_id] = total

        return


@contextmanager
def sweep_context(sweep_id):
    """
    Context manager for the System, which uses the value of the sweep_id input to perform the adjoint analysis.
    """

    # Get the token for the input sweep ID
    token = _current_sweep.set(sweep_id)

    try:
        # Yield, allowing the rest of the derivative information to process
        yield
    finally:
        # Reset the value of _current_sweep to its previous value before the sweep ID value was set
        _current_sweep.reset(token)
