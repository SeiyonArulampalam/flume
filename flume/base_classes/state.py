import numpy as np
import contextvars
from contextlib import contextmanager

# Define the active sweep for the current thread/task, which really manages which function of interest is being considered for the derivative evaluations. Deaults to None, which is serial version
_current_sweep = contextvars.ContextVar("current_sweep", default=None)


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
            self._zero_deriv()

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

    @property
    def deriv(self):
        """
        Defines the derivative property, accessing a specific key in the dictionary for the current ContextVar.
        """
        # Access and return the derivative value for the current context var
        return self._deriv_store[_current_sweep.get()]

    @deriv.setter
    def deriv(self, deriv_val):
        """
        Setter method for the deriv property, which assigns the numeric value for the input `deriv_val` in the private derivative storage dictionary
        """

        # Assign the deriv val for the current ContextVar value
        self._deriv_store[_current_sweep.get()] = deriv_val

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

        if sweep_id not in self._deriv_store:
            self._deriv_store[sweep_id] = self._zero_deriv()

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
