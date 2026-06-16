Interfaces Reference
--------------------

This page outlines the optimizer interfaces supported by *Flume*, which currently include `SciPy`_, `ParOpt`_, and `pyOptSparse`_. 

.. autoclass:: flume.interfaces.scipy_interface.FlumeScipyInterface
    :members:
    :special-members: __init__

.. autoclass:: flume.interfaces.paropt_interface.FlumeParOptInterface
    :members:
    :special-members: __init__

.. autoclass:: flume.interfaces.pyoptsparse_interface.FlumePyOptSparseInterface
    :members:
    :special-members: __init__

.. _SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
.. _ParOpt: https://github.com/smdogroup/paropt
.. _pyOptSparse: https://github.com/mdolab/pyoptsparse 