"""
Utilities for performance evaluation of a DG-FEM benchmark.

.. autofunction:: get_flop_rate
"""
import numpy as np

from arraycontext import (ArrayContext, PyOpenCLArrayContext,
                          PytatoPyOpenCLArrayContext,
                          PytatoJAXArrayContext,
                          EagerJAXArrayContext,
                          rec_multimap_array_container)
from typing import Type
from dg_benchmarks.utils import (get_benchmark_rhs,
                                 get_benchmark_ref_input_arguments_path,
                                 get_benchmark_ref_output_path)

from dg_benchmarks.perf_analysis import get_float64_flops
from time import time


def _instantiate_actx_t(actx_t: Type[ArrayContext]) -> ArrayContext:
    if issubclass(actx_t, (PyOpenCLArrayContext, PytatoPyOpenCLArrayContext)):
        import pyopencl as cl
        import pyopencl.tools as cl_tools

        ctx = cl.create_some_context()
        cq = cl.CommandQueue(ctx)
        allocator = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))
        return actx_t(cq, allocator)
    elif issubclass(actx_t, (EagerJAXArrayContext, PytatoJAXArrayContext)):
        import os
        if os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] != "false":
            raise RuntimeError("environment variable 'XLA_PYTHON_CLIENT_PREALLOCATE'"
                               " is not set 'false'. This is required so that"
                               " backends other than JAX can allocate buffers on the"
                               " device.")

        from jax.config import config
        config.update("jax_enable_x64", True)
        return actx_t()
    else:
        raise NotImplementedError(actx_t)


def get_flop_rate(actx_t: Type[ArrayContext], equation: str, dim: int,
                  degree: int) -> float:
    """
    Runs the benchmarks corresponding to *equation*, *dim*, *degree* using an
    instance of *actx_t* and returns the FLOP-through as "Total number of
    Floating Point Operations per second".
    """
    rhs_clbl = get_benchmark_rhs(equation, dim, degree)
    actx = _instantiate_actx_t(actx_t)

    with open(get_benchmark_ref_input_arguments_path(equation, dim, degree),
              "rb") as fp:
        import pickle
        np_args, np_kwargs = pickle.load(fp)

    with open(get_benchmark_ref_output_path(equation, dim, degree), "rb") as fp:
        ref_output = pickle.load()

    args, kwargs = (tuple(actx.from_numpy(arg) for arg in np_args),
                    {kw: actx.from_numpy(arg) for kw, arg in np_kwargs.items()})

    # {{{ verify correctness for actx_t

    output = rhs_clbl(actx, *args, **kwargs)
    rec_multimap_array_container(np.testing.assert_allclose,
                                 ref_output, actx.to_numpy(output))

    # }}}

    # {{{ warmup rounds

    i_warmup = 0
    t_warmup = 0

    while i_warmup < 20 and t_warmup < 2:
        t_start = time()
        rhs_clbl(actx, *args, **kwargs)
        t_end = time()
        t_warmup += (t_end - t_start)
        i_warmup += 1

    # }}}

    # {{{ warmup rounds

    i_timing = 0
    t_rhs = 0

    while i_timing < 100 and t_rhs < 5:

        t_start = time()
        for _ in range(40):
            rhs_clbl(actx, *args, **kwargs)
        t_end = time()

        t_rhs += (t_end - t_start)
        i_timing += 40

    # }}}

    flops = get_float64_flops(equation, dim, degree)

    return (flops * i_timing) / t_rhs
