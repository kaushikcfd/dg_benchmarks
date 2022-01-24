from dg_benchmarks.wave import get_wave_benchmarks  # noqa: F401
from dg_benchmarks.maxwell import get_em_benchmarks  # noqa: F401
from dg_benchmarks.euler import get_euler_benchmarks  # noqa: F401
from dg_benchmarks import (Benchmark,  # noqa: F401
                           plot_benchmarks,
                           record_to_file,
                           )

import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401
import logging
logging.basicConfig(level="CRITICAL")


def get_all_benchmarks(cl_ctx, *, dims, orders) -> npt.NDArray[Benchmark]:
    return np.array([
        get_wave_benchmarks(cl_ctx, dims=dims, orders=orders),
        get_em_benchmarks(cl_ctx, dims=dims, orders=orders),
        # get_euler_benchmarks(cl_ctx, dims=dims, orders=orders),
    ])


if __name__ == "__main__":
    # {{{ jax config

    import os
    if os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] != "false":
        raise RuntimeError("environment variable 'XLA_PYTHON_CLIENT_PREALLOCATE'"
                           " is not set 'false'. This is required so that"
                           " backends other than JAX can allocate buffers on the"
                           " device.")

    from jax.config import config
    config.update("jax_enable_x64", True)

    # }}}

    import pyopencl as cl
    cl_ctx = cl.create_some_context()

    record_to_file(get_all_benchmarks(cl_ctx,
                                      orders=(1, 2, 3, 4),
                                      dims=(2, 3),
                                      ))
