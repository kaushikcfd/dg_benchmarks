from dg_benchmarks.wave import get_wave_benchmarks
from dg_benchmarks.maxwell import get_em_benchmarks
from dg_benchmarks import Benchmark, plot_benchmarks

import numpy.typing as npt
import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools


def get_all_benchmarks(queue, allocator) -> npt.NDArray[Benchmark]:
    return np.array([get_wave_benchmarks(queue, allocator),
                     get_em_benchmarks(queue, allocator)])


if __name__ == "__main__":
    ctx = cl.create_some_context()
    cq = cl.CommandQueue(ctx)
    allocator = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))
    fig = plot_benchmarks(get_all_benchmarks(cq, allocator))
    fig.show()
