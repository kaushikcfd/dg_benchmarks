import abc
import time
import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np
import loopy as lp


from dataclasses import dataclass
from functools import cached_property

from arraycontext import thaw, freeze

from meshmode.array_context import FusionContractorArrayContext


BAR_WIDTH = 0.2
GROUP_WIDTH = 1


@dataclass(frozen=True, eq=True)
class Benchmark(abc.ABC):
    warmup_rounds: int = 5
    min_timings_rounds: int = 20
    min_timings_secs: float = 0.2

    @abc.abstractmethod
    def get_runtime(self) -> float:
        pass

    @abc.abstractmethod
    def get_nflops(self) -> int:
        pass

    @abc.abstractmethod
    def get_nbytes(self) -> int:
        pass


@lp.memoize_on_disk
def get_loopy_op_map(t_unit):
    return lp.get_op_map(t_unit, subgroup_size=32)


class GrudgeBenchmarkMixin:
    @cached_property
    def _rhs_as_loopy_knl_for_stats(self):
        if isinstance(self.actx, FusionContractorArrayContext):
            actx = self.actx
        else:
            actx = FusionContractorArrayContext(self.actx.queue,
                                                self.actx.allocator)

        rhs, fields, dt = self._setup_solver()

        from arraycontext.impl.pytato.compile import (
            _get_arg_id_to_arg_and_arg_id_to_descr)
        return (actx
               .compile(rhs)
               .program_cache[_get_arg_id_to_arg_and_arg_id_to_descr((0,
                                                                      fields),
                                                                     {})[1]]
               .pytato_program
               .program
               .default_entrypoint)

    def get_runtime(self) -> float:
        t = 0
        rhs, fields, dt = self._setup_solver()

        rhs = self.actx.compile(rhs)

        # {{{ warmup

        for _ in range(self.warmup_rounds):
            fields = thaw(freeze(fields, self.actx), self.actx)
            fields = rhs(t, fields)

        # }}}

        n_sim_rounds = 0
        total_sim_time = 0.

        while ((n_sim_rounds < self.min_timings_rounds)
               or (total_sim_time < self.min_timings_secs)):
            # {{{ Run 100 rounds

            self.actx.queue.finish()
            t_start = time.time()
            for _ in range(100):
                fields = thaw(freeze(fields, self.actx), self.actx)
                fields = rhs(t, fields)
                t += dt
            self.actx.queue.finish()
            t_end = time.time()

            # }}}

            n_sim_rounds += 100
            total_sim_time += (t_start - t_end)

        return total_sim_time / n_sim_rounds

    def get_nflops(self) -> int:
        t_unit = self._rhs_as_loopy_knl_for_stats
        op_map = get_loopy_op_map(t_unit)
        # TODO: Make sure that all our DOFs are indeed represented as f64-dtypes
        f64_ops = op_map.filter_by(dtype=[np.float64],
                                   kernel_name="_pt_kernel").eval_and_sum({})

        return f64_ops

    def get_nbytes(self) -> int:
        from pytools import product
        from loopy.kernel.array import ArrayBase

        t_unit = self._rhs_as_loopy_knl_for_stats
        knl = t_unit.default_entrypoint
        nfootprint_bytes = 0

        for ary in knl.args:
            if (isinstance(ary, ArrayBase)
                    and ary.address_space == lp.AddressSpace.GLOBAL):
                nfootprint_bytes += (product(ary.shape)
                                    * ary.dtype.itemsize)

        for ary in knl.temporary_variables.values():
            if ary.address_space == lp.AddressSpace.GLOBAL:
                # global temps would be written once and read once
                nfootprint_bytes += (2 * product(ary.shape)
                                    * ary.dtype.itemsize)

        return nfootprint_bytes


def plot_benchmarks(benchmarks: npt.NDArray[Benchmark]):
    """
    :param benchmarks: A 3-dimensional array of benchmarks where that can be
        indexed by group index by the first 2 indices and the 3rd index is
        responsible for different instances of a single group.

    .. Example::

       Something like a Maxwell's Equations DG solver would comprise a group,
       and, something like a DG solver of using polynomials of degree
       :math:`p=2` would comprise an instance of the group.
    """
    if benchmarks.ndim != 4:
        raise RuntimeError("benchmarks must be a 4-dimension np.array")

    nrows, ncols, benchmarks_in_group, bars_per_group = benchmarks.shape

    timings = np.vectorize(lambda x: x.get_runtime())(benchmarks)
    nflops = np.vectorize(lambda x: x.get_nflops())(benchmarks)
    flop_rate = (nflops / timings) * 1e-9

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)

    for irow, row in enumerate(axs):
        for icol, ax in enumerate(row):
            assert all(np.unique(nflops[irow, icol, ibench, :]).size == 1
                       for ibench in range(benchmarks_in_group))
            for ibar in range(bars_per_group):
                ax.bar(GROUP_WIDTH*np.arange(benchmarks_in_group) + ibar*BAR_WIDTH,
                       flop_rate[irow, icol, :, ibar],
                       width=BAR_WIDTH,
                       edgecolor="black"
                       # TODO
                       # label="..."
                       )

    return fig
