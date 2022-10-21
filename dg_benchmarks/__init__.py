import abc
import time
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from matplotlib import rc
# rc("text", usetex=True)
import numpy as np
import loopy as lp
import logging

from dataclasses import dataclass
from typing import Type, TYPE_CHECKING
from functools import cached_property, cache

from meshmode.array_context import FusionContractorArrayContext
from arraycontext import thaw, freeze, ArrayContext

plt.style.use("seaborn")
logger = logging.getLogger(__name__)
BAR_WIDTH = 0.2
GROUP_WIDTH = 1


if TYPE_CHECKING:
    import pyopencl as cl


@dataclass(frozen=True, eq=True)
class Benchmark(abc.ABC):
    @property
    def warmup_rounds(self) -> int:
        return 10

    @property
    def min_timing_rounds(self) -> int:
        return 100

    @property
    def min_timing_secs(self) -> float:
        return 0.5

    @abc.abstractproperty
    def xtick(self):
        pass

    @abc.abstractproperty
    def label(self) -> str:
        pass

    @abc.abstractmethod
    def get_runtime(self) -> float:
        pass

    @abc.abstractmethod
    def get_nflops(self) -> int:
        pass

    @abc.abstractmethod
    def get_nbytes(self) -> int:
        pass

    @abc.abstractproperty
    def eval_order(self) -> int:
        """
        Returns the order in which the benchmark is supposed to be run in a
        benchmark suite. Benchmarks with lower value will get run earlier.
        """
        pass


# {{{ actx to get the kernel with loop fusion, contraction

class MinimalBytesKernelException(RuntimeError):
    pass


class MinimumBytesKernelGettingActx(FusionContractorArrayContext):
    def transform_loopy_program(self, t_unit):
        from arraycontext.impl.pytato.compile import FromArrayContextCompile
        if t_unit.default_entrypoint.tags_of_type(FromArrayContextCompile):
            from meshmode.array_context import (
                fuse_same_discretization_entity_loops,
                contract_arrays, _prepare_kernel_for_parallelization)

            knl = t_unit.default_entrypoint

            # {{{ dirty hack (sorry humanity)

            knl = knl.copy(args=[arg.copy(offset=0)
                                 for arg in knl.args])

            # }}}

            # {{{ loop fusion

            knl = fuse_same_discretization_entity_loops(knl)

            # }}}

            # {{{ align kernels for fused einsums

            knl = _prepare_kernel_for_parallelization(knl)

            # }}}

            # {{{ array contraction

            knl = contract_arrays(knl, t_unit.callables_table)

            # }}}

            raise MinimalBytesKernelException(t_unit.with_kernel(knl))
        else:
            return super().transform_loopy_program(t_unit)

# }}}


# @lp.memoize_on_disk
def get_loopy_op_map(t_unit):  # noqa: E302
    # TODO: Re-enable memoize_on_disk for this routine.
    # Currently runs into: cannot pickle 'islpy._isl.Space' object
    return lp.get_op_map(t_unit, subgroup_size=1)


@dataclass(frozen=True, eq=True)
class GrudgeBenchmark(Benchmark):
    actx_class: Type[ArrayContext]
    cl_ctx: "cl.Context"
    dim: int
    order: int

    @property
    def label(self) -> str:
        return self.actx_class.__name__[:-len("ArrayContext")]

    def get_runtime(self) -> float:
        from arraycontext import (PytatoPyOpenCLArrayContext,
                                  PyOpenCLArrayContext,
                                  EagerJAXArrayContext,
                                  PytatoJAXArrayContext
                                  )
        from arraycontext.impls.pytato import _BasePytatoArrayContext
        if issubclass(self.actx_class, (PytatoPyOpenCLArrayContext,
                                        PyOpenCLArrayContext)):
            import pyopencl as cl
            import pyopencl.tools as cl_tools
            cq = cl.CommandQueue(self.cl_ctx)
            allocator = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))
            actx = self.actx_class(cq, allocator)
        elif issubclass(self.actx_class, (EagerJAXArrayContext,
                                          PytatoJAXArrayContext)):
            actx = self.actx_class()
        else:
            raise NotImplementedError(self.actx_class)

        t = 0.0
        rhs, fields, dt = self._setup_solver_properties(actx)

        rhs = actx.compile(rhs)

        # {{{ warmup

        for _ in range(self.warmup_rounds):
            fields = thaw(freeze(fields, actx), actx)
            fields = rhs(t, fields)
            t += dt

        # }}}

        n_sim_rounds = 0
        total_sim_time = 0.

        while ((n_sim_rounds < self.min_timing_rounds)
               or (total_sim_time < self.min_timing_secs)):
            # {{{ Run 100 rounds

            fields = thaw(freeze(fields, actx), actx)
            t_start = time.time()

            for _ in range(100):
                if isinstance(actx, _BasePytatoArrayContext):
                    fields = thaw(freeze(fields, actx), actx)
                fields = rhs(t, fields)
                t += dt

            fields = thaw(freeze(fields, actx), actx)
            t_end = time.time()

            # }}}

            n_sim_rounds += 100
            total_sim_time += (t_end - t_start)

        del fields
        del rhs

        if issubclass(self.actx_class, (PytatoPyOpenCLArrayContext,
                                        PyOpenCLArrayContext)):
            allocator.free_held()
            del allocator
            import gc
            import pyopencl.array as cla
            gc.collect()
            cla.zeros(cq, shape=(10,), dtype=float)

        return total_sim_time / n_sim_rounds

    @property
    def eval_order(self) -> int:
        """
        Returns the order in which the benchmark is supposed to be run in a
        benchmark suite. Benchmarks with lower value will get run earlier.
        """
        from meshmode.array_context import PyOpenCLArrayContext
        from arraycontext import PytatoJAXArrayContext

        if issubclass(self.actx_class, PyOpenCLArrayContext):
            return 1
        elif issubclass(self.actx_class, FusionContractorArrayContext):
            return 2
        elif issubclass(self.actx_class, PytatoJAXArrayContext):
            return 100
        else:
            raise NotImplementedError(self.actx_class)

    def get_nflops(self) -> int:
        raise NotImplementedError

    def get_nbytes(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True, eq=True)
class RooflineBenchmarkMixin:
    cl_ctx: "cl.Context"
    dim: int
    order: int

    @cached_property
    def _rhs_as_loopy_knl_for_stats(self):
        import pyopencl as cl
        import pyopencl.tools as cl_tools
        cq = cl.CommandQueue(self.cl_ctx)
        allocator = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(cq))
        actx = MinimumBytesKernelGettingActx(cq, allocator)

        rhs, fields, dt = self._setup_solver_properties(actx)

        compiled_rhs = actx.compile(rhs)

        try:
            compiled_rhs(0.0, fields)
        except MinimalBytesKernelException as e:
            t_unit, = e.args
            assert isinstance(t_unit, lp.TranslationUnit)
        else:
            raise RuntimeError("Was expecting a 'MinimalBytesKernelException'")

        knl = t_unit.default_entrypoint

        t_unit = t_unit.with_kernel(knl
                                    .copy(
                                        silenced_warnings=(
                                            knl.silenced_warnings
                                            + ["insn_count_subgroups_upper_bound",
                                               "summing_if_branches_ops"])))
        del rhs
        del knl
        allocator.free_held()
        del allocator

        import gc
        import pyopencl.array as cla
        gc.collect()
        cla.zeros(cq, shape=(10,), dtype=float)
        return t_unit

    @cache
    def get_nflops(self) -> int:
        t_unit = self._rhs_as_loopy_knl_for_stats
        op_map = get_loopy_op_map(t_unit)
        knl = t_unit.default_entrypoint
        # TODO: Make sure that all our DOFs are indeed represented as f64-dtypes
        c128_ops = {op_type: (op_map.filter_by(dtype=[np.complex128],
                                               name=op_type,
                                               kernel_name=knl.name)
                              .eval_and_sum({}))
                    for op_type in ["add", "mul", "div"]}
        f64_ops = (op_map.filter_by(dtype=[np.float64],
                                    kernel_name=knl.name).eval_and_sum({})
                   + (2 * c128_ops["add"]
                      + 6 * c128_ops["mul"]
                      + (6 + 3 + 2) * c128_ops["div"]))
        logger.critical(f"DONE: computing nflops for {self}.")
        return f64_ops

    @cache
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

    def get_runtime(self) -> float:
        from dg_benchmarks.device_data import (DEV_TO_PEAK_BW,
                                               DEV_TO_PEAK_F64_GFLOPS)
        dev, = self.cl_ctx.devices
        return max((self.get_nflops()*1e-9)/DEV_TO_PEAK_F64_GFLOPS[dev.name],
                   (self.get_nbytes()*1e-9)/DEV_TO_PEAK_BW[dev.name]
                   )

    @property
    def label(self) -> str:
        return "Roofline"

    @property
    def eval_order(self) -> int:
        """
        Returns the order in which the benchmark is supposed to be run in a
        benchmark suite. Benchmarks with lower value will get run earlier.
        """
        return 0


def _plot_or_record(timings: npt.NDArray[np.float64],
                    nflops: npt.NDArray[np.int64],
                    xticks: npt.NDArray[str],
                    labels: npt.NDArray[str],
                    *,
                    plot,
                    record):

    nrows, ncols, benchmarks_in_group, bars_per_group = timings.shape

    if record:
        from datetime import datetime
        import pytz

        filename = (datetime
                    .now(pytz.timezone("America/Chicago"))
                    .strftime("archive/case_%Y_%m_%d_%H%M.npz"))
        np.savez_compressed(filename,
                            timings=timings,
                            nflops=nflops,
                            xticks=xticks,
                            labels=labels)
        logger.critical(f"Data archived in '{filename}'")

    if plot:
        flop_rate = (nflops / timings) * 1e-9

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                                gridspec_kw={"hspace": 0.3})

        for irow, row in enumerate(axs):
            for icol, ax in enumerate(row):
                assert all(np.unique(nflops[irow, icol, ibench, :]).size == 1
                        for ibench in range(benchmarks_in_group))
                assert all(np.unique(xticks[irow, icol, ibench, :]).size == 1
                        for ibench in range(benchmarks_in_group))
                for ibar in range(bars_per_group):
                    label, = np.unique(labels[irow, icol, :, ibar])
                    ax.bar(
                        GROUP_WIDTH*np.arange(benchmarks_in_group) + ibar*BAR_WIDTH,
                        flop_rate[irow, icol, :, ibar],
                        width=BAR_WIDTH,
                        edgecolor="black",
                        label=label,
                        )

                ax.xaxis.set_major_locator(ticker.FixedLocator(
                    GROUP_WIDTH*np.arange(benchmarks_in_group)
                    + benchmarks_in_group*0.5*BAR_WIDTH))
                ax.xaxis.set_major_formatter(
                    ticker.FixedFormatter(xticks[irow, icol, :, 0]))
                ax.set_ylabel("GFLOPs/s")

        axs[-1, 0].legend(bbox_to_anchor=(1.1, -0.35),
                          loc="lower center",
                          ncol=bars_per_group)

        return fig


def plot_benchmarks(benchmarks: npt.NDArray[Benchmark], save=True):
    """
    :param benchmarks: A 4-dimensional array of benchmarks where that can be
        indexed by group index by the first 2 indices and the 3rd index is
        responsible for different instances of a single group.

    .. Example::

       Something like a Maxwell's Equations DG solver would comprise a group,
       and, something like a DG solver of using polynomials of degree
       :math:`p=2` would comprise an instance of the group.
    """
    if benchmarks.ndim != 4:
        raise RuntimeError("benchmarks must be a 4-dimension np.array")

    nflops = np.vectorize(lambda x: x.get_nflops(), [np.int64])(
        benchmarks[:, :, :, -1]).reshape(benchmarks.shape[:-1] + (1,))
    xticks = np.vectorize(lambda x: x.xtick, [str])(benchmarks)
    labels = np.vectorize(lambda x: x.label, [str])(benchmarks)
    timings = np.empty(benchmarks.shape, dtype=np.float64)

    for ravel_idx in np.argsort(np.vectorize(lambda x: x.eval_order,
                                             [np.int32])(benchmarks)
                                .ravel()):
        idx = np.unravel_index(ravel_idx, benchmarks.shape)
        logger.critical(f"Starting benchmark {benchmarks[idx]}")
        timings[idx] = benchmarks[idx].get_runtime()

    return _plot_or_record(timings, nflops, xticks, labels,
                           plot=True, record=save)


def record_to_file(benchmarks: npt.NDArray[Benchmark]):
    """
    :param benchmarks: A 4-dimensional array of benchmarks where that can be
        indexed by group index by the first 2 indices and the 3rd index is
        responsible for different instances of a single group.

    .. Example::

       Something like a Maxwell's Equations DG solver would comprise a group,
       and, something like a DG solver of using polynomials of degree
       :math:`p=2` would comprise an instance of the group.
    """
    if benchmarks.ndim != 4:
        raise RuntimeError("benchmarks must be a 4-dimension np.array")

    nflops = np.vectorize(lambda x: x.get_nflops(), [np.int64])(
        benchmarks[:, :, :, -1]).reshape(benchmarks.shape[:-1] + (1,))
    xticks = np.vectorize(lambda x: x.xtick, [str])(benchmarks)
    labels = np.vectorize(lambda x: x.label, [str])(benchmarks)
    timings = np.empty(benchmarks.shape, dtype=np.float64)

    for ravel_idx in np.argsort(np.vectorize(lambda x: x.eval_order,
                                             [np.int32])(benchmarks)
                                .ravel()):
        idx = np.unravel_index(ravel_idx, benchmarks.shape)
        logger.critical(f"Starting benchmark {benchmarks[idx]}")
        timings[idx] = benchmarks[idx].get_runtime()

    _plot_or_record(timings, nflops, xticks, labels,
                    plot=False, record=True)


def plot_saved_case(file):
    """
    :param file: A *file* as demanded as :func:`numpy.load`.
    """
    data = np.load(file, allow_pickle=False)
    return _plot_or_record(data["timings"], data["nflops"],
                           data["xticks"], data["labels"],
                           plot=True, record=False)
