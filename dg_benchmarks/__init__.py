import abc
import time
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from matplotlib import rc
# rc("text", usetex=True)
import numpy as np
import loopy as lp


from dataclasses import dataclass
from functools import cached_property

from arraycontext import thaw, freeze

from meshmode.array_context import FusionContractorArrayContext


BAR_WIDTH = 0.2
GROUP_WIDTH = 1
plt.style.use("seaborn")


@dataclass(frozen=True, eq=True)
class Benchmark(abc.ABC):
    @property
    def warmup_rounds(self) -> int:
        return 5

    @property
    def min_timing_rounds(self) -> int:
        return 20

    @property
    def min_timing_secs(self) -> float:
        return 0.2

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

# @lp.memoize_on_disk
def get_loopy_op_map(t_unit):  # noqa: E302
    # TODO: Re-enable memoize_on_disk for this routine.
    # Currently runs into: cannot pickle 'islpy._isl.Space' object
    return lp.get_op_map(t_unit, subgroup_size=32)


class GrudgeBenchmark(Benchmark):
    @cached_property
    def _rhs_as_loopy_knl_for_stats(self):
        if isinstance(self.actx, FusionContractorArrayContext):
            actx = self.actx
        else:
            actx = FusionContractorArrayContext(self.actx.queue,
                                                self.actx.allocator)

        rhs, fields, dt = self._setup_solver_properties

        from arraycontext.impl.pytato.compile import (
            _get_arg_id_to_arg_and_arg_id_to_descr)

        compiled_rhs = actx.compile(rhs)
        compiled_rhs(0.0, fields)

        return (compiled_rhs
                .program_cache[_get_arg_id_to_arg_and_arg_id_to_descr((0.0,
                                                                       fields),
                                                                      {})[1]]
                .pytato_program
                .program)

    @property
    def label(self) -> str:
        return type(self.actx).__name__[:-len("ArrayContext")]

    def get_runtime(self) -> float:
        t = 0.0
        rhs, fields, dt = self._setup_solver_properties

        rhs = self.actx.compile(rhs)

        # {{{ warmup

        for _ in range(self.warmup_rounds):
            fields = thaw(freeze(fields, self.actx), self.actx)
            fields = rhs(t, fields)
            t += dt

        # }}}

        n_sim_rounds = 0
        total_sim_time = 0.

        while ((n_sim_rounds < self.min_timing_rounds)
               or (total_sim_time < self.min_timing_secs)):
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
            total_sim_time += (t_end - t_start)

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


@dataclass(frozen=True, eq=True, init=False, repr=True)
class RooflineBenchmarkMixin:
    def __init__(self, **kwargs):
        cq = kwargs.pop("queue")
        allocator = kwargs.pop("allocator")
        assert "actx" not in kwargs
        kwargs["actx"] = FusionContractorArrayContext(cq, allocator)
        super().__init__(**kwargs)

    def get_runtime(self) -> float:
        from dg_benchmarks.device_data import (DEV_TO_PEAK_BW,
                                               DEV_TO_PEAK_F64_GFLOPS)
        dev = self.actx.queue.device
        return max((self.get_nflops()*1e-9)/DEV_TO_PEAK_F64_GFLOPS[dev.name],
                   (self.get_nbytes()*1e-9)/DEV_TO_PEAK_BW[dev.name]
                   )

    @property
    def label(self) -> str:
        return "Roofline"


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

    if plot:
        flop_rate = (nflops / timings) * 1e-9

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)

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

        axs[-1, 0].legend(bbox_to_anchor=(1.1, -0.5),
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

    timings = np.vectorize(lambda x: x.get_runtime(), [np.float64])(benchmarks)
    nflops = np.vectorize(lambda x: x.get_nflops(), [np.int64])(benchmarks)
    xticks = np.vectorize(lambda x: x.xtick, [str])(benchmarks)
    labels = np.vectorize(lambda x: x.label, [str])(benchmarks)

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

    timings = np.vectorize(lambda x: x.get_runtime(), [np.float64])(benchmarks)
    nflops = np.vectorize(lambda x: x.get_nflops(), [np.int64])(benchmarks)
    xticks = np.vectorize(lambda x: x.xtick, [str])(benchmarks)
    labels = np.vectorize(lambda x: x.label, [str])(benchmarks)

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
