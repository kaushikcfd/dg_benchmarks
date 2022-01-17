import abc
import numpy.typing as npt
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass


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
