from dg_benchmarks.wave import get_wave_benchmarks
from dg_benchmarks.maxwell import get_em_benchmarks
from dg_benchmarks import Benchmark, plot_benchmarks

import numpy.typing as npt
import numpy as np


def get_all_benchmarks() -> npt.NDArray[Benchmark]:
    return np.array([get_wave_benchmarks(),
                     get_em_benchmarks()])


if __name__ == "__main__":
    fig = plot_benchmarks(get_all_benchmarks())
    fig.show()
