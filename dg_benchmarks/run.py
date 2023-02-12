import argparse
import loopy as lp
import numpy as np
import datetime
import pytz

from .measure import get_flop_rate
from .perf_model import get_roofline_flop_rate
from typing import Type, Sequence
from meshmode.arraycontext import (
    BatchedEinsumPytatoPyOpenCLArrayContext,
    PyOpenCLArrayContext as BasePyOpenCLArrayContext,
)
from arraycontext import ArrayContext, PytatoJAXArrayContext, EagerJAXArrayContext


class PyOpenCLArrayContext(BasePyOpenCLArrayContext):
    def transform_loopy_program(self,
                                t_unit: lp.TranslationUnit
                                ) -> lp.TranslationUnit:
        raise NotImplementedError


def _get_actx_t_priority(actx_t):
    if issubclass(actx_t, PytatoJAXArrayContext):
        return 10
    else:
        return 1


def main(equations: Sequence[str],
         dims: Sequence[int],
         degrees: Sequence[int],
         actx_ts: Sequence[Type[ArrayContext]]):
    actx_ts = sorted(actx_ts, key=_get_actx_t_priority)

    flop_rate = np.empty([len(actx_ts), len(dims), len(equations), len(degrees)])
    roofline_flop_rate = np.empty([len(dims), len(equations), len(degrees)])

    for iactx_t, actx_t in enumerate(actx_ts):
        for idim, dim in enumerate(dims):
            for iequation, equation in enumerate(equations):
                for idegree, degree in enumerate(degrees):
                    flop_rate[iactx_t, idim, iequation, idegree] = (
                        get_flop_rate(actx_t, equation, dim, degree)
                    )

    for idim, dim in enumerate(dims):
        for iequation, equation in enumerate(equations):
            for idegree, degree in enumerate(degrees):
                roofline_flop_rate[idim, iequation, idegree] = (
                    get_roofline_flop_rate(actx_t, equation, dim, degree)
                )
    filename = (datetime
                .now(pytz.timezone("America/Chicago"))
                .strftime("archive/case_%Y_%m_%d_%H%M.npz"))

    np.savez(filename,
             equations=equations, degrees=degrees,
             dims=dims, actx_ts=actx_ts, flop_rate=flop_rate,
             roofline_flop_rate=roofline_flop_rate)


_NAME_TO_ACTX_CLASS = {
    "jax:jit": PytatoJAXArrayContext,
    "jax:nojit": EagerJAXArrayContext,
    "pytato:batched_einsum": BatchedEinsumPytatoPyOpenCLArrayContext,
    "pyopencl": PyOpenCLArrayContext,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run DG-FEM benchmarks for arraycontexts",
    )

    parser.add_argument("--equations", metavar="E", type=str,
                        nargs=1,
                        help=("comma separated strings representing which"
                              " equations to time (for ex. 'wave,euler')"),
                        required=True,
                        )
    parser.add_argument("--dims", metavar="D", type=str,
                        help=("comma separated integers representing the"
                              " topological dimensions to run the problems on"
                              " (for ex. 2,3 to run 2D and 3D versions of the"
                              " problem)"),
                        required=True,
                        )
    parser.add_argument("--degrees", metavar="G", type=str,
                        help=("comma separated integers representing the"
                              " polynomial degree of the discretizing function"
                              " spaces to run the problems on (for ex. 1,2,3"
                              " to run using P1,P2,P3 function spaces)"),
                        required=True,
                        )
    parser.add_argument("--actxs", metavar="G", type=str,
                        help=("comma separated integers representing the"
                              " polynomial degree of the discretizing function"
                              " spaces to run the problems on (for ex."
                              " 'pyopencl,jax:jit,pytato:batched_einsum')"),
                        required=True,
                        )

    args = parser.parse_args()
    main(equations=[k.strip() for k in args.equations.split(",")],
         dims=[int(k.strip()) for k in args.dims.split(",")],
         degrees=[int(k.strip()) for k in args.degrees.split(",")])
