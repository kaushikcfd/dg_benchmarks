from dg_benchmarks import GrudgeBenchmark, RooflineBenchmarkMixin
from dataclasses import dataclass
from arraycontext import freeze, thaw
from grudge import DiscretizationCollection
from meshmode.array_context import (FusionContractorArrayContext,
                                    PyOpenCLArrayContext)
from typing import Sequence

import numpy as np
import loopy as lp


@lp.memoize_on_disk
def get_loopy_op_map(t_unit):
    return lp.get_op_map(t_unit, subgroup_size=32)


def setup_em_solver(*,
                    actx,
                    dim,
                    order):
    from grudge.models.em import MaxwellOperator, get_rectangular_cavity_mode
    from meshmode.mesh.generation import generate_regular_rect_mesh

    if dim == 3:
        if order == 1:
            nel_1d = 40
        elif order == 2:
            nel_1d = 35
        elif order == 3:
            nel_1d = 30
        elif order == 4:
            nel_1d = 23
        else:
            raise NotImplementedError(order)
    elif dim == 2:
        if order == 1:
            nel_1d = 1000
        elif order == 2:
            nel_1d = 1000
        elif order == 3:
            nel_1d = 1000
        elif order == 4:
            nel_1d = 1000
        else:
            raise NotImplementedError(order)
    else:
        raise NotImplementedError

    mesh = generate_regular_rect_mesh(
            a=(0.0,)*dim,
            b=(1.0,)*dim,
            nelements_per_axis=(nel_1d,)*dim)

    dcoll = DiscretizationCollection(actx, mesh, order=order)

    if 0:
        epsilon0 = 8.8541878176e-12  # C**2 / (N m**2)
        mu0 = 4*np.pi*1e-7  # N/A**2.
        epsilon = 1*epsilon0
        mu = 1*mu0
    else:
        epsilon = 1
        mu = 1

    maxwell_operator = MaxwellOperator(
        dcoll,
        epsilon,
        mu,
        flux_type=0.5,
        dimensions=dim
    )

    def cavity_mode(x, t=0):
        if dim == 3:
            return get_rectangular_cavity_mode(actx, x, t, 1, (1, 2, 2))
        else:
            return get_rectangular_cavity_mode(actx, x, t, 1, (2, 3))

    fields = thaw(freeze(
                            1j*cavity_mode(thaw(dcoll.nodes(), actx), t=0)*-1j,
                            actx),
                  actx)

    maxwell_operator.check_bc_coverage(mesh)

    def rhs(t, w):
        return maxwell_operator.operator(t, w)

    dt = actx.to_numpy(
        maxwell_operator.estimate_rk4_timestep(actx, dcoll, fields=fields))

    return rhs, fields, dt


@dataclass(frozen=True, eq=True)
class MaxwellBenchmark(GrudgeBenchmark):
    def _setup_solver_properties(self, actx):
        return setup_em_solver(actx=actx, dim=self.dim, order=self.order)

    @property
    def xtick(self) -> str:
        return f"em.{self.dim}D.P{self.order}"


class MaxwellRooflineBenchmark(RooflineBenchmarkMixin, MaxwellBenchmark):
    pass


def get_em_benchmarks(cl_ctx, dims: Sequence[int], orders: Sequence[int]):
    from arraycontext import PytatoJAXArrayContext

    benchmarks = [
        [
            [
                MaxwellBenchmark(actx_class=PyOpenCLArrayContext, cl_ctx=cl_ctx,
                                 dim=dim, order=order),
                MaxwellBenchmark(actx_class=PytatoJAXArrayContext,
                                 cl_ctx=cl_ctx,
                                 dim=dim, order=order),
                MaxwellBenchmark(actx_class=FusionContractorArrayContext,
                                 cl_ctx=cl_ctx,
                                 dim=dim, order=order),
                MaxwellRooflineBenchmark(cl_ctx=cl_ctx, dim=dim, order=order),
            ]
            for order in orders
        ]
        for dim in dims
    ]

    return np.array(benchmarks)
