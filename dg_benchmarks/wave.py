from dg_benchmarks import GrudgeBenchmark, RooflineBenchmarkMixin
from dataclasses import dataclass
from arraycontext import thaw
from grudge import DiscretizationCollection
from pytools.obj_array import flat_obj_array
from meshmode.array_context import (FusionContractorArrayContext,
                                    PyOpenCLArrayContext)
from typing import Sequence

import numpy as np


def setup_wave_solver(*,
                      actx,
                      dim,
                      order):
    from meshmode.mesh.generation import generate_regular_rect_mesh
    if dim == 3:
        if order == 1:
            nel_1d = 50
        elif order == 2:
            nel_1d = 45
        elif order == 3:
            nel_1d = 40
        elif order == 4:
            nel_1d = 30
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
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            nelements_per_axis=(nel_1d,)*dim)

    dcoll = DiscretizationCollection(actx, mesh, order=order)

    def source_f(actx, dcoll, t=0):
        source_center = np.array([0.1, 0.22, 0.33])[:dcoll.dim]
        source_width = 0.05
        source_omega = 3
        nodes = thaw(dcoll.nodes(), actx)
        source_center_dist = flat_obj_array(
            [nodes[i] - source_center[i] for i in range(dcoll.dim)]
        )
        return (
            actx.np.sin(source_omega*t)
            * actx.np.exp(
                -np.dot(source_center_dist, source_center_dist)
                / source_width**2
            )
        )

    x = thaw(dcoll.nodes(), actx)
    ones = dcoll.zeros(actx) + 1
    c = actx.np.where(actx.np.less(np.dot(x, x), 0.15), 0.1 * ones, 0.2 * ones)

    from grudge.models.wave import VariableCoefficientWeakWaveOperator
    from meshmode.mesh import BTAG_ALL, BTAG_NONE

    wave_op = VariableCoefficientWeakWaveOperator(
        dcoll,
        c,
        source_f=source_f,
        dirichlet_tag=BTAG_NONE,
        neumann_tag=BTAG_NONE,
        radiation_tag=BTAG_ALL,
        flux_type="upwind"
    )

    fields = flat_obj_array(
        dcoll.zeros(actx),
        [dcoll.zeros(actx) for i in range(dcoll.dim)]
    )

    dt = 1/3 * actx.to_numpy(wave_op.estimate_rk4_timestep(actx, dcoll,
                                                           fields=fields))

    wave_op.check_bc_coverage(mesh)

    def rhs(t, w):
        return wave_op.operator(t, w)

    return rhs, fields, dt


@dataclass(frozen=True, eq=True, repr=True)
class WaveBenchmark(GrudgeBenchmark):
    def _setup_solver_properties(self, actx):
        return setup_wave_solver(actx=actx, dim=self.dim, order=self.order)

    @property
    def xtick(self) -> str:
        return f"wave.{self.dim}D.P{self.order}"


class WaveRooflineBenchmark(RooflineBenchmarkMixin, WaveBenchmark):
    pass


def get_wave_benchmarks(cl_ctx, dims: Sequence[int], orders: Sequence[int]):
    from arraycontext import PytatoJAXArrayContext

    benchmarks = [
        [
            [
                WaveBenchmark(actx_class=PyOpenCLArrayContext, cl_ctx=cl_ctx,
                              dim=dim, order=order),
                WaveBenchmark(actx_class=PytatoJAXArrayContext,
                              cl_ctx=cl_ctx,
                              dim=dim, order=order),
                WaveBenchmark(actx_class=FusionContractorArrayContext,
                              cl_ctx=cl_ctx,
                              dim=dim, order=order),
                WaveRooflineBenchmark(cl_ctx=cl_ctx, dim=dim, order=order),
            ]
            for order in orders
        ]
        for dim in dims
    ]

    return np.array(benchmarks)
