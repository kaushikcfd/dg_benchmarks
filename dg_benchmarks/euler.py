from dg_benchmarks import GrudgeBenchmark, RooflineBenchmarkMixin
from dataclasses import dataclass
from arraycontext import thaw
from grudge import DiscretizationCollection
from functools import cache
from meshmode.array_context import (FusionContractorArrayContext,
                                    PyOpenCLArrayContext)
from typing import Sequence

import numpy as np


def setup_euler_solver(*,
                       actx,
                       dim,
                       order):

    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import (default_simplex_group_factory,
                                                      QuadratureSimplexGroupFactory)
    from grudge.models.euler import vortex_initial_condition, EulerOperator
    from grudge.dt_utils import h_min_from_volume
    from meshmode.mesh.generation import generate_regular_rect_mesh

    # EOS-related parameters
    gamma = 1.4
    cfl = 0.01
    cn = 0.5*(order + 1)**2

    mesh = generate_regular_rect_mesh(
        a=(0, -5),
        b=(20, 5),
        nelements_per_axis=(16, 8),
        periodic=(True, True))

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(
                base_dim=mesh.dim, order=order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order)
        }
    )

    euler_operator = EulerOperator(dcoll,
                                   flux_type="central",
                                   gamma=gamma)

    fields = vortex_initial_condition(thaw(dcoll.nodes(), actx))
    dt = cfl * actx.to_numpy(h_min_from_volume(dcoll)) / cn

    def rhs(t, q):
        return euler_operator.operator(t, q)

    return rhs, fields, dt


@dataclass(frozen=True, eq=True, repr=True)
class EulerBenchmark(GrudgeBenchmark):
    @cache
    def _setup_solver_properties(self, actx):
        return setup_euler_solver(actx=actx, dim=self.dim, order=self.order)

    @property
    def xtick(self) -> str:
        return f"euler.{self.dim}D.P{self.order}"


class EulerRooflineBenchmark(RooflineBenchmarkMixin, EulerBenchmark):
    pass


def get_euler_benchmarks(cl_ctx, dims: Sequence[int], orders: Sequence[int]):

    benchmarks = [
        [
            [
                EulerBenchmark(actx_class=PyOpenCLArrayContext, cl_ctx=cl_ctx,
                               dim=dim, order=order),
                EulerBenchmark(actx_class=FusionContractorArrayContext,
                               cl_ctx=cl_ctx,
                               dim=dim, order=order),
                EulerRooflineBenchmark(cl_ctx=cl_ctx, dim=dim, order=order),
            ]
            for order in orders
        ]
        for dim in dims
    ]

    return np.array(benchmarks)
