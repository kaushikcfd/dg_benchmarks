from dg_benchmarks import GrudgeBenchmark, RooflineBenchmarkMixin
from dataclasses import dataclass
from arraycontext import ArrayContext, thaw
from grudge import DiscretizationCollection
from functools import cached_property
from meshmode.array_context import (FusionContractorArrayContext,
                                    PyOpenCLArrayContext,
                                    SingleGridWorkBalancingPytatoArrayContext)

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
    actx: ArrayContext
    dim: int
    order: int

    @cached_property
    def _setup_solver_properties(self):
        return setup_euler_solver(actx=self.actx, dim=self.dim, order=self.order)

    @property
    def xtick(self) -> str:
        return f"euler.{self.dim}D.P{self.order}"


class EulerRooflineBenchmark(RooflineBenchmarkMixin, EulerBenchmark):
    pass


def get_euler_benchmarks(cq, allocator):
    actx1 = PyOpenCLArrayContext(cq, allocator)
    actx2 = SingleGridWorkBalancingPytatoArrayContext(cq, allocator)
    actx3 = FusionContractorArrayContext(cq, allocator)
    benchmarks = [
        [
            [
                EulerBenchmark(actx=actx1, dim=dim, order=order),
                EulerBenchmark(actx=actx2, dim=dim, order=order),
                EulerBenchmark(actx=actx3, dim=dim, order=order),
                EulerRooflineBenchmark(queue=cq, allocator=allocator, dim=dim,
                                      order=order),
            ]
            for order in (1, 2, 3, 4)
        ]
        for dim in (2, 3)
    ]

    return np.array(benchmarks)
