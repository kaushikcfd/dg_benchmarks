from dg_benchmarks import Benchmark, GrudgeBenchmarkMixin
from dataclasses import dataclass
from arraycontext import ArrayContext, thaw
from grudge import DiscretizationCollection
from functools import cached_property
from meshmode.array_context import (FusionContractorArrayContext,
                                    PyOpenCLArrayContext,
                                    SingleGridWorkBalancingPytatoArrayContext)

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

    mesh = generate_regular_rect_mesh(
            a=(0.0,)*dim,
            b=(1.0,)*dim,
            nelements_per_axis=(4,)*dim)

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

    fields = cavity_mode(thaw(dcoll.nodes(), actx), t=0)

    maxwell_operator.check_bc_coverage(mesh)

    def rhs(t, w):
        return maxwell_operator.operator(t, w)

    dt = actx.to_numpy(
        maxwell_operator.estimate_rk4_timestep(actx, dcoll, fields=fields))

    return rhs, fields, dt


@dataclass(frozen=True, eq=True)
class MaxwellBenchmark(Benchmark, GrudgeBenchmarkMixin):
    actx: ArrayContext
    dim: int
    order: int

    @cached_property
    def _setup_solver(self):
        return setup_em_solver()


@dataclass(frozen=True, eq=True, init=False)
class MaxwellRooflineBenchmark(MaxwellBenchmark):
    def __init__(self, **kwargs):
        cq = kwargs.pop("queue")
        allocator = kwargs.pop("allocator")
        assert "actx" not in kwargs
        kwargs["actx"] = FusionContractorArrayContext(cq, allocator),
        super().__init__(**kwargs)

    def get_runtime(self) -> float:
        from dg_benchmarks.device_data import (DEV_TO_PEAK_BW,
                                               DEV_TO_PEAK_F64_GFLOPS)
        dev, = self.actx.queue.devices
        return max((self.get_nflops()*1e-9)/DEV_TO_PEAK_F64_GFLOPS[dev.name],
                   (self.get_nbytes()*1e-9)/DEV_TO_PEAK_BW[dev.name]
                   )


def get_em_benchmarks(cq, allocator):
    actx1 = PyOpenCLArrayContext(cq, allocator)
    actx2 = SingleGridWorkBalancingPytatoArrayContext(cq, allocator)
    actx3 = FusionContractorArrayContext(cq, allocator)
    benchmarks = [
        [
            [
                MaxwellBenchmark(actx=actx1, dim=dim, order=order),
                MaxwellBenchmark(actx=actx2, dim=dim, order=order),
                MaxwellBenchmark(actx=actx3, dim=dim, order=order),
                MaxwellRooflineBenchmark(queue=cq, allocator=allocator, dim=dim,
                                         order=order),
            ]
            for order in (1, 2, 3, 4)
        ]
        for dim in (2, 3)
    ]

    return np.array(benchmarks)
