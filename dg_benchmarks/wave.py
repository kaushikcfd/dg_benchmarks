from dg_benchmarks import GrudgeBenchmark
from dataclasses import dataclass
from arraycontext import ArrayContext, thaw
from grudge import DiscretizationCollection
from pytools.obj_array import flat_obj_array
from functools import cached_property
from meshmode.array_context import (FusionContractorArrayContext,
                                    PyOpenCLArrayContext,
                                    SingleGridWorkBalancingPytatoArrayContext)

import numpy as np


def setup_wave_solver(*,
                      actx,
                      dim,
                      order):
    from meshmode.mesh.generation import generate_regular_rect_mesh
    mesh = generate_regular_rect_mesh(
            a=(-0.5,)*dim,
            b=(0.5,)*dim,
            nelements_per_axis=(20,)*dim)

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
    actx: ArrayContext
    dim: int
    order: int

    @cached_property
    def _setup_solver_properties(self):
        return setup_wave_solver(actx=self.actx, dim=self.dim, order=self.order)


@dataclass(frozen=True, eq=True, init=False, repr=True)
class WaveRooflineBenchmark(WaveBenchmark):
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


def get_wave_benchmarks(cq, allocator):
    actx1 = PyOpenCLArrayContext(cq, allocator)
    actx2 = SingleGridWorkBalancingPytatoArrayContext(cq, allocator)
    actx3 = FusionContractorArrayContext(cq, allocator)
    benchmarks = [
        [
            [
                WaveBenchmark(actx=actx1, dim=dim, order=order),
                WaveBenchmark(actx=actx2, dim=dim, order=order),
                WaveBenchmark(actx=actx3, dim=dim, order=order),
                WaveRooflineBenchmark(queue=cq, allocator=allocator, dim=dim,
                                      order=order),
            ]
            for order in (1, 2, 3, 4)
        ]
        for dim in (2, 3)
    ]

    return np.array(benchmarks)
