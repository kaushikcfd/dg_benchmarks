from dg_benchmarks import Benchmark
from dataclasses import dataclass
from arraycontext import ArrayContext, freeze, thaw
from grudge import DiscretizationCollection
from pytools.obj_array import flat_obj_array
from functools import cached_property
from meshmode.array_context import (FusionContractorArrayContext,
                                    PyOpenCLArrayContext,
                                    SingleGridWorkBalancingPytatoArrayContext)

import numpy as np
import time
import loopy as lp


@lp.memoize_on_disk
def get_loopy_op_map(t_unit):
    return lp.get_op_map(t_unit, subgroup_size=32)


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


@dataclass(frozen=True, eq=True)
class WaveBenchmark(Benchmark):
    actx: ArrayContext
    dim: int
    order: int

    @cached_property
    def _setup_solver(self):
        return setup_wave_solver()

    @cached_property
    def _rhs_as_loopy_knl_for_stats(self):
        if isinstance(self.actx, FusionContractorArrayContext):
            actx = self.actx
        else:
            actx = FusionContractorArrayContext(self.actx.queue,
                                                self.actx.allocator)

        rhs, fields, dt = self._setup_solver()

        from arraycontext.impl.pytato.compile import (
            _get_arg_id_to_arg_and_arg_id_to_descr)
        return (actx
               .compile(rhs)
               .program_cache[_get_arg_id_to_arg_and_arg_id_to_descr((0,
                                                                      fields),
                                                                     {})[1]]
               .pytato_program
               .program
               .default_entrypoint)

    def get_runtime(self) -> float:
        t = 0
        rhs, fields, dt = self._setup_solver()

        rhs = self.actx.compile(rhs)

        n_sim_rounds = 0
        total_sim_time = 0.

        while ((n_sim_rounds < self.min_timings_rounds)
               or (total_sim_time < self.min_timings_secs)):
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
            total_sim_time += (t_start - t_end)

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


@dataclass(frozen=True, eq=True, init=False)
class WaveRooflineBenchmark(WaveBenchmark):
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
