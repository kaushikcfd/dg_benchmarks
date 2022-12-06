__copyright__ = """
Copyright (C) 2021 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# Copied almost verbatim from the "Acoustic Pulse" example in Grudge

from dg_benchmarks import GrudgeBenchmark, RooflineBenchmarkMixin
from dataclasses import dataclass
from pytools import memoize_on_first_arg
from pytools.obj_array import make_obj_array
from functools import cache
from typing import Sequence

import numpy as np


def gaussian_profile(
        x_vec, t=0, rho0=1.0, rhoamp=1.0, p0=1.0, gamma=1.4,
        center=None, velocity=None):

    from grudge.models.euler import ConservedEulerField

    dim = len(x_vec)
    if center is None:
        center = np.zeros(shape=(dim,))
    if velocity is None:
        velocity = np.zeros(shape=(dim,))

    lump_loc = center + t * velocity

    # coordinates relative to lump center
    rel_center = make_obj_array(
        [x_vec[i] - lump_loc[i] for i in range(dim)]
    )
    actx = x_vec[0].array_context
    r = actx.np.sqrt(np.dot(rel_center, rel_center))
    expterm = rhoamp * actx.np.exp(1 - r ** 2)

    mass = expterm + rho0
    mom = velocity * mass
    energy = (p0 / (gamma - 1.0)) + np.dot(mom, mom) / (2.0 * mass)

    return ConservedEulerField(mass=mass, energy=energy, momentum=mom)


def make_pulse(amplitude, r0, w, r):
    dim = len(r)
    r_0 = np.zeros(dim)
    r_0 = r_0 + r0
    rel_center = make_obj_array(
        [r[i] - r_0[i] for i in range(dim)]
    )
    actx = r[0].array_context
    rms2 = w * w
    r2 = np.dot(rel_center, rel_center) / rms2
    return amplitude * actx.np.exp(-.5 * r2)


def acoustic_pulse_condition(x_vec, t=0):
    from grudge.models.euler import ConservedEulerField

    dim = len(x_vec)
    vel = np.zeros(shape=(dim,))
    orig = np.zeros(shape=(dim,))
    uniform_gaussian = gaussian_profile(
        x_vec, t=t, center=orig, velocity=vel, rhoamp=0.0)

    amplitude = 1.0
    width = 0.1
    pulse = make_pulse(amplitude, orig, width, x_vec)

    return ConservedEulerField(
        mass=uniform_gaussian.mass,
        energy=uniform_gaussian.energy + pulse,
        momentum=uniform_gaussian.momentum
    )


@memoize_on_first_arg
def setup_euler_solver(actx,
                       *,
                       dim,
                       order):
    from meshmode.mesh import BTAG_ALL
    from grudge.models.euler import EulerOperator, InviscidWallBC
    from meshmode.mesh.generation import generate_regular_rect_mesh

    if dim == 3:
        if order == 1:
            nel_1d = 45
        elif order == 2:
            nel_1d = 30
        elif order == 3:
            nel_1d = 25
        elif order == 4:
            nel_1d = 20
        else:
            raise NotImplementedError(order)
    elif dim == 2:
        if order == 1:
            nel_1d = 1000
        elif order == 2:
            nel_1d = 500
        elif order == 3:
            nel_1d = 500
        elif order == 4:
            nel_1d = 500
        else:
            raise NotImplementedError(order)
    else:
        raise NotImplementedError

    # eos-related parameters
    gamma = 1.4
    overintegration = False

    # {{{ discretization

    box_ll = -0.5
    box_ur = 0.5
    mesh = generate_regular_rect_mesh(
        a=(box_ll,)*dim,
        b=(box_ur,)*dim,
        nelements_per_axis=(nel_1d,)*dim)

    from grudge import DiscretizationCollection
    from grudge.dof_desc import DISCR_TAG_BASE, DISCR_TAG_QUAD
    from meshmode.discretization.poly_element import \
        (default_simplex_group_factory,
         QuadratureSimplexGroupFactory)

    if overintegration:
        quad_tag = DISCR_TAG_QUAD
    else:
        quad_tag = None

    dcoll = DiscretizationCollection(
        actx, mesh,
        discr_tag_to_group_factory={
            DISCR_TAG_BASE: default_simplex_group_factory(
                base_dim=mesh.dim, order=order),
            DISCR_TAG_QUAD: QuadratureSimplexGroupFactory(2*order)
        }
    )

    # }}}

    # {{{ Euler operator

    euler_operator = EulerOperator(
        dcoll,
        bdry_conditions={BTAG_ALL: InviscidWallBC()},
        flux_type="lf",
        gamma=gamma,
        quadrature_tag=quad_tag
    )

    def rhs(t, q):
        return euler_operator.operator(t, q)

    from grudge.dt_utils import h_min_from_volume

    cfl = 0.125
    cn = 0.5*(order + 1)**2
    dt = cfl * actx.to_numpy(h_min_from_volume(dcoll)) / cn

    fields = acoustic_pulse_condition(actx.thaw(dcoll.nodes()))

    # }}}

    return rhs, fields, dt


@dataclass(frozen=True, eq=True, repr=True)
class EulerBenchmark(GrudgeBenchmark):
    def _setup_solver_properties(self, actx):
        return setup_euler_solver(actx, dim=self.dim, order=self.order)

    @property
    def xtick(self) -> str:
        return f"euler.{self.dim}D.P{self.order}"


class EulerRooflineBenchmark(RooflineBenchmarkMixin, EulerBenchmark):
    pass


def get_euler_benchmarks(cl_ctx, dims: Sequence[int], orders: Sequence[int]):

    from arraycontext import PytatoJAXArrayContext
    from meshmode.array_context import (FusionContractorArrayContext,
                                        PyOpenCLArrayContext)

    benchmarks = [
        [
            [
                EulerBenchmark(actx_class=PyOpenCLArrayContext, cl_ctx=cl_ctx,
                               dim=dim, order=order),
                EulerBenchmark(actx_class=FusionContractorArrayContext,
                               cl_ctx=cl_ctx,
                               dim=dim, order=order),
                EulerBenchmark(actx_class=PytatoJAXArrayContext,
                               cl_ctx=cl_ctx,
                               dim=dim, order=order),
                EulerRooflineBenchmark(cl_ctx=cl_ctx, dim=dim, order=order),
            ]
            for order in orders
        ]
        for dim in dims
    ]

    return np.array(benchmarks)
