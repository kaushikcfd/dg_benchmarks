"""Prediction-adjacent performance tester."""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
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
import logging
import time
import numpy as np
import math
from functools import partial

from meshmode.mesh import BTAG_ALL, BTAG_NONE  # noqa
from grudge.shortcuts import make_visualizer
from grudge.dof_desc import BoundaryDomainTag
from mirgecom.discretization import create_discretization_collection


from mirgecom.euler import euler_operator
from mirgecom.navierstokes import ns_operator
from mirgecom.simutil import (
    get_sim_timestep,
    generate_and_distribute_mesh,
    write_visfile,
    force_evaluation
)
from mirgecom.io import make_init_message
from mirgecom.mpi import mpi_entry_point
from mirgecom.initializers import (
    MixtureInitializer,
    Uniform
)
from mirgecom.eos import (
    PyrometheusMixture,
    IdealSingleGas
)
from mirgecom.transport import SimpleTransport
from mirgecom.gas_model import GasModel
from mirgecom.artificial_viscosity import (
    av_laplacian_operator,
    smoothness_indicator
)
import cantera

logger = logging.getLogger(__name__)


class MyRuntimeError(RuntimeError):
    """Simple exception for fatal driver errors."""

    pass


# Box grid generator widget lifted from @majosm and slightly bent
def _get_box_mesh(dim, a, b, n, t=None, periodic=None):
    if periodic is None:
        periodic = (False)*dim

    dim_names = ["x", "y", "z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, n=n, boundary_tag_to_face=bttf, mesh_type=t,
               periodic=periodic)


class InitSponge:
    r"""Solution initializer for flow in the ACT-II facility.

    This initializer creates a physics-consistent flow solution
    given the top and bottom geometry profiles and an EOS using isentropic
    flow relations.

    The flow is initialized from the inlet stagnations pressure, P0, and
    stagnation temperature T0.

    geometry locations are linearly interpolated between given data points

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, *, x0, thickness, amplitude):
        r"""Initialize the sponge parameters.

        Parameters
        ----------
        x0: float
            sponge starting x location
        thickness: float
            sponge extent
        amplitude: float
            sponge strength modifier
        """
        self._x0 = x0
        self._thickness = thickness
        self._amplitude = amplitude

    def __call__(self, x_vec, *, time=0.0):
        """Create the sponge intensity at locations *x_vec*.

        Parameters
        ----------
        x_vec: numpy.ndarray
            Coordinates at which solution is desired
        time: float
            Time at which solution is desired. The strength is (optionally)
            dependent on time
        """
        xpos = x_vec[0]
        actx = xpos.array_context
        zeros = 0*xpos
        x0 = zeros + self._x0

        return self._amplitude * actx.np.where(
            actx.np.greater(xpos, x0),
            (zeros + ((xpos - self._x0)/self._thickness)
             * ((xpos - self._x0) / self._thickness)),
            zeros + 0.0
        )


@mpi_entry_point
def main(dim, order, actx, *, single_gas_only):
    """Drive example."""

    casename = "mirgecom.comboozle"

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    assert nproc == 1

    print(f"Main start: {time.ctime(time.time())}")
    from mirgecom.simutil import global_reduce as _global_reduce
    global_reduce = partial(_global_reduce, comm=comm)

    # {{{ Some discretization parameters

    # - scales the size of the domain
    x_scale = 1
    y_scale = 1
    z_scale = 1

    # - params for unscaled npts/axis
    domain_xlen = .01
    domain_ylen = .01
    domain_zlen = .01
    chlen = .0025  # default to 4 elements/axis = x_len/chlen

    # }}} discretization params

    # {{{ Time stepping control

    # This example runs only 3 steps by default (to keep CI ~short)
    # With the mixture defined below, equilibrium is achieved at ~40ms
    # To run to equilibrium, set t_final >= 40ms.

    # Time loop control parameters
    t_final = 2e-12
    current_cfl = 0.05
    current_dt = 1e-13
    current_t = 0.5
    constant_cfl = False
    integrator = "euler"

    # i.o frequencies
    nstatus = 100
    nviz = 100
    nhealth = 100
    nrestart = 1000
    do_checkpoint = 0
    boundary_report = 0
    do_callbacks = 0

    # }}}  Time stepping control

    # {{{ Some solution parameters

    dummy_rhs_only = 0
    timestepping_on = 1
    av_on = 1
    sponge_on = 1
    health_pres_min = 0.
    health_pres_max = 10000000.
    init_temperature = 1500.0
    init_pressure = 101325.
    init_density = 0.23397065362031969

    # }}}

    # {{{ Boundary configuration params

    adiabatic_boundary = 0
    periodic_boundary = 1
    multiple_boundaries = False

    # }}}

    # {{{ Simulation control parameters

    grid_only = 0
    discr_only = 0
    inviscid_only = 0
    inert_only = 0
    init_only = 0
    nspecies = 7
    use_cantera = 0

    # }}}

    # coarse-scale grid/domain control

    weak_scale = 1  # scales domain uniformly, keeping dt constant

    # AV / Shock-capturing parameters
    alpha_sc = 0.5
    s0_sc = -5.0
    kappa_sc = 0.5

    # sponge parameters
    sponge_thickness = 0.09
    sponge_amp = 1.0/current_dt/1000.
    sponge_x0 = 0.9

    # param sanity check
    allowed_integrators = ["rk4", "euler", "lsrk54", "lsrk144", "compiled_lsrk45"]
    if integrator not in allowed_integrators:
        error_message = "Invalid time integrator: {}".format(integrator)
        raise RuntimeError(error_message)

    print("#### Simluation control data: ####")
    print(f"\tCasename: {casename}")
    print("----- run control ------")
    print(f"\t{grid_only=},{discr_only=},{inert_only=}")
    print(f"\t{single_gas_only=},{dummy_rhs_only=}")
    print(f"\t{periodic_boundary=},{adiabatic_boundary=}")
    print(f"\t{timestepping_on=}, {inviscid_only=}")
    print(f"\t{av_on=}, {sponge_on=}, {do_callbacks=}")
    print(f"\t{nspecies=}, {init_only=}")
    print(f"\t{health_pres_min=}, {health_pres_max=}")
    print("---- timestepping ------")
    print(f"\tcurrent_dt = {current_dt}")
    print(f"\tt_final = {t_final}")
    print(f"\tconstant_cfl = {constant_cfl}")
    print(f"\tTime integration {integrator}")
    if constant_cfl:
        print(f"\tcfl = {current_cfl}")
    print("---- i/o frequencies -----")
    print(f"\tnviz = {nviz}")
    print(f"\tnrestart = {nrestart}")
    print(f"\tnhealth = {nhealth}")
    print(f"\tnstatus = {nstatus}")
    print("----- domain ------")
    print(f"\tspatial dimension: {dim}")
    print(f"\tdomain_xlen = {domain_xlen}")
    print(f"\tx_scale = {x_scale}")
    if dim == 2:
        print(f"\tdomain_ylen = {domain_xlen}")
        print(f"\ty_scale = {y_scale}")
    elif dim == 3:
        print(f"\tdomain_zlen = {domain_xlen}")
        print(f"\tz_scale = {z_scale}")
    else:
        raise NotImplementedError(dim)

    print("\t----- discretization ----")
    print(f"\tchar_len = {chlen}")
    print(f"\torder = {order}")

    if av_on:
        print(f"\tShock capturing parameters: {alpha_sc=}, "
              f" {s0_sc=}, {kappa_sc=}")

    if sponge_on:
        print(f"Sponge parameters: {sponge_amp=}, {sponge_thickness=},"
              f" {sponge_x0=}")

    print("\tDependent variable logging is OFF.")
    print("#### Simluation control data: ####")

    xsize = domain_xlen*x_scale*weak_scale
    ysize = domain_ylen*y_scale*weak_scale
    zsize = domain_zlen*z_scale*weak_scale

    ncx = int(xsize / chlen)
    ncy = int(ysize / chlen)
    ncz = int(zsize / chlen)

    if dim == 2:
        raise NotImplementedError
        # GLOBAL_NDOFS = 3e6
        # n_refine = math.ceil(
        #     (GLOBAL_NDOFS / (2 * ncx * ncy * ndofs_per_cell))
        #     ** (1/2))
    elif dim == 3:
        if order == 1:
            ndofs_per_cell = 4
        elif order == 2:
            ndofs_per_cell = 10
        elif order == 3:
            ndofs_per_cell = 20
        elif order == 4:
            ndofs_per_cell = 35
        else:
            raise NotImplementedError

        GLOBAL_NDOFS = 1e6
        n_refine = math.ceil((GLOBAL_NDOFS / (6 * ncx * ncy * ncz * ndofs_per_cell))
                             ** (1/3))
    else:
        raise NotImplementedError

    npts_x = ncx * n_refine + 1
    npts_y = ncy * n_refine + 1
    npts_z = ncz * n_refine + 1

    x0 = xsize/2
    y0 = ysize/2
    z0 = zsize/2

    xleft = x0 - xsize/2
    xright = x0 + xsize/2
    ybottom = y0 - ysize/2
    ytop = y0 + ysize/2
    zback = z0 - zsize/2
    zfront = z0 + zsize/2

    npts_axis = (npts_x,)
    box_ll = (xleft,)
    box_ur = (xright,)
    if dim == 2:
        npts_axis = (npts_x, npts_y)
        box_ll = (xleft, ybottom)
        box_ur = (xright, ytop)
    elif dim == 3:
        npts_axis = (npts_x, npts_y, npts_z)
        box_ll = (xleft, ybottom, zback)
        box_ur = (xright, ytop, zfront)
    else:
        raise NotImplementedError(dim)

    periodic = (periodic_boundary == 1,)*dim
    print(f"---- Mesh generator inputs -----\n"
          f"\tDomain: [{box_ll}, {box_ur}], {periodic=}\n"
          f"\tNpts/axis: {npts_axis}")

    if single_gas_only:
        inert_only = 1

    wall_temperature = init_temperature
    temperature_seed = init_temperature
    debug = False

    print(f"ACTX setup start: {time.ctime(time.time())}")

    rst_path = "restart_data/"
    rst_pattern = (
        rst_path + "{cname}-{step:04d}-{rank:04d}.pkl"
    )
    generate_mesh = partial(_get_box_mesh, dim, a=box_ll, b=box_ur, n=npts_axis,
                            periodic=periodic)

    local_mesh, global_nelements = generate_and_distribute_mesh(comm,
                                                                generate_mesh)
    local_nelements = local_mesh.nelements

    print(f"{dim=},{order=},{local_nelements=},{global_nelements=}")
    if grid_only:
        return 0

    dcoll = create_discretization_collection(actx, local_mesh, order)
    nodes = actx.thaw(dcoll.nodes())
    ones = dcoll.zeros(actx) + 1.0

    print(f"ACTX Setup end -> Solution init start: {time.ctime(time.time())}")

    def vol_min(x):
        from grudge.op import nodal_min
        return actx.to_numpy(nodal_min(dcoll, "vol", x))[()]

    def vol_max(x):
        from grudge.op import nodal_max
        return actx.to_numpy(nodal_max(dcoll, "vol", x))[()]

    from grudge.dt_utils import characteristic_lengthscales
    length_scales = characteristic_lengthscales(actx, dcoll)
    h_min = vol_min(length_scales)
    h_max = vol_max(length_scales)

    quadrature_tag = None

    ones = dcoll.zeros(actx) + 1.0

    print("----- Discretization info ----")
    print(f"Discr: {nodes.shape=}, {order=}, {h_min=}, {h_max=}")
    print(f"{local_nelements=},{global_nelements=}")

    if discr_only:
        return 0

    vis_timer = None

    casename = f"{casename}-d{dim}p{order}e{global_nelements}n"

    if single_gas_only:
        nspecies = 0
        init_y = 0
    elif use_cantera:

        # {{{  Set up initial state using Cantera

        # Use Cantera for initialization
        # -- Pick up a CTI for the thermochemistry config
        # --- Note: Users may add their own CTI file by dropping it into
        # ---       mirgecom/mechanisms alongside the other CTI files.
        from mirgecom.mechanisms import get_mechanism_input
        mech_input = get_mechanism_input("uiuc")
        cantera_soln = cantera.Solution(name="gas", yaml=mech_input)
        nspecies = cantera_soln.n_species

        # Initial temperature, pressure, and mixutre mole fractions are needed
        # set up the initial state in Cantera.
        # Parameters for calculating the amounts of fuel, oxidizer, and inert species
        equiv_ratio = 1.0
        ox_di_ratio = 0.21
        stoich_ratio = 3.0
        # Grab array indices for the specific species, ethylene, oxygen, and nitrogen
        i_fu = cantera_soln.species_index("C2H4")
        i_ox = cantera_soln.species_index("O2")
        i_di = cantera_soln.species_index("N2")
        x = np.zeros(nspecies)
        # Set the species mole fractions according to our desired fuel/air mixture
        x[i_fu] = (ox_di_ratio*equiv_ratio)/(stoich_ratio+ox_di_ratio*equiv_ratio)
        x[i_ox] = stoich_ratio*x[i_fu]/equiv_ratio
        x[i_di] = (1.0-ox_di_ratio)*x[i_ox]/ox_di_ratio
        # Uncomment next line to make pylint fail when it can't find cantera.one_atm
        one_atm = cantera.one_atm  # pylint: disable=no-member

        # Let the user know about how Cantera is being initilized
        print(f"Input state (T,P,X) = ({temperature_seed}, {one_atm}, {x})")
        # Set Cantera internal gas temperature, pressure, and mole fractios
        cantera_soln.TPX = temperature_seed, one_atm, x
        # Pull temperature, total density, mass fractions, and pressure from Cantera
        # We need total density, mass fractions to initialize the fluid/gas state.
        can_t, can_rho, can_y = cantera_soln.TDY
        can_p = cantera_soln.P
        init_pressure = can_p
        init_density = can_rho
        init_temperature = can_t
        init_y = can_y
        print(f"Cantera state (rho,T,P,Y) = ({can_rho}, {can_t}, {can_p}, {can_y}")
        # *can_t*, *can_p* should not differ (much) from user's initial data,
        # but we want to ensure that we use the same starting point as Cantera,
        # so we use Cantera's version of these data.

        # }}}

    # {{{ Create Pyrometheus thermochemistry object & EOS

    # Create a Pyrometheus EOS with the Cantera soln. Pyrometheus uses Cantera and
    # generates a set of methods to calculate chemothermomechanical properties and
    # states for this particular mechanism.
    if inert_only or single_gas_only:
        eos = IdealSingleGas()
    else:
        if use_cantera:
            from mirgecom.thermochemistry import make_pyrometheus_mechanism_class
            pyro_mechanism = make_pyrometheus_mechanism_class(cantera_soln)(actx.np)
            eos = PyrometheusMixture(pyro_mechanism,
                                     temperature_guess=temperature_seed)
        else:
            from mirgecom.thermochemistry \
                import get_thermochemistry_class_by_mechanism_name
            pyro_mechanism = \
                get_thermochemistry_class_by_mechanism_name("uiuc")(actx.np)
            nspecies = pyro_mechanism.num_species
            # species_names = pyro_mechanism.species_names
            eos = PyrometheusMixture(pyro_mechanism,
                                     temperature_guess=temperature_seed)
            init_y = [0.06372925, 0.21806609, 0., 0., 0., 0., 0.71820466]

    # {{{ Initialize simple transport model

    kappa = 10
    spec_diffusivity = 0 * np.ones(nspecies)
    sigma = 1e-5
    if inviscid_only:
        transport_model = None
        gas_model = GasModel(eos=eos)
    else:
        transport_model = SimpleTransport(viscosity=sigma,
                                          thermal_conductivity=kappa,
                                          species_diffusivity=spec_diffusivity)
        gas_model = GasModel(eos=eos, transport=transport_model)

    # }}}

    from pytools.obj_array import make_obj_array

    if inert_only:
        compute_temperature_update = None
    else:
        def get_temperature_update(cv, temperature):
            y = cv.species_mass_fractions
            e = gas_model.eos.internal_energy(cv) / cv.mass
            return pyro_mechanism.get_temperature_update_energy(e, temperature, y)
        compute_temperature_update = actx.compile(get_temperature_update)

    from mirgecom.gas_model import make_fluid_state

    def get_fluid_state(cv, tseed):
        return make_fluid_state(cv=cv, gas_model=gas_model,
                                temperature_seed=tseed)

    construct_fluid_state = actx.compile(get_fluid_state)

    # }}}

    # {{{ MIRGE-Com state initialization

    # Initialize the fluid/gas state with Cantera-consistent data:
    # (density, pressure, temperature, mass_fractions)
    velocity = np.zeros(shape=(dim,))
    if single_gas_only or inert_only:
        initializer = Uniform(dim=dim, p=init_pressure, rho=init_density,
                              velocity=velocity, nspecies=nspecies)
    else:
        initializer = MixtureInitializer(dim=dim, nspecies=nspecies,
                                         pressure=init_pressure,
                                         temperature=init_temperature,
                                         massfractions=init_y, velocity=velocity)

    from mirgecom.boundary import (
        AdiabaticNoslipWallBoundary,
        IsothermalWallBoundary
    )
    adiabatic_wall = AdiabaticNoslipWallBoundary()
    isothermal_wall = IsothermalWallBoundary(wall_temperature=wall_temperature)
    if adiabatic_boundary:
        wall = adiabatic_wall
    else:
        wall = isothermal_wall

    boundaries = {}  # periodic-compatible
    if not periodic:
        if multiple_boundaries:
            for idir in range(dim):
                boundaries[BoundaryDomainTag(f"+{idir}")] = wall
                boundaries[BoundaryDomainTag(f"-{idir}")] = wall
        else:
            boundaries = {BTAG_ALL: wall}

    if boundary_report:
        from mirgecom.simutil import boundary_report
        boundary_report(dcoll, boundaries, f"{casename}_boundaries.yaml")

    # Set the current state from time 0
    current_cv = initializer(eos=gas_model.eos, x_vec=nodes)
    temperature_seed = temperature_seed * ones

    # The temperature_seed going into this function is:
    # - At time 0: the initial temperature input data (maybe from Cantera)
    # - On restart: the restarted temperature seed from restart file (saving
    #               the *seed* allows restarts to be deterministic
    current_fluid_state = construct_fluid_state(current_cv, temperature_seed)
    current_dv = current_fluid_state.dv
    temperature_seed = current_dv.temperature

    if sponge_on:
        sponge_sigma = InitSponge(x0=sponge_x0, thickness=sponge_thickness,
                                 amplitude=sponge_amp)(x_vec=nodes)
        sponge_ref_cv = initializer(eos=gas_model.eos, x_vec=nodes)

        # sponge function
        def _sponge(cv):
            return sponge_sigma*(sponge_ref_cv - cv)

    # Inspection at physics debugging time
    if debug:
        print("Initial MIRGE-Com state:")
        print(f"Initial DV pressure: {current_fluid_state.pressure}")
        print(f"Initial DV temperature: {current_fluid_state.temperature}")

    # }}}

    visualizer = make_visualizer(dcoll)
    initname = initializer.__class__.__name__
    eosname = gas_model.eos.__class__.__name__
    init_message = make_init_message(dim=dim, order=order,
                                     nelements=local_nelements,
                                     global_nelements=global_nelements,
                                     dt=current_dt, t_final=t_final, nstatus=nstatus,
                                     nviz=nviz, cfl=current_cfl,
                                     constant_cfl=constant_cfl, initname=initname,
                                     eosname=eosname, casename=casename)

    if inert_only == 0 and use_cantera:
        # Cantera equilibrate calculates the expected end
        # state @ chemical equilibrium
        # i.e. the expected state after all reactions
        cantera_soln.equilibrate("UV")
        eq_temperature, eq_density, eq_mass_fractions = cantera_soln.TDY
        eq_pressure = cantera_soln.P

        # Report the expected final state to the user
        logger.info(init_message)
        logger.info(f"Expected equilibrium state:"
                    f" {eq_pressure=}, {eq_temperature=},"
                    f" {eq_density=}, {eq_mass_fractions=}")

    def my_write_status(dt, cfl, dv=None):
        status_msg = f"------ {dt=}" if constant_cfl else f"----- {cfl=}"
        if ((dv is not None)):

            temp = dv.temperature
            press = dv.pressure

            from grudge.op import nodal_min_loc, nodal_max_loc
            tmin = global_reduce(actx.to_numpy(nodal_min_loc(dcoll, "vol", temp)),
                                 op="min")
            tmax = global_reduce(actx.to_numpy(nodal_max_loc(dcoll, "vol", temp)),
                                 op="max")
            pmin = global_reduce(actx.to_numpy(nodal_min_loc(dcoll, "vol", press)),
                                 op="min")
            pmax = global_reduce(actx.to_numpy(nodal_max_loc(dcoll, "vol", press)),
                                 op="max")
            dv_status_msg = f"\nP({pmin}, {pmax}), T({tmin}, {tmax})"
            status_msg = status_msg + dv_status_msg

        logger.info(status_msg)

    def my_write_viz(step, t, cv, dv):
        viz_fields = [("cv", cv), ("dv", dv)]
        write_visfile(dcoll, viz_fields, visualizer, vizname=casename,
                      step=step, t=t, overwrite=True, vis_timer=vis_timer)

    def my_write_restart(step, t, state, temperature_seed):
        rst_fname = rst_pattern.format(cname=casename, step=step, rank=0)
        rst_data = {
            "local_mesh": local_mesh,
            "cv": state.cv,
            "temperature_seed": temperature_seed,
            "t": t,
            "step": step,
            "order": order,
            "global_nelements": global_nelements,
            "num_parts": nproc
        }
        from mirgecom.restart import write_restart_file
        write_restart_file(actx, rst_data, rst_fname, comm)

    def my_health_check(cv, dv):
        import grudge.op as op
        health_error = False

        pressure = dv.pressure
        temperature = dv.temperature

        from mirgecom.simutil import check_naninf_local
        if check_naninf_local(dcoll, "vol", pressure):
            health_error = True
            logger.info("Invalid pressure data found.")

        if check_naninf_local(dcoll, "vol", temperature):
            health_error = True
            logger.info("Invalid temperature data found.")

        if inert_only == 0:
            # This check is the temperature convergence check
            temp_resid = compute_temperature_update(cv, temperature) / temperature
            temp_err = (actx.to_numpy(op.nodal_max_loc(dcoll, "vol", temp_resid)))
            if temp_err > 1e-8:
                health_error = True
                logger.info(f"Temperature is not converged {temp_resid=}.")

        return health_error

    def compute_av_alpha_field(state):
        """Scale alpha by the element characteristic length."""
        return alpha_sc*state.speed*length_scales

    def my_pre_step(step, t, dt, state):
        cv, tseed = state

        try:

            if do_checkpoint:

                fluid_state = construct_fluid_state(cv, tseed)
                dv = fluid_state.dv

                from mirgecom.simutil import check_step
                do_viz = check_step(step=step, interval=nviz)
                do_restart = check_step(step=step, interval=nrestart)
                do_health = check_step(step=step, interval=nhealth)
                do_status = check_step(step=step, interval=nstatus)

                # If we plan on doing anything with the state, then
                # we need to make sure it is evaluated first.
                if any([do_viz, do_restart, do_health, do_status, constant_cfl]):
                    fluid_state = force_evaluation(actx, fluid_state)

                dt = get_sim_timestep(dcoll, fluid_state, t=t, dt=dt,
                                      cfl=current_cfl, t_final=t_final,
                                      constant_cfl=constant_cfl)

                if do_health:
                    health_errors = global_reduce(my_health_check(cv, dv), op="lor")
                    if health_errors:
                        logger.info("Fluid solution failed health check.")
                        raise MyRuntimeError("Failed simulation health check.")

                if do_status:
                    my_write_status(dt=dt, cfl=current_cfl, dv=dv)

                if do_restart:
                    my_write_restart(step=step, t=t, state=fluid_state,
                                 temperature_seed=tseed)

                if do_viz:
                    my_write_viz(step=step, t=t, cv=cv, dv=dv)

        except MyRuntimeError:
            logger.info("Errors detected; attempting graceful exit.")
            raise

        return state, dt

    def my_post_step(step, t, dt, state):
        cv, tseed = state

        return state, dt

    from mirgecom.inviscid import inviscid_facial_flux_rusanov

    def dummy_pre_step(step, t, dt, state):
        return state, dt

    def dummy_post_step(step, t, dt, state):
        return state, dt

    from mirgecom.flux import num_flux_central
    from mirgecom.gas_model import make_operator_fluid_states
    from mirgecom.navierstokes import grad_cv_operator

    def rhs(t, state):
        cv, tseed = state
        from mirgecom.gas_model import make_fluid_state
        fluid_state = make_fluid_state(cv=cv, gas_model=gas_model,
                                       temperature_seed=tseed)
        fluid_operator_states = make_operator_fluid_states(dcoll, fluid_state,
                                                           gas_model, boundaries,
                                                           quadrature_tag)

        if inviscid_only:
            fluid_rhs = \
                euler_operator(
                    dcoll, state=fluid_state, time=t,
                    boundaries=boundaries, gas_model=gas_model,
                    inviscid_numerical_flux_func=inviscid_facial_flux_rusanov,
                    quadrature_tag=quadrature_tag,
                    operator_states_quad=fluid_operator_states)
        else:
            grad_cv = grad_cv_operator(dcoll, gas_model, boundaries, fluid_state,
                                       time=t,
                                       numerical_flux_func=num_flux_central,
                                       quadrature_tag=quadrature_tag,
                                       operator_states_quad=fluid_operator_states)
            fluid_rhs = \
                ns_operator(
                    dcoll, state=fluid_state, time=t, boundaries=boundaries,
                    gas_model=gas_model, quadrature_tag=quadrature_tag,
                    inviscid_numerical_flux_func=inviscid_facial_flux_rusanov)

        if inert_only:
            chem_rhs = 0*fluid_rhs
        else:
            chem_rhs = eos.get_species_source_terms(cv, fluid_state.temperature)

        if av_on:
            alpha_f = compute_av_alpha_field(fluid_state)
            indicator = smoothness_indicator(dcoll, fluid_state.mass_density,
                                             kappa=kappa_sc, s0=s0_sc)
            av_rhs = av_laplacian_operator(
                dcoll, fluid_state=fluid_state, boundaries=boundaries, time=t,
                gas_model=gas_model, grad_cv=grad_cv,
                operator_states_quad=fluid_operator_states,
                alpha=alpha_f, s0=s0_sc, kappa=kappa_sc,
                indicator=indicator)
        else:
            av_rhs = 0*fluid_rhs

        if sponge_on:
            sponge_rhs = _sponge(fluid_state.cv)
        else:
            sponge_rhs = 0*fluid_rhs

        fluid_rhs = fluid_rhs + chem_rhs + av_rhs + sponge_rhs
        tseed_rhs = fluid_state.temperature - tseed

        return make_obj_array([fluid_rhs, tseed_rhs])

    def dummy_rhs(t, state):
        cv, tseed = state
        return make_obj_array([0*cv, 0*tseed])

    if dummy_rhs_only:
        my_rhs = dummy_rhs
    else:
        my_rhs = rhs

    current_dt = get_sim_timestep(dcoll, current_fluid_state, current_t, current_dt,
                                  current_cfl, t_final, constant_cfl)

    current_state = make_obj_array([current_cv, temperature_seed])
    current_state = actx.thaw(actx.freeze(current_state))

    print(f"Stepping start time: {time.ctime(time.time())}")

    compiled_rhs = actx.compile(my_rhs)
    compiled_rhs(current_t, current_state)

# vim: foldmethod=marker
