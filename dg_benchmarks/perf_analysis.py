from functools import cache
from meshmode.array_context import BatchedEinsumPytatoPyOpenCLArrayContext

import numpy as np
import loopy as lp
import pyopencl as cl


# {{{ actx to get the kernel with loop fusion, contraction


class _MinimalBytesKernelException(RuntimeError):
    pass


class BatchedEinsumKernelGettingActx(BatchedEinsumPytatoPyOpenCLArrayContext):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def transform_loopy_program(self, t_unit):
        from arraycontext.impl.pytato.compile import FromArrayContextCompile
        if t_unit.default_entrypoint.tags_of_type(FromArrayContextCompile):

            # loop fusion
            from meshmode.arraycontext_extras.batched_einsum.utils import (
                apply_kennedy_fusion_with_batched_einsum_extension)
            t_unit = apply_kennedy_fusion_with_batched_einsum_extension(
                t_unit, self.loop_fusion_axis_tag_t,
                self.fused_loop_name_prefix_getter)

            # array contraction
            from meshmode.arraycontext_extras.batched_einsum.utils import (
                contract_arrays)
            t_unit = contract_arrays(t_unit)

            raise _MinimalBytesKernelException(t_unit)
        else:
            return super().transform_loopy_program(t_unit)

# }}}


@cache
def _get_batched_einsum_kernel(equation: str,
                               dim: int,
                               degree: int) -> lp.TranslationUnit:
    from dg_benchmarks.utils import (get_benchmark_rhs,
                                     get_benchmark_ref_input_arguments_path)
    rhs_clbl = get_benchmark_rhs(equation, dim, degree)
    cl_ctx = cl.create_some_context()
    cq = cl.CommandQueeu(cl_ctx)

    actx = _MinimalBytesKernelException(cq)

    with open(get_benchmark_ref_input_arguments_path(equation, dim, degree),
              "rb") as fp:
        import pickle
        np_args, np_kwargs = pickle.load(fp)

    args, kwargs = (tuple(actx.from_numpy(arg) for arg in np_args),
                    {kw: actx.from_numpy(arg) for kw, arg in np_kwargs.items()})

    try:
        rhs_clbl(0.0, *args, **kwargs)
    except _MinimalBytesKernelException as e:
        t_unit, = e.args
        assert isinstance(t_unit, lp.TranslationUnit)
        return t_unit
    else:
        raise RuntimeError("Was expecting a 'MinimalBytesKernelException'")


@cache
def get_float64_flops(equation: str, dim: int, degree: int) -> int:
    t_unit = _get_batched_einsum_kernel(equation, dim, degree)
    op_map = lp.get_op_map(t_unit, subgroup_size=1)
    knl = t_unit.default_entrypoint

    c128_ops = {op_type: (op_map.filter_by(dtype=[np.complex128],
                                           name=op_type,
                                           kernel_name=knl.name)
                          .eval_and_sum({}))
                for op_type in ["add", "mul", "div"]}
    f64_ops = (op_map.filter_by(dtype=[np.float64],
                                kernel_name=knl.name).eval_and_sum({})
               + (2 * c128_ops["add"]
                  + 6 * c128_ops["mul"]
                  + (6 + 3 + 2) * c128_ops["div"]))

    c64_ops = {op_type: (op_map.filter_by(dtype=[np.complex64],
                                          name=op_type,
                                          kernel_name=knl.name)
                         .eval_and_sum({}))
               for op_type in ["add", "mul", "div"]}
    f32_ops = (op_map.filter_by(dtype=[np.float32],
                                kernel_name=knl.name).eval_and_sum({})
               + (2 * c64_ops["add"]
                  + 6 * c64_ops["mul"]
                  + (6 + 3 + 2) * c64_ops["div"]))

    if f32_ops:
        raise RuntimeError("Single precision FLOPS != 0 => failed assumption.")

    return f64_ops


@cache
def get_footprint_bytes(equation: str, dim: int, degree: int) -> int:
    from pytools import product
    from loopy.kernel.array import ArrayBase

    t_unit = _get_batched_einsum_kernel(equation, dim, degree)

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


@cache
def get_roofline_flop_rate(equation: str, dim: int, degree: int,
                           roofline_model: str = "batched_einsum:global_ai",
                           ) -> float:
    from dg_benchmarks.consts import (DEV_TO_PEAK_BW,
                                      DEV_TO_PEAK_F64_GFLOPS)

    if roofline_model == "batched_einsum:global_ai":
        import pyopencl as cl
        cl_ctx = cl.create_some_context()
        cq = cl.CommandQueue(cl_ctx)
        device_name, = {dev.name for dev in cq.devices}

        try:
            t_runtime = max(
                ((get_float64_flops(equation, dim, degree)*1e-9)
                 / DEV_TO_PEAK_F64_GFLOPS[device_name]),
                ((get_footprint_bytes(equation, dim, degree)*1e-9)
                 / DEV_TO_PEAK_BW[device_name])
            )
        except KeyError:
            return np.nan
        else:
            return get_float64_flops(equation, dim, degree)/t_runtime
    else:
        raise NotImplementedError("Unknown roofline model:", roofline_model)
