from functools import cache


@cache
def get_flops(equation: str, dim: int, degree: int) -> int:
    raise NotImplementedError


@cache
def get_nbytes(equation: str, dim: int, degree: int) -> int:
    raise NotImplementedError


@cache
def get_roofline_flop_rate(equation: str, dim: int, degree: int,
                           roofline_model: str = "batched_einsum:global_ai"
                           ) -> float:

    if roofline_model == "batched_einsum:global_ai":
        import pyopencl as cl
        cl_ctx = cl.create_some_context()
        cq = cl.CommandQueue(cl_ctx)
        device_name, = {dev.name for dev in cq.devices}


        ...
    else:
        raise NotImplementedError("Unknown roofline model:", model_name)
