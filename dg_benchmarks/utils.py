from typing import Any, Callable


def _get_benchmark_directory(equation: str, dim: int, degree: int) -> str:
    import importlib.utils
    import os

    dir_path = os.path.abspath(
        os.path.join(importlib.util.find_spec("dg_benchmarks").origin,
                     "suite", f"{equation}_{dim}D_P{degree}"))

    assert os.path.isdir(dir_path)
    return dir_path


def get_benchmark_main_file(equation: str, dim: int, degree: int) -> str:
    import os
    return os.path.join(
        _get_benchmark_directory(equation, dim, degree), "main.py")


def get_benchmark_literals_path(equation: str, dim: int, degree: int) -> str:
    import os
    return os.path.join(
        _get_benchmark_directory(equation, dim, degree), "literals.npz")


def get_benchmark_ref_input_arguments_path(
        equation: str, dim: int, degree: int) -> str:
    import os
    return os.path.join(
        _get_benchmark_directory(equation, dim, degree), "ref_input_args.pkl")


def get_benchmark_ref_output_path(
        equation: str, dim: int, degree: int) -> str:
    import os
    return os.path.join(
        _get_benchmark_directory(equation, dim, degree), "ref_outputs.pkl")


def get_benchmark_output_template_path(
        equation: str, dim: int, degree: int) -> str:
    import os
    return os.path.join(
        _get_benchmark_directory(equation, dim, degree), "template_output.pkl")


def get_benchmark_rhs(equation: str, dim: int, degree: int
                      ) -> Callable[..., Any]:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "main",
        _get_benchmark_directory(equation, dim, degree))

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot find benchmark for equation '{equation}'"
                           f"-{dim}D-P{degree}")

    benchmark_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_module)
    rhs_clbl = benchmark_module.rhs

    assert callable(rhs_clbl)
    return rhs_clbl
