import pytato as pt

from arraycontext import PytatoJAXArrayContext
from typing import Callable, Any, Type
from arraycontext.impl.pytato.compile import (BaseLazilyCompilingFunctionCaller,
                                              CompiledFunction)
from meshmode.array_context import BatchedEinsumArrayContext


def generate_arraycontext(expr: pt.DictOfNamedArrays):
    ...


class LazilyArraycontextCompilingFunctionCaller(BaseLazilyCompilingFunctionCaller):
    @property
    def compiled_function_returning_array_container_class(
            self) -> Type[CompiledFunction]:
        raise NotImplementedError

    @property
    def compiled_function_returning_array_class(self) -> Type[CompiledFunction]:
        raise NotImplementedError

    def _dag_to_transformed_pytato_prg(self, dict_of_named_arrays, *, prg_id=None):
        dict_of_named_arrays = pt.transform.deduplicate_data_wrappers(
            dict_of_named_arrays)
        pytato_program = generate_arraycontext(dict_of_named_arrays)
        # simply print the program for now.
        1/0


class SuiteGeneratingArraycontext(PytatoJAXArrayContext):

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        return LazilyArraycontextCompilingFunctionCaller(self, f)

# vim: fdm=marker
