r"""
Helpers to generate python code compatible with any
:class:`~arraycontext.ArrayContext`\ 's compile method.

.. autoclass:: SuiteGeneratingArraycontext
"""

import os
import ast
import pytato as pt
import numpy as np
import re
import sys

from arraycontext import is_array_container_type
from arraycontext.container.traversal import (rec_keyed_map_array_container,
                                              rec_multimap_array_container)
from typing import Callable, Any, Type, Dict
from arraycontext.impl.pytato.compile import (BaseLazilyCompilingFunctionCaller,
                                              CompiledFunction)
from dg_benchmarks.utils import get_dg_benchmarks_path, is_dataclass_array_container
from meshmode.dof_array import array_context_for_pickling
import autoflake
import black
from pathlib import Path
from meshmode.array_context import (
    BatchedEinsumPytatoPyOpenCLArrayContext as BatchedEinsumArrayContext)


class LazilyArraycontextCompilingFunctionCaller(BaseLazilyCompilingFunctionCaller):
    """
    Traces :attr:`BaseLazilyCompilingFunctionCaller.f` to translate the array
    operations to python code that calls equivalent methods of
    :class:`arraycontext.ArrayContext` / :class:`arraycontext.FakeNumpyNamespace`.
    """
    @property
    def compiled_function_returning_array_container_class(
            self) -> Type[CompiledFunction]:
        # This is purposefully left unimplemented to ensure that we do not run
        # into potential mishaps by using the super-class' implementation.
        # TODO: Maybe fix the abstract class' implementation so that it does
        # not rely on us overriding these routines.
        raise NotImplementedError

    @property
    def compiled_function_returning_array_class(self) -> Type[CompiledFunction]:
        # This is purposefully left unimplemented to ensure that we do not run
        # into potential mishaps by using the super-class' implementation.
        # TODO: Maybe fix the abstract class' implementation so that it does
        # not rely on us overriding these routines.
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Performs the following operations:

        #. Writes the generated code to disk at the location
            :attr:`SuiteGeneratingArraycontext.main_file_path`.
        #. Compiles the generated code and executes it with the arguments
            *args*, *kwargs* and returns the output.

        .. note::

            The behavior of this routine emulates calling :attr:`f` itself.
        """
        from arraycontext.impl.pytato.compile import (
            _get_arg_id_to_arg_and_arg_id_to_descr,
            _ary_container_key_stringifier,
            _get_f_placeholder_args,
        )
        arg_id_to_arg, arg_id_to_descr = _get_arg_id_to_arg_and_arg_id_to_descr(
            args, kwargs)

        try:
            compiled_f = self.program_cache[arg_id_to_descr]
        except KeyError:
            pass
        else:
            return compiled_f(arg_id_to_arg)

        dict_of_named_arrays = {}
        input_id_to_name_in_program = {
            arg_id: f"_actx_in_{_ary_container_key_stringifier(arg_id)}"
            for arg_id in arg_id_to_arg}

        output_template = self.f(
                *[_get_f_placeholder_args(arg, iarg,
                                          input_id_to_name_in_program, self.actx)
                    for iarg, arg in enumerate(args)],
                **{kw: _get_f_placeholder_args(arg, kw,
                                               input_id_to_name_in_program,
                                               self.actx)
                    for kw, arg in kwargs.items()})

        if (not (is_array_container_type(output_template.__class__)
                 or isinstance(output_template, pt.Array))):
            # TODO: We could possibly just short-circuit this interface if the
            # returned type is a scalar. Not sure if it's worth it though.
            raise NotImplementedError(
                f"Function '{self.f.__name__}' to be compiled "
                "did not return an array container or pt.Array,"
                f" but an instance of '{output_template.__class__}' instead.")

        def _as_dict_of_named_arrays(keys, ary):
            name = "_pt_out_" + _ary_container_key_stringifier(keys)
            dict_of_named_arrays[name] = ary
            return ary

        rec_keyed_map_array_container(
            _as_dict_of_named_arrays, output_template)

        from .pytato_target import generate_arraycontext_code
        pt_dict_of_named_arrays = pt.transform.deduplicate_data_wrappers(
            pt.make_dict_of_named_arrays(dict_of_named_arrays))
        inner_code_prg = generate_arraycontext_code(pt_dict_of_named_arrays,
                                                    function_name="_rhs_inner",
                                                    actx=self.actx,
                                                    show_code=False)

        host_code = f"""
        {ast.unparse(ast.fix_missing_locations(
            ast.Module(list(inner_code_prg.import_statements), type_ignores=[])))}
        from pytools import memoize_on_first_arg, keyed_memoize_on_first_arg
        from functools import cache
        from immutables import Map
        from arraycontext import is_array_container_type
        from arraycontext.container.traversal import (rec_map_array_container,
                                                      rec_keyed_map_array_container)
        from dg_benchmarks.utils import get_dg_benchmarks_path


        {ast.unparse(ast.fix_missing_locations(
            ast.Module([inner_code_prg.function_def], type_ignores=[])))}


        @keyed_memoize_on_first_arg(key=lambda a, b: b)
        def _from_numpy(actx, npzfile, name):
            return actx.freeze(actx.from_numpy(npzfile[name]))


        @memoize_on_first_arg
        def _get_compiled_rhs_inner(actx):
            from functools import partial
            import os
            npzfile = np.load(
                os.path.join(get_dg_benchmarks_path(),
                             "{os.path.relpath(self.actx.datawrappers_path,
                                               start=get_dg_benchmarks_path())}")
            )
            return actx.compile(partial(_rhs_inner, actx=actx, npzfile=npzfile))


        @memoize_on_first_arg
        def _get_output_template(actx):
            import os
            import pytato as pt
            from pickle import load
            from meshmode.dof_array import array_context_for_pickling

            fpath = os.path.join(get_dg_benchmarks_path(),
                                "{os.path.relpath(self.actx.pickled_ref_output_path,
                                                  start=get_dg_benchmarks_path())}")
            with open(fpath, "rb") as fp:
                with array_context_for_pickling(actx):
                    output_template = load(fp)

            def _convert_to_symbolic_array(ary):
                return pt.zeros(ary.shape, ary.dtype)

            # convert to symbolic array to not free the memory corresponding to
            # output_template
            return rec_map_array_container(_convert_to_symbolic_array,
                                           output_template)


        @memoize_on_first_arg
        def _get_key_to_pos_in_output_template(actx):
            from arraycontext.impl.pytato.compile import (
                _ary_container_key_stringifier)

            output_keys = set()
            output_template = _get_output_template(actx)

            def _as_dict_of_named_arrays(keys, ary):
                output_keys.add(keys)
                return ary

            rec_keyed_map_array_container(_as_dict_of_named_arrays,
                                          output_template)

            return Map({{output_key: i
                        for i, output_key in enumerate(sorted(
                                output_keys, key=_ary_container_key_stringifier))}})

        @cache
        def _get_rhs_inner_argument_names():
            return {{
                '{"', '".join(sorted(inner_code_prg.argument_names))}'
            }}


        def rhs(actx, *args, **kwargs):
            from arraycontext.impl.pytato.compile import (
                _get_arg_id_to_arg_and_arg_id_to_descr,
                _ary_container_key_stringifier)
            arg_id_to_arg, _ = _get_arg_id_to_arg_and_arg_id_to_descr(args, kwargs)
            input_kwargs_to_rhs_inner = {{
                "_actx_in_" + _ary_container_key_stringifier(arg_id): arg
                for arg_id, arg in arg_id_to_arg.items()}}

            input_kwargs_to_rhs_inner = {{
                kw: input_kwargs_to_rhs_inner[kw]
                for kw in _get_rhs_inner_argument_names()
            }}

            compiled_rhs_inner = _get_compiled_rhs_inner(actx)
            result_as_np_obj_array = compiled_rhs_inner(**input_kwargs_to_rhs_inner)

            output_template = _get_output_template(actx)

            if is_array_container_type(output_template.__class__):
                keys_to_pos = _get_key_to_pos_in_output_template(actx)

                def to_output_template(keys, _):
                    return result_as_np_obj_array[keys_to_pos[keys]]

                return rec_keyed_map_array_container(to_output_template,
                                                     _get_output_template(actx))
            else:
                from pytato.array import Array
                assert isinstance(output_template, Array)
                assert result_as_np_obj_array.shape == (1,)
                return result_as_np_obj_array[0]
        """
        host_code = re.sub(r"^        (?P<rest_of_line>.+)$", r"\g<rest_of_line>",
                           host_code, flags=re.MULTILINE)

        from pytools.codegen import remove_common_indentation
        host_code = remove_common_indentation(host_code)

        with open(f"{self.actx.main_file_path}", "w") as fp:
            fp.write(host_code)

        autoflake._main(["--remove-unused-variables",
                         "--imports", "loopy,arraycontext",
                         "--in-place",
                         self.actx.main_file_path,
                         ],
                        standard_out=None,
                        standard_error=sys.stderr,
                        standard_input=sys.stdin,
                        )
        black.format_file_in_place(Path(self.actx.main_file_path),
                                   fast=False,
                                   mode=black.Mode(line_length=80),
                                   write_back=black.WriteBack.YES)

        with open(f"{self.actx.datawrappers_path}", "wb") as fp:
            np.savez(fp, **inner_code_prg.numpy_arrays_to_store)

        with open(f"{self.actx.pickled_ref_input_args_path}", "wb") as fp:
            import pickle

            if (all(is_dataclass_array_container(arg) or np.isscalar(arg)
                    for arg in args)
                    and all(is_dataclass_array_container(arg) or np.isscalar(arg)
                            for arg in kwargs.values())):
                with array_context_for_pickling(self.actx):
                    pickle.dump((args, kwargs), fp)
            elif (any(is_dataclass_array_container(arg) for arg in args)
                    or any(is_dataclass_array_container(arg)
                           for arg in kwargs.values())):
                raise NotImplementedError("Pickling not implemented for input"
                                          " types.")
            else:
                np_args = tuple(self.actx.to_numpy(arg)
                                for arg in args)
                np_kwargs = {kw: self.actx.to_numpy(arg)
                             for kw, arg in kwargs.items()}
                pickle.dump((np_args, np_kwargs), fp)

        ref_out = self.actx.thaw(self.actx.freeze(self.f(*args, **kwargs)))

        with open(f"{self.actx.pickled_ref_output_path}", "wb") as fp:
            import pickle
            if is_dataclass_array_container(ref_out):
                with array_context_for_pickling(self.actx):
                    pickle.dump(ref_out, fp)
            else:
                pickle.dump(self.actx.to_numpy(ref_out), fp)

        # {{{ get 'rhs' callable

        variables_after_execution: Dict[str, Any] = {
            "_MODULE_SOURCE_CODE": host_code,  # helps pudb
        }
        exec(host_code, variables_after_execution)
        assert callable(variables_after_execution["rhs"])
        compiled_func = variables_after_execution["rhs"]

        # }}}

        from functools import partial
        self_actx_clone = self.actx.clone()
        self.program_cache[arg_id_to_descr] = partial(compiled_func, self_actx_clone)

        # {{{ test that the codegen was successful

        output = self.program_cache[arg_id_to_descr](*args, **kwargs)

        rec_multimap_array_container(
            np.testing.assert_allclose,
            self_actx_clone.to_numpy(output),
            self.actx.to_numpy(ref_out)
        )

        # }}}

        return output


class SuiteGeneratingArraycontext(BatchedEinsumArrayContext):
    """
    Overrides the :meth:`compile` method of
    :class:`arraycontext.PytatoJAXArrayContext` to generate python code that is
    compatible to run with any :class:`ArrayContext` and then executes the
    generated code.
    """
    def __init__(self,
                 queue,
                 allocator,
                 *,
                 main_file_path: str,
                 datawrappers_path: str,
                 pickled_ref_input_args_path: str,
                 pickled_ref_output_path: str,
                 ) -> None:

        super().__init__(queue, allocator)

        if any(not os.path.isabs(filepath)
               for filepath in [main_file_path, datawrappers_path,
                                pickled_ref_input_args_path,
                                pickled_ref_output_path]):
            raise ValueError("Absolute paths are expected.")

        self.main_file_path = main_file_path
        self.datawrappers_path = datawrappers_path
        self.pickled_ref_input_args_path = pickled_ref_input_args_path
        self.pickled_ref_output_path = pickled_ref_output_path

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        return LazilyArraycontextCompilingFunctionCaller(self, f)

    def clone(self):
        return BatchedEinsumArrayContext(self.queue, self.allocator)

# vim: fdm=marker
