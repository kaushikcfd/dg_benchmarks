import pytato as pt
import numpy as np

from arraycontext import PytatoJAXArrayContext
from typing import Callable, Any, Type, Optional, Dict
from arraycontext.impl.pytato.compile import (BaseLazilyCompilingFunctionCaller,
                                              CompiledFunction)
# from meshmode.array_context import BatchedEinsumArrayContext


class LazilyArraycontextCompilingFunctionCaller(BaseLazilyCompilingFunctionCaller):
    @property
    def compiled_function_returning_array_container_class(
            self) -> Type[CompiledFunction]:
        raise NotImplementedError

    @property
    def compiled_function_returning_array_class(self) -> Type[CompiledFunction]:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from arraycontext.impl.pytato.compile import (
            _get_arg_id_to_arg_and_arg_id_to_descr)
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
            return pt.make_placeholder(name, shape=ary.shape, dtype=ary.dtype)

        placeholder_out_template = rec_keyed_map_array_container(_as_dict_of_named_arrays,
                                                                 output_template)

        from .pytato_target import generate_arraycontext_code
        inner_code = generate_arraycontext_code(dict_of_named_arrays,
                                                function_name="_rhs_inner",
                                                actx=self.actx,
                                                show_code=True)

        inner_code_prg = generate_arraycontext_code(dict_of_named_arrays)


        host_code = f"""
        {ast.unparse(inner_code_prg.import_statements)}
        from pytools imort memoize_on_first_arg
        from functools import cache
        from immutables import Map


        {ast.unparse(inner_code_prg.function_def)}

        @memoize_on_first_arg
        def _get_compiled_rhs_inner(actx):
            npzfile = np.load("{self.actx.save_directory}/datawrappers.npz")
            return actx.compile(
                lambda *args: **kwargs: _rhs_inner(actx, npzfile, *args, **kwargs))

        @cache
        def _get_output_template():
            from pickle import load
            with open("{self.actx.save_directory}/out_template.pkl", "rb") as fp:
                output_template = load(fp)

            return output_template

        @cache
        def _get_pos_for_key_in_output_template():
            output_keys = set()

            def _as_dict_of_named_arrays(keys, ary):
                output_keys.add(keys)
                return ary

            rec_keyed_map_array_container(_as_dict_of_named_arrays,
                                          output_template)

            return Map({{i: output_key
                        for i, output_key in enumerate(sorted(
                                output_keys, key=_ary_container_key_stringifier))}})

        def rhs(actx, *args, **kwargs):
            from arraycontext.impl.context.pytato.compile import (
                _get_arg_id_to_arg_and_arg_id_to_descr)
            arg_id_to_arg, _ = _get_arg_id_to_arg_and_arg_id_to_descr(args, kwargs)

            compiled_rhs_inner = _get_compiled_rhs_inner(actx)
            result_as_np_obj_array = compiled_rhs_inner(**arg_id_to_arg)

            output_template = _get_output_template()

            if is_array_container(output_template):
                keys_to_pos = _get_pos_for_key_in_output_template()
                def to_output_template(keys, _):
                    return result_as_np_obj_array[keys_to_pos[keys]]

                return rec_keyed_map_array_container(to_output_template,
                                                     self.output_template)
            else:
                from pytato.array import Array
                assert isinstance(output_template, Array)
                assert result_as_np_obj_array.shape == (1,)
                return result_as_np_obj_array[0]
        """

        from pytools.codegen import remove_common_indentation
        host_code = remove_common_indentation(host_code)

        with open(f"{self.actx.save_directory}/main.py", "w") as fp:
            fp.write(host_code)

        with open(f"{self.actx.save_directory}/datawrappers.npz", "w") as fp:
            np.savez(fp, **inner_code_prg.numpy_arrays_to_store)

        with open(f"{self.actx.save_directory}/ref_input.pkl", "wb") as fp:
            import pickle
            np_args = tuple(self.actx.to_numpy(arg) for arg in args)
            np_kwargs = tuple(self.actx.to_numpy(arg) for arg in args)
            pickle.dump(fp, (np_args, np_kwargs))

        with open(f"{self.actx.save_directory}/out_template.pkl", "wb") as fp:
            import pickle
            pickle.dump(fp, placeholder_out_template)

        ref_out = self.actx.to_numpy(self.f(*args, **kwargs))

        with open(f"{self.actx.save_directory}/ref_output.pkl", "wb") as fp:
            import pickle
            pickle.dump(ref_out)

        # {{{ get 'rhs' callable

        variables_after_execution: Dict[str, Any] = {
            "_MODULE_SOURCE_CODE": host_code,  # helps pudb
        }
        exec(host_code, variables_after_execution)
        assert callable(variables_after_execution["rhs"])
        compiled_func = variables_after_execution["rhs"]

        # }}}

        self.program_cache[arg_id_to_descr] = (
            lambda *args, **kwargs: compiled_func(PytatoJAXArrayContext(),
                                                  *args, **kwargs))

        # {{{ test that the codegen was successful

        output = PytatoJAXAarrayContext().to_numpy(
            self.program_cache[arg_id_to_descr](*args, **kwargs))

        rec_multimap_array_container(np.testing.assert_allclose, output,
                                     ref_out)

        # }}}

        return compiled_func(arg_id_to_arg)


# TODO: derive from PytatoPyOpenCLArrayContext instead of PytatoJAXArrayContext
class SuiteGeneratingArraycontext(PytatoJAXArrayContext):
    def __init__(self, save_directory: str,
                 *,
                 compile_trace_callback: Optional[
                     Callable[[Any, str, Any], None]] = None
                 ) -> None:
        import os
        if not os.path.isabs(save_directory):
            raise ValueError(f"'{save_directory}' does not represent an"
                             " absolute path.")
        self.save_directory = save_directory
        super().__init__()

    def compile(self, f: Callable[..., Any]) -> Callable[..., Any]:
        return LazilyArraycontextCompilingFunctionCaller(self, f)

# vim: fdm=marker
