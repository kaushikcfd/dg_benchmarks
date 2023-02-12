__copyright__ = """
Copyright (C) 2023 Kaushik Kulkarni
Copyright (C) 2023 Mit Kotak
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

import ast
import sys
import os
import numpy as np
import loopy as lp
import islpy as isl

from loopy.types import LoopyType
from dataclasses import dataclass, fields
from typing import (Callable, Optional, Mapping, Dict, TypeVar, Iterable,
                    cast, List, Set, Type, Any, Union)

from pytools import UniqueNameGenerator
from pytato.transform import CachedMapper, ArrayOrNames
from pytato.array import (Stack, Concatenate, IndexLambda, DataWrapper,
                          Placeholder, SizeParam, Roll,
                          AxisPermutation, Einsum,
                          Reshape, Array, DictOfNamedArrays, IndexBase,
                          NormalizedSlice, ShapeComponent,
                          IndexExpr, ArrayOrScalar,
                          AbstractResultWithNamedArrays,
                          AdvancedIndexInContiguousAxes,
                          AdvancedIndexInNoncontiguousAxes)
from pytools.tag import Tag
from pytato.scalar_expr import SCALAR_CLASSES
from pytato.utils import are_shape_components_equal, are_shapes_equal
from pytato.raising import BinaryOpType, C99CallOp

from pytato.target.python import BoundPythonProgram
from pytato.reductions import (ReductionOperation, SumReductionOperation,
                               ProductReductionOperation,
                               MaxReductionOperation, MinReductionOperation,
                               AllReductionOperation, AnyReductionOperation)
from arraycontext import ArrayContext


T = TypeVar("T")


def _can_colorize_output() -> bool:
    try:
        import pygments  # noqa: F401
        return True
    except ImportError:
        return False


def _get_default_colorize_code() -> bool:
    return (sys.stdout.isatty() and "NO_COLOR" not in os.environ)


def get_t_unit_for_index_lambda(expr: IndexLambda) -> lp.TranslationUnit:
    # Based Mit Kotak's CUDAGraph Target

    from pymbolic import var
    dim_to_bounds = {f"_{i}": (0, dim) for i, dim in enumerate(expr.shape)}
    all_dims = ", ".join(list(dim_to_bounds))
    bounds = " and ".join([f"{lbound} <= {dim} < {ubound}"
                        for dim, (lbound, ubound)
                        in list(dim_to_bounds.items())])
    out_var = var("out")[tuple(var(f"_{i}") for i in range(expr.ndim))]

    #FIXME : Need to remove the if condition for null domains
    domain = "{ [%s]: %s }" % (all_dims, bounds) if dim_to_bounds else "{:}"

    knl = lp.make_kernel(
        domains=domain,
        instructions=[lp.Assignment(out_var,
                                    expr.expr,
                                    within_inames=frozenset(
                                        {f"_{i}" for i in range(expr.ndim)}
                                    ))],
        kernel_data=[lp.GlobalArg("out", shape=expr.shape, dtype=expr.dtype),
                     *[lp.GlobalArg(name, shape=bnd.shape, dtype=bnd.dtype)
                       for name, bnd in sorted(expr.bindings.items())]],
        lang_version=(2018, 2))

    return knl


MISMATCHED_C99_CALL_TO_NP_FUNC = {
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "atan2": "arctan2"}


def _c99_callop_numpy_name(hlo: C99CallOp) -> str:
    if hlo.function in MISMATCHED_C99_CALL_TO_NP_FUNC:
        return MISMATCHED_C99_CALL_TO_NP_FUNC[hlo.function]
    else:
        return hlo.function


def first_true(iterable: Iterable[T], default: T,
               pred: Optional[Callable[[T], bool]] = None) -> T:
    """
    Returns the first true value in *iterable*. If no true value is found,
    returns *default* If *pred* is not None, returns the first item for which
    pred(item) is true.
    """
    # Taken from <https://docs.python.org/3/library/itertools.html#itertools-recipes>
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)


def _get_einsum_subscripts(einsum: Einsum) -> str:
    from pytato.array import EinsumElementwiseAxis, EinsumAxisDescriptor

    idx_stream = (chr(i) for i in range(ord("i"), ord("z")))
    idx_gen: Callable[[], str] = lambda: next(idx_stream)  # noqa: E731
    axis_descr_to_idx: Dict[EinsumAxisDescriptor, str] = {}
    input_specs = []
    for access_descr in einsum.access_descriptors:
        spec = ""
        for axis_descr in access_descr:
            try:
                spec += axis_descr_to_idx[axis_descr]
            except KeyError:
                axis_descr_to_idx[axis_descr] = idx_gen()
                spec += axis_descr_to_idx[axis_descr]

        input_specs.append(spec)

    output_spec = "".join(axis_descr_to_idx[EinsumElementwiseAxis(i)]
                          for i in range(einsum.ndim))

    return f"{', '.join(input_specs)} -> {output_spec}"


def _is_slice_trivial(slice_: NormalizedSlice,
                      dim: ShapeComponent) -> bool:
    """
    Return *True* only if *slice_* is equivalent to the trivial slice i.e.
    traverses an axis of length *dim* in unit steps.
    """
    return (are_shape_components_equal(slice_.start, 0)
            and are_shape_components_equal(slice_.stop, dim)
            and slice_.step == 1)


@dataclass(frozen=True)
class ArraycontextProgram:
    import_statements: ast.Module
    inner_function: ast.Module
    numpy_arrays_to_store: Mapping[str, np.ndarray]


SIMPLE_BINOP_TO_AST_OP = {BinaryOpType.ADD:         ast.Add,
                          BinaryOpType.SUB:         ast.Sub,
                          BinaryOpType.MULT:        ast.Mult,
                          BinaryOpType.TRUEDIV:     ast.Div,
                          BinaryOpType.FLOORDIV:    ast.FloorDiv,
                          BinaryOpType.MOD:         ast.Mod,
                          BinaryOpType.POWER:       ast.Pow,
                          BinaryOpType.BITWISE_OR:  ast.BitOr,
                          BinaryOpType.BITWISE_XOR: ast.BitXor,
                          BinaryOpType.BITWISE_AND: ast.BitAnd,
                          }

COMPARISON_OP_TO_CALL = {BinaryOpType.LESS:    "less",
                         BinaryOpType.GREATER: "greater",
                         BinaryOpType.LESS_EQUAL: "less_equal",
                         BinaryOpType.GREATER_EQUAL: "greater_equal",
                         BinaryOpType.EQUAL: "equal",
                         BinaryOpType.NOT_EQUAL: "not_equal",
                         }

LOGICAL_OP_TO_CALL = {BinaryOpType.LOGICAL_OR:  "logical_or",
                      BinaryOpType.LOGICAL_AND: "logical_and",
                      }

PYTATO_REDUCTION_TO_NP_REDUCTION: Mapping[Type[ReductionOperation], str] = {
    SumReductionOperation: "sum",
    ProductReductionOperation: "product",
    MaxReductionOperation: "max",
    MinReductionOperation: "min",
    AllReductionOperation: "all",
    AnyReductionOperation: "any",
}


class ArraycontextCodegenMapper(CachedMapper[ArrayOrNames]):
    """
    .. note::

        - This mapper stores mutable state for building the program. The same
          mapper instance must be re-used with care.
    """
    def __init__(self, actx: ArrayContext, vng: Callable[[str], str]):
        super().__init__()

        # Immutable state
        # ---------------
        self.vng = vng
        self.actx = actx
        self.numpy = "np"
        self.actx_arg_name = "actx"
        self.npzfile_arg_name = "npzfile"

        # Mutable state
        # -------------
        self.lines: List[ast.stmt] = []
        self.arg_names: Set[str] = set()
        self.numpy_arrays: Dict[str, np.ndarray] = {}
        self.seen_tags_to_names: Dict[Type[Tag], str] = {}
        self.seen_tunits_to_names: Dict[lp.TranslationUnit, str] = {}

    def get_t_unit_var_name(self, t_unit: lp.TranslationUnit) -> str:
        try:
            return self.seen_tunits_to_names[t_unit]
        except KeyError:
            new_var_name = self.vng("_pt_t_unit")
            self.seen_tunits_to_names[t_unit] = new_var_name
            return new_var_name

    def _get_tag_expr(self, tag: Tag) -> ast.expr:

        if tag.__class__ not in self.seen_tags_to_names:
            self.seen_tags_to_names[tag.__class__] = self.vng(
                tag.__class__.__name__)

        class_name = self.seen_tags_to_names[tag.__class__]

        args = []
        kwargs = []

        for field in fields(tag):
            field_val = getattr(tag, field.name)
            if isinstance(field_val, (int, str)):
                field_val_expr = ast.Constant(field_val)
            else:
                raise NotImplementedError()

            if field.kw_only:
                args.append(field_val_expr)
            else:
                kwargs.append(ast.keyword(arg=field.name,
                                          value=field_val_expr))

        return ast.Call(ast.Name(class_name),
                        args=args,
                        keywords=kwargs)

    def rec(self, expr: ArrayOrNames) -> Any:
        lhs = super().rec(expr)

        if isinstance(expr, Array):
            if expr.tags:
                rhs = ast.Call(ast.Attribute(ast.Name(self.actx_arg_name), "tag"),
                               args=[ast.Tuple(elts=[self._get_tag_expr(tag)
                                                     for tag in expr.tags])],
                               keywords=[],
                               )
                lhs = self._record_line_and_return_lhs(lhs, rhs)

            for iaxis, axis in enumerate(expr.axes):
                if axis.tags:
                    rhs = ast.Call(
                        ast.Attribute(ast.Name(self.actx_arg_name), "tag_axis"),
                        args=[ast.Constant(iaxis),
                              ast.Tuple(elts=[self._get_tag_expr(tag)
                                              for tag in axis.tags]),
                              ast.Name(lhs)],
                        keywords=[],
                    )
                    lhs = self._record_line_and_return_lhs(lhs, rhs)
        else:
            assert isinstance(expr, AbstractResultWithNamedArrays)
            # arraycontext does not currently allowing tagging such types.
            assert not expr.tags

        return lhs

    def actx_np(self) -> ast.expr:
        return ast.Attribute(value=ast.Name(self.actx_arg_name),
                             attr="np")

    def _record_line_and_return_lhs(self,
                                    lhs: str, rhs: ast.expr) -> str:
        self.lines.append(ast.Assign(targets=[ast.Name(lhs)],
                                     value=rhs))
        return lhs

    def map_index_lambda(self, expr: IndexLambda) -> str:
        from pytato.raising import index_lambda_to_high_level_op
        from pytato.raising import (FullOp, BinaryOp, WhereOp,
                                    BroadcastOp, ReduceOp,
                                    BinaryOpType)
        hlo = index_lambda_to_high_level_op(expr)
        lhs = self.vng("_pt_tmp")
        rhs: ast.expr

        def _rec_ary_or_constant(e: ArrayOrScalar) -> ast.expr:
            if isinstance(e, Array):
                return ast.Name(self.rec(e))
            else:
                assert isinstance(e, SCALAR_CLASSES)
                if np.isnan(e):
                    # generates code like: `np.float64("nan")`.
                    return ast.Call(
                        func=ast.Attribute(value=ast.Name(self.numpy),
                                           attr=e.dtype.type.__name__),
                        args=[ast.Constant(value="nan")],
                        keywords=[])
                else:
                    return ast.Constant(e)

        if isinstance(hlo, FullOp):
            if hlo.fill_value == 0:
                rhs = ast.Call(
                    ast.Attribute(ast.Name(self.actx_arg_name),
                                  "zeros"),
                    args=[ast.Tuple(elts=[ast.Constant(d)
                                          for d in expr.shape])],
                    keywords=[ast.keyword(
                        arg="dtype",
                        value=ast.Attribute(ast.Name(self.numpy),
                                            expr.dtype.type.__name__)
                    )])
            else:
                rhs = ast.BinOp(
                    left=ast.Call(
                        ast.Attribute(self.actx_np, "full"),
                        args=[ast.Tuple(elts=[ast.Constant(d)
                                              for d in expr.shape]),
                              _rec_ary_or_constant(hlo.fill_value),
                              ],
                        keywords=[ast.keyword(
                            arg="dtype",
                            value=ast.Attribute(ast.Name(self.numpy),
                                                expr.dtype.type.__name__),
                        )]),
                    op=ast.Add(),
                    right=_rec_ary_or_constant(hlo.fill_value))
        elif isinstance(hlo, BinaryOp):
            if hlo.binary_op in {BinaryOpType.ADD, BinaryOpType.SUB,
                                 BinaryOpType.MULT, BinaryOpType.POWER,
                                 BinaryOpType.TRUEDIV, BinaryOpType.FLOORDIV,
                                 BinaryOpType.MOD, BinaryOpType.BITWISE_OR,
                                 BinaryOpType.BITWISE_XOR,
                                 BinaryOpType.BITWISE_AND,
                                 }:
                rhs = ast.BinOp(left=_rec_ary_or_constant(hlo.x1),
                                op=SIMPLE_BINOP_TO_AST_OP[hlo.binary_op](),
                                right=_rec_ary_or_constant(hlo.x2))
            elif hlo.binary_op in {BinaryOpType.EQUAL, BinaryOpType.NOT_EQUAL,
                                   BinaryOpType.LESS, BinaryOpType.LESS_EQUAL,
                                   BinaryOpType.GREATER,
                                   BinaryOpType.GREATER_EQUAL}:
                rhs = ast.Call(ast.Attribute(self.actx_np,
                                             COMPARISON_OP_TO_CALL[hlo.binary_op]),
                               args=[_rec_ary_or_constant(hlo.x1),
                                     _rec_ary_or_constant(hlo.x2)],
                               keywords=[])
            elif hlo.binary_op in {BinaryOpType.LOGICAL_OR,
                                   BinaryOpType.LOGICAL_AND}:
                rhs = ast.Call(ast.Attribute(self.actx_np,
                                             LOGICAL_OP_TO_CALL[hlo.binary_op]),
                               args=[_rec_ary_or_constant(hlo.x1),
                                     _rec_ary_or_constant(hlo.x2)],
                               keywords=[])
            else:
                raise NotImplementedError(hlo.binary_op)

            if (isinstance(hlo.x1, Array)
                    and isinstance(hlo.x2, Array)
                    and not are_shapes_equal(hlo.x1.shape, hlo.x2.shape)):
                t_unit = get_t_unit_for_index_lambda(expr)
                t_unit_var_name = self.get_t_unit_var_name(t_unit)
                rhs = ast.IfExp(
                    test=ast.Attribute(ast.Name(self.actx_arg_name),
                                       "supports_nonscalar_broadcasting",),
                    body=rhs,
                    orelse=ast.Subscript(
                        ast.Call(ast.Attribute(ast.Name(self.actx_arg_name),
                                               "call_loopy"),
                                 args=[ast.Name(t_unit_var_name)],
                                 keywords=[
                                     ast.keyword(k, ast.Name(self.rec(v)))
                                     for k, v in sorted(expr.bindings.items())]
                                 ),
                        ast.Constant("out"),
                    )
                )

        elif isinstance(hlo, C99CallOp):
            rhs = ast.Call(ast.Attribute(self.actx_np,
                                         _c99_callop_numpy_name(hlo)),
                           args=[_rec_ary_or_constant(arg)
                                 for arg in hlo.args],
                           keywords=[])
        elif isinstance(hlo, WhereOp):
            rhs = ast.Call(ast.Attribute(self.actx_np, "where"),
                           args=[_rec_ary_or_constant(hlo.condition),
                                 _rec_ary_or_constant(hlo.then),
                                 _rec_ary_or_constant(hlo.else_)],
                           keywords=[])
            from itertools import combinations

            if any((isinstance(where_arg0, Array)
                    and isinstance(where_arg1, Array)
                    and not are_shapes_equal(where_arg0.shape,
                                             where_arg1.shape))
                   for where_arg0, where_arg1 in combinations((hlo.condition,
                                                               hlo.then,
                                                               hlo.else_), 2)):
                t_unit = get_t_unit_for_index_lambda(expr)
                t_unit_var_name = self.get_t_unit_var_name(t_unit)
                rhs = ast.IfExp(
                    test=ast.Attribute(ast.Name(self.actx_arg_name),
                                       "supports_nonscalar_broadcasting",),
                    body=rhs,
                    orelse=ast.Subscript(
                        ast.Call(ast.Attribute(ast.Name(self.actx_arg_name),
                                               "call_loopy"),
                                 args=[ast.Name(t_unit_var_name)],
                                 keywords=[
                                     ast.keyword(k, ast.Name(self.rec(v)))
                                     for k, v in sorted(expr.bindings.items())]
                                 ),
                        ast.Constant("out"),
                    )
                )

        elif isinstance(hlo, BroadcastOp):
            if not all(isinstance(d, int) for d in expr.shape):
                raise NotImplementedError("Parametric shape in broadcast_to")

            rhs = ast.Call(ast.Attribute(self.actx_np, "broadcast_to"),
                           args=[ast.Name(self.rec(hlo.x)),
                                 ast.Tuple(elts=[ast.Constant(d)
                                                 for d in expr.shape])],
                           keywords=[])

            t_unit = get_t_unit_for_index_lambda(expr)
            t_unit_var_name = self.get_t_unit_var_name(t_unit)
            rhs = ast.IfExp(
                test=ast.Attribute(ast.Name(self.actx_arg_name),
                                   "supports_nonscalar_broadcasting",),
                body=rhs,
                orelse=ast.Subscript(
                    ast.Call(ast.Attribute(ast.Name(self.actx_arg_name),
                                           "call_loopy"),
                             args=[ast.Name(t_unit_var_name)],
                             keywords=[
                                 ast.keyword(k, ast.Name(self.rec(v)))
                                 for k, v in sorted(expr.bindings.items())]
                             ),
                    ast.Constant("out"),
                )
            )
        elif isinstance(hlo, ReduceOp):
            if type(hlo.op) not in PYTATO_REDUCTION_TO_NP_REDUCTION:
                raise NotImplementedError(hlo.op)
            np_fn_name = PYTATO_REDUCTION_TO_NP_REDUCTION[type(hlo.op)]
            if all(i in hlo.axes for i in range(hlo.x.ndim)):
                rhs = ast.Call(ast.Attribute(self.actx_np, np_fn_name),
                               args=[ast.Name(self.rec(hlo.x))],
                               keywords=[])
            else:
                if len(hlo.axes) == 1:
                    axis, = hlo.axes.keys()
                    axis_ast: ast.expr = ast.Constant(axis)
                else:
                    axis_ast = ast.Tuple(elts=[ast.Constant(e)
                                               for e in sorted(hlo.axes.keys())])
                rhs = ast.Call(ast.Attribute(self.actx_np, np_fn_name),
                               args=[ast.Name(self.rec(hlo.x))],
                               keywords=[ast.keyword(arg="axis",
                                                     value=axis_ast)])
        else:
            raise NotImplementedError(type(hlo))

        return self._record_line_and_return_lhs(lhs, rhs)

    def map_placeholder(self, expr: Placeholder) -> str:
        self.arg_names.add(expr.name)
        return expr.name

    def map_stack(self, expr: Stack) -> str:
        assert isinstance(expr.axis, int)

        rec_ids = [self.rec(ary) for ary in expr.arrays]
        lhs = self.vng("_pt_tmp")
        rhs = ast.Call(ast.Attribute(self.actx_np, "stack"),
                       args=[ast.List([ast.Name(id_)
                                       for id_ in rec_ids])],
                       keywords=[ast.keyword(arg="axis",
                                             value=ast.Constant(expr.axis))])

        return self._record_line_and_return_lhs(lhs, rhs)

    def map_concatenate(self, expr: Concatenate) -> str:
        assert isinstance(expr.axis, int)

        rec_ids = [self.rec(ary) for ary in expr.arrays]
        lhs = self.vng("_pt_tmp")
        rhs = ast.Call(ast.Attribute(self.actx_np, "concatenate"),
                       args=[ast.List([ast.Name(id_)
                                       for id_ in rec_ids])],
                       keywords=[ast.keyword(arg="axis",
                                             value=ast.Constant(expr.axis))])

        return self._record_line_and_return_lhs(lhs, rhs)

    def map_roll(self, expr: Roll) -> str:
        lhs = self.vng("_pt_tmp")
        rhs = ast.Call(ast.Attribute(self.actx_np, "roll"),
                       args=[ast.Name(self.rec(expr.array)),
                             ],
                       keywords=[ast.keyword(arg="shift",
                                             value=ast.Constant(expr.shift)),
                                 ast.keyword(arg="axis",
                                             value=ast.Constant(expr.axis))])

        return self._record_line_and_return_lhs(lhs, rhs)

    def map_axis_permutation(self, expr: AxisPermutation) -> str:
        lhs = self.vng("_pt_tmp")
        if expr.axis_permutation == tuple(range(expr.ndim))[::-1]:
            rhs: ast.expr = ast.Attribute(ast.Name(self.rec(expr.array)), "T")
        else:
            rhs = ast.Call(ast.Attribute(self.actx_np, "transpose"),
                           args=[ast.Name(self.rec(expr.array))],
                           keywords=[ast.keyword(
                               arg="axes",
                               value=ast.List(elts=[ast.Constant(a)
                                                    for a in expr.axis_permutation]))
                                     ])

        return self._record_line_and_return_lhs(lhs, rhs)

    def _map_index_base(self, expr: IndexBase) -> str:

        last_non_trivial_index = first_true(
            range(expr.array.ndim)[::-1],
            default=-1,
            pred=lambda i: not (isinstance(expr.indices[i], NormalizedSlice)
                                and _is_slice_trivial(
                                        cast(NormalizedSlice, expr.indices[i]),
                                        expr.array.shape[i]))
        )

        if last_non_trivial_index == -1:
            return self.rec(expr.array)  # type: ignore[no-any-return]

        lhs = self.vng("_pt_tmp")

        def _rec_idx(idx: IndexExpr, dim: ShapeComponent) -> ast.expr:
            if isinstance(idx, int):
                return ast.Constant(idx)
            elif isinstance(idx, NormalizedSlice):
                step = idx.step if idx.step != 1 else None
                if idx.step > 0:
                    start = (None
                             if are_shape_components_equal(0,
                                                           idx.start)
                             else idx.start)

                    stop = (None
                            if are_shape_components_equal(dim, idx.stop)
                            else idx.stop)
                else:
                    start = (None
                             if are_shape_components_equal(dim-1, idx.start)
                             else idx.start)

                    stop = (None
                            if are_shape_components_equal(-1, idx.stop)
                            else idx.stop)

                kwargs = {}
                if step is not None:
                    assert isinstance(step, int)
                    kwargs["step"] = ast.Constant(step)
                if start is not None:
                    assert isinstance(start, int)
                    kwargs["lower"] = ast.Constant(start)
                if stop is not None:
                    assert isinstance(stop, int)
                    kwargs["upper"] = ast.Constant(stop)

                return ast.Slice(**kwargs)
            else:
                assert isinstance(idx, Array)
                return ast.Name(self.rec(idx))

        rhs = ast.Subscript(value=ast.Name(self.rec(expr.array)),
                            slice=ast.Tuple(
                                elts=[
                                    _rec_idx(idx, dim)
                                    for idx, dim in zip(
                                            expr.indices[:last_non_trivial_index+1],
                                            expr.array.shape)]))

        if isinstance(expr, (AdvancedIndexInContiguousAxes,
                             AdvancedIndexInNoncontiguousAxes)):
            from pytato.transform import lower_to_index_lambda
            t_unit = get_t_unit_for_index_lambda(lower_to_index_lambda(expr))
            t_unit_var_name = self.get_t_unit_var_name(t_unit)
            rhs = ast.IfExp(
                test=ast.Attribute(ast.Name(self.actx_arg_name),
                                   "permits_advanced_indexing",),
                body=rhs,
                orelse=ast.Subscript(
                    ast.Call(ast.Attribute(ast.Name(self.actx_arg_name),
                                           "call_loopy"),
                             args=[ast.Name(t_unit_var_name)],
                             keywords=[
                                 ast.keyword(k, ast.Name(self.rec(v)))
                                 for k, v in sorted(expr.bindings.items())]
                             ),
                    ast.Constant("out"),
                )
            )

        return self._record_line_and_return_lhs(lhs, rhs)

    map_basic_index = _map_index_base
    map_contiguous_advanced_index = _map_index_base
    map_non_contiguous_advanced_index = _map_index_base

    def map_data_wrapper(self, expr: DataWrapper) -> str:
        lhs = self.vng("_pt_data") if expr.name is None else expr.name
        self.numpy_arrays[lhs] = self.actx.to_numpy(expr)
        rhs = ast.Subscript(ast.Name(self.npzfile_arg_name), ast.Constant(lhs))
        return self._record_line_and_return_lhs(lhs, rhs)

    def map_size_param(self, expr: SizeParam) -> str:
        # would demand a more complicated BoundProgram implementation.
        raise NotImplementedError("SizeParams not yet supported  in numpy-targets.")

    def map_einsum(self, expr: Einsum) -> str:
        lhs = self.vng("_pt_tmp")
        args = [ast.Name(self.rec(arg)) for arg in expr.args]
        rhs = ast.Call(ast.Attribute(ast.Name(self.actx_arg_name), "einsum"),
                        args=[ast.Constant(_get_einsum_subscripts(expr)),
                              *args],
                       keywords=[],
                       )

        return self._record_line_and_return_lhs(lhs, rhs)

    def map_reshape(self, expr: Reshape) -> str:
        lhs = self.vng("_pt_tmp")
        if not all(isinstance(d, int) for d in expr.shape):
            raise NotImplementedError("Non-integral reshapes.")
        rhs = ast.Call(ast.Attribute(self.actx_np, "reshape"),
                       args=[ast.Name(self.rec(expr.array)),
                             ast.Tuple(elts=[ast.Constant(d)
                                             for d in expr.shape])],
                       keywords=[],
                       )

        return self._record_line_and_return_lhs(lhs, rhs)

    def map_dict_of_named_arrays(self, expr: DictOfNamedArrays) -> str:
        lhs = self.vng("_pt_tmp")

        values = []
        for name, subexpr in sorted(expr._data.items()):
            values.append(ast.Name(self.rec(subexpr)))

        # our final goal is to return a type that `ArrayContext.compile` can
        # digest.
        rhs = ast.Call(ast.Name("make_obj_array"),
                       args=[ast.List(elts=values)],
                       keywords=[])

        return self._record_line_and_return_lhs(lhs, rhs)


def generate_arraycontext_code(
    expr: Union[Array, Mapping[str, Array], DictOfNamedArrays],
    actx: ArrayContext,
    function_name: str,
    show_code: bool,
    colorize_show_code: Optional[bool] = None,
) -> BoundPythonProgram:
    from pytato.transform import InputGatherer
    import collections

    if ((not isinstance(expr, DictOfNamedArrays))
            and isinstance(expr, collections.abc.Mapping)):
        from pytato.array import make_dict_of_named_arrays
        expr = make_dict_of_named_arrays(dict(expr))

    assert isinstance(expr, DictOfNamedArrays)
    var_name_gen = UniqueNameGenerator()
    var_name_gen.add_names({input_expr.name
                            for input_expr in InputGatherer()(expr)
                            if isinstance(input_expr,
                                          (Placeholder, SizeParam, DataWrapper))
                            if input_expr.name is not None})
    if isinstance(expr, DictOfNamedArrays):
        var_name_gen.add_names(expr)

    var_name_gen.add_names({function_name})

    # assumptions made by ArraycontextCodegenMapper
    var_name_gen.add_names({"actx", "npzfile", "np", "make_obj_array", "lp"})

    cgen_mapper = ArraycontextCodegenMapper(actx, vng=var_name_gen)
    result_var = cgen_mapper(expr)

    lines = cgen_mapper.lines
    lines.append(ast.Return(ast.Name(result_var)))

    # {{{ define the translation units

    define_t_unit_lines: List[ast.Assign] = []

    for t_unit, name in sorted(cgen_mapper.seen_tunits_to_names.items(),
                               key=lambda k, v: v):
        knl = t_unit.default_entrypoint
        if len(knl.domains) != 1 and len(knl.instructions) != 1:
            raise NotImplementedError

        domain, = knl.domains
        domain_str = str(domain) if domain.dim(isl.dim_type.set) else "{ : }"

        insn, = knl.instructions
        insn_str = f"{insn.assignee} = {insn.expression}"

        knl_args: List[ast.expr] = []
        for arg in knl.args:
            if not all(isinstance(d, int) for d in arg.shape):
                raise NotImplementedError()

            if not isinstance(arg.dtype, LoopyType):
                raise NotImplementedError()

            numpy_type_name = arg.dtype.numpy_dtype.type.__name__

            knl_args.append(
                ast.Call(
                    ast.Attribute(ast.Name("lp"), "GlobalArg"),
                    args=[ast.Constant(arg.name), 1/0],
                    keywords=[ast.keyword("dtype",
                                          ast.Attribute(value=ast.Name("np"),
                                                        attr=numpy_type_name)),
                              ast.keyword("shape",
                                          ast.Tuple(elts=[ast.Constant(d)
                                                          for d in arg.shape])),
                              ]
                )
            )

        rhs = ast.Call(ast.Name("make_loopy_program"),
                       args=[ast.Constant(domain_str), ast.Constant(insn_str),
                             ast.List(elts=knl_args)],
                       keywords=[])

        define_t_unit_lines.append(ast.Assign(targets=[ast.Name(name)],
                                              value=rhs))

    # }}}

    # {{{ handle tag imports

    tag_t_define_lines: List[ast.expr] = []

    for tag_t, name in sorted(cgen_mapper.seen_tags_to_names.items(),
                              key=lambda k, v: v):
        if name != tag_t.__name__:
            tag_t_define_lines.append(
                ast.ImportFrom(module=tag_t.__module__,
                               names=[ast.alias(tag_t.__name__, asname=name)])
            )
        else:
            tag_t_define_lines.append(
                ast.ImportFrom(module=tag_t.__module__,
                               names=[ast.alias(tag_t.__name__)])
            )

    # }}}

    import_statements = (ast.ImportFrom("pytools.obj_array",
                                        [ast.alias(name="make_obj_array")],
                                        level=0),
                         ast.Import(names=[ast.alias(name="numpy", asname="np")]),
                         ast.Import(names=[ast.alias(name="arraycontext",
                                                     asname="make_loopy_program")]),
                         ast.Import(names=[ast.alias(name="loopy", asname="lp")]),
                         *tag_t_define_lines,)
    function_def = ast.FunctionDef(
        name=function_name,
        posonlyargs=[],
        args=ast.arguments(
            args=[ast.arg(arg="actx"), ast.arg(arg="npzfile")],
            posonlyargs=[],
            kwonlyargs=[ast.arg(arg=name)
                        for name in cgen_mapper.arg_names],
            kw_defaults=[None for _ in cgen_mapper.arg_names],
            defaults=[]),
        body=define_t_unit_lines + lines,
        decorator_list=[],
    )

    if show_code:
        module = ast.Module(
            body=[*import_statements, function_def],
            type_ignores=[]
        )

        program = ast.unparse(ast.fix_missing_locations(module))

        if colorize_show_code is None:
            colorize_show_code = _get_default_colorize_code()

        assert colorize_show_code
        assert isinstance(colorize_show_code, bool)

        if _can_colorize_output() and colorize_show_code:
            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import TerminalTrueColorFormatter
            print(highlight(program,
                            formatter=TerminalTrueColorFormatter(),
                            lexer=PythonLexer()))
        else:
            print(program)

    return ArraycontextProgram(ast.Module(body=[*import_statements],
                                          type_ignores=[]),
                               ast.Module(body=[function_def],
                                          type_ignores=[]),
                               cgen_mapper.numpy_arrays)
