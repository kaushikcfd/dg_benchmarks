from pytools.obj_array import make_obj_array
import numpy as np
from arraycontext import make_loopy_program
import loopy as lp
from meshmode.transform_metadata import DiscretizationAmbientDimAxisTag
from meshmode.transform_metadata import DiscretizationDOFAxisTag
from meshmode.transform_metadata import DiscretizationElementAxisTag
from meshmode.transform_metadata import DiscretizationFaceAxisTag
from meshmode.transform_metadata import DiscretizationTopologicalDimAxisTag
from meshmode.transform_metadata import FirstAxisIsElementsTag
from pytato.tags import PrefixNamed
from pytools import memoize_on_first_arg
from functools import cache
from immutables import Map
from arraycontext import is_array_container_type
from arraycontext.container.traversal import (
    rec_map_array_container,
    rec_keyed_map_array_container,
)
from dg_benchmarks.utils import get_dg_benchmarks_path


def _rhs_inner(
    actx,
    npzfile,
    *,
    _actx_in_1_v_2_0,
    _actx_in_1_u_0,
    _actx_in_1_v_0_0,
    _actx_in_1_v_1_0
):
    _pt_t_unit = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 29999 and 0 <= _1 <= 2 }",
        "out[_0, _1] = in_0[in_1[_0] % 4, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(30000, 3)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(4, 3)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(30000,)),
        ],
    )
    _pt_t_unit_0 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 29999 and 0 <= _1 <= 2 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 750000, in_2[_0, _1] % 4]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(30000, 3)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(750000, 4)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(30000, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(30000, 3)),
        ],
    )
    _pt_t_unit_1 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 29999 and 0 <= _1 <= 2 }",
        "out[_0, _1] = _in0[_0, _1]*_in1[_0, 0]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(30000, 3)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(30000, 3)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(30000, 1)),
        ],
    )
    _pt_t_unit_10 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 2999999 and 0 <= _1 <= 2 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 2970000, in_2[_0, _1] % 3]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(3000000, 3)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(2970000, 3)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(3000000, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(3000000, 3)),
        ],
    )
    _pt_t_unit_2 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 2999999 and 0 <= _1 <= 2 }",
        "out[_0, _1] = in_0[0, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(3000000, 3)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(1, 3)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(3000000,)),
        ],
    )
    _pt_t_unit_3 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 2999999 and 0 <= _1 <= 2 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 30000, in_2[_0, _1] % 3]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(3000000, 3)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(30000, 3)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(3000000, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(3000000, 3)),
        ],
    )
    _pt_t_unit_4 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 2999999 and 0 <= _1 <= 2 }",
        "out[_0, _1] = _in1[_0, _1] if _in0[_0, 0] else 0",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(3000000, 3)),
            lp.GlobalArg("_in0", dtype=np.int8, shape=(3000000, 1)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(3000000, 3)),
        ],
    )
    _pt_t_unit_5 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 2969999 and 0 <= _1 <= 2 }",
        "out[_0, _1] = in_0[in_1[_0] % 4, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(2970000, 3)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(4, 3)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(2970000,)),
        ],
    )
    _pt_t_unit_6 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 2969999 and 0 <= _1 <= 2 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 750000, in_2[_0, _1] % 4]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(2970000, 3)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(750000, 4)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(2970000, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(2970000, 3)),
        ],
    )
    _pt_t_unit_7 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 2969999 and 0 <= _1 <= 2 }",
        "out[_0, _1] = in_0[in_1[_0] % 3, _1]",
        [
            lp.GlobalArg("out", dtype=np.int32, shape=(2970000, 3)),
            lp.GlobalArg("in_0", dtype=np.int32, shape=(3, 3)),
            lp.GlobalArg("in_1", dtype=np.int8, shape=(2970000,)),
        ],
    )
    _pt_t_unit_8 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 2969999 and 0 <= _1 <= 2 }",
        "out[_0, _1] = in_0[in_1[_0, 0] % 2970000, in_2[_0, _1] % 3]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(2970000, 3)),
            lp.GlobalArg("in_0", dtype=np.float64, shape=(2970000, 3)),
            lp.GlobalArg("in_1", dtype=np.int32, shape=(2970000, 1)),
            lp.GlobalArg("in_2", dtype=np.int32, shape=(2970000, 3)),
        ],
    )
    _pt_t_unit_9 = make_loopy_program(
        "{ [_0, _1] : 0 <= _0 <= 2969999 and 0 <= _1 <= 2 }",
        "out[_0, _1] = _in0[_0, _1]*_in1[_0, 0]",
        [
            lp.GlobalArg("out", dtype=np.float64, shape=(2970000, 3)),
            lp.GlobalArg("_in0", dtype=np.float64, shape=(2970000, 3)),
            lp.GlobalArg("_in1", dtype=np.float64, shape=(2970000, 1)),
        ],
    )
    _pt_data = actx.from_numpy(npzfile["_pt_data"])
    _pt_data = actx.tag((PrefixNamed(prefix="area_el_vol"),), _pt_data)
    _pt_data = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data)
    _pt_data = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data)
    _pt_tmp_1 = 1.0 / _pt_data
    _pt_data_0 = actx.from_numpy(npzfile["_pt_data_0"])
    _pt_data_0 = actx.tag_axis(0, (DiscretizationDOFAxisTag(),), _pt_data_0)
    _pt_data_1 = actx.from_numpy(npzfile["_pt_data_1"])
    _pt_data_1 = actx.tag(
        (PrefixNamed(prefix="inv_metric_deriv_vol"),), _pt_data_1
    )
    _pt_data_1 = actx.tag_axis(
        0, (DiscretizationAmbientDimAxisTag(),), _pt_data_1
    )
    _pt_data_1 = actx.tag_axis(
        1, (DiscretizationTopologicalDimAxisTag(),), _pt_data_1
    )
    _pt_data_1 = actx.tag_axis(2, (DiscretizationElementAxisTag(),), _pt_data_1)
    _pt_data_1 = actx.tag_axis(3, (DiscretizationDOFAxisTag(),), _pt_data_1)
    _pt_data_2 = actx.from_numpy(npzfile["_pt_data_2"])
    _pt_data_2 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_2)
    _actx_in_1_v_0_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_v_0_0
    )
    _actx_in_1_v_0_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_v_0_0
    )
    _actx_in_1_v_1_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_v_1_0
    )
    _actx_in_1_v_1_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_v_1_0
    )
    _actx_in_1_v_2_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_v_2_0
    )
    _actx_in_1_v_2_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_v_2_0
    )
    _pt_tmp_5 = actx.np.stack(
        [_actx_in_1_v_0_0, _actx_in_1_v_1_0, _actx_in_1_v_2_0], axis=0
    )
    _pt_tmp_4 = actx.einsum(
        "ijkl, jml, ikl -> km", _pt_data_1, _pt_data_2, _pt_tmp_5
    )
    _pt_tmp_4 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_4)
    _pt_tmp_3 = -1 * _pt_tmp_4
    _pt_data_3 = actx.from_numpy(npzfile["_pt_data_3"])
    _pt_data_3 = actx.tag_axis(0, (DiscretizationDOFAxisTag(),), _pt_data_3)
    _pt_data_3 = actx.tag_axis(2, (DiscretizationDOFAxisTag(),), _pt_data_3)
    _pt_data_4 = actx.from_numpy(npzfile["_pt_data_4"])
    _pt_data_4 = actx.tag(
        (PrefixNamed(prefix="area_el_b_face_restr_all"),), _pt_data_4
    )
    _pt_data_4 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_4)
    _pt_data_4 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_4)
    _pt_tmp_7 = actx.np.reshape(_pt_data_4, (4, 750000, 1))
    _pt_tmp_7 = actx.tag_axis(1, (DiscretizationElementAxisTag(),), _pt_tmp_7)
    _pt_data_5 = actx.from_numpy(npzfile["_pt_data_5"])
    _pt_data_5 = actx.tag((PrefixNamed(prefix="from_el_present"),), _pt_data_5)
    _pt_tmp_12 = actx.np.reshape(_pt_data_5, (3000000, 1))
    _pt_tmp_12 = actx.tag((PrefixNamed(prefix="from_el_present"),), _pt_tmp_12)
    _pt_data_6 = actx.from_numpy(npzfile["_pt_data_6"])
    _pt_data_6 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_data_6)
    _pt_tmp_23 = actx.np.reshape(_pt_data_6, (30000, 1))
    _pt_tmp_23 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_23)
    _pt_data_7 = actx.from_numpy(npzfile["_pt_data_7"])
    _pt_data_7 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_7)
    _pt_data_8 = actx.from_numpy(npzfile["_pt_data_8"])
    _pt_data_8 = actx.tag(
        (PrefixNamed(prefix="dof_pick_list_indices"),), _pt_data_8
    )
    _pt_tmp_24 = (
        _pt_data_7[_pt_data_8,]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit, in_0=_pt_data_7, in_1=_pt_data_8)[
            "out"
        ]
    )
    _pt_tmp_22 = (
        _actx_in_1_v_0_0[_pt_tmp_23, _pt_tmp_24]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_v_0_0,
            in_1=_pt_tmp_23,
            in_2=_pt_tmp_24,
        )["out"]
    )
    _pt_tmp_22 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_22)
    _pt_tmp_22 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_22)
    _pt_tmp_21 = 0 + _pt_tmp_22
    _pt_tmp_20 = _pt_tmp_21 + _pt_tmp_21
    _pt_tmp_19 = 0.5 * _pt_tmp_20
    _pt_data_9 = actx.from_numpy(npzfile["_pt_data_9"])
    _pt_data_9 = actx.tag((PrefixNamed(prefix="normal_1_b_all"),), _pt_data_9)
    _pt_data_9 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_data_9)
    _pt_data_9 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_9)
    _pt_tmp_18 = (
        _pt_tmp_19 * _pt_data_9
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_19, _in1=_pt_data_9)[
            "out"
        ]
    )
    _pt_tmp_29 = (
        _actx_in_1_v_1_0[_pt_tmp_23, _pt_tmp_24]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_v_1_0,
            in_1=_pt_tmp_23,
            in_2=_pt_tmp_24,
        )["out"]
    )
    _pt_tmp_29 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_29)
    _pt_tmp_29 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_29)
    _pt_tmp_28 = 0 + _pt_tmp_29
    _pt_tmp_27 = _pt_tmp_28 + _pt_tmp_28
    _pt_tmp_26 = 0.5 * _pt_tmp_27
    _pt_data_10 = actx.from_numpy(npzfile["_pt_data_10"])
    _pt_data_10 = actx.tag((PrefixNamed(prefix="normal_2_b_all"),), _pt_data_10)
    _pt_data_10 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _pt_data_10
    )
    _pt_data_10 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_10)
    _pt_tmp_25 = (
        _pt_tmp_26 * _pt_data_10
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_26, _in1=_pt_data_10)[
            "out"
        ]
    )
    _pt_tmp_17 = _pt_tmp_18 + _pt_tmp_25
    _pt_tmp_34 = (
        _actx_in_1_v_2_0[_pt_tmp_23, _pt_tmp_24]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0,
            in_0=_actx_in_1_v_2_0,
            in_1=_pt_tmp_23,
            in_2=_pt_tmp_24,
        )["out"]
    )
    _pt_tmp_34 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_34)
    _pt_tmp_34 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_34)
    _pt_tmp_33 = 0 + _pt_tmp_34
    _pt_tmp_32 = _pt_tmp_33 + _pt_tmp_33
    _pt_tmp_31 = 0.5 * _pt_tmp_32
    _pt_data_11 = actx.from_numpy(npzfile["_pt_data_11"])
    _pt_data_11 = actx.tag((PrefixNamed(prefix="normal_4_b_all"),), _pt_data_11)
    _pt_data_11 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _pt_data_11
    )
    _pt_data_11 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_11)
    _pt_tmp_30 = (
        _pt_tmp_31 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_31, _in1=_pt_data_11)[
            "out"
        ]
    )
    _pt_tmp_16 = _pt_tmp_17 + _pt_tmp_30
    _actx_in_1_u_0 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _actx_in_1_u_0
    )
    _actx_in_1_u_0 = actx.tag_axis(
        1, (DiscretizationDOFAxisTag(),), _actx_in_1_u_0
    )
    _pt_tmp_39 = (
        _actx_in_1_u_0[_pt_tmp_23, _pt_tmp_24]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_0, in_0=_actx_in_1_u_0, in_1=_pt_tmp_23, in_2=_pt_tmp_24
        )["out"]
    )
    _pt_tmp_39 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_39)
    _pt_tmp_39 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_39)
    _pt_tmp_38 = 0 + _pt_tmp_39
    _pt_tmp_37 = -1 * _pt_tmp_38
    _pt_tmp_36 = _pt_tmp_37 - _pt_tmp_38
    _pt_tmp_35 = 0.5 * _pt_tmp_36
    _pt_tmp_15 = _pt_tmp_16 + _pt_tmp_35
    _pt_tmp_14 = 1 * _pt_tmp_15
    _pt_data_12 = actx.from_numpy(npzfile["_pt_data_12"])
    _pt_data_12 = actx.tag(
        (PrefixNamed(prefix="from_el_indices"),), _pt_data_12
    )
    _pt_tmp_40 = actx.np.reshape(_pt_data_12, (3000000, 1))
    _pt_tmp_40 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_40)
    _pt_data_13 = actx.from_numpy(npzfile["_pt_data_13"])
    _pt_data_13 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_13)
    _pt_tmp_42 = actx.zeros((3000000,), dtype=np.int32)
    _pt_tmp_41 = (
        _pt_data_13[_pt_tmp_42,]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_2, in_0=_pt_data_13, in_1=_pt_tmp_42)[
            "out"
        ]
    )
    _pt_tmp_13 = (
        _pt_tmp_14[_pt_tmp_40, _pt_tmp_41]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_3, in_0=_pt_tmp_14, in_1=_pt_tmp_40, in_2=_pt_tmp_41
        )["out"]
    )
    _pt_tmp_11 = (
        actx.np.where(_pt_tmp_12, _pt_tmp_13, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_12, _in1=_pt_tmp_13)[
            "out"
        ]
    )
    _pt_tmp_11 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_11)
    _pt_tmp_11 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_11)
    _pt_tmp_10 = 0 + _pt_tmp_11
    _pt_data_14 = actx.from_numpy(npzfile["_pt_data_14"])
    _pt_data_14 = actx.tag(
        (PrefixNamed(prefix="from_el_present"),), _pt_data_14
    )
    _pt_tmp_46 = actx.np.reshape(_pt_data_14, (3000000, 1))
    _pt_tmp_46 = actx.tag((PrefixNamed(prefix="from_el_present"),), _pt_tmp_46)
    _pt_data_15 = actx.from_numpy(npzfile["_pt_data_15"])
    _pt_data_15 = actx.tag(
        (PrefixNamed(prefix="from_el_indices"),), _pt_data_15
    )
    _pt_tmp_57 = actx.np.reshape(_pt_data_15, (2970000, 1))
    _pt_tmp_57 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_57)
    _pt_data_16 = actx.from_numpy(npzfile["_pt_data_16"])
    _pt_data_16 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_16)
    _pt_data_17 = actx.from_numpy(npzfile["_pt_data_17"])
    _pt_data_17 = actx.tag(
        (PrefixNamed(prefix="dof_pick_list_indices"),), _pt_data_17
    )
    _pt_tmp_58 = (
        _pt_data_16[_pt_data_17,]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_5, in_0=_pt_data_16, in_1=_pt_data_17)[
            "out"
        ]
    )
    _pt_tmp_56 = (
        _actx_in_1_v_0_0[_pt_tmp_57, _pt_tmp_58]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_6,
            in_0=_actx_in_1_v_0_0,
            in_1=_pt_tmp_57,
            in_2=_pt_tmp_58,
        )["out"]
    )
    _pt_tmp_56 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_56)
    _pt_tmp_56 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_56)
    _pt_tmp_55 = 0 + _pt_tmp_56
    _pt_data_18 = actx.from_numpy(npzfile["_pt_data_18"])
    _pt_data_18 = actx.tag(
        (PrefixNamed(prefix="from_el_indices"),), _pt_data_18
    )
    _pt_tmp_61 = actx.np.reshape(_pt_data_18, (2970000, 1))
    _pt_tmp_61 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_61)
    _pt_data_19 = actx.from_numpy(npzfile["_pt_data_19"])
    _pt_data_19 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_19)
    _pt_data_20 = actx.from_numpy(npzfile["_pt_data_20"])
    _pt_data_20 = actx.tag(
        (PrefixNamed(prefix="dof_pick_list_indices"),), _pt_data_20
    )
    _pt_tmp_62 = (
        _pt_data_19[_pt_data_20,]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_7, in_0=_pt_data_19, in_1=_pt_data_20)[
            "out"
        ]
    )
    _pt_tmp_60 = (
        _pt_tmp_55[_pt_tmp_61, _pt_tmp_62]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_pt_tmp_55, in_1=_pt_tmp_61, in_2=_pt_tmp_62
        )["out"]
    )
    _pt_tmp_60 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_60)
    _pt_tmp_60 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_60)
    _pt_tmp_59 = 0 + _pt_tmp_60
    _pt_tmp_54 = _pt_tmp_55 + _pt_tmp_59
    _pt_tmp_53 = 0.5 * _pt_tmp_54
    _pt_data_21 = actx.from_numpy(npzfile["_pt_data_21"])
    _pt_data_21 = actx.tag(
        (PrefixNamed(prefix="normal_1_b_face_restr_interior"),), _pt_data_21
    )
    _pt_data_21 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _pt_data_21
    )
    _pt_data_21 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_21)
    _pt_tmp_52 = (
        _pt_tmp_53 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_53, _in1=_pt_data_21)[
            "out"
        ]
    )
    _pt_tmp_67 = (
        _actx_in_1_v_1_0[_pt_tmp_57, _pt_tmp_58]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_6,
            in_0=_actx_in_1_v_1_0,
            in_1=_pt_tmp_57,
            in_2=_pt_tmp_58,
        )["out"]
    )
    _pt_tmp_67 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_67)
    _pt_tmp_67 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_67)
    _pt_tmp_66 = 0 + _pt_tmp_67
    _pt_tmp_69 = (
        _pt_tmp_66[_pt_tmp_61, _pt_tmp_62]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_pt_tmp_66, in_1=_pt_tmp_61, in_2=_pt_tmp_62
        )["out"]
    )
    _pt_tmp_69 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_69)
    _pt_tmp_69 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_69)
    _pt_tmp_68 = 0 + _pt_tmp_69
    _pt_tmp_65 = _pt_tmp_66 + _pt_tmp_68
    _pt_tmp_64 = 0.5 * _pt_tmp_65
    _pt_data_22 = actx.from_numpy(npzfile["_pt_data_22"])
    _pt_data_22 = actx.tag(
        (PrefixNamed(prefix="normal_2_b_face_restr_interior"),), _pt_data_22
    )
    _pt_data_22 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _pt_data_22
    )
    _pt_data_22 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_22)
    _pt_tmp_63 = (
        _pt_tmp_64 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_64, _in1=_pt_data_22)[
            "out"
        ]
    )
    _pt_tmp_51 = _pt_tmp_52 + _pt_tmp_63
    _pt_tmp_74 = (
        _actx_in_1_v_2_0[_pt_tmp_57, _pt_tmp_58]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_6,
            in_0=_actx_in_1_v_2_0,
            in_1=_pt_tmp_57,
            in_2=_pt_tmp_58,
        )["out"]
    )
    _pt_tmp_74 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_74)
    _pt_tmp_74 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_74)
    _pt_tmp_73 = 0 + _pt_tmp_74
    _pt_tmp_76 = (
        _pt_tmp_73[_pt_tmp_61, _pt_tmp_62]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_pt_tmp_73, in_1=_pt_tmp_61, in_2=_pt_tmp_62
        )["out"]
    )
    _pt_tmp_76 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_76)
    _pt_tmp_76 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_76)
    _pt_tmp_75 = 0 + _pt_tmp_76
    _pt_tmp_72 = _pt_tmp_73 + _pt_tmp_75
    _pt_tmp_71 = 0.5 * _pt_tmp_72
    _pt_data_23 = actx.from_numpy(npzfile["_pt_data_23"])
    _pt_data_23 = actx.tag(
        (PrefixNamed(prefix="normal_4_b_face_restr_interior"),), _pt_data_23
    )
    _pt_data_23 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _pt_data_23
    )
    _pt_data_23 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_data_23)
    _pt_tmp_70 = (
        _pt_tmp_71 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_71, _in1=_pt_data_23)[
            "out"
        ]
    )
    _pt_tmp_50 = _pt_tmp_51 + _pt_tmp_70
    _pt_tmp_82 = (
        _actx_in_1_u_0[_pt_tmp_57, _pt_tmp_58]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_6, in_0=_actx_in_1_u_0, in_1=_pt_tmp_57, in_2=_pt_tmp_58
        )["out"]
    )
    _pt_tmp_82 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_82)
    _pt_tmp_82 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_82)
    _pt_tmp_81 = 0 + _pt_tmp_82
    _pt_tmp_80 = (
        _pt_tmp_81[_pt_tmp_61, _pt_tmp_62]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_8, in_0=_pt_tmp_81, in_1=_pt_tmp_61, in_2=_pt_tmp_62
        )["out"]
    )
    _pt_tmp_80 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_80)
    _pt_tmp_80 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_80)
    _pt_tmp_79 = 0 + _pt_tmp_80
    _pt_tmp_78 = _pt_tmp_79 - _pt_tmp_81
    _pt_tmp_77 = 0.5 * _pt_tmp_78
    _pt_tmp_49 = _pt_tmp_50 + _pt_tmp_77
    _pt_tmp_48 = 1 * _pt_tmp_49
    _pt_data_24 = actx.from_numpy(npzfile["_pt_data_24"])
    _pt_data_24 = actx.tag(
        (PrefixNamed(prefix="from_el_indices"),), _pt_data_24
    )
    _pt_tmp_83 = actx.np.reshape(_pt_data_24, (3000000, 1))
    _pt_tmp_83 = actx.tag((PrefixNamed(prefix="from_el_indices"),), _pt_tmp_83)
    _pt_data_25 = actx.from_numpy(npzfile["_pt_data_25"])
    _pt_data_25 = actx.tag((PrefixNamed(prefix="dof_pick_lists"),), _pt_data_25)
    _pt_tmp_84 = (
        _pt_data_25[_pt_tmp_42,]
        if actx.permits_advanced_indexing
        else actx.call_loopy(_pt_t_unit_2, in_0=_pt_data_25, in_1=_pt_tmp_42)[
            "out"
        ]
    )
    _pt_tmp_47 = (
        _pt_tmp_48[_pt_tmp_83, _pt_tmp_84]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_48, in_1=_pt_tmp_83, in_2=_pt_tmp_84
        )["out"]
    )
    _pt_tmp_45 = (
        actx.np.where(_pt_tmp_46, _pt_tmp_47, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_46, _in1=_pt_tmp_47)[
            "out"
        ]
    )
    _pt_tmp_45 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_45)
    _pt_tmp_45 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_45)
    _pt_tmp_44 = 0 + _pt_tmp_45
    _pt_tmp_43 = 0 + _pt_tmp_44
    _pt_tmp_9 = _pt_tmp_10 + _pt_tmp_43
    _pt_tmp_8 = actx.np.reshape(_pt_tmp_9, (4, 750000, 3))
    _pt_tmp_8 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_8)
    _pt_tmp_6 = actx.einsum(
        "ijk, jlk, jlk -> li", _pt_data_3, _pt_tmp_7, _pt_tmp_8
    )
    _pt_tmp_6 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_6)
    _pt_tmp_2 = _pt_tmp_3 + _pt_tmp_6
    _pt_tmp_0 = actx.einsum(
        "ij, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_2
    )
    _pt_tmp_0 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_0)
    _pt_tmp_89 = actx.einsum(
        "ijkl, jml, kl -> ikm", _pt_data_1, _pt_data_2, _actx_in_1_u_0
    )
    _pt_tmp_89 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_89)
    _pt_tmp_88 = _pt_tmp_89[0,]
    _pt_tmp_87 = -1 * _pt_tmp_88
    _pt_tmp_100 = _pt_tmp_38 + _pt_tmp_37
    _pt_tmp_99 = 0.5 * _pt_tmp_100
    _pt_tmp_98 = (
        _pt_tmp_99 * _pt_data_9
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_99, _in1=_pt_data_9)[
            "out"
        ]
    )
    _pt_tmp_106 = _pt_tmp_21 - _pt_tmp_21
    _pt_tmp_105 = (
        _pt_tmp_106 * _pt_data_9
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_106, _in1=_pt_data_9)[
            "out"
        ]
    )
    _pt_tmp_108 = _pt_tmp_28 - _pt_tmp_28
    _pt_tmp_107 = (
        _pt_tmp_108 * _pt_data_10
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_108, _in1=_pt_data_10)[
            "out"
        ]
    )
    _pt_tmp_104 = _pt_tmp_105 + _pt_tmp_107
    _pt_tmp_110 = _pt_tmp_33 - _pt_tmp_33
    _pt_tmp_109 = (
        _pt_tmp_110 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_110, _in1=_pt_data_11)[
            "out"
        ]
    )
    _pt_tmp_103 = _pt_tmp_104 + _pt_tmp_109
    _pt_tmp_102 = 0.5 * _pt_tmp_103
    _pt_tmp_101 = (
        _pt_tmp_102 * _pt_data_9
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_102, _in1=_pt_data_9)[
            "out"
        ]
    )
    _pt_tmp_97 = _pt_tmp_98 + _pt_tmp_101
    _pt_tmp_96 = 1 * _pt_tmp_97
    _pt_tmp_95 = (
        _pt_tmp_96[_pt_tmp_40, _pt_tmp_41]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_3, in_0=_pt_tmp_96, in_1=_pt_tmp_40, in_2=_pt_tmp_41
        )["out"]
    )
    _pt_tmp_94 = (
        actx.np.where(_pt_tmp_12, _pt_tmp_95, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_12, _in1=_pt_tmp_95)[
            "out"
        ]
    )
    _pt_tmp_94 = actx.tag_axis(0, (DiscretizationElementAxisTag(),), _pt_tmp_94)
    _pt_tmp_94 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_94)
    _pt_tmp_93 = 0 + _pt_tmp_94
    _pt_tmp_119 = _pt_tmp_81 + _pt_tmp_79
    _pt_tmp_118 = 0.5 * _pt_tmp_119
    _pt_tmp_117 = (
        _pt_tmp_118 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_118, _in1=_pt_data_21)[
            "out"
        ]
    )
    _pt_tmp_125 = _pt_tmp_59 - _pt_tmp_55
    _pt_tmp_124 = (
        _pt_tmp_125 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_125, _in1=_pt_data_21)[
            "out"
        ]
    )
    _pt_tmp_127 = _pt_tmp_68 - _pt_tmp_66
    _pt_tmp_126 = (
        _pt_tmp_127 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_127, _in1=_pt_data_22)[
            "out"
        ]
    )
    _pt_tmp_123 = _pt_tmp_124 + _pt_tmp_126
    _pt_tmp_129 = _pt_tmp_75 - _pt_tmp_73
    _pt_tmp_128 = (
        _pt_tmp_129 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_129, _in1=_pt_data_23)[
            "out"
        ]
    )
    _pt_tmp_122 = _pt_tmp_123 + _pt_tmp_128
    _pt_tmp_121 = 0.5 * _pt_tmp_122
    _pt_tmp_120 = (
        _pt_tmp_121 * _pt_data_21
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_121, _in1=_pt_data_21)[
            "out"
        ]
    )
    _pt_tmp_116 = _pt_tmp_117 + _pt_tmp_120
    _pt_tmp_115 = 1 * _pt_tmp_116
    _pt_tmp_114 = (
        _pt_tmp_115[_pt_tmp_83, _pt_tmp_84]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_115, in_1=_pt_tmp_83, in_2=_pt_tmp_84
        )["out"]
    )
    _pt_tmp_113 = (
        actx.np.where(_pt_tmp_46, _pt_tmp_114, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_46, _in1=_pt_tmp_114)[
            "out"
        ]
    )
    _pt_tmp_113 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _pt_tmp_113
    )
    _pt_tmp_113 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_113)
    _pt_tmp_112 = 0 + _pt_tmp_113
    _pt_tmp_111 = 0 + _pt_tmp_112
    _pt_tmp_92 = _pt_tmp_93 + _pt_tmp_111
    _pt_tmp_91 = actx.np.reshape(_pt_tmp_92, (4, 750000, 3))
    _pt_tmp_91 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_91)
    _pt_tmp_90 = actx.einsum(
        "ijk, jlk, jlk -> li", _pt_data_3, _pt_tmp_7, _pt_tmp_91
    )
    _pt_tmp_90 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_90)
    _pt_tmp_86 = _pt_tmp_87 + _pt_tmp_90
    _pt_tmp_85 = actx.einsum(
        "ij, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_86
    )
    _pt_tmp_85 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_85)
    _pt_tmp_133 = _pt_tmp_89[1,]
    _pt_tmp_132 = -1 * _pt_tmp_133
    _pt_tmp_142 = (
        _pt_tmp_99 * _pt_data_10
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_99, _in1=_pt_data_10)[
            "out"
        ]
    )
    _pt_tmp_143 = (
        _pt_tmp_102 * _pt_data_10
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_102, _in1=_pt_data_10)[
            "out"
        ]
    )
    _pt_tmp_141 = _pt_tmp_142 + _pt_tmp_143
    _pt_tmp_140 = 1 * _pt_tmp_141
    _pt_tmp_139 = (
        _pt_tmp_140[_pt_tmp_40, _pt_tmp_41]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_3, in_0=_pt_tmp_140, in_1=_pt_tmp_40, in_2=_pt_tmp_41
        )["out"]
    )
    _pt_tmp_138 = (
        actx.np.where(_pt_tmp_12, _pt_tmp_139, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_12, _in1=_pt_tmp_139)[
            "out"
        ]
    )
    _pt_tmp_138 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _pt_tmp_138
    )
    _pt_tmp_138 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_138)
    _pt_tmp_137 = 0 + _pt_tmp_138
    _pt_tmp_150 = (
        _pt_tmp_118 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_118, _in1=_pt_data_22)[
            "out"
        ]
    )
    _pt_tmp_151 = (
        _pt_tmp_121 * _pt_data_22
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_121, _in1=_pt_data_22)[
            "out"
        ]
    )
    _pt_tmp_149 = _pt_tmp_150 + _pt_tmp_151
    _pt_tmp_148 = 1 * _pt_tmp_149
    _pt_tmp_147 = (
        _pt_tmp_148[_pt_tmp_83, _pt_tmp_84]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_148, in_1=_pt_tmp_83, in_2=_pt_tmp_84
        )["out"]
    )
    _pt_tmp_146 = (
        actx.np.where(_pt_tmp_46, _pt_tmp_147, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_46, _in1=_pt_tmp_147)[
            "out"
        ]
    )
    _pt_tmp_146 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _pt_tmp_146
    )
    _pt_tmp_146 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_146)
    _pt_tmp_145 = 0 + _pt_tmp_146
    _pt_tmp_144 = 0 + _pt_tmp_145
    _pt_tmp_136 = _pt_tmp_137 + _pt_tmp_144
    _pt_tmp_135 = actx.np.reshape(_pt_tmp_136, (4, 750000, 3))
    _pt_tmp_135 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_135)
    _pt_tmp_134 = actx.einsum(
        "ijk, jlk, jlk -> li", _pt_data_3, _pt_tmp_7, _pt_tmp_135
    )
    _pt_tmp_134 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_134)
    _pt_tmp_131 = _pt_tmp_132 + _pt_tmp_134
    _pt_tmp_130 = actx.einsum(
        "ij, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_131
    )
    _pt_tmp_130 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_130)
    _pt_tmp_155 = _pt_tmp_89[2,]
    _pt_tmp_154 = -1 * _pt_tmp_155
    _pt_tmp_164 = (
        _pt_tmp_99 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_99, _in1=_pt_data_11)[
            "out"
        ]
    )
    _pt_tmp_165 = (
        _pt_tmp_102 * _pt_data_11
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_1, _in0=_pt_tmp_102, _in1=_pt_data_11)[
            "out"
        ]
    )
    _pt_tmp_163 = _pt_tmp_164 + _pt_tmp_165
    _pt_tmp_162 = 1 * _pt_tmp_163
    _pt_tmp_161 = (
        _pt_tmp_162[_pt_tmp_40, _pt_tmp_41]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_3, in_0=_pt_tmp_162, in_1=_pt_tmp_40, in_2=_pt_tmp_41
        )["out"]
    )
    _pt_tmp_160 = (
        actx.np.where(_pt_tmp_12, _pt_tmp_161, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_12, _in1=_pt_tmp_161)[
            "out"
        ]
    )
    _pt_tmp_160 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _pt_tmp_160
    )
    _pt_tmp_160 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_160)
    _pt_tmp_159 = 0 + _pt_tmp_160
    _pt_tmp_172 = (
        _pt_tmp_118 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_118, _in1=_pt_data_23)[
            "out"
        ]
    )
    _pt_tmp_173 = (
        _pt_tmp_121 * _pt_data_23
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_9, _in0=_pt_tmp_121, _in1=_pt_data_23)[
            "out"
        ]
    )
    _pt_tmp_171 = _pt_tmp_172 + _pt_tmp_173
    _pt_tmp_170 = 1 * _pt_tmp_171
    _pt_tmp_169 = (
        _pt_tmp_170[_pt_tmp_83, _pt_tmp_84]
        if actx.permits_advanced_indexing
        else actx.call_loopy(
            _pt_t_unit_10, in_0=_pt_tmp_170, in_1=_pt_tmp_83, in_2=_pt_tmp_84
        )["out"]
    )
    _pt_tmp_168 = (
        actx.np.where(_pt_tmp_46, _pt_tmp_169, 0)
        if actx.supports_nonscalar_broadcasting
        else actx.call_loopy(_pt_t_unit_4, _in0=_pt_tmp_46, _in1=_pt_tmp_169)[
            "out"
        ]
    )
    _pt_tmp_168 = actx.tag_axis(
        0, (DiscretizationElementAxisTag(),), _pt_tmp_168
    )
    _pt_tmp_168 = actx.tag_axis(1, (DiscretizationDOFAxisTag(),), _pt_tmp_168)
    _pt_tmp_167 = 0 + _pt_tmp_168
    _pt_tmp_166 = 0 + _pt_tmp_167
    _pt_tmp_158 = _pt_tmp_159 + _pt_tmp_166
    _pt_tmp_157 = actx.np.reshape(_pt_tmp_158, (4, 750000, 3))
    _pt_tmp_157 = actx.tag_axis(0, (DiscretizationFaceAxisTag(),), _pt_tmp_157)
    _pt_tmp_156 = actx.einsum(
        "ijk, jlk, jlk -> li", _pt_data_3, _pt_tmp_7, _pt_tmp_157
    )
    _pt_tmp_156 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_156)
    _pt_tmp_153 = _pt_tmp_154 + _pt_tmp_156
    _pt_tmp_152 = actx.einsum(
        "ij, jk, ik -> ij", _pt_tmp_1, _pt_data_0, _pt_tmp_153
    )
    _pt_tmp_152 = actx.tag((FirstAxisIsElementsTag(),), _pt_tmp_152)
    _pt_tmp = make_obj_array([_pt_tmp_0, _pt_tmp_85, _pt_tmp_130, _pt_tmp_152])
    return _pt_tmp


@memoize_on_first_arg
def _get_compiled_rhs_inner(actx):
    from functools import partial
    import os

    npzfile = np.load(
        os.path.join(get_dg_benchmarks_path(), "suite/wave_3D_P1/literals.npz")
    )
    return actx.compile(partial(_rhs_inner, actx=actx, npzfile=npzfile))


@memoize_on_first_arg
def _get_output_template(actx):
    import os
    import pytato as pt
    from pickle import load
    from meshmode.dof_array import array_context_for_pickling

    fpath = os.path.join(
        get_dg_benchmarks_path(), "suite/wave_3D_P1/ref_outputs.pkl"
    )
    with open(fpath, "rb") as fp:
        with array_context_for_pickling(actx):
            output_template = load(fp)

    def _convert_to_symbolic_array(ary):
        return pt.zeros(ary.shape, ary.dtype)

    # convert to symbolic array to not free the memory corresponding to
    # output_template
    return rec_map_array_container(_convert_to_symbolic_array, output_template)


@memoize_on_first_arg
def _get_key_to_pos_in_output_template(actx):
    from arraycontext.impl.pytato.compile import _ary_container_key_stringifier

    output_keys = set()
    output_template = _get_output_template(actx)

    def _as_dict_of_named_arrays(keys, ary):
        output_keys.add(keys)
        return ary

    rec_keyed_map_array_container(_as_dict_of_named_arrays, output_template)

    return Map(
        {
            output_key: i
            for i, output_key in enumerate(
                sorted(output_keys, key=_ary_container_key_stringifier)
            )
        }
    )


@cache
def _get_rhs_inner_argument_names():
    return {
        "_actx_in_1_u_0",
        "_actx_in_1_v_0_0",
        "_actx_in_1_v_1_0",
        "_actx_in_1_v_2_0",
    }


def rhs(actx, *args, **kwargs):
    from arraycontext.impl.pytato.compile import (
        _get_arg_id_to_arg_and_arg_id_to_descr,
        _ary_container_key_stringifier,
    )

    arg_id_to_arg, _ = _get_arg_id_to_arg_and_arg_id_to_descr(args, kwargs)
    input_kwargs_to_rhs_inner = {
        "_actx_in_" + _ary_container_key_stringifier(arg_id): arg
        for arg_id, arg in arg_id_to_arg.items()
    }

    input_kwargs_to_rhs_inner = {
        kw: input_kwargs_to_rhs_inner[kw]
        for kw in _get_rhs_inner_argument_names()
    }

    compiled_rhs_inner = _get_compiled_rhs_inner(actx)
    result_as_np_obj_array = compiled_rhs_inner(**input_kwargs_to_rhs_inner)

    output_template = _get_output_template(actx)

    if is_array_container_type(output_template.__class__):
        keys_to_pos = _get_key_to_pos_in_output_template(actx)

        def to_output_template(keys, _):
            return result_as_np_obj_array[keys_to_pos[keys]]

        return rec_keyed_map_array_container(
            to_output_template, _get_output_template(actx)
        )
    else:
        from pytato.array import Array

        assert isinstance(output_template, Array)
        assert result_as_np_obj_array.shape == (1,)
        return result_as_np_obj_array[0]
