from dg_benchmarks.codegen import SuiteGeneratingArraycontext
import tempfile


def _get_suite_generating_actx():
    tempdir = tempfile.mkdtemp()

    return SuiteGeneratingArraycontext(
        f"{tempdir}/main.py",
        f"{tempdir}/datawrappers.npz",
        f"{tempdir}/ref_input_args.npz",
        f"{tempdir}/ref_output.npz",
        f"{tempdir}/ref_output_template.npz",
    )


def test_array_returning_function():

    def f(x):
        return 2*x

    actx = _get_suite_generating_actx()

    a = actx.zeros(10, "float64")
    actx.compile(f)(a+42)  # internally asserts that the result is correct

    a = actx.zeros(10, "float32")
    actx.compile(f)(actx.thaw(actx.freeze(a+1729)))


def test_array_container_returning_function():

    def f(x):
        from pytools.obj_array import make_obj_array
        return make_obj_array([2*x, 3*x, x**2])

    actx = _get_suite_generating_actx()

    a = actx.zeros(10, "float64")
    actx.compile(f)(a+42)  # internally asserts that the result is correct

    a = actx.zeros(10, "float32")
    actx.compile(f)(actx.thaw(actx.freeze(a+1729)))
