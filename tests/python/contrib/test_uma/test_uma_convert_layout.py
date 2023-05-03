import tvm
from tvm import relay
from tvm.relay.backend.contrib.uma.api.utils import PassPhase
from tvm.relay.transform import ConvertLayout
from .test_uma_vanilla_accelerator import VanillaAcceleratorBackend


class AlternativeLayoutBackend(VanillaAcceleratorBackend):
    def __init__(self):
        super().__init__()
        self._register_relay_pass(
            PassPhase.POST_PARTITIONING_1,
            layout_converter(),
        )


def layout_converter():
    return ConvertLayout({"nn.conv2d": ["NHWC", "OIHW"]})


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_convert_layout():
    def _data_weights():
        dtype = "float32"
        ishape = (1, 32, 32, 32)
        wshape = (32, 32, 3, 3)
        data0 = relay.var("data", shape=ishape, dtype=dtype)
        weight0 = relay.var("weight", shape=wshape, dtype=dtype)
        return data0, weight0

    def before():
        data0, weight0 = _data_weights()
        y = relay.nn.conv2d(
            data0,
            weight0,
            kernel_size=(3, 3),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.Function([data0, weight0], y)
        return y

    def expected():
        data0, weight0 = _data_weights()
        data0 = relay.layout_transform(data0, "NCHW", "NHWC")
        y = relay.nn.conv2d(
            data0,
            weight0,
            kernel_size=(3, 3),
            data_layout="NHWC",
            kernel_layout="OIHW",
        )
        y = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = run_opt_pass(before(), layout_converter())
    b = run_opt_pass(expected(), relay.transform.InferType())
    assert tvm.ir.structural_equal(a, b)

    uma_backend = AlternativeLayoutBackend()
    uma_backend.register()
    a = uma_backend.partition(tvm.IRModule.from_expr(before()))
    assert tvm.ir.structural_equal(a, b)
