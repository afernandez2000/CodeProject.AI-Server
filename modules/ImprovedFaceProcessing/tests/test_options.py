import importlib.util, os
BASE = os.path.dirname(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("options", os.path.join(BASE,"intelligencelayer","options.py"))
opt = importlib.util.module_from_spec(spec); spec.loader.exec_module(opt)

def test_tier_selection():
    assert opt.select_tier(True, 32*10**9, "auto") == "accurate"   # 5090
    assert opt.select_tier(True,  8*10**9, "auto") == "accurate"   # 3070 (>=6GB)
    assert opt.select_tier(True,  4*10**9, "auto") == "fast"       # low-VRAM GPU
    assert opt.select_tier(False, 0,       "auto") == "fast"       # CPU
    assert opt.select_tier(True,  4*10**9, "accurate") == "accurate"  # override wins
    assert opt.select_tier(True, 32*10**9, "fast") == "fast"

def test_onnx_providers():
    assert opt.onnx_providers(True)[0] == "CUDAExecutionProvider"
    assert opt.onnx_providers(True)[-1] == "CPUExecutionProvider"
    assert opt.onnx_providers(False) == ["CPUExecutionProvider"]
