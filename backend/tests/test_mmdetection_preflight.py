from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "examples/mmdetection/scripts/preflight_official_models.py"
    spec = spec_from_file_location("mmdetection_preflight", path)
    module = module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_preflight_reports_artifact_hashes_and_model_identity(tmp_path, monkeypatch):
    module = _module()
    config = tmp_path / "config.py"; checkpoint = tmp_path / "model.pth"
    config.write_text("model = {}", encoding="utf-8"); checkpoint.write_bytes(b"fixture-checkpoint")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _: object())
    report = module.inspect_bundle("rtmdet-tiny", str(config), str(checkpoint))
    assert report["status"] == "ready"
    assert report["family"] == "RTMDet-tiny"
    assert report["artifacts"]["checkpoint"]["sha256"]


def test_preflight_distinguishes_missing_artifact(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _: object())
    report = module.inspect_bundle("yolox-s", str(tmp_path / "missing.py"), str(tmp_path / "missing.pth"))
    assert report["status"] == "missing_artifacts"
    assert report["missing"] == ["config", "checkpoint"]
