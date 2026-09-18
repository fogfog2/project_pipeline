# ONNX smoke fixtures

These tiny CPU-only files are deliberately simple and are used to verify the
profile and evaluation plumbing without downloading a real model:

- `classification_mean.onnx` emits three channel means as class scores.
- `detection_constant.onnx` emits one `boxes`/`scores`/`labels` prediction.
- `images/` and `records/` contain the corresponding inputs.

Regenerate them after changing the recipe with:

```bash
python examples/fixtures/create_fixtures.py
```

Register each ONNX file with an explicit `onnx_profile`; the detector profile
must map `boxes`, `scores`, and `labels` by name. The output is a fixture for
adapter/evaluator checks and is not a model-quality benchmark.
