---
name: vision-lifecycle-onboard
description: Connect an existing Vision AI dataset, training output, model bundle, quantization result, or board benchmark to this lifecycle registry. Use for RTMDet, YOLOX, YOLO-style detection, classification, and compatible external evaluation results.
---

# Vision Lifecycle Onboarding

Use this skill when a user supplies Vision AI files or paths and wants them connected to the local lifecycle system.

## Workflow

1. Inspect the provided paths without moving or modifying source assets.
2. Identify dataset format, task, annotation source, model artifact, config, class mapping, evaluation data, and any calibration or board results.
3. Report missing provenance explicitly. Preserve unknown values; never infer a dataset, class order, model output layout, or board metric.
4. Map recognized data into the registry entities: DatasetVersion, ModelVersion, Run, Artifact, and optional TargetProfile/BoardBenchmark.
5. Validate a small sample before registering a finalized dataset or official evaluation. Keep native training, ONNX export, quantization, and board measurements as distinct runs.
6. Update the project's setup notes with paths, adapters, environment versions, and known limits.

## Contract-first registration

Use the checked-in `schemas/v1/` contracts (or `GET /api/v1/schemas`) when creating manifests. Validate the JSON before calling the API or CLI, preserve `schema_version`, and keep unknown provenance fields explicit rather than inventing values. `visionops schemas --output contracts.json` writes the same registry for offline agents.

## MMDetection detection

For RTMDet or YOLOX, preserve the source config, resolved config, checkpoint, MMDetection version, and `metainfo.classes`. Validate those classes against COCO `categories`; category IDs can be non-contiguous and are not model label indices. Use a native runner only in its declared environment.

For MMDeploy output, retain the full deployment directory including ONNX and deployment metadata. Record it as an ExportRun, not as quantization, unless the conversion changes precision.

Read `examples/mmdetection/README.md` when onboarding the bundled RTMDet/YOLOX path.

## Completion standard

Finish with the registered IDs, validation result, remaining unknowns, and whether the item is only registered, locally runnable, or eligible for an official comparison. Do not execute upload, download, board, or external commands without a specific user request.
