# RTMDet · YOLOX 온보딩 예제

이 폴더는 모델이나 GPU 없이 lifecycle 흐름을 확인하는 작은 COCO fixture를 제공한다.
`visionops demo` 또는 UI의 **MMDetection 데모 시작**을 실행하면 같은 메타데이터가 SQLite registry에 등록된다.

## 실제 MMDetection 모델 연결

1. MMDetection 3.3.0 환경에서 RTMDet-tiny 또는 YOLOX-s config와 `.pth` checkpoint를 준비한다.
2. UI의 모델 등록 또는 `POST /api/v1/projects/{project_id}/models`로 config와 checkpoint 경로를 model bundle에 등록한다.
3. COCO annotation의 `categories`와 config의 `metainfo.classes`를 대조한다. COCO category ID와 model label index는 별도의 mapping으로 보관한다.
4. native evaluation 결과를 `evaluation` run으로 가져온다.
5. MMDeploy를 사용한 ONNX export는 `export` run으로 등록한다. ONNX 결과는 원본 checkpoint 결과와 같은 evaluation set에서 평가한다.

이 fixture의 metric은 전체 COCO benchmark가 아니며, 모의 보드 결과도 실제 하드웨어 측정값이 아니다.

## 공식 pretrained artifact 준비

MMDetection 3.3.0 환경에서 아래 script를 실행하면 RTMDet-tiny와 YOLOX-s의 model-zoo config·checkpoint를 별도 폴더에 준비한다.

```bash
bash examples/mmdetection/scripts/prepare-official-models.sh /path/to/model-zoo
```

다운로드 대상과 config identifier는 `models.json`에 기록한다. 준비 후 UI의 **MMDetection 모델 연결**에서 config와 `.pth` 경로를 각각 등록한다. checkpoint는 저장소 또는 Pages snapshot에 포함하지 않는다.

## 등록 전 preflight

다운로드한 bundle이 어느 단계에서 막혔는지 먼저 확인한다. 이 명령은 registry를 수정하지 않고 config/checkpoint의 크기·SHA-256과 MMDetection 의존성만 검사한다.

```bash
python examples/mmdetection/scripts/preflight_official_models.py rtmdet-tiny \
  --config /path/to/model-zoo/rtmdet-tiny/rtmdet_tiny_8xb32-300e_coco.py \
  --checkpoint /path/to/model-zoo/rtmdet-tiny/model.pth \
  --output /tmp/rtmdet-preflight.json
```

`status`가 `ready`이면 UI에서 config와 checkpoint를 별도 artifact로 등록한다. `missing_artifacts`이면 경로를 고치거나 준비 script를 다시 실행하고, `missing_dependencies`이면 MMDetection 3.3.0 환경을 활성화한다. report의 SHA-256을 모델 등록 결과와 함께 보관하면 이후 파일 교체를 감지할 수 있다. YOLOX-s도 `yolox-s` 인수로 같은 절차를 따른다.
