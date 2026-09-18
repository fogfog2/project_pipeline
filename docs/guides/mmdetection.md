# RTMDet · YOLOX 온보딩 가이드

## 1. 데모 실행

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
visionops demo
uvicorn vision_lifecycle.main:app --app-dir backend --reload
```

8000번 포트가 사용 중이면 API를 8001번으로 실행하고, 프런트엔드 proxy에도 같은 포트를 지정한다.

```bash
uvicorn vision_lifecycle.main:app --app-dir backend --reload --port 8001
```

다른 터미널에서 UI를 실행한다.

```bash
cd frontend
npm install
npm run dev
```

API를 8001번으로 실행한 경우에는 마지막 줄을 `VITE_API_PORT=8001 npm run dev`로 바꾼다.

UI에서 **빈 프로젝트 만들기** 또는 **MMDetection · YOLOX 실습 시작(빈 상태)**을 선택한다. 프로젝트와 실습 recipe만 생성하며 모델 checkpoint, 이미지, 점수, 실제 board 결과는 자동 등록하지 않는다. 이후 연결·설정에서 storage root를 등록하고 데이터·모델·평가를 단계별로 추가한다.

공식 pretrained RTMDet-tiny·YOLOX-s artifact가 필요하면 `examples/mmdetection/scripts/prepare-official-models.sh`를 실행한다. 이 download는 registry 등록과 별개이며, 완료 후 사용자가 config·checkpoint 경로를 명시적으로 연결한다.

## 2. 실제 RTMDet 또는 YOLOX 연결

준비물은 이미지 디렉터리, COCO annotation JSON, MMDetection config, `.pth` checkpoint이다. 평가하려면 validation/test split과 config에서 사용하는 class mapping도 필요하다.

1. Project를 만들고 storage root를 연결한다.
2. COCO annotation을 DatasetVersion으로 등록한다. `images`, `annotations`, `categories`와 실제 이미지 존재 여부를 검사한다.
3. `metainfo.classes`와 COCO categories를 대조한다. COCO category ID와 model index는 별도 mapping으로 저장한다.
4. config와 checkpoint를 ModelVersion bundle로 등록한다. `_base_` 의존성과 custom import는 runner 환경에서만 해석한다.
5. native MMDetection 평가 결과를 EvaluationRun으로 가져온다.
6. MMDeploy 산출물은 ONNX와 metadata 파일을 함께 등록한다. 변환 결과도 같은 평가 set에서 평가한다.
7. baseline·candidate를 지정하고 비교한다.

## 3. 오류 처리

| 증상 | 확인할 항목 |
|---|---|
| class 결과가 어긋남 | `metainfo.classes`, COCO categories, label-index mapping |
| config를 찾지 못함 | `_base_` 상대 경로와 원본 MMDetection checkout |
| ONNX 결과가 다름 | deploy config, resize/pad, RGB/BGR, NMS, threshold |
| 비교 불가 | dataset ID와 evaluator version이 같은지 확인 |

RTMDet와 YOLOX는 동일 task라도 전처리·output decoding이 다를 수 있으므로 하나의 YOLO adapter로 자동 처리하지 않는다.

## 4. Native preview 환경

프로젝트 기본 의존성에는 MMDetection을 포함하지 않는다. native config·checkpoint preview가 필요하면 runner 환경에 선택 의존성을 설치한다.

```bash
python -m pip install -e '.[mmdetection]'
```

`GET /api/v1/environment`에서 `mmdetection.available`를 확인한다. custom module이 있는 config는 해당 module도 같은 runner 환경에 있어야 한다.
## 5. MMDetection · MMDeploy 연결 가이드

## 지원 범위

- MMDetection 3.3 계열의 RTMDet-tiny와 YOLOX-s checkpoint/config bundle을 `mmdetection-pytorch`으로 등록한다.
- 변환된 MMDeploy 모델 디렉터리는 `mmdeploy` 형식으로 별도 ModelVersion으로 등록한다. 원본 FP32와 변환/양자화 결과를 같은 모델로 덮어쓰지 않는다.
- COCO prediction JSON은 빠른 onboarding AP50 또는 공식 `coco_full` AP@[.50:.95] 중 하나를 명시적으로 선택해 평가한다.

## MMDeploy 등록 예시

MMDeploy 변환은 target SDK와 호환되는 별도 환경에서 수행한다. 이 서비스는 변환 명령을 추측하거나 자동 실행하지 않는다. 변환이 끝나면 model bundle 경로와 사용한 deploy config, backend, target runtime 정보를 등록한다.

```json
{
  "name": "RTMDet-tiny TensorRT",
  "version": "int8-v1",
  "family": "RTMDet-tiny",
  "task_kind": "detection",
  "format": "mmdeploy",
  "precision": "int8",
  "artifact_path": "/models/rtmdet-tiny-int8",
  "source_dataset_id": "DS-...",
  "source_run_id": "RUN-quantization",
  "metadata_json": {
    "class_mapping_version": "coco-v1",
    "mmdeploy_profile": {"device_name": "cuda", "device_id": 0},
    "deploy_config_artifact": "sha256:..."
  }
}
```

`/api/v1/environment`에서 MMDeploy runtime 상태를 먼저 확인한다. 해당 패키지는 target runtime과 ABI를 맞춰 설치해야 하므로 기본 개발 환경에는 포함하지 않는다.
