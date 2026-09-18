# Vision Lifecycle

Vision Lifecycle는 외부 학습 환경을 바꾸지 않고도 Vision AI 데이터, 학습 결과, 모델 bundle, ONNX export, 양자화, 평가, 보드 benchmark를 하나의 lineage로 연결하는 로컬 관리 시스템이다. 새 프로젝트와 실습 프로젝트는 모두 빈 상태에서 시작한다.

첫 온보딩 예제는 MMDetection 3.3.0 계열의 **RTMDet-tiny**와 **YOLOX-s**다. 두 모델을 동일 COCO evaluation set에서 baseline/candidate로 비교하는 흐름을 제공한다.

## 제공 기능

- SQLite 기반 Project, DatasetVersion, ModelVersion, Run, Job registry
- 읽기 전용 Storage mapping 등록·재검사·root 내부 탐색, bounded file inventory와 annotation/manifest fingerprint 변경 감지
- Storage mapping의 root 수정과 원본 삭제 없는 보관/복원, 보관 중 browse·inventory 실행 차단
- DatasetVersion별 canonical snapshot, parent version 연결, COCO category/image/annotation diff 조회
- SplitVersion 검증에서 item 중복·알 수 없는 item·group 누수를 검사하고 `require_complete` 분할 미할당을 표시
- COCO annotation validation과 external prediction JSON의 onboarding AP50 또는 공식 COCO AP@[.50:.95] 평가
- 분류 prediction record 기반 Top-1/Top-K·macro F1·confusion matrix 평가
- 분류 평가의 class mapping·빈 label·중복 image·top-k 입력 검증
- COCO·YOLO TXT·classification folder/CSV 경로 검사
- 승인된 local runner profile과 모의 보드 runner의 작업 상태·로그 관리
- 서비스 재시작 시 active job을 `interrupted`로 보존하고 자동 재실행하지 않는 복구 처리
- 실패·timeout·cancelled·interrupted 작업을 원본 입력 snapshot으로 명시적으로 재시도
- 작업 화면에서 전체 로그와 exit code/runner 결과를 펼쳐 보고, external worker 모드에서는 재시도 작업을 queue에 남겨 worker가 가져가도록 처리
- typed QuantizationRun과 BoardBenchmark 등록·조회, calibration/model/target/evaluation lineage 검증
- 실험 화면에서 외부 training Run의 dataset·config·metrics·environment·external ID를 등록하고 모델 연결에 재사용
- 명시적 FP32·QuantSim·Target 평가 계약 검증과 Quantization Loss·Target Gap 비교 결과 저장
- Dataset·Model·Run·Quantization·Board·Release를 연결하는 lineage API와 리포트 화면
- recipe별 onboarding session/step progress와 evidence 저장·재개
- 실제 Storage/Dataset/Model/Run/Release 근거가 없으면 onboarding 완료 처리를 거부
- 대용량 Storage inventory를 별도 Job으로 실행하고 로그·취소·재시도 흐름으로 관리
- classification prediction records 평가 UI와 detection COCO 평가 UI
- 검출 평가에서 invalid bbox/image/class/score를 성공 지표와 분리하고 최대 50건의 오류 이유를 보존
- 명시적 metric 규칙을 사용하는 Release gate와 민감 경로를 제거한 결과 export
- Release 생성 시 evaluation/board/quantization/artifact evidence를 snapshot·content hash로 고정하고 필수 evidence 누락을 INCOMPLETE으로 표시
- 감사 로그 화면에서 프로젝트·Dataset·Storage·모델·외부 Run·Release의 생성/수정/보관/복원과 변경 전후 값을 확인하고, export에는 경로·명령·비밀값을 제거한 이벤트를 포함
- regression gate는 baseline의 dataset·evaluator·protocol·scope·class mapping 계약이 맞을 때만 비교하며, 근거가 없으면 INCOMPLETE
- 명시적 ONNX input/output profile을 요구하는 local CPU inference preview adapter
- ONNX image record manifest 기반 batch 평가: classification Top-K/혼동행렬, COCO detection AP, 입력 manifest artifact 보존
- 평가 Run에 per-class/confusion/error 상세 결과와 evaluator scope를 보존하고 재조회하는 결과 provenance
- 모델·Dataset·Run 소유 entity에 연결된 artifact hash와 lineage graph 노드
- 등록된 파일/디렉터리를 `.vision-lifecycle/artifacts`에 관리 사본으로 보존하고 원본·관리 경로를 별도 기록(`VISION_LIFECYCLE_ARTIFACT_ROOT`로 위치 변경 가능)
- 모델 화면의 checkpoint/config hash 재검증과 drift 시 inference preview 차단
- 원본 model mount가 없어도 drift가 아닌 경우 관리 artifact 사본으로 inference preview를 재현하고 결과에 source/managed provenance를 표시
- 모델·config 교체 시 과거 artifact를 `superseded`로 보존하고 현재 provenance와 구분
- baseline/candidate alias 변경을 별도 이력으로 보존하고 변경 사유를 모델 화면에서 조회
- 모델 등록 화면에서 ONNX/MMDeploy/MMDetection 형식·정밀도·profile을 선택하고 샘플 inference preview 실행
- 선택 의존성 환경에서 MMDetection native config·checkpoint preview adapter
- 선택 의존성 환경에서 MMDeploy runtime model directory preview adapter와 versioned target profile
- versioned external result manifest import와 외부 run ID의 idempotency/conflict 검사
- 데이터·모델·평가 결과의 명시적 lineage
- 모델 등록 화면에서 학습 DatasetVersion과 외부 `training` Run을 각각 선택하며, lineage graph가 `training run → model` 관계를 표시
- 등록 가능한 checkpoint/config 파일의 SHA-256 provenance 기록
- label schema, split, evaluation set, calibration set의 versioned contract와 content hash
- COCO annotation image/category/bbox 샘플 preview
- baseline/candidate의 호환성 검사와 metric delta 비교
- 비교 시 완료된 평가 중 동일 dataset·평가 설정·class mapping 계약을 만족하는 최신 재현 가능 pair를 선택하고, 실패/불일치 실행은 공식 delta에서 제외
- FastAPI `/api/v1`와 `visionops` CLI
- React/Vite 한국어 UI 골격
- MMDetection fixture와 RTMDet·YOLOX 온보딩 문서
- 결과 export API와 GitHub Pages용 정적 snapshot 경로(생성 시각·overview·artifact hash·평가 상세 포함, 원본 경로 제외)
- 기존 자료를 연결하는 `vision-lifecycle-onboard` agent skill

## 빠른 시작

Python과 Node.js/npm이 설치된 Linux 환경에서 실행한다.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
visionops demo
uvicorn vision_lifecycle.main:app --app-dir backend --reload
```

작업을 API 프로세스와 분리하려면 API와 별도 터미널에서 external worker 모드를 사용한다. 이 모드에서는 API가 작업을 queue에 저장하고 worker가 하나씩 claim한다.

```bash
VISION_LIFECYCLE_EXTERNAL_WORKER=true uvicorn vision_lifecycle.main:app --app-dir backend --reload
visionops worker --poll-seconds 1
```

검증이나 일회성 실행은 `visionops worker --once`를 사용한다. 서비스나 worker가 재시작되면 실행 중이던 작업은 `interrupted`로 남고 자동 재실행되지 않는다.

기본 API 포트 8000이 다른 로컬 서비스에서 사용 중이면 포트를 바꿔 실행한다.

```bash
uvicorn vision_lifecycle.main:app --app-dir backend --reload --port 8001
```

별도 터미널에서 UI를 실행한다.

```bash
cd frontend
npm install
npm run dev
```

API를 8001번으로 실행한 경우에는 `VITE_API_PORT=8001 npm run dev`를 사용한다.

브라우저에서 `http://127.0.0.1:5173`를 열고 **빈 프로젝트 만들기** 또는 **MMDetection · YOLOX 실습 시작(빈 상태)**을 선택한다. 먼저 **연결·설정**에서 이미지·annotation·모델이 있는 서버/NAS root를 Storage mapping으로 등록·검사한 뒤, 데이터와 모델을 단계별로 연결한다. API 문서는 `http://127.0.0.1:8000/docs`에서 확인한다.

GitHub Pages는 Actions의 `workflow_dispatch`로 배포한다. workflow는 fixture registry를 임시 SQLite에 만들고 `visionops export`로 Pages-safe `snapshot.json`을 생성한 뒤 정적 UI를 빌드한다. 실제 프로젝트 결과를 게시할 때는 로컬에서 검토한 export를 `frontend/public/snapshot.json`에 넣어 선택한 공개 데이터만 포함한다.

Storage mapping은 서버가 접근할 수 있는 경로를 등록하는 기능이다. 브라우저에서 고른 로컬 파일을 자동 업로드하지 않으며, 등록·검사만으로 외부 명령을 실행하지 않는다. Dataset 초안에 연결한 annotation/manifest의 내용이 바뀌면 prediction 평가를 중단하므로 새 버전을 만든 뒤 다시 확정한다.

## CLI 온보딩

UI를 열 수 없는 runner에서도 같은 registry를 사용할 수 있다. 경로 검사는 등록이나 외부 명령 실행을 하지 않는다.

```bash
visionops inspect /data/instances_val.json
visionops validate-coco /data/instances_val.json
visionops create-project project.json
visionops create-dataset PRJ-... dataset-version.json
visionops create-model PRJ-... model-version.json
visionops register-artifact PRJ-... artifact.json
visionops export PRJ-... --output pages/snapshot.json
visionops backup --output backups/registry.sqlite
visionops restore --input backups/registry.sqlite
```

`export` 결과는 API export와 같은 redaction 규칙을 사용한다. 원본·관리 artifact·annotation·checkpoint의 절대 경로, 명령과 환경변수 이름은 포함하지 않는다.

모델을 baseline 또는 candidate로 승격·교체할 때는 `PATCH /api/v1/projects/{project_id}/models/{model_id}`에 `alias`와 선택적인 `alias_reason`을 보내고, 변경 이력은 `GET /api/v1/projects/{project_id}/models/{model_id}/alias-history`에서 확인한다. 기존 모델을 덮어쓰지 않고 이전 alias와 사유를 남기므로 비교 기준의 이동을 재현할 수 있다.

`backup`/`restore`는 로컬 SQLite registry의 ID와 lineage를 보존하는 운영 백업이다. Pages 공개용 결과를 만들 때는 `export`를 사용하며, backup 파일에는 로컬 경로와 설정이 포함될 수 있으므로 공개 저장소에 올리지 않는다.

## RTMDet·YOLOX 사용 흐름

1. COCO annotation과 이미지 경로를 DatasetVersion으로 등록한다.
2. RTMDet 또는 YOLOX config와 checkpoint를 ModelVersion bundle로 등록한다.
3. config classes와 COCO categories의 mapping을 검증한다.
4. native evaluation 결과를 EvaluationRun으로 가져온다.
5. MMDeploy 변환 결과와 target profile, 평가 결과를 별도 ExportRun/EvaluationRun/BoardRun으로 연결한다.
6. 동일 dataset·evaluator 버전을 쓴 모델을 baseline과 candidate로 지정해 비교한다.

상세 절차는 [MMDetection 가이드](docs/guides/mmdetection.md)를 참고한다. fixture의 metric은 전체 COCO benchmark나 실제 보드 성능이 아니다.

공식 pretrained RTMDet-tiny·YOLOX-s는 [models.json](examples/mmdetection/models.json)과 [준비 script](examples/mmdetection/scripts/prepare-official-models.sh)로 별도 다운로드한다. 모델 binary는 이 저장소나 Pages snapshot에 포함하지 않는다.

## Agent 사용

실제 모델이나 데이터가 있다면 다음처럼 agent에 요청한다.

```text
skills/vision-lifecycle-onboard/SKILL.md를 사용해서 다음 자료를 lifecycle registry에 연결해줘.
데이터: /path/to/images, /path/to/instances_val.json
모델: /path/to/config.py, /path/to/best.pth
평가 결과: /path/to/metrics.json
```

skill은 자료를 조사하고, 확인 가능한 lineage와 누락된 provenance를 분리해 등록하도록 설계되어 있다.

## 개발 상태와 한계

최신 제품 설계는 [빈 프로젝트에서 시작하는 실습형 Lifecycle 계획](docs/guided-lifecycle-plan.md)이다. 가이드 단계별 실제 연결·평가·변경 비교, 수정/보관, 디렉터리 연결과 작업 로그 UX를 포함한다. 이는 후속 개발 계획이며 현재 UI가 해당 흐름을 모두 제공하는 것은 아니다.

원본 계획의 전체 시나리오는 아직 구현되지 않았다. 현재 코드에 근거한 지원 상태와 다음 개발 순서는 [Lifecycle 시나리오 점검 및 추가 개발 계획](docs/lifecycle-gap-analysis.md)을 기준으로 한다. 통합 ONNX batch 평가, 양자화 손실·target gap과 고급 Release evidence가 남아 있다.

테스트는 프로세스별 임시 SQLite DB를 사용하며 기존 `.vision-lifecycle/registry.db`를 변경하지 않는다.

이 첫 버전은 registry, demo, COCO annotation 검증, external prediction JSON 기반 onboarding AP50 및 공식 COCO 평가, 비교 API, 경로 검사, runner profile·모의 보드 job, target profile, release gate, UI와 Pages workflow를 제공한다. 실제 MMDetection/MMDeploy 변환은 각 target SDK 환경에서 수행해 산출물과 결과 manifest를 연결한다. file upload wizard와 특정 vendor board recipe는 하드웨어·운영 환경이 정해진 뒤 추가할 확장 지점이다.

백엔드 테스트는 프로젝트 가상환경에서 실행한다.

```bash
.venv/bin/python -m pytest -q
```

UI build and static Pages build are verified with:

```bash
cd frontend
npm ci
npm run build
VITE_STATIC_MODE=true VITE_BASE_PATH=/your-repository/ npm run build
```
