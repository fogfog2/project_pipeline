# Vision Lifecycle

Vision Lifecycle는 외부 학습 환경을 바꾸지 않고도 Vision AI 데이터, 학습 결과, 모델 bundle, ONNX export, 양자화, 평가, 보드 benchmark를 하나의 lineage로 연결하는 로컬 관리 시스템이다. 새 프로젝트와 실습 프로젝트는 모두 빈 상태에서 시작한다.

첫 온보딩 예제는 MMDetection 3.3.0 계열의 **RTMDet-tiny**와 **YOLOX-s**다. 두 모델을 동일 COCO evaluation set에서 baseline/candidate로 비교하는 흐름을 제공한다.

## 제공 기능

- SQLite 기반 Project, DatasetVersion, ModelVersion, Run, Job registry
- 읽기 전용 Storage mapping 등록·재검사·root 내부 탐색, bounded file inventory와 annotation/manifest fingerprint 변경 감지
- COCO annotation validation과 external prediction JSON의 onboarding AP50 또는 공식 COCO AP@[.50:.95] 평가
- 분류 prediction record 기반 Top-1/Top-K·macro F1·confusion matrix 평가
- COCO·YOLO TXT·classification folder/CSV 경로 검사
- 승인된 local runner profile과 모의 보드 runner의 작업 상태·로그 관리
- 서비스 재시작 시 active job을 `interrupted`로 보존하고 자동 재실행하지 않는 복구 처리
- 실패·timeout·cancelled·interrupted 작업을 원본 입력 snapshot으로 명시적으로 재시도
- 명시적 metric 규칙을 사용하는 Release gate와 민감 경로를 제거한 결과 export
- 명시적 ONNX input/output profile을 요구하는 local CPU inference preview adapter
- 선택 의존성 환경에서 MMDetection native config·checkpoint preview adapter
- 선택 의존성 환경에서 MMDeploy runtime model directory preview adapter와 versioned target profile
- versioned external result manifest import와 외부 run ID의 idempotency/conflict 검사
- 데이터·모델·평가 결과의 명시적 lineage
- 등록 가능한 checkpoint/config 파일의 SHA-256 provenance 기록
- label schema, split, evaluation set, calibration set의 versioned contract와 content hash
- COCO annotation image/category/bbox 샘플 preview
- baseline/candidate의 호환성 검사와 metric delta 비교
- FastAPI `/api/v1`와 `visionops` CLI
- React/Vite 한국어 UI 골격
- MMDetection fixture와 RTMDet·YOLOX 온보딩 문서
- 결과 export API와 GitHub Pages용 정적 snapshot 경로
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

Storage mapping은 서버가 접근할 수 있는 경로를 등록하는 기능이다. 브라우저에서 고른 로컬 파일을 자동 업로드하지 않으며, 등록·검사만으로 외부 명령을 실행하지 않는다. Dataset 초안에 연결한 annotation/manifest의 내용이 바뀌면 prediction 평가를 중단하므로 새 버전을 만든 뒤 다시 확정한다.

## CLI 온보딩

UI를 열 수 없는 runner에서도 같은 registry를 사용할 수 있다. 경로 검사는 등록이나 외부 명령 실행을 하지 않는다.

```bash
visionops inspect /data/instances_val.json
visionops validate-coco /data/instances_val.json
visionops create-project project.json
visionops create-dataset PRJ-... dataset-version.json
visionops create-model PRJ-... model-version.json
visionops export PRJ-... --output pages/snapshot.json
```

`export` 결과는 API export와 같은 redaction 규칙을 사용한다. 원본·annotation·checkpoint의 절대 경로, 명령과 환경변수 이름은 포함하지 않는다.

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

원본 계획의 전체 시나리오는 아직 구현되지 않았다. 현재 코드에 근거한 지원 상태와 다음 개발 순서는 [Lifecycle 시나리오 점검 및 추가 개발 계획](docs/lifecycle-gap-analysis.md)을 기준으로 한다. 데이터 snapshot·label/split/calibration 버전, artifact 보존, 통합 평가·양자화 비교·Release 흐름이 남아 있다.

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
