# Vision Lifecycle 통합 개발 계획

> 원본 계획과 현재 코드의 시나리오 감사를 반영했다. 현재는 초기 골격이며 전체 Lifecycle 시나리오가 완료된 상태가 아니다. 상세 근거·우선순위·인수 기준은 [추가 개발 계획](docs/lifecycle-gap-analysis.md)에 정리한다.

## 최신 설계 기준: 빈 프로젝트에서 실습하며 연결하기

최신 사용자 요구사항을 반영한 전체 계획은 [실습형 Lifecycle 통합 계획](docs/guided-lifecycle-plan.md)을 따른다. 이 문서는 기존 코드 감사 결과를 포함하고, 제품의 진입 방식과 UI 구현 순서를 재구성한다.

- 첫 실행은 빈 화면과 새 프로젝트 안내다. 모델·데이터·성능 점수를 자동 등록하지 않는다.
- 내 자료 연결과 선택형 실습 recipe가 동일 UI/API를 사용한다. RTMDet-tiny와 YOLOX는 framework 종속 없는 제품 위에 추가하는 recipe다.
- 가이드를 따라 자료 연결→검사→데이터 확정→외부 학습/모델 연결→실제 평가→변경→재평가·비교를 수행한다.
- 수정/새 버전/보관·복원, 디렉터리 연결·검사·재연결, 작업 로그·취소·재시도와 이력을 각 단계에 포함한다.
- 진행 상태는 실제 entity/artifact/job 근거로 저장하고 재개한다. 미평가·unknown·예제·실측을 구분한다.
- 기존 사용자 자료와 데모는 보존한다. 기존 수기 데모 점수를 새로운 가이드 실행 결과로 취급하지 않는다.

### 구현 순서

U0 신뢰성 보수 → U1 빈 프로젝트·범용 CRUD UI → U2 디렉터리·데이터 버전 → U3 worker·로그·recipe engine → U4 모델 연결·실제 평가·변경 비교 → U5 RTMDet/YOLOX recipe → U6 양자화·보드·Release → U7 도입 문서·운영 검증.

현재 구현 진행: U0의 테스트 DB 격리, gate 방향·규칙 검증, 교차 프로젝트 참조 차단, 빈/recipe 프로젝트 생성, Dataset 초안 저장, 명시적 모델 데이터 선택, project/dataset/model 보관·복원 API와 기본 UI를 반영했다. U2의 첫 조각으로 읽기 전용 storage mapping 등록·재검사·root 내부 탐색과 root remap, DataAsset 파일 inventory·SHA-256, dataset annotation/manifest fingerprint, 원본 변경 시 prediction 평가 차단, Pages export의 경로·fingerprint 경로 제거를 반영했다. 모델 checkpoint/config SHA-256 provenance, 형식·정밀도·profile 선택과 샘플 inference preview, 평가 화면의 외부 COCO prediction 평가, Release 화면의 평가 run 기반 gate 판정, label schema/split/evaluation/calibration versioned contract와 UI, COCO annotation bbox preview, typed QuantizationRun·BoardBenchmark와 UI, Dataset·Model·Run·Quantization·Board·Release lineage graph와 리포트, recipe별 onboarding session/step evidence 저장·재개를 연결했다. onboarding 완료는 실제 Storage/Dataset/Model/Run/Release 근거가 없으면 거부한다. API queue와 별도 `visionops worker` 프로세스의 claim/실행 모드, SQLite backup/restore CLI를 추가했으며, 서비스 재시작 시 active job을 interrupted로 보존하고 자동 재실행하지 않는다. 실패·timeout·취소 작업은 UI/API에서 새 입력 snapshot으로 재시도할 수 있다. 대형 검사 Job은 다음 작업이다.

아래 A–G는 기술 작업 분류로 유지하며 실제 구현은 최신 통합 계획의 U0–U7에서 사용자 흐름별로 묶어 수행한다.

## 목표

외부 학습 시스템을 유지하면서 Vision AI 데이터, config, 모델, export·양자화, 평가, 보드 benchmark를 추적 가능한 lineage로 연결한다. 초기 지원 task는 classification과 bbox detection이며, 첫 완결형 예제는 MMDetection RTMDet-tiny와 YOLOX-s다.

## 제품 구조

- Local: React·TypeScript·Vite UI, FastAPI API, SQLite registry, local worker.
- Static: 선택한 안전한 JSON snapshot만 사용하는 GitHub Pages 조회 UI.
- Registry: Project, DatasetVersion, ModelVersion, Run, Job을 중심으로 확장한다.
- Model bundle: weight, config, preprocessing/postprocessing, class mapping, deployment metadata를 함께 보존한다.
- Run: training, export, quantization, evaluation, board benchmark를 별도 kind로 기록한다.

## 핵심 규칙

1. finalized data snapshot은 변경하지 않는다. 변경은 새 DatasetVersion이다.
2. 확정 비교에는 같은 dataset, annotation/class mapping version, evaluator version이 필요하다.
3. quantization loss와 target gap은 lineage가 확인된 단계에서만 계산한다.
4. 출처를 모르는 정보는 `unknown`으로 기록하고 추정하지 않는다.
5. 설정 저장이나 연결 검사는 외부 명령을 실행하지 않는다.
6. 외부 runner는 구조화된 command와 input/output contract로 연결한다.

## RTMDet·YOLOX 온보딩

선택형 recipe에서 빈 실습 프로젝트를 만든다. 사용자가 공통 COCO subset을 연결·검증하고 RTMDet-tiny config·checkpoint를 등록·평가한 뒤 YOLOX-s를 추가해 비교한다. native evaluation과 MMDeploy ONNX evaluation을 분리해 기록하고, 같은 evaluator 기준에서 비교한다. COCO category ID는 model label index와 같다고 가정하지 않는다. 두 모델과 점수가 미리 채워진 프로젝트를 기본 제품 흐름으로 사용하지 않는다.

## UI

상단에는 project selector와 상태, 좌측에는 개요·데이터·실험·모델·평가/비교·실행/보드·release/report·설정·가이드를 둔다. overview는 다음 행동을 안내하고, models/runs는 MLOps registry 패턴으로 검색·비교한다. evaluation은 metric에서 오류 사례와 lineage로 이동한다. 상세 규격은 `docs/ui-spec.md`에 둔다.

## 단계별 구현

1. Registry/API/CLI와 RTMDet·YOLOX metadata fixture. **부분 구현** — 공통 검증·artifact 보존·실모델 예제 필요.
2. COCO·YOLO·classification 경로 검사. **부분 구현** — 원본 hash·snapshot·label/split/evaluation/calibration 버전 관리 필요.
3. 외부 prediction 평가와 ONNX profile adapter. **부분 구현** — dataset 전체 추론·상세 결과 저장·동일 조건 비교 강화 필요.
4. runner·mock board·gate·report export. **부분 구현** — 독립 worker·복구·gate 정확성·공개 export 규격 보완 필요.
5. Pages workflow와 UI onboarding. **부분 구현** — 실제 export 직접 소비·wizard·각 관리 화면의 입력/실행 흐름 필요.
6. MMDetection/MMDeploy adapter와 COCO evaluator, target profile. **부분 구현** — 실모델 검증·보드 측정 계약·양자화 lineage 및 loss/gap 계산 필요.
7. vendor board recipe·외부 MLOps connector. **환경 의존 후속 확장** — 위 핵심 미구현 항목과 별도로 관리.

## 추가 개발 순서

1. A / P0: 테스트 DB 격리, 참조·비교·gate 검증, 공개 export 보수, 공통 service와 migration.
2. B / P1: DataAsset·Artifact·storage mapping, immutable dataset와 label/split/evaluation/calibration 버전 및 데이터 UI.
3. C / P1: 외부 학습·모델·양자화의 typed lineage, alias 이력, Quantization Loss/Target Gap·matrix 화면.
4. D / P1: dataset 전체 평가 Job, 예측/오류/per-class artifact, RTMDet·YOLOX 및 분류 예제·평가 UI.
5. E / P1: 독립 worker, 보드 결과 계약, 다중 근거 Release gate·승인 UI.
6. F / P1–P2: wizard·CLI·skill 도입, 백업/복구·Pages·최소 field feedback loop.
7. G / P2: 환경이 정해진 vendor/live connector와 선택적 운영 확장.

각 단계의 구체적 작업과 A01–A12 인수 시나리오는 상세 개발 계획을 따른다. 기존 수용 기준은 목표이며 현재 충족을 선언하지 않는다.

## 수용 기준

- 새 환경에서 문서만 따라 RTMDet·YOLOX demo를 등록·비교한다.
- 실제 config/checkpoint/dataset을 넣으면 provenance 누락과 class mapping 오류를 안내한다.
- 다른 evaluator 조건의 delta를 막고 이유를 표시한다.
- model/dataset/run의 lineage를 역추적할 수 있다.
- 실제 board 결과와 fixture 결과를 구분한다.
