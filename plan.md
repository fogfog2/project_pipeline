# Vision Lifecycle 구현 기준

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

공통 COCO subset을 DatasetVersion으로 등록하고, RTMDet-tiny와 YOLOX-s config·checkpoint를 각각 ModelVersion bundle로 등록한다. native evaluation과 MMDeploy ONNX evaluation을 분리해 기록하고, 같은 evaluator 기준에서 두 모델을 comparison한다. COCO category ID는 model label index와 같다고 가정하지 않는다.

## UI

상단에는 project selector와 상태, 좌측에는 개요·데이터·실험·모델·평가/비교·실행/보드·release/report·설정·가이드를 둔다. overview는 다음 행동을 안내하고, models/runs는 MLOps registry 패턴으로 검색·비교한다. evaluation은 metric에서 오류 사례와 lineage로 이동한다. 상세 규격은 `docs/ui-spec.md`에 둔다.

## 단계별 구현

1. Registry/API/CLI와 RTMDet·YOLOX fixture. **완료**
2. COCO·YOLO·classification 경로 검사와 validation. **완료**
3. external prediction import, detection AP50와 classification Top-K/F1 evaluator, 명시적 ONNX profile adapter. **완료**
4. runner worker, mock board contract, gate, 안전한 report export. **완료**
5. Pages workflow와 UI onboarding. **완료**
6. MMDetection native/MMDeploy runtime adapter, 전체 COCO evaluator, versioned target profile과 결과 manifest 연결. **완료**
7. 특정 vendor board recipe와 live external MLOps API connector는 실제 하드웨어·계정·SDK가 정해진 뒤 adapter로 추가한다. **환경 의존 후속 확장**

## 수용 기준

- 새 환경에서 문서만 따라 RTMDet·YOLOX demo를 등록·비교한다.
- 실제 config/checkpoint/dataset을 넣으면 provenance 누락과 class mapping 오류를 안내한다.
- 다른 evaluator 조건의 delta를 막고 이유를 표시한다.
- model/dataset/run의 lineage를 역추적할 수 있다.
- 실제 board 결과와 fixture 결과를 구분한다.
