# 원본 Lifecycle 계획 대비 구현 점검 및 추가 개발 계획

> 후속 UI 점검과 빈 프로젝트 기반 실습 요구사항은 [실습형 Lifecycle 통합 계획](guided-lifecycle-plan.md)에 통합했다. 본 문서는 코드 감사 근거와 A–G 기술 작업 분류로 유지하며, 최신 사용자 흐름과 구현 순서는 통합 계획을 따른다.

최신 보수: 프로젝트/Git/기준 브랜치 설정 편집, 버전 있는 recipe 계약과 recipe 조회 API, POSIX runner process-group 취소, cursor 기반 Job 로그 API와 WebSocket 스트림, overview의 저장된 근거별 readiness와 모델 provenance 누락 표시, Alembic baseline 및 `visionops migrate`, 다중 worker의 원자적 queue claim·lease owner·만료 시각·시도 횟수·만료 lease 재큐잉·실행 중 heartbeat 갱신과 작업 표 표시, API·CLI가 공유하는 외부 결과 manifest import service, COCO prediction·ONNX batch·classification prediction 평가의 외부 worker Job 경로, Release의 특정 evidence ID 필수 참조가 추가됐다. 아래 표의 항목은 여전히 전체 사용자 시나리오 기준의 남은 범위를 나타낸다.

점검 기준: `586b534` 커밋의 소스와 테스트. 기준 문서: `../On-device_Vision_AI_Lifecycle_Management_Plan.md` 전체 22개 절 및 사용자 후속 요구사항.

## 1. 결론과 범위

현재 시스템으로 원본 계획의 모든 시나리오를 수행할 수 없다. 현재 단계는 registry·평가기·연결 어댑터·대시보드의 초기 골격이다. 원본 데이터부터 승인된 모델까지 재현 가능한 lineage를 보장하는 운영 기준선은 아직 완성되지 않았다.

이전 `plan.md`의 일괄적인 **완료** 표시는 API나 함수 존재를 사용자 시나리오 완료로 해석한 것으로 수정이 필요하다. 남은 일이 vendor SDK나 외부 계정 연동뿐이라는 기존 설명도 정정한다. 핵심 누락 대부분은 실제 보드 없이 개발할 수 있다.

원본 문서 자체를 실행 지시로 취급하지 않는다. 이후 사용자가 확정한 범위를 적용한다.

- 학습·양자화·target 변환은 외부 수행, 이 시스템은 입력·설정·산출물·평가·승인 이력을 관리한다.
- Linux, SQLite, 로컬/NAS, CPU 추론을 기본으로 유지한다. PostgreSQL·DVC·MLflow는 필수 전제조건이 아니다.
- 실제 제품 배포는 수행하지 않는다. Release는 결정과 증거를 보관한다.
- 원본의 제품명·클래스·보드·수치 기준은 예시다.
- Field loop의 기본 이력 연결은 목표에 포함한다. annotation 편집기, embedding 기반 검색, active learning·drift 자동화, 팀 인증은 후속 단계다.

## 2. 점검 방법과 검증 범위

DB 모델, API, CLI, worker, evaluator, 추론 adapter, 프런트엔드, 예제와 기존 테스트를 읽고 사용자 흐름의 연결 여부를 확인했다. 이후 U0의 일부 보수와 빈 프로젝트 UI가 반영되었으며, 아래 표와 결함 목록은 전체 누락을 추적하는 기준이다.

기존 테스트는 `backend/tests/conftest.py`가 프로세스별 `/tmp` DB를 주입한 뒤 해당 테스트 DB에 `drop_all()`을 호출한다. 사용자 `.vision-lifecycle/registry.db`는 대상이 아니다. 현재 19개 테스트 통과는 P0 회귀와 storage/fingerprint fixture 검증이며 전체 시나리오 완료의 근거가 아니다.

DB를 사용하지 않는 gate 함수로 아래 문제를 직접 재현했다.

| 입력 | 현재 출력 | 필요한 동작 |
|---|---|---|
| 알 수 없는 규칙 `unsupported_rule` | U0에서 422/`GateConfigError` | 규격 오류로 거부 |
| latency 20 → 30ms, 허용 회귀 2ms | U0에서 `FAIL`, 회귀 10 | 낮을수록 좋은 지표 방향 적용 |

## 3. 원본 시나리오 대응표

부분 구현은 데이터 입력란·함수·API 일부가 있다는 뜻이며 전체 사용자 흐름 완료를 의미하지 않는다.

| ID | 원본 절 / 사용자 시나리오 | 현재 상태 | 남은 핵심 작업 / 코드 근거 |
|---|---|---|---|
| S01 | §4: 미라벨 원본 이미지·영상부터 등록 | 부분 구현 | Storage inventory와 DatasetVersion 등록, source fingerprint와 canonical snapshot, root 수정·보관/복원을 제공한다. 영상/metadata 상태와 item-level DataAsset 자동 결합은 남아 있다 |
| S02 | §3–4: immutable dataset, 이전 버전 재현·diff | 부분 구현 | finalized version immutable, parent version, COCO·YOLO TXT·classification folder/CSV·공통 JSONL snapshot과 content hash, category/image/annotation/item diff, 형식별 validator 통과 후 확정, 원본 변경 시 평가 차단을 제공한다. 모든 형식의 full manifest diff는 남아 있다 |
| S03 | §5: 클래스 분리·통합·폐기와 legacy 평가 | 부분 구현 | Dataset class mapping과 분류 prediction의 unknown label을 검증하고, LabelSchemaVersion 등록 시 class ID/name 중복과 mapping 대상 누락을 차단한다. parent와 `supersedes` lineage로 mapping history를 보존하며, ID 기준 added/removed/renamed/mapping diff API와 record 오류를 상세로 보존한다. 고정 class ID·계층·legacy 변환 규칙과 UI diff 화면은 남아 있다 |
| S04 | §6: 그룹 단위 split·누수 방지 | 부분 구현 | SplitVersion에 assignments/splits 계약, item 중복·unknown item·group leak·require_complete 미할당 검증과 UI 재검사를 제공한다. event/device/session metadata 자동 추출과 대규모 검증 job은 남아 있다 |
| S05 | §7: Core/Field/Hard/Regression 고정 평가 세트 | 부분 구현 | EvaluationSetVersion을 평가 API/Run.config에 명시적으로 연결하고 Dataset 불일치를 거부하며 비교 계약에 포함한다. definition의 `items`/`image_ids`를 분류·COCO·ONNX batch 입력에 실제 적용하고 선택 개수/평가 레코드 수를 결과에 남긴다. 생성·재검증 시 중복/누락/범위/빈 목록 상태도 저장한다. 완료 Run을 EvaluationSet별로 묶는 API·Pages snapshot·UI 리포트를 제공하며, 다중 set 승인과 slice별 리포트는 남아 있다 |
| S06 | §8: field 실패 사례를 다음 dataset으로 연결 | 부분 구현 | FieldDataBatch가 원본 모델·Dataset, 원본/예측/수정 label artifact, candidate DatasetVersion과 sample/failure 수를 연결하고 lineage에 표시한다. label 검수 workflow·candidate 승격 이력·개인정보 상태는 남아 있다 |
| S07 | §9: 외부 학습 결과 등록 | 부분 구현 | 실험 화면/API에서 typed `training` provenance(framework/commit/seed/split/label schema/unknown)와 Run.config/environment/details 및 외부 ID를 등록하고, 참조의 프로젝트·Dataset 일치를 검증한다. 외부 config 경로를 `training-config` 관리 artifact로 복사·hash하고 lineage에 연결하며, 동일 내용은 idempotent, 다른 내용은 충돌로 처리한다. 모델 등록 화면에서 외부 `training` Run을 명시적으로 선택해 연결한다. loss typed metric, parent 참조 검증은 남아 있다. `importer.py`, `main.py` |
| S08 | §10: model bundle·alias·ONNX provenance | 부분 구현 | 모델·config 파일의 SHA-256 provenance, 소유 entity, 원본과 관리 artifact 사본, 교체 시 superseded 이력, 재검증과 drift 시 inference 차단, alias 변경 사유·이력 조회, 원본 mount 부재 시 관리 사본 inference fallback을 제공한다. ONNX/MMDeploy profile이 없으면 runnable을 차단하고 adapter required 상태를 기록하며, ONNX profile 구조도 등록·수정 시 검증한다. ONNX metadata 삽입/검증은 남아 있다 |
| S09 | §11: 독립 calibration 버전·통계 | 부분 구현 | CalibrationSetVersion에 sample 수·중복·Dataset item·strategy/seed·전처리 선언 validation/statistics를 저장하고 재검사 API/UI를 제공한다. 연결된 COCO 이미지의 bounded decoded sample에 대해 실제 파일 누락·decode 오류·해상도·채널·pixel 평균/표준편차와 선언된 resize/color/normalization 적용 후 tensor 평균·표준편차를 `/inspect` API와 UI로 기록한다. 대규모 sampling job은 남아 있다 |
| S10 | §11: 양자화 matrix·encoding·QuantSim lineage | 부분 구현 | source/output model, calibration, encoding과 명시적 source/output role을 저장하고 matrix UI의 기본 입력을 제공한다. encoding 파일은 별도 artifact로 hash/managed copy를 보존하고 UI에서 재검증할 수 있으며, 중복 tensor 이름·NaN/무한대·잘못된 scale/range를 거부한다. method별 matrix와 세부 encoding 내용 schema 검증은 남아 있다 |
| S11 | §11: Quantization Loss·Target Gap 계산 | 부분 구현 | baseline·QuantSim·target 평가의 dataset/evaluator/protocol/scope/class mapping 계약을 검사해 Quantization Loss·Target Gap을 Run으로 저장한다. 다중 metric/critical class와 target 측정 범위 검증은 남아 있다 |
| S12 | §12: 분류·검출 공통 평가 | 부분 구현 | 외부 prediction 기반 분류, AP50, pycocotools bbox 평가와 invalid image/class/bbox/score 검사, 명시적 ONNX image record batch 평가가 존재한다. COCO prediction·ONNX batch·classification records 평가는 동기 API와 동일 계약의 외부 worker Job/API 경로를 제공한다. 고급 전처리/후처리 profile은 남아 있다 |
| S13 | §12: per-class/confusion/error/slice 보고서 | 부분 구현 | 분류 confusion/per-class와 COCO per-class/invalid 결과를 Run.details에 저장하고 재조회 가능하며, classification record의 명시적 confidence에 대해 ECE/bin과 `slice`/`slices`별 지표를 계산한다. micro 지표, 고정 Top-K 규약, detection slice·오류 파일·시각화 artifact는 남아 있다 |
| S14 | §12,17: baseline/candidate 공식 비교 | 부분 구현·정확성 보완 필요 | 완료된 평가 중 동일 dataset·전체 평가 config·evaluator·class mapping 계약을 만족하는 최신 pair를 선택하고 latency 계약을 별도 검사한다. 다중 세트·slice·전체 설정 hash/승인 이력은 남아 있다. `service.py:compare_models` |
| S15 | §13: target profile·외부 보드 결과 | 부분 구현 | target 행 등록·board import 연결과 metric 유한값/`source`·`scope`·batch·warmup·iterations·units 측정 계약 검증을 제공한다. TargetProfile에 hardware와 firmware/accelerator/runtime metadata를 별도 JSON으로 보존하고 UI에서 입력·조회한다. 보드 raw output 파일도 artifact로 hash/managed copy를 보존하고 재검증한다. board 결과에서 공통 평가 연결은 남아 있다 |
| S16 | §13: 외부 작업 실행·취소·복구 | 부분 구현 | subprocess runner와 독립 worker 모드, 재시작 복구, 전체 로그·exit code 표시, 외부 worker 재시도 queue 보존을 제공한다. API 재시작은 running만 `interrupted`로 바꾸고 queued는 claim 가능 상태로 둔다. queue claim은 worker owner·lease 만료 시각·attempt count를 기록하고 조건부 update로 중복 claim을 막으며 만료된 running lease는 다음 claim 전에 queue로 되돌린다. 실행 중 worker는 소유자 조건으로 lease를 heartbeat 갱신한다. WebSocket 로그 스트림은 cursor 기반 snapshot/delta와 terminal close를 제공하고 UI가 자동 연결한다. 취소 의도를 먼저 저장하고 POSIX process group을 종료하며 timeout도 하위 프로세스까지 정리한다. 인증과 연결 끊김 후 지수 backoff는 운영 확장 항목이다. `runner.py`, `main.py` |
| S17 | §14: 다단계 release gate·승인 | 부분 구현·정확성 보완 필요 | 단일 평가 scalar 규칙과 baseline dataset/evaluator/protocol/scope/class mapping 호환성 검사를 제공하며, `critical_classes`로 `details.per_class` 클래스별 minimum/maximum을 검사한다. Release 생성 시 evaluation/board/quantization/artifact evidence snapshot/hash, 종류별 required evidence와 `required_evidence_refs`의 특정 ID 누락 INCOMPLETE를 기록한다. PASS Release는 승인자·시각·근거를 immutable approval history와 AuditEvent에 기록하며 중복 승인을 차단한다. 다중 세트·인증 시스템 연동은 남아 있다 |
| S18 | §15,20: Production부터 원본까지 drill-down | 부분 구현 | Dataset·Model·Run 소유 artifact 노드와 `has_artifact` edge를 lineage API/화면에서 조회 가능. 양자화·Release 증거의 상세 drill-down과 변경 이력은 남아 있다 |
| S19 | 후속 요구: 처음 사용자 UI만으로 온보딩 | 부분 구현 | 프로젝트/모델/target 입력, 경로 검사, recipe별 저장·재개 wizard와 단계별 readiness/막힌 이유 표시를 제공한다. contracts 단계는 이제 LabelSchema·Split·EvaluationSet을 각각 요구하고 누락 목록을 표시한다. dataset 등록·확정, Git/storage/runner 편집, 평가 실행·release 폼의 통합 wizard 흐름은 남아 있다. `frontend/src/main.tsx` |
| S20 | 후속 요구: RTMDet·YOLOX 실제 예제 | 부분 구현·실모델 미검증 | COCO annotation·수기 예측 fixture 및 다운로드 recipe, 별도 classification/detection 실행 ONNX smoke fixture와 입력 records, 공식 checkpoint/config 경로·SHA-256·의존성 preflight CLI/API/UI를 제공한다. 이 환경에 공식 MMDetection/MMDeploy와 checkpoint가 없어 예제 이미지/실제 두 모델의 native→ONNX→평가 E2E는 아직 검증하지 못했다. MMDetection/MMDeploy 어댑터 존재와 실행 성공은 별개 |
| S21 | 후속 요구: API/CLI/agent 동일 서비스 | 부분 구현 | `schemas/v1` JSON Schema와 API/CLI schema registry, versioned adapter registry를 제공하고 agent가 같은 계약을 사용할 수 있다. LabelSchema 생성·diff도 `visionops` CLI에 연결했고, CLI의 Dataset/parent 프로젝트 참조 검증도 API와 맞췄다. API와 `visionops import-result`가 공유하는 result manifest service에서 검증·idempotency·lineage 참조·audit event를 동일하게 처리한다. 전체 registry 명령의 공통 service화와 전체 contract suite는 남아 있다 |
| S22 | 후속 요구: Pages 실제 결과 조회 | 부분 구현·정확성 보완 필요 | API/CLI export와 Pages workflow가 생성 시각·overview·lineage graph·artifact hash·평가 상세를 포함하고 경로/중첩 details의 path·command·secret 계열 키를 제거한다. 로컬 Release 화면은 redacted JSON·HTML 요약·Run CSV를 제공하며 Pages에서도 Dataset snapshot diff, lineage graph와 Release evidence snapshot을 조회한다. workflow_dispatch에서 demo 또는 커밋된 특정 export snapshot 경로를 선택하고 필수 키/redaction 계약을 검사한다. 중첩 자유값 전체 allowlist와 정적 화면의 상세 리포트는 추가 필요 |
| S23 | 후속 요구: 백업·복구·경로 이동 | 부분 구현 | CLI backup/restore가 SQLite integrity와 프로젝트·Storage ID·artifact hash manifest를 함께 검증하고, managed artifact 파일을 `<backup>.artifacts`에 함께 보관·복원하며 파일 hash를 확인한다. Alembic baseline 및 `visionops migrate`로 새 DB/기존 0.1 registry schema를 버전화한다. Storage root 이동 후 모든 참조 파일을 자동 재검증하고, 다른 머신으로 managed artifact root를 재배치하는 UX는 남아 있다 |

§1–3·15·17·19–21의 목표/아키텍처/수용 기준은 위 S01–S23의 통합 완료로 판단한다. §16의 추천 도구는 의무 설치 항목이 아니며 §18의 일정은 기존 예시로만 취급한다. §22 외부 참고 링크의 현재 제품 기능은 이번 코드 감사에서 재검증하지 않았다.

## 4. 먼저 해결해야 하는 결함

### P0: 결과 신뢰성과 사용자 자료 보존

1. **테스트 DB 격리**: 1차 해결. `backend/tests/conftest.py`가 프로세스별 `/tmp` DB를 주입하고 SQLite FK를 활성화했다. CI 병렬성과 migration 회귀는 추가 검증한다.
2. **잘못된 lineage 생성**: 1차 해결. UI가 dataset 첫 항목을 자동 연결하지 않고 사용자의 선택/unknown을 저장한다. 실제 content hash 검증은 남아 있다.
3. **참조 무결성**: 1차 해결. model/dataset/run/parent/result/release의 프로젝트 소속과 source 존재를 검사한다. API·CLI 공통 service 통합과 cycle/richer kind 검증은 남아 있다.
4. **비교·gate 공통 계약**: gate의 unknown section, 빈 rules, lower-is-better latency 회귀는 보수했다. protocol/IoU/label mapping/evaluation set 계약과 release baseline 호환성 검사는 남아 있다.
5. **정적 export 계약**: 상위 경로와 평가 details의 path·command·secret 계열 키를 제거하는 보수적 redaction을 1차 적용했다. 공개 필드 allowlist와 schema 검증, overview·생성시각·선택한 결과를 포함하는 규격, export snapshot 정적 UI 검증은 남아 있다.
6. **입력 오류를 성공으로 처리하지 않기**: bbox NaN/무한대·음수/0 크기, 알 수 없는 image/class, score 오류를 검사하고 최대 50건의 이유를 `Run.details`에 보존한다. 중복 prediction ID, classification 정답과 dataset 불일치, 공식 evaluator의 undefined category 구분은 남아 있다.

## 5. 추가 개발 순서와 완료 조건

작업 순서는 A → B → C → D → E → F이다. UI는 마지막에 몰아서 만들지 않고 각 단계에서 해당 API와 함께 완성한다. 외부 학습 시스템/실제 보드가 없는 경우에도 A–F의 관리·평가 흐름은 fixture와 외부 결과 manifest로 검증할 수 있어야 한다.

### A. 신뢰성 보수와 운영 기반 — P0

- 위 P0 1–6 수정, 테스트 DB 격리 후 회귀 테스트 추가.
- API/CLI 공통 service·Pydantic 규격 적용, migration 도입, JSON Schema 생성 및 schema_version 정책.
- 기본 dataset/model/run/target 이름·버전 중복 정책 및 import 트랜잭션 정의.
- 공개 snapshot 소비 규격과 read-only UI 정리, 로컬 UI build 정적 asset 제공 경로 검증.
- 완료: 기존 등록 DB 보존, 교차 프로젝트 참조 거부, 모든 불일치 비교 차단, 잘못된 gate PASS 방지, export JSON으로 API 없는 UI 동작.

### B. 재현 가능한 데이터와 artifact — P1

- DataAsset, Annotation, DatasetItem, StorageMapping, Artifact 추가. manifest는 storage ID + 상대 경로를 사용한다.
- image/video 원본 등록, metadata/label 상태, SHA-256 중복·손상·누락 검증. 기존 원본은 참조하되 annotation/config/model은 관리 artifact로 보존한다.
- dataset draft 편집 → 검증 → 확정 및 parent version, label/item/content hash, 추가·삭제·라벨 변경 diff.
- LabelClass/LabelSchemaVersion, class index↔category mapping, parent/legacy mapping history.
- SplitVersion, EvaluationSetVersion, CalibrationSetVersion 분리. event/device/session 그룹 누수 차단, sampling seed와 분포 기록.
- UI: 경로/브라우저 업로드 구분, 데이터 목록·이미지/annotation preview, mapping·split 검증 및 확정 화면.
- 완료: 미라벨 원본→label 추가→dataset v1→v2 생성 후 v1 재현. 파일 1바이트 변경 시 이전 snapshot 평가 차단. 클래스 분리 후 legacy 평가 가능. 그룹 중복 split 거부. storage 루트 이동 후 ID 유지.

### C. 외부 학습·모델·양자화 lineage — P1

- 공통 Run을 유지하되 kind별 typed payload와 input/output artifact 관계를 정의한다. entity를 무조건 별도 테이블로 나누기보다 참조·검증·조회 규격을 우선한다.
- TrainingRun에 dataset/label/split, commit/config/seed/loss/environment를 참조시키고 unknown을 필드별 표시한다.
- Model bundle에 hash, 실행 profile, class mapping, source training/export/quantization run을 연결한다. alias 전환은 별도 이력 테이블과 변경 사유로 보존하며, 관리 artifact 디렉터리와 ONNX metadata 검증을 추가한다.
- QuantizationRun에 source/output model, calibration version, weight/activation dtype, method/PCQ/clipping, encoding/converter를 연결한다.
- FP32/QuantSim/Target 역할을 명시하고 공통 evaluation contract로 Quantization Loss·Target Gap 계산. 역할·조건 누락 시 INCOMPLETE.
- lineage graph·양자화 experiment matrix·모델 상세 페이지 구현. ONNX provenance metadata 읽기/검증 및 새 export artifact에 기록.
- 완료: release 후보에서 원본 image·config·calibration·encoding까지 조회 가능. 외부 동일 ID 재등록은 idempotent, 다른 내용은 충돌. 이름/precision만으로 QuantSim을 추정하지 않는다.

### D. 통합 평가와 RTMDet·YOLOX 온보딩 — P1

- dataset batch 추론→표준 prediction→평가→artifact 저장의 단일 Job 흐름.
- versioned preprocessing/postprocessing: resize/letterbox 역변환, RGB/BGR, normalization, bbox 단위/좌표계, NMS, threshold, class mapping. 알 수 없는 출력은 adapter required로 표시.
- ONNX 기본 분류·검출 adapter와 MMDetection/MMDeploy profile을 검증하고 실행 provider/package 버전 기록.
- Top-K의 K, micro/macro/per-class 지표, confusion pair, slices, confidence calibration을 규격화. COCO AP/AP50/AP75/AR 및 undefined 값 처리 통일.
- predictions 입력은 hash artifact로 등록하고 metrics, per_class, confusion, invalid/error 상세는 `Run.details`에 저장해 API/UI에서 재조회. slice_metrics와 comparison HTML/CSV artifact는 후속 구현.
- 합성 이미지·실행 가능한 classification/detection ONNX fixture 제공. RTMDet-tiny·YOLOX-s 공식 모델 recipe는 다운로드 분리, 호환 환경 버전과 검증된 출력 profile 명시.
- UI: 평가 모델·세트·profile 선택, 미리보기 bbox 확인, 실행 job, 클래스 회귀·오류 사례 비교. 데이터셋과 calibration 버전 비교도 제공.
- 완료: 정답/오답/빈 검출/배경/invalid bbox/잘못된 mapping fixture 검증, 실제 두 모델 sample 검증 결과 별도 기록. synthetic 성공을 실제 RTMDet/YOLOX 성공으로 표시하지 않는다.

### E. 외부 작업·보드·Release — P1

- 독립 worker, DB queue claim/lease, 재시작 시 interrupted 상태, 자동 재실행 금지. job input/output contract, 전체 로그, exit code, POSIX process group 취소·timeout 및 비밀값 마스킹.
- runner 등록/검사/실행 UI, 업로드·다운로드는 명시적 실행 버튼으로 수행. 실행 파일과 구조화 인수·작업 폴더·환경변수 이름·timeout을 저장한다.
- TargetProfile의 hardware/OS/firmware/runtime/accelerator/execution version 규격화, BoardBenchmark에 측정 범위·batch·warmup·횟수·단위·raw output hash 기록.
- target prediction은 공통 평가기로, summary metric은 external measurement로 구분. mock worker 결과는 실제 성능에서 제외.
- 다중 평가 세트/critical class/양자화 손실/target gap/latency/메모리 등 근거를 고정한 ReleaseEvidence와 GateConfigVersion, 수동 결정·사유·시각·승인자 기록.
- UI: 결과 수집 상태·누락 항목·gate별 근거와 INCOMPLETE 사유. 승인 후 실제 제품 배포는 하지 않는다.
- 완료: 성공/실패/timeout/실행 전후 취소/API·worker 중단 테스트. 근거가 바뀌어도 과거 release 판정은 보존. 실제 보드 recipe는 target/SDK가 주어졌을 때 별도 인수 시험.

### F. 초보자 도입·운영·Field 이력 — P1/P2

- wizard 저장·재개: 프로젝트/task/storage → 자료 탐지 → class/split 매핑 → Git/외부 코드 → runner/skill → 소규모 평가 → 확정.
- 모든 등록·평가·import/export·backup/restore 명령을 공통 service 기반 CLI에 제공. adapter protocol과 버전 관리·contract fixture·템플릿 문서화.
- agent skill에 실제 지원 범위·필수 증거·unknown 처리·소규모 검증·등록 계획/결과를 연결. API/CLI를 우회한 DB 수정 금지.
- FieldDataBatch와 failure record를 등록해 새 dataset candidate로 연결하는 최소 feedback loop. 개인정보 상태·검증 label·source model 이력 포함.
- backup/restore는 SQLite와 manifest/artifact hash·storage mapping을 함께 검증. README를 빈 환경부터 실제 자료 연결까지 따라가기 검증.
- Pages는 가이드/데모 기본, 선택 프로젝트의 공개 결과만 export. 생성 시각·demo 표시·모든 버튼의 정적 동작 확인.
- 완료: 비개발자가 문서와 UI만으로 새 프로젝트를 등록→평가→리포트 출력. 백업 복원 및 경로 변경 후 lineage 유지. field 실패에서 새 candidate까지 역추적.

### G. 환경 의존 확장 — 선택적 P2

- 지정 보드의 ADB/SSH·vendor profiling recipe 및 실측 인수 시험.
- 필요해진 외부 MLOps 서비스의 connector와 인증/재시도/중복 처리.
- 팀 인증·공유 DB, annotation 편집기, perceptual/embedding 검색, active learning·drift 분석은 별도 범위로 산정한다.

## 6. UI 완료 기준

| 화면 | 구현해야 할 사용자 행동 | 주요 의존 단계 |
|---|---|---|
| 초기 연결·설정 | 프로젝트·Git·storage·runner 저장/수정/검사·재개, agent 요청문 | A/B/E/F |
| 데이터 | 원본/annotation preview, class·split·calibration 설정, snapshot 확정·diff | B |
| 실험·모델 | 외부 manifest 미리보기/import, 정확한 source 선택, unknown 및 alias 이력 | C |
| 양자화·보드 | matrix, calibration·encoding 확인, target profile, job 실행/취소/결과 수집 | C/E |
| 평가·비교 | 평가 job, profile preview, per-class/confusion/slice/오류, 조건 불일치 표시 | D |
| lineage·Release | 항목 drill-down, evidence 기반 gate, 결정 기록·HTML/JSON/CSV export | C/D/E |
| Pages | 로컬 API 없이 선택된 공개 snapshot 조회·가이드 이동 | A/F |

화면마다 loading/empty/partial/error/static 상태, 필드별 오류·해결 행동, 키보드 접근성과 반응형 레이아웃을 검증한다. 버튼과 제목만 있는 화면을 기능 완료로 표시하지 않는다.

## 7. 최종 인수 시나리오

| ID | 시험 | 통과 증거 |
|---|---|---|
| A01 | 미라벨 이미지 등록→label 추가→dataset v1/v2 | 원본 보존, 각 버전 manifest/hash와 diff |
| A02 | class 분리/통합·그룹 split | fine/legacy 결과, train/test 그룹 누수 거부 |
| A03 | 외부 학습 모델·설정 등록 | 파일 hash·commit/unknown·학습 데이터 관계, 교차 프로젝트 거부 |
| A04 | RTMDet·YOLOX 및 분류 추론 | 실행 가능한 fixture와 별도의 실모델 smoke 결과 |
| A05 | 평가 입력 이상치·원본 변경 | 잘못된 bbox/class/reference 차단, 내용 변경 시 실행 거부 |
| A06 | calibration v1/v2·양자화 설정 비교 | encoding/source/output 참조, 3단계 호환 결과일 때만 loss/gap |
| A07 | 불일치 비교·gate | 빈/실패/다른 조건 결과의 공식 delta 차단, unknown rule 거부, latency 방향 검증 |
| A08 | worker 생명주기 | 성공/실패/timeout/취소/재시작 interrupted와 로그·exit code |
| A09 | Release 결정 | 다중 세트·class·board 증거와 승인 이력의 immutable snapshot |
| A10 | Pages·백업·복구 | 중첩 secret·경로 제외, export 직접 소비, storage 이동 뒤 lineage 유지 |
| A11 | field feedback | 실패 사례→검증 label→dataset candidate→새 모델 관계 |
| A12 | README/CLI/skill 도입 | 신규 환경에서 UI/CLI 동일 결과·모든 필수 단계의 실제 실행 기록 |

각 단계는 구현 코드뿐 아니라 해당 인수 증거·사용법 문서·지원 범위 표시까지 완료되어야 닫는다. 현재 기능 수나 테스트 개수로 전체 완료율을 계산하지 않는다. 다음 실제 개발 착수점은 **A: 테스트 DB 격리와 잘못된 lineage/gate/export 방지**다.
