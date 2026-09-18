# Vision Lifecycle: 빈 프로젝트에서 시작하는 실습형 관리 시스템 계획

상태: 설계안. 이 문서는 후속 구현의 통합 기준이며 현재 구현 완료를 의미하지 않는다.
기존 요구사항은 `../On-device_Vision_AI_Lifecycle_Management_Plan.md`, 코드 감사 근거는 `lifecycle-gap-analysis.md`를 참고한다.

구현 현황(2026-09-18): U0/U1과 U2의 시작점을 반영했다. 빈 프로젝트/선택형 실습, 초안 등록·보관, storage mapping의 읽기 전용 탐색·재검사, DataAsset 파일 inventory, annotation/manifest fingerprint와 변경 감지, label/split/evaluation/calibration contract, COCO bbox preview, typed QuantizationRun·BoardBenchmark, lineage graph, 외부 COCO prediction 평가 및 Release gate UI까지는 동작한다. 서비스 재시작 시 active job을 interrupted로 보존하고 실패·timeout·취소·interrupted 작업을 명시적으로 재시도하는 처리도 동작한다. recipe 상태 엔진은 아직 계획 항목이다.

## 1. 제품 목표와 변경 방향

사용자는 빈 프로젝트를 만들고, 안내를 따라 자신이 가진 이미지·annotation·학습 결과·모델·config를 연결한다. 연결한 자료를 실제 검사·평가한 뒤 데이터나 모델, 설정을 변경해 결과를 비교하면서 시스템 사용법을 익힌다.

RTMDet-tiny·YOLOX가 미리 등록된 대시보드를 제품 기본 화면으로 사용하지 않는다. 모델 종류와 무관하게 같은 작업 흐름을 제공하고, 두 모델은 사용자가 선택할 수 있는 독립적인 실습 recipe로 제공한다. 예제를 완료한 사용자가 자신의 프로젝트에서도 같은 버튼과 절차를 이용할 수 있어야 한다.

학습·양자화·보드 측정은 외부 시스템에서 수행한다. 이 시스템은 입력과 산출물을 연결하고 로컬 추론·평가·비교·승인 이력을 관리한다. 학습 자체 실행과 실제 제품 배포는 범위에 포함하지 않는다.

## 2. 현재 UI 점검 결과

점검은 실행 중인 `http://127.0.0.1:5173`의 개요·데이터·실행/보드 화면과 프런트엔드/API 소스를 함께 확인했다. 자료 등록·작업 실행·삭제는 하지 않았다. 모바일 문제는 현재 CSS에 근거하며 별도 모바일 실기 검증은 하지 않았다.

| 현재 관찰 | 사용자에게 생기는 문제 | 설계 변경 |
|---|---|---|
| 사이드바에 MMDetection 데모 시작, 기존 데모 프로젝트·점수 표시 | 내 자료를 어디서부터 연결하는지 불명확 | 빈 시작 화면과 프로젝트 생성, 선택형 실습 센터 |
| 개요에 lineage 100% 표시 | config 경로와 dataset ID만 있어도 완전 추적처럼 보임 | 검증된 근거 항목별 완료/누락/unknown 표시 |
| 데이터 목록과 경로 검사만 제공 | 검사 후 등록·수정·확정으로 진행 불가 | 검사 결과에서 초안 생성→매핑→확정 흐름 |
| 모델 폼이 MMDetection 중심이고 첫 dataset 자동 선택 | 다른 모델 등록이 어렵고 잘못된 학습 이력 생성 | task/adapter 기반 폼, 사용자가 source 직접 선택 또는 unknown |
| 등록 항목의 상세·수정·삭제 버튼 없음 | 잘못 입력한 경로와 이름을 UI에서 고칠 수 없음 | 공통 상세 패널·초안 편집·새 버전·보관/복원 |
| 디렉터리를 문자열로 입력, 검사 결과는 JSON | 서버 경로/내 PC 경로 혼동, 실패 원인 파악 어려움 | storage 연결 카드와 범위 제한 탐색·구조화된 진단 |
| 작업 로그가 표의 한 칸, 자동 갱신·취소 UI 없음 | 진행/실패/중단 상태와 다음 행동을 알기 어려움 | 작업 상세 로그 뷰어·진행 상태·취소·재시도 |
| 설정은 agent 요청문 기능만 제공 | Git·경로·명령·skill 설정 관리 불가 | 탭별 설정과 연결 검사·변경 영향 안내 |
| 비교 결과는 상단 메시지, Release는 API 주소 표시 | 클래스별 개선·회귀를 탐색하거나 결정 기록 불가 | 결과 비교 화면과 증거 기반 Release 폼 |
| 메뉴 선택은 React 메모리 상태 | 새로고침·뒤로 가기·상세 링크 공유 불가 | project/entity ID를 포함한 URL routing |
| 1024px 이하 메뉴 글자 숨김, 680px 이하 메뉴 제거 | 작은 창에서 탐색 수단 상실 | 접근 가능한 메뉴 drawer·아이콘/레이블·키보드 탐색 |
| placeholder 중심 입력과 공통 메시지 한 개 | 필드 의미·오류·저장 여부가 불분명 | 고정 label, 필드별 오류, 저장/실행 상태 및 재시도 |

## 3. 첫 실행과 프로젝트 구조

첫 실행에는 데이터·모델·평가·성능 수치를 넣지 않는다. 프로젝트가 없으면 “아직 프로젝트가 없습니다”와 다음 행동을 표시한다. 안내를 보기 위해 가짜 프로젝트를 자동 생성하지 않는다.

시작 선택지는 세 가지다.

1. **내 프로젝트 시작**: 이름, 설명, task(미정/분류/검출), 작업 폴더를 지정하고 빈 프로젝트 생성.
2. **예제로 사용법 익히기**: recipe를 선택한 후 별도 실습 프로젝트 생성. 생성 시점에는 자료와 점수가 비어 있고 단계별 작업으로 채워진다.
3. **기존 프로젝트 열기**: 등록된 프로젝트 목록에서 검색·최근 사용·보관됨 필터로 선택.

실습 프로젝트와 실제 프로젝트는 같은 CRUD·평가 API를 사용한다. 실습 여부와 자료 출처를 명시하고 결과를 섞지 않는다. 사용자가 만든 프로젝트에는 RTMDet·YOLOX 이름이나 기본 baseline을 넣지 않는다. 새 프로젝트 생성은 프로젝트가 이미 있어도 항상 접근 가능해야 한다.

기존 DB의 데모와 사용자 자료는 보존한다. 이전 데모는 legacy demo로 표시하고 기존 수기 점수는 reference/imported example로 구분한다. 가이드 완료로 소급 인정하지 않으며 새 실습을 시작하려면 별도 빈 프로젝트를 만든다.

## 4. 안내와 실제 작업을 연결하는 단계

각 단계는 **목적 → 준비물 → 입력 → 연결/검사 → 실행 → 결과 → 다음 단계** 구조다. 체크박스를 누르는 것으로 완료되지 않고 저장된 entity/artifact/job 결과를 근거로 완료한다. 읽기 전용 가이드와 작업 폼을 같은 화면에서 연결한다.

| 단계 | 사용자가 하는 일 | 저장되는 근거 | 화면의 변화 |
|---|---|---|---|
| 0 프로젝트 | task·저장 위치 선택, 미정 허용 | Project, StorageMapping 초안 | 빈 개요에 다음 행동 표시 |
| 1 자료 연결 | 이미지·annotation 경로 선택 또는 업로드 | DataAsset·파일 inventory·검사 Job | 개수·샘플·누락/중복 표시 |
| 2 데이터 확정 | class mapping·split·평가 세트 구성 | LabelSchema/Split/Dataset/EvaluationSet 버전 | 데이터 통계와 확정 상태 표시 |
| 3 외부 학습 연결 | architecture·config·commit·학습 결과 입력 | TrainingRun·config artifact·unknown 목록 | 학습 구성 카드와 추적 관계 생성 |
| 4 모델 연결 | 모델 형식·adapter·전후처리·출력 매핑 선택 | ModelVersion·profile·model hash | 소규모 추론 결과와 bbox/score preview |
| 5 첫 평가 | evaluation set과 모델을 확인해 실행 | EvaluationRun·prediction·상세 metric artifact | 실제 계산된 점수가 처음 나타남 |
| 6 기준 지정 | 검증된 모델·평가를 baseline으로 선택 | Alias 이력·기준 evaluation ID | 기준 모델/평가 조건 고정 |
| 7 변경 실습 | 모델/설정/데이터 중 하나를 바꾸어 새 버전 생성 | Candidate와 입력 diff | 변경 전후 설정·자료 비교 |
| 8 재평가·비교 | 같은 평가 조건으로 candidate 평가 | 호환성 판정·comparison report | 정확도·클래스별 회귀·오류 사례 표시 |
| 9 양자화 연결(선택) | calibration·encoding·외부 양자화 결과 등록 | CalibrationSet·QuantizationRun·output model | 조건 충족 시 loss 계산 |
| 10 보드 결과(선택) | target·측정 조건·prediction/summary 연결 | BoardBenchmark·TargetProfile·artifact | target gap/latency와 측정 범위 표시 |
| 11 리포트·결정 | gate 설정과 증거 확인, 결과 내보내기 | ReleaseEvidence·결정·안전한 export | 재현 가능한 최종 비교 보고서 |

task 미정일 때도 원본 catalog 등록을 허용한다. 모델의 학습 데이터가 불명확하면 unknown으로 연결하고 외부 pretrained 모델 평가를 허용하되, 완전한 학습 lineage로 표시하지 않는다. calibration·board를 건너뛰면 해당 기능만 미완료로 남기고 기본 모델 평가 실습은 완료할 수 있다.

## 5. 변경·비교 실습 규칙

첫 실습은 원인 이해를 위해 하나의 변경 축을 선택하도록 안내한다. 이후 복합 변경도 허용하지만 모든 차이를 기록하고 성능 변화의 원인을 단정하지 않는다.

- **모델/학습 구조 변경**: RTMDet-tiny → YOLOX-s 또는 사용자 모델 A → B. architecture, input size, optimizer/loss/augmentation/config를 구조화 비교한다. 알 수 없는 값은 unknown이며 성능 개선을 보장하지 않는다.
- **추론 설정 변경**: threshold·NMS·resize 등을 새 profile로 저장한다. profile 차이를 보고서에 표시하고 동일 평가 계약에서 비교 허용되는 변경인지 판정한다. evaluator 임계값 자체를 바꾼 결과는 공식 delta에서 제외한다.
- **학습 데이터 변경**: dataset v1 → v2로 외부 학습한 결과를 연결한다. 변경 내용과 모델을 비교하되 고정 evaluation set을 유지한다. 이 시스템이 재학습을 실행한 것처럼 표시하지 않는다.
- **평가 데이터 변경**: 다른 evaluation set 결과는 나란히 탐색할 수 있으나 공식 delta는 계산하지 않는다. 공통 평가 세트로 재실행 버튼을 제공한다.
- **양자화 변경**: calibration/precision/method를 바꾼 외부 결과를 등록하고 FP32/QuantSim/Target 역할과 lineage를 검증한다.

baseline 지정은 후보 등록과 분리한다. 실습 중 새로운 모델을 연결했다고 기준을 자동 변경하지 않는다. 비교는 모델의 최신 실행을 임의 선택하지 않고 사용자가 확인한 evaluation ID를 고정한다.

## 6. 선택형 recipe와 예제 데이터

### 6.1 기본 recipe

- 가벼운 합성 분류: 생성 이미지와 작은 ONNX 모델로 등록→추론→평가→설정 변경 흐름을 CPU에서 검증.
- 가벼운 합성 검출: bbox·배경 이미지·오류 사례가 포함된 fixture로 동일 흐름 검증.
- MMDetection 실습: RTMDet-tiny를 연결하고 첫 평가 후 YOLOX-s를 추가해 실제 두 모델 비교. 공통 COCO subset과 모델별 명시적 profile 사용.
- 내 자료 연결: 사용자 경로/manifest/외부 결과를 동일 단계에 매핑. 특정 framework를 강제하지 않는다.

합성 fixture는 synthetic, 외부 저장 결과는 imported, 직접 실행 결과는 measured로 출처를 구분한다. 다운로드 없는 기본 실습과 공식 pretrained 모델 다운로드 실습을 구분한다. 다운로드 전에 용량·저장 위치·출처·라이선스 안내를 제공한다. 모델을 다운로드하거나 등록하는 것만으로 점수를 생성하지 않는다.

### 6.2 Recipe 규격

`recipes/<recipe-id>/recipe.yaml`에 recipe version, task, prerequisites, 자료 목록, 단계와 의존성, form schema, 허용 adapter/action ID, 기대 artifact/검증 조건, 문제 해결 안내를 둔다. 프런트엔드는 특정 family 문자열로 분기하지 않는다.

recipe는 임의 shell을 실행하는 문서가 아니다. action은 등록된 API/runner ID만 호출한다. 동일 API service와 검증기를 사용하며 가이드 전용 DB 쓰기는 만들지 않는다. agent도 같은 규격으로 새 recipe/adapter와 contract fixture를 추가한다.

## 7. 수정·삭제·복원 정책

| 대상 | 수정 방법 | 제거 방법과 영향 |
|---|---|---|
| 프로젝트 이름·설명·사용 편의 설정 | 직접 수정, 변경 이력 | 프로젝트 보관/복원. 원본 파일은 유지 |
| dataset/label/split/calibration 초안 | 수정·검사 후 저장 | 참조 없는 초안만 삭제 가능, 영향 목록 확인 |
| 확정 dataset/model/profile/target 구성 | 복제하여 새 버전, 변경 diff | 기본은 보관. 참조된 artifact와 과거 증거는 유지 |
| 모델 표시명·메모·alias | 표시 metadata 편집 또는 명시적 alias 전환 | production/baseline 참조 시 영향 확인·전환 후 보관 |
| 완료 Run·Evaluation·Release | 메모 추가, 재실행/새 결정으로 갱신 | 결과 덮어쓰기 금지. 조회 보관은 가능, 참조 증거 보존 |
| storage 연결 | 루트 재연결·hash 검증, 논리 ID 유지 | 연결 해제. 사용 중 참조와 영향 범위 안내 |
| runner 설정 | 초안 수정 후 version 확정 | 비활성화, 실행 중 job에는 시작 시 snapshot 사용 |
| 실행 중 job | 설정 수정 불가 | 취소 요청, 종료 확인 뒤 필요시 새 job 재시도 |

UI에서는 “보관”, “등록 해제”, “연결 해제”, “파일 영구 삭제”를 혼용하지 않는다. 기본 범위에 원본 파일 영구 삭제 기능은 넣지 않는다. 관리 artifact 정리는 참조 검사·보존 정책·별도 확인이 있는 후속 관리 기능이다. 삭제 확인에는 이름·ID·영향받는 모델/run 수·복원 가능 여부를 표시한다. 변경 충돌은 revision/ETag로 감지하고 새 값을 조용히 덮어쓰지 않는다.

## 8. 디렉터리 연결 UX

연결 화면은 이미지, annotation, 모델, config, calibration, 작업 산출물의 용도를 각각 표시한다. 연결마다 논리 storage ID, 실제 backend 경로, 읽기/쓰기 상태, 마지막 검사 시각, 사용 중 entity 수를 보여준다.

1. **서버 경로 참조**와 **브라우저 파일 업로드**를 먼저 구분한다. 다른 PC의 브라우저에서 선택한 폴더를 서버의 절대 경로로 오인하지 않는다.
2. 직접 경로 입력/붙여넣기와 서버 폴더 탐색을 제공한다. 탐색은 사용자가 연결한 허용 root 안에서만 수행하며 `..`·symlink 탈출을 차단한다.
3. 경로 존재·권한·파일 개수·형식·누락 annotation·샘플을 검사하고 결과를 표와 preview로 보여준다. 큰 디렉터리는 진행/취소가 가능한 검사 Job을 사용한다.
4. 성공 시 “이 자료로 초안 만들기”로 연결한다. 검사나 연결 저장만으로 다운로드/업로드/외부 명령을 실행하지 않는다.
5. NAS mount가 끊기면 연결 불가와 영향 항목을 표시하고 재검사/루트 재연결을 제공한다. 경로 변경만으로 새 데이터 버전을 만들지 않고 content hash 일치를 검증한다.

검사 오류는 경로 없음·권한 없음·빈 폴더·지원하지 않는 형식·annotation 불일치로 구분하며 해결 방법과 문제 파일을 표시한다. 기존 연결 편집과 최근 경로 재사용을 지원한다.

## 9. 작업 로그와 변경 이력

작업 목록은 이름/유형, 연결 대상, 상태, 시작/종료, 경과 시간, 진행률 또는 처리 개수, 실행자를 표시한다. 전체 로그를 표 한 칸에 넣지 않고 행 선택 시 작업 상세로 이동한다.

작업 상세 탭은 **요약 / 입력·설정 / 로그 / 결과 / 변경·재시도 관계**다. 로그에는 시각·단계·수준·stdout/stderr를 구분하며 검색, 수준 필터, 자동 스크롤 일시정지, 끝으로 이동, 복사·다운로드, 마지막 수신 시각을 제공한다. 진행률을 알 수 없으면 임의 퍼센트 대신 실행 중과 처리 건수를 표시한다.

로그는 cursor 기반 SSE 또는 polling으로 증분 수신하고 재접속 시 이어받는다. 네트워크 끊김과 job 실패를 구분하고 수동 새로고침을 제공한다. 대용량 로그는 제한된 페이지/가상 목록으로 렌더링한다. stdout에 나온 비밀값은 저장 전에 마스킹하며 Pages에는 원본 실행 로그를 기본 포함하지 않는다.

취소 버튼은 요청중→종료 상태를 구분한다. 재시도는 원래 Job을 수정하지 않고 입력 snapshot을 보여준 뒤 새 Job을 만든다. API/worker 재시작으로 중단된 작업은 interrupted로 표시하고 사용자가 재실행을 선택한다.

실행 로그와 별도로 audit 이력에 생성·수정·보관·복원·연결 변경·alias 전환·승인 기록 및 변경 전후 값을 남긴다. 초기 단일 사용자 환경에서는 local operator로 표시하고 인증된 사용자처럼 표현하지 않는다.

## 10. 공통 화면 규격

- 상단: 프로젝트 선택/새 프로젝트, task, 로컬/정적/실습 모드, 연결 상태, 실행 중 작업 수.
- 좌측: 시작 가이드, 개요, 데이터, 학습·모델, 양자화·보드, 평가·비교, 작업·로그, lineage·리포트, 설정.
- 본문: 제목·설명·현재 단계, 주요 행동 1개, 목록/상세. 안내 패널은 접을 수 있고 익숙한 사용자는 메뉴로 바로 작업한다.
- 상세 화면: 요약/입력/산출물/lineage/이력 탭과 ID 복사·관련 항목 이동·편집·새 버전·보관 메뉴.
- 목록: 검색·정렬·페이지 이동·상태/task/version 필터·선택 비교·보관됨 보기. 선택 일괄 작업은 항목별 실패/성공과 영향 확인을 제공한다.
- 입력: label·예시·필수 표시, 필드별 검증, 저장 중 중복 제출 방지, 저장 성공/실패, 미저장 변경 안내. 경로와 값이 긴 경우 줄바꿈/복사 지원.
- 빈 화면: 무엇이 없는지, 왜 필요한지, 연결할 자료, 첫 행동을 안내. 0점과 미평가를 구분하며 미평가는 “아직 평가하지 않음”으로 표시한다.
- 오류: 사용자 설명·해결 행동·재시도·오류 상세 펼치기. 하나의 전역 메시지로 모든 화면 상태를 덮지 않는다.
- routing: `#/projects/:id/...` 형태의 상세 URL로 새로고침/뒤로 가기/Pages 하위 경로 지원. 실제 구현 시 routing 선택은 이 동작을 충족해야 한다.
- 작은 화면: 메뉴 drawer와 접근 가능한 label, 2열→1열 전환, 표 내부 가로 스크롤. 키보드 focus, dialog focus 복원, 상태 live region 적용.

설정 탭은 프로젝트/Git, storage, adapters·환경 진단, 외부 명령, agent 연결, 백업·복구, 보관함으로 나눈다. Git URL·기준 브랜치는 탐지값을 확인하고 사용자가 수정할 수 있다. skill 경로 등록과 skill 실행을 구분한다.

## 11. 안내 진행 상태와 서비스 설계

추가 entity는 `OnboardingSession`, `RecipeVersion`, `StepProgress`, `AuditEvent`다. 진행 상태는 프로젝트·recipe version·단계별로 저장하고 사용자가 나중에 이어서 진행할 수 있다.

단계 상태는 not_started / ready / running / blocked / completed / skipped / needs_revalidation을 사용한다. completed에는 검증한 entity version, artifact hash, job ID를 기록한다. 데이터/profile이 바뀌면 영향을 받는 후속 단계만 재검증 필요로 표시하고 과거 결과를 삭제하지 않는다.

UI와 agent는 공통 service를 통해 `connect → inspect → draft → validate → finalize → run → compare`를 호출한다. 실습 설명을 닫거나 직접 API를 사용해도 저장된 근거로 단계 상태를 계산한다. step 완료 버튼만으로 validation을 우회할 수 없다.

추가 API 계약: draft PATCH/revision, archive/restore, dependency impact, storage browse/validate/remap, recipe/session/progress, Job log cursor/cancel/retry, comparison/evidence 조회. 기존 등록 API의 project/reference 검증부터 통합한다. 독립 worker·artifact store·Alembic과 데이터 버전 규격은 기존 감사 계획의 A–F를 따른다.

## 12. 개발 순서

| 단계 | 작업 | 사용자에게 제공되는 결과 | 의존성 |
|---|---|---|---|
| U0 / P0 | 테스트 DB 격리, lineage/gate/export 보수, demo provenance 정리 | 잘못된 연결·점수·승인 방지 | 감사 계획 A |
| U1 / P1 | 범용 shell·빈 화면·새 프로젝트·상세 routing·공통 폼, CRUD/보관/이력 | 예제 없이 자신의 프로젝트 관리 시작 | U0, 공통 service |
| U2 / P1 | storage 탐색·검사·재연결, dataset/label/split 초안→확정 및 preview | 첫 자료 연결 실습 완료 | U1, 감사 B |
| U3 / P1 | 독립 worker·작업 상세·로그/취소/복구, recipe/session engine, 합성 recipe | 안내를 따라 실제 작업 수행·재개 | U2, 감사 E의 worker 부분 |
| U4 / P1 | 외부 학습·모델/profile 연결, 첫 평가·artifact·baseline/candidate 비교 | 자신의 모델/데이터/설정 변경 학습 | U3, 감사 C/D |
| U5 / P1 | RTMDet-tiny→YOLOX 선택형 recipe, 실모델 sample 검증 | framework 예제로 동일 흐름 익히기 | U4, 별도 모델 실행 환경 |
| U6 / P1 | calibration·양자화·target·Release evidence 및 고급 비교 | 외부 결과 연결까지 확장 | U4, 감사 C/E |
| U7 / P1–P2 | README/skill, 백업/복구, Pages, field 이력·사용성 E2E | 예제에서 실제 프로젝트로 전환·운영 | U5/U6, 감사 F |

UI만 먼저 채우고 버튼을 비활성 상태로 남기는 방식으로 완료 판정하지 않는다. 각 단계는 API·저장·검증·UI·문서를 하나의 사용자 흐름으로 완료한다. 하드웨어가 없어도 U0–U4와 U6의 외부 manifest 흐름은 구현 가능하다. vendor별 실측 연동은 별도 인수 시험으로 남긴다.

## 13. 완료 검증

1. 빈 설치에서 프로젝트/모델/점수 자동 생성 없이 새 프로젝트를 만들 수 있다.
2. 프로젝트가 있는 상태에서도 다른 프로젝트 생성·전환·검색·보관·복원이 가능하다.
3. 실습 recipe를 골라도 해당 단계 실행 전 dataset/model/metric이 생성되지 않는다.
4. 서버 경로 참조와 브라우저 업로드를 구분하고 권한/누락 오류에서 해결 행동으로 이동한다.
5. draft 수정은 저장되고, 확정 데이터 수정은 새 버전이 되며 이전 모델의 snapshot은 유지된다.
6. 등록 해제/보관이 원본 파일을 삭제하지 않고 사용 중 참조를 숨기거나 끊지 않는다.
7. 사용자가 선택하지 않은 dataset을 모델의 학습 이력으로 연결하지 않는다.
8. 안내를 닫거나 새로고침·서비스 재시작 후에도 진행 상태와 근거를 이어서 확인한다.
9. 첫 평가가 완료되어야 점수가 표시되고 평가 실패는 0점/성공으로 표시되지 않는다.
10. 모델 구조/config/data의 변경 diff와 같은 evaluation set에서 얻은 실제 결과를 나란히 확인한다.
11. 조건이 다른 평가의 공식 delta는 차단되고 공통 조건 재평가 경로를 안내한다.
12. 작업 진행/실패/취소/timeout/interrupted와 로그·exit code·재시도 관계를 UI에서 확인한다.
13. 검색/필터/페이지 이동, 1440·1024·768·390px 탐색, 키보드 입력/대화상자 동작을 검증한다.
14. 정적 Pages는 demo/read-only/생성시각을 표시하고 로컬 경로 입력·실행·수정 버튼을 제공하지 않는다. 인터랙티브 실습은 로컬 설치 안내로 연결한다.
15. README만 따라 합성 recipe 완료 후 새 빈 프로젝트에서 자신의 자료를 연결한다. MMDetection 실습은 별도로 실제 RTMDet/YOLOX 실행 증거를 남긴다.

이번 변경은 계획 문서 작성까지다. 현재 DB·예제·UI의 실제 변경은 후속 구현에서 위 순서와 보존 정책에 따라 수행한다.
