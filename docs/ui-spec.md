# UI 규격

> 이 문서의 초기 UI 골격을 확장한 최신 규격은 [실습형 Lifecycle 통합 계획](guided-lifecycle-plan.md)의 §2–11을 따른다. 특히 초기 진입, CRUD/보관, 디렉터리 연결, 작업 로그, recipe 진행 상태는 해당 규격이 우선한다. 아래 demo 시작은 자동 사전 등록이 아닌 선택형 실습 시작으로 변경한다.

## Shell

- Desktop 기준 폭은 1440px이며 좌측 탐색, 상단 프로젝트 선택, 본문, 비교 바를 사용한다.
- 1024px 미만에서는 좌측 탐색을 접고, 모바일은 읽기·조회 작업에 집중한다.
- 모든 상태는 색상과 텍스트를 함께 제공한다: loading, empty, setup-required, partial, running, failed, static.

| Route / screen | Primary user goal | Key data | Main action |
|---|---|---|---|
| Overview | 현재 프로젝트 상태 파악 | dataset/model/run 수, lineage completeness, aliases | demo 시작, 비교 |
| Data | snapshot과 품질 확인 | format, split, classes, validation | 등록, version 확정 |
| Runs | 외부 결과 탐색 | inputs, config, metrics, environment | import, compare |
| Models | 모델 bundle 추적 | checkpoint, config, precision, source dataset | register, set alias |
| Evaluation | 성능·오류 분석 | metrics, per-class, error samples | evaluate, compare |
| Jobs & Board | 외부 runner 관리 | target, status, logs, output completeness | launch, cancel |
| Setup | 연결 정보 검증 | storage, git, plugins, runners | validate, copy agent prompt |
| Guide | 온보딩 수행 | prerequisite, example, recovery actions | start a recipe |

## Comparison

선택 가능한 모델은 2~4개이고 baseline은 하나다. 정확도 delta는 dataset version, annotation/class mapping, evaluator version이 같을 때만 보인다. Latency delta는 target profile, runtime, batch, warm-up, measurement scope도 같아야 한다. 불일치할 때는 수치 대신 원인과 해결 행동을 보여준다.

## Form and job behavior

저장과 연결 검사는 외부 명령을 실행하지 않는다. 실행 요청은 등록된 runner ID와 구조화된 인수만 전달한다. 오래 걸리는 요청은 Job을 생성하고 상태, 로그, exit code, output completeness를 표시한다.
