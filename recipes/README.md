# Recipe 계약

각 recipe는 처음 사용하는 사람이 빈 프로젝트에서 실제 등록·검증·평가를 따라가도록 하는 선언 파일이다. recipe는 shell 문자열을 실행하지 않으며, `actions`에는 API 경로 또는 프로젝트에 등록된 runner ID만 기록한다. 새 recipe를 추가할 때는 `id`, `version`, `task`, `prerequisites`, `steps`, `expected_evidence`, `troubleshooting`을 모두 채운다.

프런트엔드의 기본 선택 목록은 `/api/v1/recipes`에서 제공하고, 이 디렉터리의 파일은 문서·Agent가 단계별 입력과 완료 근거를 확인할 때 사용하는 기준이다. 실제 자료를 포함하지 않는 recipe는 `fixtures`와 `source`를 명시하고, synthetic/imported/measured 출처를 구분한다.
