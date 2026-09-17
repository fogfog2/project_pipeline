# MLOps 레퍼런스와 채택 패턴

| Reference | Adopted pattern | Local implementation |
|---|---|---|
| MLflow Tracking | run 검색, metrics/params/artifacts 탐색 | Run 목록, config·metrics JSON, model/dataset 참조 |
| MLflow Model Registry | version, alias, source lineage | ModelVersion, baseline/candidate alias, source run/dataset |
| FiftyOne | data filtering and error drill-down | Dataset validation과 향후 sample overlay API |
| ClearML | model comparison and queued work | baseline compare endpoint, Job status/log model |
| MMDetection/MMDeploy | config/checkpoint/deploy bundle | MMDetection model bundle, ExportRun, RTMDet/YOLOX recipes |

이 프로젝트는 위 제품을 설치하거나 내부 DB 형식을 모방하지 않는다. 사용자에게 필요한 traceability, comparison, visualization, and runner state 패턴만 독립 entity와 API로 구현한다.
