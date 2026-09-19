# Database migrations

새 설치는 `visionops migrate` 또는 `alembic upgrade head`로 현재 registry schema를 생성한다. 기존 0.1.x SQLite 파일은 애플리케이션 시작 시 호환 보정을 먼저 적용할 수 있으며, 이후 Alembic revision으로 변경 이력을 관리한다.

```bash
VISION_LIFECYCLE_DB=.vision-lifecycle/registry.db alembic upgrade head
```

Migration은 사용자 원본·artifact 파일을 이동하거나 삭제하지 않는다. 운영 DB를 변경하기 전에 `visionops backup`을 실행한다.
