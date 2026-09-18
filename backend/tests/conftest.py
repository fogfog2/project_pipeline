"""Keep destructive schema setup inside a process-local test database."""

import os
from pathlib import Path


os.environ.setdefault("VISION_LIFECYCLE_DB", str(Path("/tmp") / f"vision-lifecycle-test-{os.getpid()}.db"))
