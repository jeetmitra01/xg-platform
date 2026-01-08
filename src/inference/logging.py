import json
import time
from pathlib import Path
from typing import Any, Dict

LOG_DIR = Path("artifacts/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "inference.jsonl"

def log_event(record: Dict[str, Any]) -> None:
    record = dict(record)
    record["ts"] = time.time()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
