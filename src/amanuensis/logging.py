"""JSON-lines structured logging so improvement-loop agents can read logs."""
import json
import sys
import time


def log(event: str, **fields) -> None:
    print(json.dumps({"ts": round(time.time(), 3), "event": event, **fields}), file=sys.stderr, flush=True)
