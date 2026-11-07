"""Coordinate Azure OpenAI request pacing across local worker processes."""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from filelock import FileLock


class AzureRateLimiter:
    def __init__(self, endpoint: str, requests_per_minute: float | None = None):
        rpm = float(os.getenv("ROBOBENCH_AZURE_RPM", "60")) if requests_per_minute is None else requests_per_minute
        if rpm <= 0:
            raise ValueError("ROBOBENCH_AZURE_RPM must be positive")
        self.interval = 60.0 / rpm
        digest = hashlib.sha256(endpoint.rstrip("/").encode()).hexdigest()[:20]
        self.path = Path(tempfile.gettempdir()) / f"robobench-azure-{digest}.json"
        self.lock = FileLock(str(self.path) + ".lock")

    def _read(self, now: float) -> dict:
        try:
            state = json.loads(self.path.read_text())
            if now < state.get("updated", now) or now - state.get("updated", now) > 300:
                return {}
            return state
        except (FileNotFoundError, ValueError):
            return {}

    def _write(self, state: dict, now: float) -> None:
        state["updated"] = now
        self.path.write_text(json.dumps(state))

    def before_request(self, request) -> None:
        while True:
            now = time.monotonic()
            with self.lock:
                state = self._read(now)
                wait = max(state.get("next", now), state.get("blocked_until", now)) - now
                if wait <= 0:
                    state["next"] = now + self.interval
                    self._write(state, now)
                    return
            time.sleep(min(wait, 5.0))

    def after_response(self, response) -> None:
        headers = response.headers

        def number(name):
            try:
                return float(headers.get(name, ""))
            except (TypeError, ValueError):
                return None

        delays = []
        if response.status_code == 429:
            delays.append(max(number("retry-after") or 0,
                              (number("retry-after-ms") or 0) / 1000, 1))
        remaining_requests = number("x-ratelimit-remaining-requests")
        if remaining_requests is not None and remaining_requests <= 1:
            delays.append(number("x-ratelimit-reset-requests") or 60)
        remaining_tokens = number("x-ratelimit-remaining-tokens")
        if remaining_tokens is not None and remaining_tokens < 3000:
            delays.append(number("x-ratelimit-reset-tokens") or 60)
        if delays:
            now = time.monotonic()
            with self.lock:
                state = self._read(now)
                state["blocked_until"] = max(state.get("blocked_until", now), now + max(delays))
                self._write(state, now)
