from __future__ import annotations

import threading
import time


class SnowflakeIdGenerator:
    """Small Snowflake-compatible ID generator for chunk and document metadata."""

    def __init__(self, worker_id: int = 1, datacenter_id: int = 1) -> None:
        self.worker_id = worker_id & 0x1F
        self.datacenter_id = datacenter_id & 0x1F
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()
        self.epoch_ms = 1704067200000

    def next_id(self) -> int:
        with self.lock:
            timestamp = self._timestamp_ms()
            if timestamp < self.last_timestamp:
                timestamp = self.last_timestamp
            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & 0xFFF
                if self.sequence == 0:
                    timestamp = self._wait_next_ms(timestamp)
            else:
                self.sequence = 0
            self.last_timestamp = timestamp
            return (
                ((timestamp - self.epoch_ms) << 22)
                | (self.datacenter_id << 17)
                | (self.worker_id << 12)
                | self.sequence
            )

    def next_id_str(self) -> str:
        return str(self.next_id())

    @staticmethod
    def _timestamp_ms() -> int:
        return int(time.time() * 1000)

    def _wait_next_ms(self, current: int) -> int:
        timestamp = self._timestamp_ms()
        while timestamp <= current:
            timestamp = self._timestamp_ms()
        return timestamp


snowflake = SnowflakeIdGenerator()
